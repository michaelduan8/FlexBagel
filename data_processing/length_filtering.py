"""
filter_vlm_lengths.py

Measures per-example VLM sequence lengths for a multimodal dataset and
filters out examples that exceed a configurable token-length budget.

Supports Qwen2 / Qwen2.5-VL processors.
"""

import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import load_dataset, Dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Paths
    dataset_name: str = "traces/multimodal/pubmed_vision/pubmed_vision_train"
    input_dataset: str = field(init=False)
    output_dataset: str = field(init=False)
    removed_dataset: str = field(init=False)
    image_root: Optional[str] = None

    # Model
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Sampling (set to None to use the full dataset)
    sample_size: Optional[int] = None
    seed: int = 2026

    # Filtering: at least one must be set; if both are set, the stricter wins
    max_vlm_length: Optional[int] = 4096
    max_vlm_length_percentile: Optional[float] = None

    # Minimum image dimensions — examples with any image smaller than this are
    # removed before length measurement and saved separately
    min_image_width: int = 56
    min_image_height: int = 56

    # Output
    keep_length_columns: bool = True
    num_proc: int = 24
    map_batch_size: int = 128

    # Sentinel value written when length measurement fails
    bad_length: int = int(1e12)

    def __post_init__(self):
        self.input_dataset = f"{self.dataset_name}.jsonl"
        self.output_dataset = f"{self.dataset_name}_w_length.jsonl"
        self.removed_dataset = f"{self.dataset_name}_removed_long_examples.jsonl"
        self.small_image_dataset = f"{self.dataset_name}_removed_small_images.jsonl"

        if self.max_vlm_length is None and self.max_vlm_length_percentile is None:
            raise ValueError(
                "At least one of max_vlm_length or max_vlm_length_percentile must be set."
            )


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize_qwen(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """
    Mirror of Qwen2-VL / Qwen2.5-VL smart-resize logic.

    Returns the (height, width) the model would resize the image to,
    which determines the visual-token grid size.
    """
    if height < factor and width < factor:
        raise ValueError(
            f"Image too small: height={height}, width={width}, factor={factor}"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Extreme aspect ratio: height={height}, width={width}")

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)

    return int(h_bar), int(w_bar)


def resolve_image_path(
    img_item,
    image_root: Optional[str] = None,
    input_parent: Optional[Path] = None,
) -> Path:
    """Resolve an image reference to an absolute Path."""
    img_path = img_item.get("path") if isinstance(img_item, dict) else img_item

    if img_path is None:
        raise ValueError(f"Cannot resolve image path from {img_item!r}")

    img_path = Path(img_path)

    if img_path.is_absolute():
        return img_path
    if image_root is not None:
        return Path(image_root) / img_path
    if input_parent is not None:
        candidate = input_parent / img_path
        if candidate.exists():
            return candidate

    return img_path


def get_image_info(
    img_item,
    processor: AutoProcessor,
    image_root: Optional[str] = None,
    input_parent: Optional[Path] = None,
) -> tuple[int, int, int]:
    """
    Return (width, height, token_count) for a single image without loading pixel data.

    Reads only image dimensions, then applies Qwen's resize math to compute
    the patch grid token count.
    """
    img_path = resolve_image_path(img_item, image_root=image_root, input_parent=input_parent)

    with Image.open(img_path) as img:
        width, height = img.size

    ip = processor.image_processor
    patch_size = getattr(ip, "patch_size", 14)
    merge_size = getattr(ip, "merge_size", 2)
    min_pixels = getattr(ip, "min_pixels", 56 * 56)
    max_pixels = getattr(ip, "max_pixels", 28 * 28 * 1280)
    factor = patch_size * merge_size

    resized_h, resized_w = smart_resize_qwen(height, width, factor, min_pixels, max_pixels)

    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    grid_t = 1  # single frame (not video)
    token_count = int((grid_t * grid_h * grid_w) // (merge_size ** 2))

    return width, height, token_count


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def build_messages(example: dict) -> list[dict]:
    """
    Convert a dataset example into the chat-template message format used
    during training.  Mirrors the training-time convert_row() logic.
    """
    assert "images" in example, "Example must contain `images`."
    assert "conversation" in example, "Example must contain `conversation`."

    conversation = example["conversation"]
    images = example["images"]
    ex_id = example.get("id", "UNKNOWN")

    if conversation[-1]["role"] != "assistant":
        raise ValueError(f"Last turn is not `assistant` for id={ex_id}")

    prompt = []
    for turn in conversation[:-1]:
        role = turn["role"]
        text = turn["content"]
        img_loc = turn.get("img_loc")

        if role == "user" and img_loc is not None:
            text_part = [{"type": "text", "text": text}]
            image_parts = [{"type": "image"} for _ in images]

            if img_loc == "before":
                content = image_parts + text_part
            elif img_loc == "after":
                content = text_part + image_parts
            else:
                raise ValueError(f"Unsupported img_loc={img_loc!r} for id={ex_id}")
        else:
            content = [{"type": "text", "text": text}]

        prompt.append({"role": role, "content": content})

    completion = [{
        "role": "assistant",
        "content": [{"type": "text", "text": conversation[-1]["content"]}],
    }]

    return prompt + completion


def count_image_placeholders(messages: list[dict]) -> int:
    """Count `{"type": "image"}` parts across all turns."""
    return sum(
        1
        for msg in messages
        for part in msg["content"]
        if part["type"] == "image"
    )


# ---------------------------------------------------------------------------
# Image token cache
# ---------------------------------------------------------------------------

def build_image_token_cache(
    dataset: Dataset,
    processor: AutoProcessor,
    cfg: "Config",
    input_parent: Optional[Path],
) -> tuple[dict[str, int], set[str]]:
    """
    Pre-compute visual-token counts for every unique image path in the dataset.

    Uses a thread pool (I/O-bound) so image metadata reads run in parallel.

    Returns:
        image_token_cache: path str → token count (only for images that pass size check)
        undersized_paths:  set of path strs whose dimensions are below cfg minimums
    """
    unique_items: dict[str, Any] = {}
    for img_list in dataset["images"]:  # fast column access, no per-row overhead
        for img_item in img_list:
            key = str(resolve_image_path(img_item, cfg.image_root, input_parent))
            if key not in unique_items:
                unique_items[key] = img_item

    n_unique = len(unique_items)
    n_total = sum(len(imgs) for imgs in dataset["images"])
    print(f"Found {n_unique} unique images across {n_total} image references ({len(dataset)} examples)")

    image_token_cache: dict[str, int] = {}
    undersized_paths: set[str] = set()
    failed: list[str] = []

    def _process(key: str, img_item) -> tuple[str, int | None, bool]:
        """Returns (key, token_count_or_None, is_undersized)."""
        w, h, token_count = get_image_info(img_item, processor, cfg.image_root, input_parent)
        too_small = w < cfg.min_image_width and h < cfg.min_image_height
        return key, None if too_small else token_count, too_small

    with ThreadPoolExecutor(max_workers=cfg.num_proc) as pool:
        futures = {
            pool.submit(_process, key, img_item): key
            for key, img_item in unique_items.items()
        }
        for future in tqdm(as_completed(futures), total=n_unique, desc="Caching image info"):
            key = futures[future]
            try:
                _, token_count, too_small = future.result()
                if too_small:
                    undersized_paths.add(key)
                else:
                    image_token_cache[key] = token_count
            except Exception as exc:
                print(f"[ERROR] {key}: {exc!r}")
                failed.append(key)

    print(
        f"Cache built — valid: {len(image_token_cache)}, "
        f"undersized (<{cfg.min_image_width}x{cfg.min_image_height}): {len(undersized_paths)}, "
        f"errors: {len(failed)}"
    )
    return image_token_cache, undersized_paths


# ---------------------------------------------------------------------------
# Image-size filtering
# ---------------------------------------------------------------------------

def split_by_image_size(
    dataset: Dataset,
    undersized_paths: set[str],
    image_root: Optional[str],
    input_parent: Optional[Path],
) -> tuple[Dataset, Dataset]:
    """
    Split dataset into (valid, removed) based on whether any image in an
    example falls below the minimum dimensions.

    An example is removed if at least one of its images is in undersized_paths.
    """
    valid_idx, removed_idx = [], []
    for i, img_list in enumerate(dataset["images"]):
        has_small = any(
            str(resolve_image_path(img, image_root, input_parent)) in undersized_paths
            for img in img_list
        )
        (removed_idx if has_small else valid_idx).append(i)

    return dataset.select(valid_idx), dataset.select(removed_idx)


# ---------------------------------------------------------------------------
# Length measurement
# ---------------------------------------------------------------------------

def _append_bad(results: dict, error_msg: str, num_images: int, bad_length: int) -> None:
    results["vlm_length"].append(bad_length)
    results["vlm_text_length"].append(bad_length)
    results["vlm_image_token_estimate"].append(bad_length)
    results["vlm_num_images"].append(num_images)
    results["vlm_measurement_error"].append(error_msg)


def compute_vlm_lengths_batched(
    examples: dict,
    processor: AutoProcessor,
    image_token_cache: dict[str, int],
    bad_length: int,
    image_root: Optional[str],
    input_parent: Optional[Path],
) -> dict:
    """
    Estimate VLM token lengths for a batch of examples.

    Optimisations vs the single-example version:
      - apply_chat_template runs on the full batch in one call
      - tokenizer runs on the full batch in one call (major speedup)
      - image token counts are cache lookups — zero file I/O

    Formula per example:
        full_length = text_tokens + sum(image_tokens_i - 1)
    """
    batch_size = len(examples["images"])
    results: dict[str, list] = {
        "vlm_length": [], "vlm_text_length": [], "vlm_image_token_estimate": [],
        "vlm_num_images": [], "vlm_measurement_error": [],
    }

    # --- Step 1: build message lists, skip broken examples ---
    valid: list[tuple[int, list]] = []   # (original_index, messages)
    build_errors: dict[int, str] = {}

    for i in range(batch_size):
        example = {k: v[i] for k, v in examples.items()}
        try:
            valid.append((i, build_messages(example)))
        except Exception as exc:
            build_errors[i] = repr(exc)

    # --- Step 2: batch apply_chat_template + batch tokenize ---
    valid_indices = [i for i, _ in valid]
    valid_messages = [msgs for _, msgs in valid]

    texts: list[str] = processor.apply_chat_template(
        valid_messages, tokenize=False, add_generation_prompt=False
    )
    token_ids_batch: list[list[int]] = processor.tokenizer(
        texts, return_tensors=None, add_special_tokens=False
    )["input_ids"]

    text_len_by_idx = {idx: len(ids) for idx, ids in zip(valid_indices, token_ids_batch)}
    messages_by_idx = {i: msgs for i, msgs in valid}

    # --- Step 3: assemble per-example results using the cache ---
    for i in range(batch_size):
        example = {k: v[i] for k, v in examples.items()}
        num_images = len(example.get("images", []))

        if i in build_errors:
            _append_bad(results, build_errors[i], num_images, bad_length)
            continue

        try:
            messages = messages_by_idx[i]
            num_placeholders = count_image_placeholders(messages)

            if num_placeholders != num_images:
                raise ValueError(
                    f"Placeholder/image mismatch for id={example.get('id')}: "
                    f"placeholders={num_placeholders}, images={num_images}"
                )

            image_token_counts = []
            for img_item in example["images"]:
                key = str(resolve_image_path(img_item, image_root, input_parent))
                count = image_token_cache.get(key)
                if count is None:
                    raise ValueError(f"Image not in cache: {key}")
                image_token_counts.append(count)

            base_text_len = text_len_by_idx[i]
            full_len = base_text_len + sum(t - 1 for t in image_token_counts)

            results["vlm_length"].append(int(full_len))
            results["vlm_text_length"].append(int(base_text_len))
            results["vlm_image_token_estimate"].append(int(sum(image_token_counts)))
            results["vlm_num_images"].append(num_images)
            results["vlm_measurement_error"].append("")

        except Exception as exc:
            print(f"[ERROR] id={example.get('id', 'UNKNOWN')}: {exc!r}")
            _append_bad(results, repr(exc), num_images, bad_length)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_length_stats(lengths: list[int], title: str, bad_length: int) -> None:
    arr = np.asarray(lengths, dtype=np.int64)
    valid = arr[arr < bad_length]

    print(f"\n=== {title} ===")
    print(f"  total:   {len(arr)}")
    print(f"  valid:   {len(valid)}")
    print(f"  bad:     {len(arr) - len(valid)}")

    if len(valid) == 0:
        print("  (no valid lengths)")
        return

    for label, pct in [("mean", None), ("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99), ("p99.5", 99.5), ("max", None)]:
        if label == "mean":
            print(f"  mean:    {valid.mean():.1f}")
        elif label == "max":
            print(f"  max:     {valid.max()}")
        else:
            print(f"  {label}:    {np.percentile(valid, pct):.1f}")


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def load_dataset_from_path(path_or_name: str) -> Dataset:
    """Load a HuggingFace dataset from a local file or hub name."""
    if path_or_name.endswith((".jsonl", ".json")):
        return load_dataset("json", data_files=path_or_name, split="train")
    if path_or_name.endswith(".parquet"):
        return load_dataset("parquet", data_files=path_or_name, split="train")
    return load_dataset(path_or_name, split="train")


def save_dataset(dataset: Dataset, output_path: str) -> None:
    """Save a dataset to .jsonl or .parquet based on file extension."""
    if output_path.endswith(".parquet"):
        dataset.to_parquet(output_path)
    else:
        dataset.to_json(output_path, orient="records", lines=True, force_ascii=False)


def drop_length_columns(dataset: Dataset, columns: list[str]) -> Dataset:
    to_drop = [c for c in columns if c in dataset.column_names]
    return dataset.remove_columns(to_drop) if to_drop else dataset


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

LENGTH_COLUMNS = [
    "vlm_length",
    "vlm_text_length",
    "vlm_image_token_estimate",
    "vlm_num_images",
    "vlm_measurement_error",
]


def compute_length_cutoff(lengths: np.ndarray, cfg: Config) -> int:
    """Derive the token-length cutoff from config, applying the stricter threshold."""
    valid = lengths[lengths < cfg.bad_length]
    thresholds = []

    if cfg.max_vlm_length is not None:
        thresholds.append(cfg.max_vlm_length)

    if cfg.max_vlm_length_percentile is not None:
        p = cfg.max_vlm_length_percentile
        if not (0 < p <= 100):
            raise ValueError("max_vlm_length_percentile must be in (0, 100].")
        pct_cutoff = int(np.percentile(valid, p))
        print(f"Percentile cutoff p{p}: {pct_cutoff}")
        thresholds.append(pct_cutoff)

    return min(thresholds)


def split_by_length(dataset: Dataset, cutoff: int) -> tuple[Dataset, Dataset]:
    """
    Split dataset into (kept, removed) in a single pass.

    Uses dataset.select on a pre-built index list, which is faster than
    running two separate dataset.filter passes.
    """
    kept_idx, removed_idx = [], []
    for i, length in enumerate(dataset["vlm_length"]):
        (kept_idx if length <= cutoff else removed_idx).append(i)

    return dataset.select(kept_idx), dataset.select(removed_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config) -> None:
    input_parent = Path(cfg.input_dataset).parent if Path(cfg.input_dataset).exists() else None

    print(f"Loading processor: {cfg.model_name}")
    processor = AutoProcessor.from_pretrained(cfg.model_name, use_fast=True)

    print(f"Loading dataset: {cfg.input_dataset}")
    dataset = load_dataset_from_path(cfg.input_dataset)
    print(f"Dataset size: {len(dataset)}")

    if cfg.sample_size is not None and cfg.sample_size < len(dataset):
        print(f"Subsampling to {cfg.sample_size} examples...")
        dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.sample_size))

    print("Building image token cache...")
    image_token_cache, undersized_paths = build_image_token_cache(dataset, processor, cfg, input_parent)

    # --- Size filter (before length measurement to avoid wasted work) ---
    if undersized_paths:
        print(f"\nFiltering examples with images smaller than {cfg.min_image_width}x{cfg.min_image_height}...")
        dataset, removed_small = split_by_image_size(dataset, undersized_paths, cfg.image_root, input_parent)
        print(
            f"Size filter: {len(dataset) + len(removed_small)} → {len(dataset)} "
            f"(removed {len(removed_small)}, "
            f"{len(removed_small) / max(len(dataset) + len(removed_small), 1) * 100:.2f}%)"
        )
        print(f"Saving size-filtered examples → {cfg.small_image_dataset}")
        save_dataset(removed_small, cfg.small_image_dataset)
    else:
        print("No undersized images found — skipping size filter.")

    # --- Length measurement ---
    print("\nMeasuring VLM sequence lengths...")
    dataset = dataset.map(
        compute_vlm_lengths_batched,
        batched=True,
        batch_size=cfg.map_batch_size,
        num_proc=cfg.num_proc,
        fn_kwargs={
            "processor": processor,
            "image_token_cache": image_token_cache,
            "bad_length": cfg.bad_length,
            "image_root": cfg.image_root,
            "input_parent": input_parent,
        },
        desc="Measuring VLM sequence lengths",
    )

    lengths = np.asarray(dataset["vlm_length"], dtype=np.int64)
    print_length_stats(lengths, "Full dataset — text + image tokens", cfg.bad_length)

    if (lengths < cfg.bad_length).sum() == 0:
        raise RuntimeError("No valid examples after length measurement.")

    cutoff = compute_length_cutoff(lengths, cfg)
    print(f"\nApplying cutoff: {cutoff} tokens")

    kept, removed_long = split_by_length(dataset, cutoff)

    n_before, n_after = len(dataset), len(kept)
    print(
        f"Length filter: {n_before} → {n_after}  "
        f"(removed {n_before - n_after}, "
        f"{(n_before - n_after) / max(n_before, 1) * 100:.2f}%)"
    )
    print_length_stats(kept["vlm_length"], "Kept", cfg.bad_length)
    print_length_stats(removed_long["vlm_length"], "Removed (too long)", cfg.bad_length)

    if not cfg.keep_length_columns:
        kept = drop_length_columns(kept, LENGTH_COLUMNS)
        removed_long = drop_length_columns(removed_long, LENGTH_COLUMNS)

    print(f"Saving kept examples → {cfg.output_dataset}")
    save_dataset(kept, cfg.output_dataset)

    print(f"Saving length-filtered examples → {cfg.removed_dataset}")
    save_dataset(removed_long, cfg.removed_dataset)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a multimodal dataset by VLM token length.")
    parser.add_argument("dataset_name", help="Dataset name or path prefix (without .jsonl extension)")
    args = parser.parse_args()

    main(Config(dataset_name=args.dataset_name))