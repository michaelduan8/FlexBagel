"""
filter_vlm_lengths_memory_peak.py

Measures per-example VLM sequence lengths and estimates potential activation
memory pressure for Qwen2.5-VL-style multimodal training.

Compared with the original length-only script, this version tracks two separate
pressure sources:
  1. LLM-side sequence length after image-token merging.
  2. Vision-tower-side raw patch tokens before spatial merge.

It then estimates approximate activation-memory peaks for the LLM part, the
vision part, and max(LLM, vision), and can filter by any combination of:
  - max_vlm_length
  - max_vision_patch_tokens
  - max_memory_peak_gb
  - max_llm_peak_gb
  - max_vision_peak_gb
  - percentile versions of those cutoffs

The memory estimator is intentionally conservative and meant for dataset
filtering / ranking, not for exact allocator accounting. It uses Qwen2.5-VL-3B
architecture defaults, but will try to read compatible values from the provided
model config when possible.
"""

import argparse
import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import load_dataset, Dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Helpers for argparse
# ---------------------------------------------------------------------------

def none_or_int(x: Optional[str]) -> Optional[int]:
    if x is None or str(x).lower() in {"none", "null", ""}:
        return None
    return int(x)


def none_or_float(x: Optional[str]) -> Optional[float]:
    if x is None or str(x).lower() in {"none", "null", ""}:
        return None
    return float(x)


def str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in {"1", "true", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {x!r}")


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
    small_image_dataset: str = field(init=False)
    image_root: Optional[str] = None

    # Model
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Sampling (set to None to use the full dataset)
    sample_size: Optional[int] = None
    seed: int = 2026

    # Existing LLM-side VLM length filtering.
    # If both absolute and percentile are set, the stricter one is used.
    max_vlm_length: Optional[int] = 4096
    max_vlm_length_percentile: Optional[float] = None

    # New vision-side raw patch-token filtering.
    # This catches examples whose images are expensive inside the ViT even if
    # their post-merge LLM sequence length is not too large.
    max_vision_patch_tokens: Optional[int] = None
    max_vision_patch_tokens_percentile: Optional[float] = None

    # New memory-peak filtering. Units are approximate activation GB.
    # max_memory_peak_gb applies to max(LLM peak, vision peak).
    max_memory_peak_gb: Optional[float] = None
    max_memory_peak_percentile: Optional[float] = None
    max_llm_peak_gb: Optional[float] = None
    max_vision_peak_gb: Optional[float] = None

    # Minimum image dimensions — examples with any image smaller than this are
    # removed before length/memory measurement and saved separately.
    min_image_width: int = 56
    min_image_height: int = 56

    # Memory-estimation knobs.
    # moe_top_k is the number of active experts per token in your MoE variants.
    # For the original dense MLP, keep moe_top_k=1.
    moe_top_k: int = 1
    moe_num_experts: int = 2
    dtype_bytes: int = 2  # bf16/fp16 activation bytes
    assumed_batch_size: int = 1
    activation_checkpointing: bool = True
    attn_score_memory_factor: float = 0.0
    memory_safety_factor: float = 1.0

    # Output / performance
    stats_only: bool = False
    keep_length_columns: bool = True
    num_proc: int = 24
    map_batch_size: int = 128
    top_k_report: int = 20

    # Sentinel value written when measurement fails
    bad_length: int = int(1e12)
    bad_memory_gb: float = 1e12

    def __post_init__(self):
        self.input_dataset = f"{self.dataset_name}.jsonl"
        self.output_dataset = f"{self.dataset_name}_w_length_memory.jsonl"
        self.removed_dataset = f"{self.dataset_name}_removed_memory_or_long_examples.jsonl"
        self.small_image_dataset = f"{self.dataset_name}_removed_small_images.jsonl"

        has_any_filter = any(
            value is not None
            for value in [
                self.max_vlm_length,
                self.max_vlm_length_percentile,
                self.max_vision_patch_tokens,
                self.max_vision_patch_tokens_percentile,
                self.max_memory_peak_gb,
                self.max_memory_peak_percentile,
                self.max_llm_peak_gb,
                self.max_vision_peak_gb,
            ]
        )
        if not has_any_filter and not self.stats_only:
            raise ValueError(
                "Set at least one length/memory filter: max_vlm_length, "
                "max_vision_patch_tokens, max_memory_peak_gb, max_llm_peak_gb, "
                "max_vision_peak_gb, or a percentile counterpart. "
                "Use --stats_only true to inspect statistics without filtering."
            )

        if self.moe_top_k < 1:
            raise ValueError("moe_top_k must be >= 1")
        if self.moe_num_experts < self.moe_top_k:
            raise ValueError("moe_num_experts must be >= moe_top_k")
        if self.assumed_batch_size < 1:
            raise ValueError("assumed_batch_size must be >= 1")
        if self.dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")
        if self.attn_score_memory_factor < 0:
            raise ValueError("attn_score_memory_factor must be >= 0")
        if self.memory_safety_factor <= 0:
            raise ValueError("memory_safety_factor must be positive")


@dataclass
class Qwen25VLMemorySpec:
    # Qwen2.5-VL-3B text defaults
    text_hidden_size: int = 2048
    text_intermediate_size: int = 11008
    text_num_hidden_layers: int = 36
    text_num_attention_heads: int = 16
    text_num_key_value_heads: int = 2

    # Qwen2.5-VL-3B vision defaults
    vision_hidden_size: int = 1280
    vision_intermediate_size: int = 3420
    vision_depth: int = 32
    vision_num_heads: int = 16
    vision_out_hidden_size: int = 2048
    vision_patch_size: int = 14
    vision_spatial_merge_size: int = 2
    vision_window_size: int = 112
    vision_fullatt_block_indexes: tuple[int, ...] = (7, 15, 23, 31)

    @property
    def text_head_dim(self) -> int:
        return self.text_hidden_size // self.text_num_attention_heads

    @property
    def vision_window_patch_tokens_per_side(self) -> int:
        return max(1, self.vision_window_size // self.vision_patch_size)

    @property
    def vision_window_patch_tokens(self) -> int:
        s = self.vision_window_patch_tokens_per_side
        return s * s


def load_memory_spec(model_name: str) -> Qwen25VLMemorySpec:
    """
    Try to read Qwen2.5-VL dimensions from AutoConfig. If the config cannot be
    loaded, fall back to Qwen/Qwen2.5-VL-3B-Instruct defaults.
    """
    spec = Qwen25VLMemorySpec()
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        text_cfg = getattr(cfg, "text_config", cfg)
        vision_cfg = getattr(cfg, "vision_config", None)

        spec.text_hidden_size = int(getattr(text_cfg, "hidden_size", spec.text_hidden_size))
        spec.text_intermediate_size = int(getattr(text_cfg, "intermediate_size", spec.text_intermediate_size))
        spec.text_num_hidden_layers = int(getattr(text_cfg, "num_hidden_layers", spec.text_num_hidden_layers))
        spec.text_num_attention_heads = int(getattr(text_cfg, "num_attention_heads", spec.text_num_attention_heads))
        spec.text_num_key_value_heads = int(getattr(text_cfg, "num_key_value_heads", spec.text_num_key_value_heads))

        if vision_cfg is not None:
            spec.vision_hidden_size = int(getattr(vision_cfg, "hidden_size", spec.vision_hidden_size))
            spec.vision_intermediate_size = int(getattr(vision_cfg, "intermediate_size", spec.vision_intermediate_size))
            spec.vision_depth = int(getattr(vision_cfg, "depth", spec.vision_depth))
            spec.vision_num_heads = int(getattr(vision_cfg, "num_heads", spec.vision_num_heads))
            spec.vision_out_hidden_size = int(getattr(vision_cfg, "out_hidden_size", spec.vision_out_hidden_size))
            spec.vision_patch_size = int(getattr(vision_cfg, "patch_size", spec.vision_patch_size))
            spec.vision_spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", spec.vision_spatial_merge_size))
            spec.vision_window_size = int(getattr(vision_cfg, "window_size", spec.vision_window_size))
            fullatt = getattr(vision_cfg, "fullatt_block_indexes", spec.vision_fullatt_block_indexes)
            spec.vision_fullatt_block_indexes = tuple(int(x) for x in fullatt)

        print("Loaded memory spec from model config.")
    except Exception as exc:
        print(f"[WARN] Could not load AutoConfig for {model_name!r}: {exc!r}")
        print("[WARN] Falling back to Qwen2.5-VL-3B default memory spec.")

    print("\n=== Memory-estimation architecture spec ===")
    print(
        f"Text: layers={spec.text_num_hidden_layers}, hidden={spec.text_hidden_size}, "
        f"intermediate={spec.text_intermediate_size}, heads={spec.text_num_attention_heads}, "
        f"kv_heads={spec.text_num_key_value_heads}"
    )
    print(
        f"Vision: depth={spec.vision_depth}, hidden={spec.vision_hidden_size}, "
        f"intermediate={spec.vision_intermediate_size}, heads={spec.vision_num_heads}, "
        f"patch={spec.vision_patch_size}, merge={spec.vision_spatial_merge_size}, "
        f"window={spec.vision_window_size}, fullatt={spec.vision_fullatt_block_indexes}"
    )
    return spec


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

    Returns the resized (height, width), which determines the visual-token grid.
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


def read_image_size_fast(img_path: Path) -> tuple[int, int]:
    """
    Return (width, height) without decoding image pixels.
    For PNG, read IHDR directly so Pillow does not parse huge text metadata.
    Falls back to Pillow for non-PNG files.
    """
    with img_path.open("rb") as f:
        header = f.read(24)

    # PNG signature + IHDR chunk
    if (
        len(header) >= 24
        and header[:8] == b"\x89PNG\r\n\x1a\n"
        and header[12:16] == b"IHDR"
    ):
        width, height = struct.unpack(">II", header[16:24])
        return int(width), int(height)

    with Image.open(img_path) as img:
        return img.size


def get_image_info(
    img_item,
    processor: AutoProcessor,
    spec: Qwen25VLMemorySpec,
    image_root: Optional[str] = None,
    input_parent: Optional[Path] = None,
) -> dict[str, int]:
    """
    Return image dimensions and both post-merge LLM tokens and pre-merge ViT
    patch tokens, without loading pixel data.
    """
    img_path = resolve_image_path(img_item, image_root=image_root, input_parent=input_parent)
    width, height = read_image_size_fast(img_path)

    ip = processor.image_processor
    patch_size = int(getattr(ip, "patch_size", spec.vision_patch_size))

    # HF image processor may call this `merge_size`; model config calls it
    # `spatial_merge_size`. Prefer processor value when present because it is
    # what preprocessing will actually use.
    merge_size = int(
        getattr(
            ip,
            "merge_size",
            getattr(ip, "spatial_merge_size", spec.vision_spatial_merge_size),
        )
    )
    min_pixels = int(getattr(ip, "min_pixels", 56 * 56))
    max_pixels = int(getattr(ip, "max_pixels", 28 * 28 * 1280))
    factor = patch_size * merge_size

    resized_h, resized_w = smart_resize_qwen(height, width, factor, min_pixels, max_pixels)

    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    grid_t = 1  # single image, not video

    vision_patch_tokens = int(grid_t * grid_h * grid_w)
    llm_image_tokens = int(vision_patch_tokens // (merge_size ** 2))

    window_side = max(1, spec.vision_window_size // patch_size)
    num_windows = int(math.ceil(grid_h / window_side) * math.ceil(grid_w / window_side) * grid_t)

    return {
        "width": int(width),
        "height": int(height),
        "resized_width": int(resized_w),
        "resized_height": int(resized_h),
        "grid_t": int(grid_t),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "llm_image_tokens": int(llm_image_tokens),
        "vision_patch_tokens": int(vision_patch_tokens),
        "vision_windows": int(num_windows),
    }


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def build_messages(example: dict) -> list[dict]:
    """
    Convert a dataset example into the chat-template message format used
    during training. Mirrors the training-time convert_row() logic.
    """
    assert "images" in example, "Example must contain `images`."
    assert "conversation" in example, "Example must contain `conversation`."

    conversation = example["conversation"]
    images = example["images"]
    ex_id = example.get("id", "UNKNOWN")

    if conversation[-1]["role"] != "assistant":
        raise ValueError(f"Last turn is not `assistant` for id={ex_id}")

    prompt = []
    used_image = False
    for turn in conversation[:-1]:
        role = turn["role"]
        text = turn["content"]
        img_loc = turn.get("img_loc")

        content = [{"type": "text", "text": text}]
        if role == "user" and img_loc is not None and images and not used_image:
            image_parts = [{"type": "image"} for _ in images]
            if img_loc == "before":
                content = image_parts + content
            elif img_loc == "after":
                content = content + image_parts
            else:
                raise ValueError(f"Unsupported img_loc={img_loc!r} for id={ex_id}")
            used_image = True

        prompt.append({"role": role, "content": content})

    completion = [{
        "role": "assistant",
        "content": [{"type": "text", "text": conversation[-1]["content"]}],
    }]

    return prompt + completion


def count_image_placeholders(messages: list[dict]) -> int:
    """Count `{\"type\": \"image\"}` parts across all turns."""
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
    cfg: Config,
    spec: Qwen25VLMemorySpec,
    input_parent: Optional[Path],
) -> tuple[dict[str, dict[str, int]], set[str]]:
    """
    Pre-compute image dimensions/tokens for every unique image path.

    Returns:
        image_token_cache: path str -> image info dict
        undersized_paths: set of path strs whose dimensions fail min size check
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

    image_token_cache: dict[str, dict[str, int]] = {}
    undersized_paths: set[str] = set()
    failed: list[str] = []

    def _process(key: str, img_item) -> tuple[str, dict[str, int] | None, bool]:
        info = get_image_info(img_item, processor, spec, cfg.image_root, input_parent)
        # Treat either dimension below threshold as undersized.
        too_small = info["width"] < cfg.min_image_width or info["height"] < cfg.min_image_height
        return key, info, too_small

    with ThreadPoolExecutor(max_workers=cfg.num_proc) as pool:
        futures = {
            pool.submit(_process, key, img_item): key
            for key, img_item in unique_items.items()
        }
        for future in tqdm(as_completed(futures), total=n_unique, desc="Caching image info"):
            key = futures[future]
            try:
                _, info, too_small = future.result()
                if too_small:
                    undersized_paths.add(key)
                    # In stats-only mode, keep undersized images in the cache so
                    # the dataset can be measured without actually filtering rows.
                    if cfg.stats_only:
                        assert info is not None
                        image_token_cache[key] = info
                else:
                    assert info is not None
                    image_token_cache[key] = info
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
    example is in undersized_paths.
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
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_llm_peak_gb(seq_len: int, cfg: Config, spec: Qwen25VLMemorySpec) -> tuple[float, float]:
    """
    Estimate LLM activation memory pressure for one example, scaled by
    cfg.assumed_batch_size.

    Returns:
        (peak_gb, explicit_attention_score_gb)

    Notes:
      - moe_top_k scales MLP activation because k experts are active per token.
      - If activation_checkpointing=True, estimate saved layer inputs plus one
        layer's compute peak. Otherwise, estimate retained per-layer activations.
      - attn_score_memory_factor controls how much explicit T^2 attention-score
        storage to include. Use 0 for flash-like kernels, 1 for conservative
        math-attention-like accounting.
    """
    t = int(seq_len) * cfg.assumed_batch_size
    h = spec.text_hidden_size
    inter = spec.text_intermediate_size
    layers = spec.text_num_hidden_layers
    heads = spec.text_num_attention_heads
    kv_heads = spec.text_num_key_value_heads
    head_dim = spec.text_head_dim
    b = cfg.dtype_bytes

    # Q projection is full hidden; K/V are smaller with GQA unless the runtime
    # explicitly repeats KV to all attention heads.
    qkv_elems = t * (h + 2 * kv_heads * head_dim)
    attn_out_elems = t * h
    residual_norm_elems = 4 * t * h

    # SwiGLU-ish FFN: gate/up/product/down-side saved activations. This is a
    # filtering heuristic, not an exact autograd graph inventory.
    mlp_elems = cfg.moe_top_k * 3 * t * inter

    per_layer_linear_elems = qkv_elems + attn_out_elems + residual_norm_elems + mlp_elems
    attn_score_elems = heads * t * t

    per_layer_peak_bytes = b * (
        per_layer_linear_elems + cfg.attn_score_memory_factor * attn_score_elems
    )

    if cfg.activation_checkpointing:
        saved_inputs_bytes = b * layers * t * h
        peak_bytes = saved_inputs_bytes + per_layer_peak_bytes
    else:
        peak_bytes = layers * per_layer_peak_bytes

    peak_gb = cfg.memory_safety_factor * peak_bytes / GB
    attn_score_gb = cfg.memory_safety_factor * b * attn_score_elems / GB
    return float(peak_gb), float(attn_score_gb)


def estimate_vision_peak_gb(
    image_infos: list[dict[str, int]],
    merged_image_tokens: int,
    cfg: Config,
    spec: Qwen25VLMemorySpec,
) -> tuple[float, float, float]:
    """
    Estimate vision-tower activation memory pressure for one example.

    Returns:
        (peak_gb, full_attention_score_gb, window_attention_score_gb)
    """
    batch = cfg.assumed_batch_size
    p = int(sum(info["vision_patch_tokens"] for info in image_infos)) * batch
    windows = int(sum(info["vision_windows"] for info in image_infos)) * batch

    h = spec.vision_hidden_size
    inter = spec.vision_intermediate_size
    depth = spec.vision_depth
    heads = spec.vision_num_heads
    b = cfg.dtype_bytes

    # Window attention uses fixed-size windows. This estimate uses padded window
    # token count, so it slightly overestimates incomplete border windows.
    window_tokens = spec.vision_window_patch_tokens
    window_attn_score_elems = heads * windows * (window_tokens ** 2)

    # Full-attention blocks attend over each image/frame sequence. Sum over
    # images is a better pressure proxy than only max image length.
    full_attn_score_elems = heads * batch * sum(
        int(info["vision_patch_tokens"]) ** 2 for info in image_infos
    )

    qkv_elems = 3 * p * h
    attn_out_elems = p * h
    residual_norm_elems = 4 * p * h
    mlp_elems = cfg.moe_top_k * 3 * p * inter

    # Include a small merger/output term: raw patches are reduced to LLM-side
    # visual tokens and projected to text hidden size.
    merger_elems = batch * int(merged_image_tokens) * spec.vision_out_hidden_size

    per_block_linear_elems = qkv_elems + attn_out_elems + residual_norm_elems + mlp_elems + merger_elems
    window_block_bytes = b * (
        per_block_linear_elems + cfg.attn_score_memory_factor * window_attn_score_elems
    )
    full_block_bytes = b * (
        per_block_linear_elems + cfg.attn_score_memory_factor * full_attn_score_elems
    )
    per_block_peak_bytes = max(window_block_bytes, full_block_bytes)

    if cfg.activation_checkpointing:
        saved_inputs_bytes = b * depth * p * h
        peak_bytes = saved_inputs_bytes + per_block_peak_bytes
    else:
        # Most blocks are window attention; only len(fullatt) blocks use full attention.
        n_full = len(spec.vision_fullatt_block_indexes)
        n_window = max(0, depth - n_full)
        peak_bytes = n_window * window_block_bytes + n_full * full_block_bytes

    peak_gb = cfg.memory_safety_factor * peak_bytes / GB
    full_attn_score_gb = cfg.memory_safety_factor * b * full_attn_score_elems / GB
    window_attn_score_gb = cfg.memory_safety_factor * b * window_attn_score_elems / GB
    return float(peak_gb), float(full_attn_score_gb), float(window_attn_score_gb)


# ---------------------------------------------------------------------------
# Length + memory measurement
# ---------------------------------------------------------------------------

def _append_bad(results: dict, error_msg: str, num_images: int, cfg: Config) -> None:
    int_bad_cols = [
        "vlm_length",
        "vlm_text_length",
        "vlm_image_token_estimate",
        "vlm_vision_patch_tokens",
        "vlm_vision_windows",
        "vlm_max_image_width",
        "vlm_max_image_height",
        "vlm_num_images",
    ]
    float_bad_cols = [
        "vlm_est_llm_peak_gb",
        "vlm_est_vision_peak_gb",
        "vlm_est_peak_gb",
        "vlm_est_llm_attn_scores_gb",
        "vlm_est_vision_full_attn_scores_gb",
        "vlm_est_vision_window_attn_scores_gb",
    ]
    for col in int_bad_cols:
        results[col].append(cfg.bad_length if col != "vlm_num_images" else num_images)
    for col in float_bad_cols:
        results[col].append(cfg.bad_memory_gb)
    results["vlm_measurement_error"].append(error_msg)


def compute_vlm_lengths_and_memory_batched(
    examples: dict,
    processor: AutoProcessor,
    image_token_cache: dict[str, dict[str, int]],
    cfg: Config,
    spec: Qwen25VLMemorySpec,
    image_root: Optional[str],
    input_parent: Optional[Path],
) -> dict:
    """
    Estimate VLM token lengths plus LLM/vision activation-memory pressure for a
    batch of examples.
    """
    batch_size = len(examples["images"])
    results: dict[str, list] = {
        "vlm_length": [],
        "vlm_text_length": [],
        "vlm_image_token_estimate": [],
        "vlm_vision_patch_tokens": [],
        "vlm_vision_windows": [],
        "vlm_max_image_width": [],
        "vlm_max_image_height": [],
        "vlm_num_images": [],
        "vlm_est_llm_peak_gb": [],
        "vlm_est_vision_peak_gb": [],
        "vlm_est_peak_gb": [],
        "vlm_est_llm_attn_scores_gb": [],
        "vlm_est_vision_full_attn_scores_gb": [],
        "vlm_est_vision_window_attn_scores_gb": [],
        "vlm_measurement_error": [],
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

    if valid_messages:
        texts: list[str] = processor.apply_chat_template(
            valid_messages, tokenize=False, add_generation_prompt=False
        )
        token_ids_batch: list[list[int]] = processor.tokenizer(
            texts, return_tensors=None, add_special_tokens=False
        )["input_ids"]
    else:
        token_ids_batch = []

    text_len_by_idx = {idx: len(ids) for idx, ids in zip(valid_indices, token_ids_batch)}
    messages_by_idx = {i: msgs for i, msgs in valid}

    # --- Step 3: assemble per-example results using the image cache ---
    for i in range(batch_size):
        example = {k: v[i] for k, v in examples.items()}
        num_images = len(example.get("images", []))

        if i in build_errors:
            _append_bad(results, build_errors[i], num_images, cfg)
            continue

        try:
            messages = messages_by_idx[i]
            num_placeholders = count_image_placeholders(messages)

            if num_placeholders != num_images:
                raise ValueError(
                    f"Placeholder/image mismatch for id={example.get('id')}: "
                    f"placeholders={num_placeholders}, images={num_images}"
                )

            image_infos = []
            for img_item in example["images"]:
                key = str(resolve_image_path(img_item, image_root, input_parent))
                info = image_token_cache.get(key)
                if info is None:
                    raise ValueError(f"Image not in cache: {key}")
                image_infos.append(info)

            base_text_len = int(text_len_by_idx[i])
            merged_image_tokens = int(sum(info["llm_image_tokens"] for info in image_infos))
            raw_patch_tokens = int(sum(info["vision_patch_tokens"] for info in image_infos))
            vision_windows = int(sum(info["vision_windows"] for info in image_infos))
            full_len = base_text_len + sum(int(info["llm_image_tokens"]) - 1 for info in image_infos)

            llm_peak_gb, llm_attn_score_gb = estimate_llm_peak_gb(full_len, cfg, spec)
            vision_peak_gb, vision_full_score_gb, vision_window_score_gb = estimate_vision_peak_gb(
                image_infos=image_infos,
                merged_image_tokens=merged_image_tokens,
                cfg=cfg,
                spec=spec,
            )
            total_peak_gb = max(llm_peak_gb, vision_peak_gb)

            results["vlm_length"].append(int(full_len))
            results["vlm_text_length"].append(int(base_text_len))
            results["vlm_image_token_estimate"].append(int(merged_image_tokens))
            results["vlm_vision_patch_tokens"].append(int(raw_patch_tokens))
            results["vlm_vision_windows"].append(int(vision_windows))
            results["vlm_max_image_width"].append(int(max(info["width"] for info in image_infos)))
            results["vlm_max_image_height"].append(int(max(info["height"] for info in image_infos)))
            results["vlm_num_images"].append(int(num_images))
            results["vlm_est_llm_peak_gb"].append(float(llm_peak_gb))
            results["vlm_est_vision_peak_gb"].append(float(vision_peak_gb))
            results["vlm_est_peak_gb"].append(float(total_peak_gb))
            results["vlm_est_llm_attn_scores_gb"].append(float(llm_attn_score_gb))
            results["vlm_est_vision_full_attn_scores_gb"].append(float(vision_full_score_gb))
            results["vlm_est_vision_window_attn_scores_gb"].append(float(vision_window_score_gb))
            results["vlm_measurement_error"].append("")

        except Exception as exc:
            print(f"[ERROR] id={example.get('id', 'UNKNOWN')}: {exc!r}")
            _append_bad(results, repr(exc), num_images, cfg)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_value_stats(values: list[Any], title: str, bad_value: float | int) -> None:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[arr < bad_value]

    print(f"\n=== {title} ===")
    print(f"  total:   {len(arr)}")
    print(f"  valid:   {len(valid)}")
    print(f"  bad:     {len(arr) - len(valid)}")

    if len(valid) == 0:
        print("  (no valid values)")
        return

    for label, pct in [("mean", None), ("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99), ("p99.5", 99.5), ("max", None)]:
        if label == "mean":
            value = valid.mean()
        elif label == "max":
            value = valid.max()
        else:
            value = np.percentile(valid, pct)
        print(f"  {label:<6} {value:.4f}")


def print_top_examples(dataset: Dataset, column: str, n: int, bad_value: float | int) -> None:
    if n <= 0 or column not in dataset.column_names:
        return

    values = np.asarray(dataset[column], dtype=np.float64)
    valid_idx = np.where(values < bad_value)[0]
    if len(valid_idx) == 0:
        return

    top_idx = valid_idx[np.argsort(values[valid_idx])[-n:]][::-1]
    print(f"\n=== Top {min(n, len(top_idx))} examples by {column} ===")
    header = (
        "rank\tid\tpeak_gb\tllm_gb\tvision_gb\tvlm_len\tvision_patches\t"
        "img_tokens\tnum_images\tmax_wh"
    )
    print(header)
    for rank, idx in enumerate(top_idx, start=1):
        ex = dataset[int(idx)]
        ex_id = ex.get("id", ex.get("prompt_id", int(idx)))
        print(
            f"{rank}\t{ex_id}\t"
            f"{ex.get('vlm_est_peak_gb', float('nan')):.4f}\t"
            f"{ex.get('vlm_est_llm_peak_gb', float('nan')):.4f}\t"
            f"{ex.get('vlm_est_vision_peak_gb', float('nan')):.4f}\t"
            f"{ex.get('vlm_length', 'NA')}\t"
            f"{ex.get('vlm_vision_patch_tokens', 'NA')}\t"
            f"{ex.get('vlm_image_token_estimate', 'NA')}\t"
            f"{ex.get('vlm_num_images', 'NA')}\t"
            f"{ex.get('vlm_max_image_width', 'NA')}x{ex.get('vlm_max_image_height', 'NA')}"
        )


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
    "vlm_vision_patch_tokens",
    "vlm_vision_windows",
    "vlm_max_image_width",
    "vlm_max_image_height",
    "vlm_num_images",
    "vlm_est_llm_peak_gb",
    "vlm_est_vision_peak_gb",
    "vlm_est_peak_gb",
    "vlm_est_llm_attn_scores_gb",
    "vlm_est_vision_full_attn_scores_gb",
    "vlm_est_vision_window_attn_scores_gb",
    "vlm_measurement_error",
    "vlm_filter_reason",
]


def compute_numeric_cutoff(
    values: np.ndarray,
    bad_value: float | int,
    absolute: Optional[float | int],
    percentile: Optional[float],
    label: str,
) -> Optional[float]:
    """Derive a cutoff from absolute and/or percentile thresholds."""
    thresholds: list[float] = []
    valid = values[np.asarray(values, dtype=np.float64) < float(bad_value)]

    if absolute is not None:
        thresholds.append(float(absolute))

    if percentile is not None:
        if not (0 < percentile <= 100):
            raise ValueError(f"{label} percentile must be in (0, 100].")
        if len(valid) == 0:
            raise RuntimeError(f"No valid values for {label}; cannot compute percentile cutoff.")
        pct_cutoff = float(np.percentile(valid, percentile))
        print(f"Percentile cutoff for {label} p{percentile}: {pct_cutoff:.4f}")
        thresholds.append(pct_cutoff)

    if not thresholds:
        return None
    return min(thresholds)


def build_filter_cutoffs(dataset: Dataset, cfg: Config) -> dict[str, float]:
    """Build all active cutoffs. An example must satisfy all of them."""
    cutoffs: dict[str, float] = {}

    cutoff = compute_numeric_cutoff(
        np.asarray(dataset["vlm_length"], dtype=np.float64),
        cfg.bad_length,
        cfg.max_vlm_length,
        cfg.max_vlm_length_percentile,
        "vlm_length",
    )
    if cutoff is not None:
        cutoffs["vlm_length"] = cutoff

    cutoff = compute_numeric_cutoff(
        np.asarray(dataset["vlm_vision_patch_tokens"], dtype=np.float64),
        cfg.bad_length,
        cfg.max_vision_patch_tokens,
        cfg.max_vision_patch_tokens_percentile,
        "vlm_vision_patch_tokens",
    )
    if cutoff is not None:
        cutoffs["vlm_vision_patch_tokens"] = cutoff

    cutoff = compute_numeric_cutoff(
        np.asarray(dataset["vlm_est_peak_gb"], dtype=np.float64),
        cfg.bad_memory_gb,
        cfg.max_memory_peak_gb,
        cfg.max_memory_peak_percentile,
        "vlm_est_peak_gb",
    )
    if cutoff is not None:
        cutoffs["vlm_est_peak_gb"] = cutoff

    cutoff = compute_numeric_cutoff(
        np.asarray(dataset["vlm_est_llm_peak_gb"], dtype=np.float64),
        cfg.bad_memory_gb,
        cfg.max_llm_peak_gb,
        None,
        "vlm_est_llm_peak_gb",
    )
    if cutoff is not None:
        cutoffs["vlm_est_llm_peak_gb"] = cutoff

    cutoff = compute_numeric_cutoff(
        np.asarray(dataset["vlm_est_vision_peak_gb"], dtype=np.float64),
        cfg.bad_memory_gb,
        cfg.max_vision_peak_gb,
        None,
        "vlm_est_vision_peak_gb",
    )
    if cutoff is not None:
        cutoffs["vlm_est_vision_peak_gb"] = cutoff

    return cutoffs


def add_filter_reasons_batched(examples: dict, cutoffs: dict[str, float]) -> dict:
    """Vector-light batched filter reason computation for HF Dataset.map.

    The previous implementation materialized several full columns as Python lists and
    then called add_column. On large Arrow datasets this can look like a hang right
    after printing `=== Active cutoffs ===`. This batched version gives a map
    progress bar and avoids building one giant reason list in the driver process.
    """
    batch_size = len(examples["vlm_measurement_error"])
    reasons = []

    for i in range(batch_size):
        row_reasons = []
        err = examples["vlm_measurement_error"][i]
        if err:
            row_reasons.append("measurement_error")

        for col, cutoff in cutoffs.items():
            try:
                value = float(examples[col][i])
            except Exception:
                row_reasons.append(f"{col}=unreadable")
                continue
            if value > cutoff:
                row_reasons.append(f"{col}>{cutoff:.4f}")

        reasons.append(";".join(row_reasons))

    return {"vlm_filter_reason": reasons}


def split_by_cutoffs(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Split using the filter reason column, with visible progress."""
    kept_idx, removed_idx = [], []
    reasons = dataset["vlm_filter_reason"]
    for i, reason in enumerate(tqdm(reasons, desc="Building kept/removed indices")):
        (kept_idx if reason == "" else removed_idx).append(i)

    print(f"Selecting kept rows: {len(kept_idx)}", flush=True)
    kept = dataset.select(kept_idx)
    print(f"Selecting removed rows: {len(removed_idx)}", flush=True)
    removed = dataset.select(removed_idx)
    return kept, removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config) -> None:
    input_parent = Path(cfg.input_dataset).parent if Path(cfg.input_dataset).exists() else None

    print(f"Loading processor: {cfg.model_name}")
    processor = AutoProcessor.from_pretrained(cfg.model_name, use_fast=True)
    spec = load_memory_spec(cfg.model_name)

    print("\n=== Memory-estimation knobs ===")
    print(
        f"moe_top_k={cfg.moe_top_k}, moe_num_experts={cfg.moe_num_experts}, "
        f"dtype_bytes={cfg.dtype_bytes}, assumed_batch_size={cfg.assumed_batch_size}, "
        f"activation_checkpointing={cfg.activation_checkpointing}, "
        f"attn_score_memory_factor={cfg.attn_score_memory_factor}, "
        f"memory_safety_factor={cfg.memory_safety_factor}, "
        f"stats_only={cfg.stats_only}"
    )

    print(f"\nLoading dataset: {cfg.input_dataset}")
    dataset = load_dataset_from_path(cfg.input_dataset)
    print(f"Dataset size: {len(dataset)}")

    if cfg.sample_size is not None and cfg.sample_size < len(dataset):
        print(f"Subsampling to {cfg.sample_size} examples...")
        dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.sample_size))

    print("Building image token cache...")
    image_token_cache, undersized_paths = build_image_token_cache(dataset, processor, cfg, spec, input_parent)

    # --- Size filter (before length measurement to avoid wasted work) ---
    if undersized_paths:
        if cfg.stats_only:
            print(
                f"\nStats-only mode: found {len(undersized_paths)} unique images smaller than "
                f"{cfg.min_image_width}x{cfg.min_image_height}; not filtering or saving them."
            )
        else:
            print(f"\nFiltering examples with images smaller than {cfg.min_image_width}x{cfg.min_image_height}...")
            dataset, removed_small = split_by_image_size(dataset, undersized_paths, cfg.image_root, input_parent)
            print(
                f"Size filter: {len(dataset) + len(removed_small)} -> {len(dataset)} "
                f"(removed {len(removed_small)}, "
                f"{len(removed_small) / max(len(dataset) + len(removed_small), 1) * 100:.2f}%)"
            )
            print(f"Saving size-filtered examples -> {cfg.small_image_dataset}")
            save_dataset(removed_small, cfg.small_image_dataset)
    else:
        print("No undersized images found — skipping size filter.")

    # --- Length + memory measurement ---
    print("\nMeasuring VLM sequence lengths and memory estimates...")
    dataset = dataset.map(
        compute_vlm_lengths_and_memory_batched,
        batched=True,
        batch_size=cfg.map_batch_size,
        num_proc=cfg.num_proc,
        fn_kwargs={
            "processor": processor,
            "image_token_cache": image_token_cache,
            "cfg": cfg,
            "spec": spec,
            "image_root": cfg.image_root,
            "input_parent": input_parent,
        },
        desc="Measuring VLM lengths + memory",
    )

    print_value_stats(dataset["vlm_length"], "Full dataset — LLM-side text + merged image tokens", cfg.bad_length)
    print_value_stats(dataset["vlm_image_token_estimate"], "Full dataset — merged image tokens inserted into LLM", cfg.bad_length)
    print_value_stats(dataset["vlm_vision_patch_tokens"], "Full dataset — raw ViT patch tokens before merge", cfg.bad_length)
    print_value_stats(dataset["vlm_est_llm_peak_gb"], "Estimated LLM activation peak GB", cfg.bad_memory_gb)
    print_value_stats(dataset["vlm_est_vision_peak_gb"], "Estimated vision activation peak GB", cfg.bad_memory_gb)
    print_value_stats(dataset["vlm_est_peak_gb"], "Estimated max(LLM, vision) activation peak GB", cfg.bad_memory_gb)

    if (np.asarray(dataset["vlm_length"], dtype=np.int64) < cfg.bad_length).sum() == 0:
        raise RuntimeError("No valid examples after measurement.")

    print_top_examples(dataset, "vlm_est_peak_gb", cfg.top_k_report, cfg.bad_memory_gb)
    print_top_examples(dataset, "vlm_est_vision_peak_gb", cfg.top_k_report, cfg.bad_memory_gb)
    print_top_examples(dataset, "vlm_est_llm_peak_gb", cfg.top_k_report, cfg.bad_memory_gb)

    if cfg.stats_only:
        print("\nstats_only=True: finished statistics inspection. No filtering or saving was performed.")
        return

    cutoffs = build_filter_cutoffs(dataset, cfg)
    print("\n=== Active cutoffs ===")
    for col, cutoff in cutoffs.items():
        print(f"  {col} <= {cutoff:.4f}")

    if "vlm_filter_reason" in dataset.column_names:
        dataset = dataset.remove_columns(["vlm_filter_reason"])

    print("Adding filter reasons...", flush=True)
    dataset = dataset.map(
        add_filter_reasons_batched,
        batched=True,
        batch_size=cfg.map_batch_size,
        num_proc=cfg.num_proc,
        fn_kwargs={"cutoffs": cutoffs},
        desc="Adding filter reasons",
    )

    print("Splitting dataset by cutoffs...", flush=True)
    kept, removed = split_by_cutoffs(dataset)

    n_before, n_after = len(dataset), len(kept)
    print(
        f"\nCombined filter: {n_before} -> {n_after}  "
        f"(removed {n_before - n_after}, "
        f"{(n_before - n_after) / max(n_before, 1) * 100:.2f}%)"
    )

    print_value_stats(kept["vlm_est_peak_gb"], "Kept — estimated max peak GB", cfg.bad_memory_gb)
    print_value_stats(removed["vlm_est_peak_gb"], "Removed — estimated max peak GB", cfg.bad_memory_gb)

    if not cfg.keep_length_columns:
        kept = drop_length_columns(kept, LENGTH_COLUMNS)
        removed = drop_length_columns(removed, LENGTH_COLUMNS)

    print(f"Saving kept examples -> {cfg.output_dataset}")
    save_dataset(kept, cfg.output_dataset)

    print(f"Saving removed examples -> {cfg.removed_dataset}")
    save_dataset(removed, cfg.removed_dataset)

    print("Done.")


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Filter a multimodal dataset by VLM length and estimated Qwen2.5-VL memory pressure."
    )
    parser.add_argument("--dataset_name", default=defaults.dataset_name, help="Dataset path prefix without .jsonl extension")
    parser.add_argument("--model_name", default=defaults.model_name)
    parser.add_argument("--image_root", default=defaults.image_root)
    parser.add_argument("--sample_size", type=none_or_int, default=defaults.sample_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)

    parser.add_argument("--max_vlm_length", type=none_or_int, default=defaults.max_vlm_length)
    parser.add_argument("--max_vlm_length_percentile", type=none_or_float, default=defaults.max_vlm_length_percentile)
    parser.add_argument("--max_vision_patch_tokens", type=none_or_int, default=defaults.max_vision_patch_tokens)
    parser.add_argument("--max_vision_patch_tokens_percentile", type=none_or_float, default=defaults.max_vision_patch_tokens_percentile)
    parser.add_argument("--max_memory_peak_gb", type=none_or_float, default=defaults.max_memory_peak_gb)
    parser.add_argument("--max_memory_peak_percentile", type=none_or_float, default=defaults.max_memory_peak_percentile)
    parser.add_argument("--max_llm_peak_gb", type=none_or_float, default=defaults.max_llm_peak_gb)
    parser.add_argument("--max_vision_peak_gb", type=none_or_float, default=defaults.max_vision_peak_gb)

    parser.add_argument("--min_image_width", type=int, default=defaults.min_image_width)
    parser.add_argument("--min_image_height", type=int, default=defaults.min_image_height)

    parser.add_argument("--moe_top_k", "--k", type=int, default=defaults.moe_top_k,
                        help="Number of active MoE experts per token. Dense MLP equivalent is 1.")
    parser.add_argument("--moe_num_experts", type=int, default=defaults.moe_num_experts,
                        help="Total experts; only affects router-logit accounting/validation, not FFN activation much.")
    parser.add_argument("--dtype_bytes", type=int, default=defaults.dtype_bytes)
    parser.add_argument("--assumed_batch_size", type=int, default=defaults.assumed_batch_size,
                        help="Scale estimates as if this many similar examples are on one device.")
    parser.add_argument("--activation_checkpointing", type=str2bool, default=defaults.activation_checkpointing)
    parser.add_argument("--attn_score_memory_factor", type=float, default=defaults.attn_score_memory_factor,
                        help="0 for flash/SDPA-like no explicit T^2 scores; 1 for conservative explicit attention scores.")
    parser.add_argument("--memory_safety_factor", type=float, default=defaults.memory_safety_factor)

    parser.add_argument("--stats_only", nargs="?", const=True, type=str2bool, default=defaults.stats_only,
                        help="Only compute/print statistics and top examples. Do not filter rows or save output datasets. Can be passed as --stats_only or --stats_only true.")
    parser.add_argument("--keep_length_columns", type=str2bool, default=defaults.keep_length_columns)
    parser.add_argument("--num_proc", type=int, default=defaults.num_proc)
    parser.add_argument("--map_batch_size", type=int, default=defaults.map_batch_size)
    parser.add_argument("--top_k_report", type=int, default=defaults.top_k_report)

    args = parser.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    main(parse_args())