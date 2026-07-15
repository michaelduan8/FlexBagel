#!/usr/bin/env python3
"""
Fast proportional subsampling from HuggingFaceM4/FineVision.

Key speedups:
  - streaming=True, so the full dataset is never downloaded
  - parallel subset processing
  - Image(decode=False)
  - direct byte-copy when the original image is already PNG
  - low PNG compression for faster saving

Install:
  pip install -U datasets huggingface_hub pillow tqdm requests

Example:
  python fast_subsample_finevision.py \
    --target-num 10000 \
    --output-jsonl /flare/MatSciAI/xinxil/data/finevision_sample/finevision_10k.jsonl \
    --image-dir /flare/MatSciAI/xinxil/data/finevision_sample/images \
    --num-workers 8 \
    --shuffle-buffer 0 \
    --png-compress-level 0 \
    --overwrite
"""

import argparse
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image as PILImage
from tqdm import tqdm

from datasets import (
    Image,
    Sequence,
    get_dataset_config_names,
    get_dataset_infos,
    load_dataset,
)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
LOGGER = logging.getLogger("fast_finevision")


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def slugify(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text[:max_len] if text else "unknown"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dst) + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def get_split_count_from_dataset_server(
    dataset_id: str,
    config: str,
    split: str,
    token: Optional[str],
) -> Optional[int]:
    url = "https://datasets-server.huggingface.co/splits"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(
            url,
            params={"dataset": dataset_id, "config": config},
            headers=headers,
            timeout=60,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        for item in data.get("splits", []):
            if item.get("split") == split and item.get("num_examples") is not None:
                return int(item["num_examples"])
    except Exception:
        return None

    return None


def get_subset_counts(
    dataset_id: str,
    split: str,
    token: Optional[str],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
) -> Dict[str, int]:
    configs = get_dataset_config_names(dataset_id, token=token)

    if include_regex:
        pat = re.compile(include_regex)
        configs = [c for c in configs if pat.search(c)]

    if exclude_regex:
        pat = re.compile(exclude_regex)
        configs = [c for c in configs if not pat.search(c)]

    if not configs:
        raise RuntimeError("No configs left after include/exclude filtering.")

    try:
        infos = get_dataset_infos(dataset_id, token=token)
    except Exception as exc:
        LOGGER.warning("get_dataset_infos failed, falling back to datasets-server: %s", exc)
        infos = {}

    counts: Dict[str, int] = {}

    for cfg in tqdm(configs, desc="Fetching subset sizes"):
        n = None

        info = infos.get(cfg) if isinstance(infos, dict) else None
        if info is not None and getattr(info, "splits", None) is not None:
            try:
                if split in info.splits:
                    n = int(info.splits[split].num_examples)
            except Exception:
                n = None

        if n is None:
            n = get_split_count_from_dataset_server(dataset_id, cfg, split, token)

        if n is not None and n > 0:
            counts[cfg] = n
        else:
            LOGGER.warning("Skipping config with unknown or zero size: %s", cfg)

    if not counts:
        raise RuntimeError("Could not determine subset sizes.")

    return counts


def proportional_allocation(target_num: int, counts: Dict[str, int]) -> Dict[str, int]:
    total_available = sum(counts.values())

    if target_num <= 0:
        raise ValueError("--target-num must be positive.")

    if target_num > total_available:
        raise ValueError(
            f"Requested {target_num}, but only {total_available} examples are reported available."
        )

    raw = {cfg: target_num * n / total_available for cfg, n in counts.items()}
    alloc = {cfg: int(math.floor(x)) for cfg, x in raw.items()}

    remainder = target_num - sum(alloc.values())

    ranked = sorted(
        counts.keys(),
        key=lambda cfg: (raw[cfg] - alloc[cfg], counts[cfg]),
        reverse=True,
    )

    for cfg in ranked[:remainder]:
        alloc[cfg] += 1

    return {cfg: n for cfg, n in alloc.items() if n > 0}


def get_first_turn(example: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    texts = example.get("texts")
    if not texts:
        return None

    if isinstance(texts, list):
        first = texts[0]
    elif isinstance(texts, dict):
        first = texts
    else:
        return None

    if not isinstance(first, dict):
        return None

    user_text = (
        first.get("user")
        or first.get("human")
        or first.get("question")
        or first.get("prompt")
    )
    assistant_text = (
        first.get("assistant")
        or first.get("gpt")
        or first.get("answer")
        or first.get("response")
    )

    if user_text is None or assistant_text is None:
        return None

    return str(user_text), str(assistant_text)


def normalize_images(images: Any) -> Optional[List[Any]]:
    if images is None:
        return None
    if isinstance(images, list):
        return images
    return [images]


def image_source_to_bytes_or_path(img_obj: Any) -> Tuple[Optional[bytes], Optional[Path], Optional[Any]]:
    """
    Returns:
      bytes, path, pil_like

    With Image(decode=False), HF image entries are usually:
      {"bytes": ..., "path": ...}
    """
    if isinstance(img_obj, dict):
        b = img_obj.get("bytes")
        p = img_obj.get("path")

        if isinstance(b, memoryview):
            b = b.tobytes()

        if b is not None:
            return bytes(b), None, None

        if p is not None:
            return None, Path(p), None

        return None, None, None

    if isinstance(img_obj, (bytes, bytearray, memoryview)):
        return bytes(img_obj), None, None

    if isinstance(img_obj, str):
        return None, Path(img_obj), None

    if hasattr(img_obj, "save"):
        return None, None, img_obj

    return None, None, None


def save_png_fast(img_obj: Any, out_path: Path, png_compress_level: int) -> None:
    """
    Save image as PNG.

    Fast path:
      If source bytes/path are already PNG, write/copy directly.

    Slow path:
      Decode with PIL and encode as PNG with low compression.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_bytes, img_path, pil_like = image_source_to_bytes_or_path(img_obj)

    # Fast path 1: original bytes are already PNG.
    if img_bytes is not None and img_bytes.startswith(PNG_MAGIC):
        atomic_write_bytes(out_path, img_bytes)
        return

    # Fast path 2: original local file is already PNG.
    if img_path is not None and img_path.exists() and img_path.suffix.lower() == ".png":
        atomic_copy_file(img_path, out_path)
        return

    # Slow path: decode and re-encode as PNG.
    if img_bytes is not None:
        pil_img = PILImage.open(io.BytesIO(img_bytes))
    elif img_path is not None:
        pil_img = PILImage.open(img_path)
    elif pil_like is not None:
        pil_img = pil_like
    else:
        raise TypeError(f"Unsupported image object type: {type(img_obj)}")

    pil_img.load()

    # Keep common PNG-compatible modes. Convert uncommon modes for safety.
    if pil_img.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I;16"}:
        pil_img = pil_img.convert("RGB")

    tmp = Path(str(out_path) + ".tmp")
    pil_img.save(
        tmp,
        format="PNG",
        compress_level=png_compress_level,
        optimize=False,
    )
    os.replace(tmp, out_path)


def build_record(
    example: Dict[str, Any],
    sample_dir: Path,
    png_compress_level: int,
) -> Optional[Dict[str, Any]]:
    first_turn = get_first_turn(example)
    if first_turn is None:
        return None

    user_text, assistant_text = first_turn

    images = normalize_images(example.get("images"))
    if not images:
        return None

    image_paths = []
    for i, img in enumerate(images):
        out_path = sample_dir / f"image_{i:02d}.png"
        save_png_fast(img, out_path, png_compress_level=png_compress_level)
        image_paths.append(str(out_path.resolve()))

    return {
        "images": image_paths,
        "conversation": [
            {
                "content": user_text,
                "img_loc": "before",
                "role": "user",
            },
            {
                "content": assistant_text,
                "img_loc": None,
                "role": "assistant",
            },
        ],
    }


def maybe_decode_false(ds):
    try:
        return ds.cast_column("images", Sequence(Image(decode=False)))
    except Exception as exc:
        LOGGER.warning("Could not cast images to decode=False: %s", exc)
        return ds


def worker_process(task: Dict[str, Any]) -> Dict[str, Any]:
    logging.basicConfig(
        level=task["log_level"],
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    dataset_id = task["dataset_id"]
    config = task["config"]
    split = task["split"]
    needed = task["needed"]
    token = task["token"]
    seed = task["seed"]
    shuffle_buffer = task["shuffle_buffer"]
    image_dir = Path(task["image_dir"])
    shard_dir = Path(task["shard_dir"])
    rank = task["rank"]
    png_compress_level = task["png_compress_level"]

    cfg_slug = slugify(config)
    shard_path = shard_dir / f"{rank:06d}_{cfg_slug}.jsonl"
    tmp_shard_path = Path(str(shard_path) + ".tmp")

    ds = load_dataset(
        dataset_id,
        name=config,
        split=split,
        streaming=True,
        token=token,
    )

    ds = maybe_decode_false(ds)

    if shuffle_buffer > 0:
        ds = ds.shuffle(
            seed=seed + stable_int(config),
            buffer_size=shuffle_buffer,
        )

    saved = 0
    skipped = 0
    seen = 0

    shard_path.parent.mkdir(parents=True, exist_ok=True)

    with tmp_shard_path.open("w", encoding="utf-8") as writer:
        for example in ds:
            if saved >= needed:
                break

            seen += 1
            sample_dir = image_dir / cfg_slug / f"{saved:08d}"

            try:
                record = build_record(
                    example=example,
                    sample_dir=sample_dir,
                    png_compress_level=png_compress_level,
                )

                if record is None:
                    skipped += 1
                    if sample_dir.exists():
                        shutil.rmtree(sample_dir, ignore_errors=True)
                    continue

                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1

            except Exception as exc:
                skipped += 1
                LOGGER.warning("Skipping bad sample in config=%s: %s", config, exc)
                if sample_dir.exists():
                    shutil.rmtree(sample_dir, ignore_errors=True)

    os.replace(tmp_shard_path, shard_path)

    return {
        "rank": rank,
        "config": config,
        "needed": needed,
        "saved": saved,
        "skipped": skipped,
        "seen": seen,
        "shard_path": str(shard_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-id", default="HuggingFaceM4/FineVision")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-num", type=int, required=True)

    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel subset workers. Try 4, 8, or 16 depending on I/O.",
    )

    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help=(
            "Streaming shuffle buffer. 0 is fastest but samples the beginning of each subset. "
            "Use 100-1000 for approximate random sampling."
        ),
    )

    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=1,
        choices=list(range(10)),
        help=(
            "PNG compression level. 0 is fastest/largest, 1 is usually a good fast default, "
            "9 is slowest/smallest."
        ),
    )

    parser.add_argument("--include-regex", default=None)
    parser.add_argument("--exclude-regex", default=None)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_jsonl} already exists. Use --overwrite to replace it."
        )

    shard_dir = args.output_jsonl.parent / f".{args.output_jsonl.stem}_shards"

    if shard_dir.exists():
        if args.overwrite:
            shutil.rmtree(shard_dir)
        else:
            raise FileExistsError(
                f"Shard dir {shard_dir} already exists. Use --overwrite to replace it."
            )

    shard_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Fetching subset sizes...")
    counts = get_subset_counts(
        dataset_id=args.dataset_id,
        split=args.split,
        token=args.token,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
    )

    allocation = proportional_allocation(args.target_num, counts)

    metadata = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "target_num": args.target_num,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "png_compress_level": args.png_compress_level,
        "num_workers": args.num_workers,
        "counts": counts,
        "allocation": allocation,
    }

    meta_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    LOGGER.info("Usable subsets: %d", len(counts))
    LOGGER.info("Sampled subsets: %d", len(allocation))
    LOGGER.info("Total allocated: %d", sum(allocation.values()))
    LOGGER.info("Metadata written to: %s", meta_path)

    tasks = []
    for rank, (cfg, needed) in enumerate(
        sorted(allocation.items(), key=lambda x: counts[x[0]], reverse=True)
    ):
        tasks.append(
            {
                "rank": rank,
                "dataset_id": args.dataset_id,
                "config": cfg,
                "split": args.split,
                "needed": needed,
                "token": args.token,
                "seed": args.seed,
                "shuffle_buffer": args.shuffle_buffer,
                "image_dir": str(args.image_dir),
                "shard_dir": str(shard_dir),
                "png_compress_level": args.png_compress_level,
                "log_level": getattr(logging, args.log_level.upper()),
            }
        )

    results = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(worker_process, task) for task in tasks]

        with tqdm(total=args.target_num, desc="Finished samples") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                pbar.update(result["saved"])

                if result["saved"] < result["needed"]:
                    LOGGER.warning(
                        "Config %s only saved %d/%d samples. skipped=%d seen=%d",
                        result["config"],
                        result["saved"],
                        result["needed"],
                        result["skipped"],
                        result["seen"],
                    )

    results = sorted(results, key=lambda x: x["rank"])

    total_saved = 0
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for result in results:
            shard_path = Path(result["shard_path"])
            with shard_path.open("r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    total_saved += 1

    result_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".results.json")
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    LOGGER.info("Final JSONL written to: %s", args.output_jsonl)
    LOGGER.info("Total saved: %d", total_saved)
    LOGGER.info("Worker results written to: %s", result_path)

    if not args.keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)

    if total_saved != args.target_num:
        LOGGER.warning(
            "Requested %d samples but saved %d. Some streamed samples may have been malformed.",
            args.target_num,
            total_saved,
        )


if __name__ == "__main__":
    main()