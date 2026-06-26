#!/usr/bin/env python
"""
Embed your VLM training data with Qwen/Qwen3-VL-Embedding-2B or
Qwen/Qwen3-VL-Embedding-8B, without requiring the cloned Qwen3-VL-Embedding repo.

Expected input row schema, matching your training script:
{
  "id": "...",
  "images": ["/path/to/image1", ...],
  "conversation": [
    {"role": "user"|"assistant", "content": "...", "img_loc": "before"|"after"|None},
    ...,
    {"role": "assistant", "content": "final response", "img_loc": None}
  ]
}

For --embed_mode prompt:
  - uses the same prompt formatting as your SFT script
  - drops the final assistant response
  - keeps previous assistant turns if the data is multi-turn

For --embed_mode image_only:
  - ignores all text
  - embeds only the images attached to the sample

Outputs:
  output_dir/
    embeddings_prompt.npy
    metadata_prompt.jsonl
    embeddings_image_only.npy
    metadata_image_only.jsonl
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    HfArgumentParser,
)


@dataclass
class EmbedArgs:
    # Data
    datasets: List[str] = field(
        metadata={
            "help": "Dataset path(s) or HF dataset name(s). Supports json/jsonl/parquet/HF datasets."
        }
    )
    output_dir: str = field(default="./qwen3vl_embeddings")
    run_seed: int = field(default=2025)
    sample_size: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "One value = sample after merging; one value per dataset = sample each dataset before merging."
        },
    )
    filter_by_id: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Only keep rows whose id contains at least one substring."},
    )
    num_proc: int = field(default=12)

    # Multi-process / multi-GPU sharding.
    # You can either launch with torchrun/mpirun/SLURM env vars, or set these manually.
    num_shards: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of embedding shards/processes. Defaults to WORLD_SIZE/SLURM_NTASKS/PMI_SIZE if set."},
    )
    shard_rank: Optional[int] = field(
        default=None,
        metadata={"help": "This process rank in [0, num_shards). Defaults to RANK/SLURM_PROCID/PMI_RANK if set."},
    )
    local_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Local GPU index for this process. Defaults to LOCAL_RANK/SLURM_LOCALID if set."},
    )
    merge_shards: bool = field(
        default=True,
        metadata={"help": "In multi-shard mode, rank 0 waits for shard outputs and merges them."},
    )
    keep_shards: bool = field(
        default=True,
        metadata={"help": "Keep per-rank shard files after merge. Set false to delete them."},
    )
    merge_timeout_s: int = field(
        default=0,
        metadata={"help": "Rank-0 merge wait timeout in seconds. 0 means wait indefinitely."},
    )
    force_recompute: bool = field(
        default=False,
        metadata={"help": "Recompute embeddings even if final output files already exist."},
    )

    # Image loading robustness
    png_max_text_chunk_mb: int = field(
        default=256,
        metadata={"help": "PIL PNG metadata chunk limit. Increase for PNGs with oversized iCCP profiles."},
    )
    png_max_text_memory_mb: int = field(
        default=1024,
        metadata={"help": "PIL total PNG text/iCCP memory limit."},
    )
    skip_bad_images: bool = field(
        default=True,
        metadata={"help": "Skip rows whose images cannot be decoded after one-by-one retry."},
    )

    # Model
    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-Embedding-2B",
        metadata={
            "help": "Use Qwen/Qwen3-VL-Embedding-2B, Qwen/Qwen3-VL-Embedding-8B, or a local checkpoint path."
        },
    )
    dtype: str = field(default="auto", metadata={"help": "auto | bf16 | fp16 | fp32"})
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "flash_attention_2 | sdpa | eager | none"},
    )
    device: str = field(default="auto", metadata={"help": "auto | cuda | cpu | cuda:0 etc."})
    trust_remote_code: bool = field(default=False)

    # Embedding behavior
    embed_mode: str = field(
        default="prompt",
        metadata={"help": "prompt | image_only | both"},
    )
    batch_size: int = field(default=4)
    normalize: bool = field(default=True)
    output_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional MRL-style truncation dimension, e.g. 256/512/1024/2048. "
                    "For 2B full dim is 2048; for 8B full dim is 4096."
        },
    )

    # Formatting: default is intentionally close to your SFT script.
    # Your training script uses apply_chat_template(..., add_generation_prompt=False).
    add_generation_prompt: bool = field(default=False)
    instruction: str = field(
        default="",
        metadata={
            "help": "Optional system instruction. Empty string means no system instruction, matching your SFT data format."
        },
    )

    # Processor controls
    max_length: int = field(default=8192)
    min_pixels: int = field(default=4096)
    max_pixels: int = field(default=1843200)
    max_images_per_sample: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional cap on number of images used per sample. Useful when a few multi-image rows OOM."
        },
    )
    force_no_cache: bool = field(
        default=True,
        metadata={"help": "Disable transformer KV cache during embedding forward to reduce memory."},
    )
    empty_cache_every_n_batches: int = field(
        default=0,
        metadata={"help": "Call torch.*.empty_cache every N batches. 0 disables; 1 is safest but slower."},
    )

    # Debug / metadata
    dry_run_examples: int = field(
        default=0,
        metadata={"help": "Print converted examples and exit before loading model."},
    )
    save_prompt_text: bool = field(default=False)


def _normalize_sample_sizes(sample_size: Optional[Sequence[int]], n_datasets: int) -> Optional[List[int]]:
    if sample_size is None:
        return None
    sample_sizes = list(sample_size)
    if len(sample_sizes) not in (1, n_datasets):
        raise ValueError(
            f"sample_size must be either one value or one value per dataset. "
            f"Got {len(sample_sizes)} values for {n_datasets} datasets."
        )
    if any(x is not None and x < 0 for x in sample_sizes):
        raise ValueError("sample_size values must be non-negative.")
    return sample_sizes


def _load_one_dataset(dataset_name: str) -> Dataset:
    lower = dataset_name.lower()
    if lower.endswith(".jsonl") or lower.endswith(".json") or ".jsonl" in lower or ".json" in lower:
        return load_dataset("json", data_files=dataset_name, split="train")
    if lower.endswith(".parquet") or ".parquet" in lower:
        return load_dataset("parquet", data_files=dataset_name, split="train")
    return load_dataset(dataset_name, split="train")


def _content_text(turn: Dict[str, Any]) -> str:
    content = turn.get("content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for x in content:
            if isinstance(x, dict) and x.get("type") == "text":
                parts.append(str(x.get("text", "")))
            elif isinstance(x, str):
                parts.append(x)
        return "\n".join([p for p in parts if p])
    return str(content)


def _image_paths(item: Dict[str, Any]) -> List[str]:
    images = item.get("images", None)
    if images is None:
        images = item.get("image", None)
    if images is None:
        return []
    if isinstance(images, str):
        return [images]
    return [str(x) for x in images if x is not None]


def preprocess_dataset_for_embedding(dataset: Dataset, num_proc: int = 12) -> Dataset:
    """
    Convert raw rows to:
      prompt_id: str
      prompt: list[chat messages], same content structure as your SFT prompt
      image_paths: list[str]
    Final assistant response is dropped.
    """

    def convert_row(item: Dict[str, Any]) -> Dict[str, Any]:
        if "id" not in item or "conversation" not in item:
            raise KeyError("Each row must contain at least `id`, `images`, and `conversation`.")

        prompt_id = str(item["id"])
        image_paths = _image_paths(item)
        conversation = item["conversation"]

        if not isinstance(conversation, list) or len(conversation) == 0:
            return {"prompt_id": prompt_id, "prompt": [], "image_paths": image_paths, "keep": False}

        # Match your SFT assumption: final turn is supervised answer.
        # For embedding, drop only that final assistant response.
        if conversation[-1].get("role") == "assistant":
            prompt_turns_raw = conversation[:-1]
        else:
            # Be permissive if a row is already prompt-only.
            prompt_turns_raw = conversation

        prompt: List[Dict[str, Any]] = []
        for turn in prompt_turns_raw:
            role = turn.get("role", "user")
            text = _content_text(turn)
            img_loc = turn.get("img_loc", None)

            if role == "user" and img_loc is not None and len(image_paths) > 0:
                text_piece = {"type": "text", "text": text}
                image_pieces = [{"type": "image"} for _ in image_paths]

                if img_loc == "before":
                    content = image_pieces + ([text_piece] if text else [])
                else:
                    content = ([text_piece] if text else []) + image_pieces
            else:
                content = [{"type": "text", "text": text}] if text else []

            if len(content) > 0:
                prompt.append({"role": role, "content": content})

        keep = len(image_paths) > 0 and len(prompt) > 0
        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "image_paths": image_paths,
            "keep": keep,
        }

    out = dataset.map(convert_row, remove_columns=dataset.column_names, num_proc=num_proc)
    out = out.filter(lambda x: bool(x["keep"]), num_proc=num_proc)
    out = out.remove_columns(["keep"])
    return out


def prepare_datasets(args: EmbedArgs) -> Dataset:
    sample_sizes = _normalize_sample_sizes(args.sample_size, len(args.datasets))
    loaded: List[Dataset] = []

    for idx, dataset_name in enumerate(args.datasets):
        print(f"Loading dataset: {dataset_name}")
        ds = _load_one_dataset(dataset_name)

        # Per-dataset sampling before preprocessing.
        if sample_sizes is not None and len(sample_sizes) == len(args.datasets) and len(sample_sizes) > 1:
            n = sample_sizes[idx]
            if n is not None and n < len(ds):
                print(f"Sampling {n} examples from dataset[{idx}] before preprocessing")
                ds = ds.shuffle(seed=args.run_seed).select(range(n))

        ds = preprocess_dataset_for_embedding(ds, num_proc=args.num_proc)
        loaded.append(ds)

    merged = concatenate_datasets(loaded) if len(loaded) > 1 else loaded[0]

    if args.filter_by_id is not None and len(args.filter_by_id) > 0:
        before = len(merged)
        keep_substrings = tuple(args.filter_by_id)
        merged = merged.filter(lambda x: any(s in x["prompt_id"] for s in keep_substrings), num_proc=args.num_proc)
        print(f"filter_by_id: {before} -> {len(merged)} rows")

    print("Shuffling dataset")
    merged = merged.shuffle(seed=args.run_seed)

    # Global sampling after merge.
    if sample_sizes is not None and len(sample_sizes) == 1:
        n = sample_sizes[0]
        if n is not None and n < len(merged):
            print(f"Sampling {n} examples from merged dataset")
            merged = merged.select(range(n))

    print(f"Final embedding dataset size: {len(merged)}")
    return merged


def _abs_image_path(path: str) -> str:
    if path.startswith(("http://", "https://", "file://")):
        return path
    return os.path.abspath(os.path.expanduser(path))


def _image_content_piece(path: str, args: Optional[EmbedArgs] = None) -> Dict[str, Any]:
    piece: Dict[str, Any] = {"type": "image", "image": _abs_image_path(path)}
    # Important: qwen-vl-utils reads min_pixels/max_pixels from each image dict.
    # Passing these only to AutoProcessor.from_pretrained is not enough for all versions.
    if args is not None:
        piece["min_pixels"] = int(args.min_pixels)
        piece["max_pixels"] = int(args.max_pixels)
    return piece


def build_messages_for_processor(
    ex: Dict[str, Any],
    mode: str,
    instruction: str,
    args: Optional[EmbedArgs] = None,
) -> List[Dict[str, Any]]:
    """
    Build Qwen chat messages with actual image paths.
    This does not use the official Qwen3-VL-Embedding repo.
    """
    image_paths = list(ex["image_paths"])
    if args is not None and args.max_images_per_sample is not None:
        image_paths = image_paths[: max(0, int(args.max_images_per_sample))]
    messages: List[Dict[str, Any]] = []

    if instruction is not None and instruction.strip():
        messages.append({"role": "system", "content": [{"type": "text", "text": instruction.strip()}]})

    if mode == "image_only":
        # Truly image-only unless the user explicitly passed --instruction.
        messages.append(
            {
                "role": "user",
                "content": [_image_content_piece(p, args) for p in image_paths],
            }
        )
        return messages

    if mode != "prompt":
        raise ValueError(f"Unknown mode: {mode}")

    image_i = 0
    n_images = max(1, len(image_paths))

    for msg in ex["prompt"]:
        role = msg.get("role", "user")
        content_out: List[Dict[str, Any]] = []
        for piece in msg.get("content", []):
            if piece.get("type") == "image":
                img_path = image_paths[image_i % n_images]
                image_i += 1
                content_out.append(_image_content_piece(img_path, args))
            elif piece.get("type") == "text":
                text = str(piece.get("text", ""))
                if text:
                    content_out.append({"type": "text", "text": text})

        if content_out:
            messages.append({"role": role, "content": content_out})

    return messages


def prompt_to_plain_text(prompt: List[Dict[str, Any]]) -> str:
    rows = []
    for msg in prompt:
        pieces = []
        for p in msg.get("content", []):
            if p.get("type") == "image":
                pieces.append("<image>")
            elif p.get("type") == "text":
                pieces.append(str(p.get("text", "")))
        rows.append(f"{msg.get('role', 'unknown')}: " + " ".join([x for x in pieces if x]))
    return "\n".join(rows)


def _xpu_available() -> bool:
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def resolve_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # Aurora/Intel GPUs are generally intended to run bf16 workloads.
        if _xpu_available():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def _env_int(names: Sequence[str], default: Optional[int] = None) -> Optional[int]:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            try:
                return int(value)
            except ValueError:
                pass
    return default


def get_shard_info(args: EmbedArgs) -> Tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from args or common launcher env vars.

    Aurora/PALS launches often expose PALS_LOCAL_RANKID but may not expose PMI_RANK.
    The old code defaulted missing PMI_RANK/PMI_SIZE to rank=0/world_size=1, causing every
    MPI process to write the same shard file. Keep defaults as None until all launcher-
    specific fallbacks have been tried.
    """
    env_rank = _env_int(
        [
            "RANK",
            "WORLD_RANK",
            "GLOBAL_RANK",
            "SLURM_PROCID",
            "PMI_RANK",
            "PMIX_RANK",
            "PALS_RANKID",
            "PALS_RANK_ID",
            "PALS_RANK",
            "OMPI_COMM_WORLD_RANK",
            "MV2_COMM_WORLD_RANK",
        ],
        None,
    )
    env_world = _env_int(
        [
            "WORLD_SIZE",
            "SLURM_NTASKS",
            "PMI_SIZE",
            "PMIX_SIZE",
            "PALS_APSIZE",
            "PALS_APP_SIZE",
            "PALS_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "MV2_COMM_WORLD_SIZE",
        ],
        None,
    )
    env_local = _env_int(
        [
            "LOCAL_RANK",
            "SLURM_LOCALID",
            "PALS_LOCAL_RANKID",
            "MPI_LOCALRANKID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
        ],
        None,
    )
    env_local_size = _env_int(
        [
            "LOCAL_WORLD_SIZE",
            "SLURM_NTASKS_PER_NODE",
            "PALS_LOCAL_SIZE",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
            "MV2_COMM_WORLD_LOCAL_SIZE",
        ],
        None,
    )
    env_node_rank = _env_int(["SLURM_NODEID", "PALS_NODEID"], 0)

    # Explicit CLI arguments always win.
    if args.shard_rank is not None:
        rank = int(args.shard_rank)
    else:
        rank = env_rank

    if args.num_shards is not None:
        world_size = int(args.num_shards)
    else:
        world_size = env_world

    if args.local_rank is not None:
        local_rank = int(args.local_rank)
    else:
        local_rank = env_local

    # PALS sometimes gives local rank/size but not PMI_RANK/PMI_SIZE.
    # For a single-node job this is enough; for multi-node, combine node id if present.
    if rank is None and local_rank is not None:
        if env_local_size is not None and env_node_rank is not None:
            rank = int(env_node_rank) * int(env_local_size) + int(local_rank)
        else:
            rank = int(local_rank)

    if world_size is None:
        # If only local size is visible, this is correct for one-node jobs and better than
        # silently treating every process as a singleton rank=0/world_size=1.
        if env_local_size is not None:
            world_size = int(env_local_size)
        else:
            world_size = 1

    if rank is None:
        rank = 0
    if local_rank is None:
        # Use rank modulo visible devices later in resolve_device; local_rank itself is just
        # a launcher-local index.
        local_rank = rank

    rank = int(rank)
    world_size = int(world_size)
    local_rank = int(local_rank)

    if world_size < 1:
        raise ValueError(f"num_shards/world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"shard_rank/rank must be in [0, {world_size}), got {rank}. "
            "If launcher env vars are incomplete, pass --num_shards and --shard_rank explicitly."
        )

    print(
        "Resolved shard info:",
        f"rank={rank}",
        f"world_size={world_size}",
        f"local_rank={local_rank}",
        f"env_rank={env_rank}",
        f"env_world={env_world}",
        f"env_local={env_local}",
        f"env_local_size={env_local_size}",
        flush=True,
    )
    return rank, world_size, local_rank


def _visible_device_index(kind: str, local_rank: int) -> int:
    """Map launcher local_rank to the index visible inside this process.

    With ZE_AFFINITY_MASK/gpu_*_compact.sh, each rank may see exactly one XPU, so
    the correct in-process device is xpu:0 even when PALS_LOCAL_RANKID is 1, 2, ...
    Without affinity masking, modulo maps ranks over all visible devices.
    """
    if kind == "cuda":
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    elif kind == "xpu":
        count = torch.xpu.device_count() if _xpu_available() else 0
    else:
        count = 0

    if count <= 0:
        return int(local_rank)
    return int(local_rank) % int(count)


def resolve_device(device: str, local_rank: int = 0) -> torch.device:
    device = device.lower()
    if device == "auto":
        if _xpu_available():
            return torch.device(f"xpu:{_visible_device_index('xpu', local_rank)}")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{_visible_device_index('cuda', local_rank)}")
        return torch.device("cpu")

    if device == "cuda":
        return torch.device(f"cuda:{_visible_device_index('cuda', local_rank)}")
    if device == "xpu":
        return torch.device(f"xpu:{_visible_device_index('xpu', local_rank)}")
    return torch.device(device)


def set_torch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.set_device(device)


def configure_pil_image_loading(args: EmbedArgs) -> None:
    """Relax PIL PNG metadata limits so oversized iCCP chunks do not kill embedding jobs."""
    try:
        from PIL import ImageFile, PngImagePlugin
    except Exception:
        return

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    chunk_limit = max(1, int(args.png_max_text_chunk_mb)) * 1024 * 1024
    memory_limit = max(chunk_limit, int(args.png_max_text_memory_mb) * 1024 * 1024)

    if hasattr(PngImagePlugin, "MAX_TEXT_CHUNK"):
        PngImagePlugin.MAX_TEXT_CHUNK = max(PngImagePlugin.MAX_TEXT_CHUNK, chunk_limit)
    if hasattr(PngImagePlugin, "MAX_TEXT_MEMORY"):
        PngImagePlugin.MAX_TEXT_MEMORY = max(PngImagePlugin.MAX_TEXT_MEMORY, memory_limit)


def maybe_adjust_attention_for_device(args: EmbedArgs, device: torch.device) -> None:
    if device.type == "xpu" and args.attn_implementation.lower() == "flash_attention_2":
        print("XPU device detected; changing attn_implementation from flash_attention_2 to sdpa.", flush=True)
        args.attn_implementation = "sdpa"


def make_model_kwargs(args: EmbedArgs) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation.lower() != "none":
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


def disable_model_cache(model: torch.nn.Module) -> None:
    """Disable KV cache for forward-only embedding; it wastes memory because we do not generate."""
    candidates = [
        getattr(model, "config", None),
        getattr(model, "generation_config", None),
        getattr(getattr(model, "model", None), "config", None),
    ]
    for cfg in candidates:
        if cfg is not None and hasattr(cfg, "use_cache"):
            try:
                cfg.use_cache = False
            except Exception:
                pass


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass


def _try_process_vision_info(messages_batch: List[List[Dict[str, Any]]]):
    """
    Prefer qwen-vl-utils for robust image loading/resizing.
    It is not the Qwen3-VL-Embedding repo; it is the normal Qwen VLM utility package.
    """
    try:
        try:
            from qwen_vl_utils.vision_process import process_vision_info
        except Exception:
            from qwen_vl_utils import process_vision_info  # type: ignore

        try:
            return process_vision_info(
                messages_batch,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except TypeError:
            # Older qwen-vl-utils returns just images/videos.
            images, videos = process_vision_info(messages_batch)
            return images, videos, {}
    except ImportError as e:
        raise ImportError(
            "Missing qwen-vl-utils. Install it with: pip install 'qwen-vl-utils>=0.0.14'"
        ) from e


def preprocess_batch(
    processor: Any,
    messages_batch: List[List[Dict[str, Any]]],
    args: EmbedArgs,
) -> Dict[str, torch.Tensor]:
    texts = processor.apply_chat_template(
        messages_batch,
        tokenize=False,
        add_generation_prompt=args.add_generation_prompt,
    )

    vision_result = _try_process_vision_info(messages_batch)
    if len(vision_result) == 3:
        image_inputs, video_inputs, video_kwargs = vision_result
    else:
        image_inputs, video_inputs = vision_result
        video_kwargs = {}

    videos = None
    video_metadata = None
    if video_inputs is not None:
        # Newer qwen-vl-utils may return list[(video, metadata)].
        if len(video_inputs) > 0 and isinstance(video_inputs[0], tuple):
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos = video_inputs

    processor_kwargs: Dict[str, Any] = {
        "text": texts,
        "images": image_inputs,
        "videos": videos,
        "padding": True,
        "truncation": True,
        "max_length": args.max_length,
        "return_tensors": "pt",
    }

    # Some processor versions accept video_metadata; some do not.
    if video_metadata is not None:
        processor_kwargs["video_metadata"] = video_metadata

    processor_kwargs.update(video_kwargs)

    try:
        return processor(**processor_kwargs)
    except TypeError:
        processor_kwargs.pop("video_metadata", None)
        return processor(**processor_kwargs)


def move_inputs_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in inputs.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def forward_last_hidden_state(model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Get the final-layer hidden states.

    We first try the base model to avoid materializing LM logits.
    If the architecture does not expose that cleanly, fall back to the full model
    with output_hidden_states=True.
    """
    backbone = getattr(model, "model", None)
    if backbone is not None:
        try:
            outputs = backbone(**inputs, return_dict=True, use_cache=False)
        except TypeError:
            outputs = backbone(**inputs, return_dict=True)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]

    try:
        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    except TypeError:
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state

    raise RuntimeError("Could not find last hidden states in model outputs.")


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool the final non-padding token.

    This matches common Qwen embedding pooling logic:
      - if left padded, use hidden[:, -1]
      - if right padded, use hidden[batch_idx, attention_mask.sum(-1)-1]
    """
    if attention_mask[:, -1].sum().item() == attention_mask.shape[0]:
        return last_hidden_state[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


@torch.inference_mode()
def embed_conversations(
    model: torch.nn.Module,
    processor: Any,
    messages_batch: List[List[Dict[str, Any]]],
    args: EmbedArgs,
    device: torch.device,
) -> np.ndarray:
    inputs = preprocess_batch(processor, messages_batch, args)
    inputs = move_inputs_to_device(inputs, device)

    attention_mask = inputs["attention_mask"]
    autocast_dtype = resolve_dtype(args.dtype)
    autocast_enabled = device.type in {"cuda", "xpu"} and autocast_dtype in {torch.bfloat16, torch.float16}
    if autocast_enabled:
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            last_hidden = forward_last_hidden_state(model, inputs)
            embeddings = last_token_pool(last_hidden, attention_mask)
    else:
        last_hidden = forward_last_hidden_state(model, inputs)
        embeddings = last_token_pool(last_hidden, attention_mask)

    if args.output_dim is not None:
        if args.output_dim <= 0 or args.output_dim > embeddings.shape[-1]:
            raise ValueError(f"output_dim must be in [1, {embeddings.shape[-1]}], got {args.output_dim}")
        embeddings = embeddings[:, : args.output_dim]

    embeddings = embeddings.float()
    if args.normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    return embeddings.detach().cpu().numpy().astype(np.float32)


def batched_indices(n: int, batch_size: int) -> Iterable[range]:
    for start in range(0, n, batch_size):
        yield range(start, min(n, start + batch_size))


_BAD_IMAGE_ERROR_MARKERS = (
    "broken data stream",
    "cannot identify image file",
    "image file is truncated",
    "truncated file read",
    "unidentifiedimageerror",
    "failed to read image",
    "cannot read image",
    "invalid image",
    "no such file or directory",
    "file not found",
    "is a directory",
    "permission denied",
    "decompressionbomb",
    "decompressed data too large",
    "max_text_chunk",
    "max_text_memory",
    "iccp",
)


def _exception_chain_text(error: BaseException) -> str:
    """Return reprs from an exception and its chained causes/contexts."""
    parts: List[str] = []
    seen: set[int] = set()
    cur: Optional[BaseException] = error

    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur!s}")
        cur = cur.__cause__ or cur.__context__

    return " | ".join(parts)


def _is_bad_image_error(error: BaseException) -> bool:
    """Best-effort classifier for image decode/path failures that are safe to skip."""
    text = _exception_chain_text(error).lower()
    return any(marker in text for marker in _BAD_IMAGE_ERROR_MARKERS)


def _write_skipped_row(
    skip_f: Any,
    *,
    source_row_idx: int,
    ex: Dict[str, Any],
    mode: str,
    error: BaseException,
) -> None:
    skip_row = {
        "source_row_idx": source_row_idx,
        "prompt_id": ex.get("prompt_id"),
        "image_paths": ex.get("image_paths", []),
        "embed_mode": mode,
        "error_type": type(error).__name__,
        "error": _exception_chain_text(error),
    }
    skip_f.write(json.dumps(skip_row, ensure_ascii=False) + "\n")


def _compact_memmap(
    *,
    full_path: Path,
    final_path: Path,
    written: int,
    batch_size: int,
) -> None:
    """Replace the full-size temporary .npy with a compact .npy containing only written rows."""
    full = np.load(full_path, mmap_mode="r")
    dim = int(full.shape[1])
    trimmed_path = final_path.with_name(f".{final_path.stem}.trimmed.tmp.npy")

    trimmed = np.lib.format.open_memmap(
        trimmed_path, mode="w+", dtype=np.float32, shape=(written, dim)
    )
    copy_chunk = max(batch_size * 1024, 1024)
    for start in range(0, written, copy_chunk):
        end = min(written, start + copy_chunk)
        trimmed[start:end] = full[start:end]

    trimmed.flush()
    del trimmed
    del full
    os.replace(trimmed_path, final_path)
    full_path.unlink(missing_ok=True)


def embed_one_mode(
    dataset: Dataset,
    model: torch.nn.Module,
    processor: Any,
    args: EmbedArgs,
    mode: str,
    device: torch.device,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        shard_dir = out_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"rank{rank:05d}-of-{world_size:05d}"
        emb_path = shard_dir / f"embeddings_{mode}.{suffix}.npy"
        tmp_emb_path = shard_dir / f".embeddings_{mode}.{suffix}.full.tmp.npy"
        meta_path = shard_dir / f"metadata_{mode}.{suffix}.jsonl"
        skip_path = shard_dir / f"skipped_{mode}.{suffix}.jsonl"
        done_path = shard_dir / f"done_{mode}.{suffix}.json"
    else:
        emb_path = out_dir / f"embeddings_{mode}.npy"
        tmp_emb_path = out_dir / f".embeddings_{mode}.full.tmp.npy"
        meta_path = out_dir / f"metadata_{mode}.jsonl"
        skip_path = out_dir / f"skipped_{mode}.jsonl"
        done_path = None

    n = len(dataset)
    mmap = None
    written = 0
    skipped_bad_images = 0

    tmp_emb_path.unlink(missing_ok=True)
    emb_path.unlink(missing_ok=True)
    if done_path is not None:
        done_path.unlink(missing_ok=True)

    with meta_path.open("w", encoding="utf-8") as meta_f, skip_path.open("w", encoding="utf-8") as skip_f:
        for idx_range in tqdm(
            batched_indices(n, args.batch_size),
            total=(n + args.batch_size - 1) // args.batch_size,
            desc=f"{mode} rank={rank}/{world_size}",
            disable=(rank != 0),  # only show rank 0 to avoid messy logs
        ):
            local_indices = [int(i) for i in idx_range]
            examples = [dataset[i] for i in local_indices]
            source_indices = [int(ex.get("global_row_idx", local_i)) for local_i, ex in zip(local_indices, examples)]
            messages_batch = [
                build_messages_for_processor(ex, mode=mode, instruction=args.instruction, args=args)
                for ex in examples
            ]
            valid_examples: List[tuple[int, Dict[str, Any]]] = list(zip(source_indices, examples))

            try:
                emb = embed_conversations(model, processor, messages_batch, args, device)
            except Exception as batch_error:
                print(
                    f"Batch failed at rows {idx_range.start}:{idx_range.stop}; "
                    f"retrying one-by-one. Error: {_exception_chain_text(batch_error)}"
                )
                emb_list = []
                valid_examples = []

                for source_row_idx, ex in zip(source_indices, examples):
                    one_messages = [build_messages_for_processor(ex, mode=mode, instruction=args.instruction, args=args)]
                    try:
                        one = embed_conversations(model, processor, one_messages, args, device)
                        emb_list.append(one[0])
                        valid_examples.append((source_row_idx, ex))
                    except Exception as one_error:
                        if (not args.skip_bad_images) or (not _is_bad_image_error(one_error)):
                            raise RuntimeError(
                                f"Embedding failed for row_idx={source_row_idx}, "
                                f"prompt_id={ex['prompt_id']}, images={ex['image_paths']}. "
                                "This did not look like a bad-image decode/path error, or --skip_bad_images is disabled."
                            ) from one_error

                        skipped_bad_images += 1
                        _write_skipped_row(
                            skip_f,
                            source_row_idx=source_row_idx,
                            ex=ex,
                            mode=mode,
                            error=one_error,
                        )
                        print(
                            f"[{mode}] skipping bad-image row_idx={source_row_idx}, "
                            f"prompt_id={ex['prompt_id']}: {_exception_chain_text(one_error)}"
                        )

                if len(emb_list) == 0:
                    processed = idx_range.stop
                    if processed % max(args.batch_size * 10, 1) == 0 or processed == n:
                        print(
                            f"[{mode}] processed {processed}/{n}; "
                            f"embedded {written}; skipped_bad_images {skipped_bad_images}"
                        )
                    continue

                emb = np.stack(emb_list, axis=0).astype(np.float32)

            if mmap is None:
                dim = emb.shape[-1]
                print(f"Creating temporary {tmp_emb_path} with max shape ({n}, {dim})")
                mmap = np.lib.format.open_memmap(
                    tmp_emb_path, mode="w+", dtype=np.float32, shape=(n, dim)
                )

            bsz = emb.shape[0]
            mmap[written : written + bsz] = emb

            for j, (source_row_idx, ex) in enumerate(valid_examples):
                row = {
                    "row_idx": written + j,
                    "source_row_idx": source_row_idx,
                    "shard_rank": rank,
                    "world_size": world_size,
                    "prompt_id": ex["prompt_id"],
                    "image_paths": ex["image_paths"],
                    "embed_mode": mode,
                    "model_name_or_path": args.model_name_or_path,
                    "normalized": args.normalize,
                    "output_dim": int(emb.shape[-1]),
                    "add_generation_prompt": args.add_generation_prompt,
                    "instruction": args.instruction,
                }
                if args.save_prompt_text:
                    row["prompt_text"] = prompt_to_plain_text(ex["prompt"])
                meta_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            written += bsz
            if args.empty_cache_every_n_batches > 0:
                batch_no = (idx_range.start // args.batch_size) + 1
                if batch_no % int(args.empty_cache_every_n_batches) == 0:
                    _empty_device_cache(device)
            processed = idx_range.stop
            if processed % max(args.batch_size * 10, 1) == 0 or processed == n:
                print(
                    f"[{mode}] processed {processed}/{n}; "
                    f"embedded {written}; skipped_bad_images {skipped_bad_images}"
                )

    if mmap is not None:
        mmap.flush()
        del mmap

    if written == 0:
        tmp_emb_path.unlink(missing_ok=True)
        emb_path.unlink(missing_ok=True)
        if world_size == 1:
            raise RuntimeError(
                f"No embeddings were written for mode={mode}. "
                f"Skipped {skipped_bad_images} bad-image row(s). See: {skip_path}"
            )
    elif written < n:
        print(f"[{mode}] compacting embeddings from max shape ({n}, dim) to ({written}, dim)")
        _compact_memmap(
            full_path=tmp_emb_path,
            final_path=emb_path,
            written=written,
            batch_size=args.batch_size,
        )
    else:
        os.replace(tmp_emb_path, emb_path)

    if skipped_bad_images == 0:
        skip_path.unlink(missing_ok=True)

    result = {
        "mode": mode,
        "rank": rank,
        "world_size": world_size,
        "n_shard_rows": n,
        "written": written,
        "skipped_bad_images": skipped_bad_images,
        "emb_path": str(emb_path) if emb_path.exists() else None,
        "meta_path": str(meta_path),
        "skip_path": str(skip_path) if skip_path.exists() else None,
    }

    if done_path is not None:
        tmp_done = done_path.with_suffix(".json.tmp")
        tmp_done.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_done, done_path)

    if emb_path.exists():
        print(f"Saved embeddings to: {emb_path}")
    print(f"Saved metadata to:   {meta_path}")
    print(f"[{mode} rank={rank}/{world_size}] skipped bad-image rows: {skipped_bad_images}")
    if skipped_bad_images > 0:
        print(f"Saved skipped-row report to: {skip_path}")

    return result



def add_global_row_indices(dataset: Dataset, num_proc: int) -> Dataset:
    if "global_row_idx" in dataset.column_names:
        return dataset
    return dataset.map(lambda _ex, idx: {"global_row_idx": int(idx)}, with_indices=True, num_proc=num_proc)


def shard_dataset_for_rank(dataset: Dataset, rank: int, world_size: int) -> Dataset:
    if world_size == 1:
        return dataset
    # contiguous=True lets rank-order concatenation recover the same order as the full shuffled dataset.
    return dataset.shard(num_shards=world_size, index=rank, contiguous=True)


def _shard_suffix(rank: int, world_size: int) -> str:
    return f"rank{rank:05d}-of-{world_size:05d}"


def _wait_for_done_files(
    out_dir: Path,
    mode: str,
    world_size: int,
    timeout_s: int,
    min_mtime: float,
) -> List[Path]:
    shard_dir = out_dir / "shards"
    done_files = [shard_dir / f"done_{mode}.{_shard_suffix(rank, world_size)}.json" for rank in range(world_size)]
    start = time.time()
    while True:
        missing = [
            p for p in done_files
            if (not p.exists()) or p.stat().st_mtime < min_mtime
        ]
        if not missing:
            return done_files
        if timeout_s > 0 and time.time() - start > timeout_s:
            missing_s = ", ".join(str(p) for p in missing[:5])
            raise TimeoutError(f"Timed out waiting for {len(missing)} fresh shard(s) for mode={mode}: {missing_s}")
        print(f"[merge {mode}] waiting for {len(missing)} shard(s) to finish...", flush=True)
        time.sleep(30)


def merge_mode_shards(args: EmbedArgs, mode: str, world_size: int, min_mtime: float = 0.0) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    shard_dir = out_dir / "shards"
    done_files = _wait_for_done_files(out_dir, mode, world_size, args.merge_timeout_s, min_mtime=min_mtime)

    infos: List[Dict[str, Any]] = []
    for p in done_files:
        infos.append(json.loads(p.read_text(encoding="utf-8")))

    total_written = int(sum(int(info.get("written", 0)) for info in infos))
    total_skipped = int(sum(int(info.get("skipped_bad_images", 0)) for info in infos))
    if total_written == 0:
        raise RuntimeError(f"No embeddings were written for mode={mode}; skipped_bad_images={total_skipped}")

    first_emb = next(info["emb_path"] for info in infos if info.get("emb_path"))
    first_arr = np.load(first_emb, mmap_mode="r")
    dim = int(first_arr.shape[1])
    del first_arr

    final_emb_path = out_dir / f"embeddings_{mode}.npy"
    final_meta_path = out_dir / f"metadata_{mode}.jsonl"
    final_skip_path = out_dir / f"skipped_{mode}.jsonl"

    tmp_final_emb = out_dir / f".embeddings_{mode}.merge.tmp.npy"
    tmp_final_emb.unlink(missing_ok=True)
    merged = np.lib.format.open_memmap(tmp_final_emb, mode="w+", dtype=np.float32, shape=(total_written, dim))

    row_cursor = 0
    with final_meta_path.open("w", encoding="utf-8") as meta_out:
        for info in infos:
            emb_path = info.get("emb_path")
            if emb_path:
                arr = np.load(emb_path, mmap_mode="r")
                n = int(arr.shape[0])
                merged[row_cursor: row_cursor + n] = arr
                del arr
            else:
                n = 0

            meta_path = info.get("meta_path")
            if meta_path and Path(meta_path).exists():
                with Path(meta_path).open("r", encoding="utf-8") as meta_in:
                    for line in meta_in:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        row["row_idx"] = row_cursor
                        meta_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        row_cursor += 1
            else:
                row_cursor += n

    merged.flush()
    del merged
    os.replace(tmp_final_emb, final_emb_path)

    wrote_any_skip = False
    with final_skip_path.open("w", encoding="utf-8") as skip_out:
        for info in infos:
            skip_path = info.get("skip_path")
            if skip_path and Path(skip_path).exists():
                with Path(skip_path).open("r", encoding="utf-8") as skip_in:
                    for line in skip_in:
                        if line.strip():
                            skip_out.write(line)
                            wrote_any_skip = True
    if not wrote_any_skip:
        final_skip_path.unlink(missing_ok=True)

    if not args.keep_shards:
        for info in infos:
            for key in ("emb_path", "meta_path", "skip_path"):
                path = info.get(key)
                if path:
                    Path(path).unlink(missing_ok=True)
        for p in done_files:
            p.unlink(missing_ok=True)
        # Remove shard directory only if empty.
        try:
            shard_dir.rmdir()
        except OSError:
            pass

    result = {
        "mode": mode,
        "written": total_written,
        "skipped_bad_images": total_skipped,
        "emb_path": str(final_emb_path),
        "meta_path": str(final_meta_path),
        "skip_path": str(final_skip_path) if final_skip_path.exists() else None,
    }
    print(f"[merge {mode}] wrote {total_written} embeddings to {final_emb_path}")
    print(f"[merge {mode}] skipped bad-image rows across shards: {total_skipped}")
    return result


def save_average_embedding(output_dir: Path, mode: str) -> None:
    emb_path = output_dir / f"embeddings_{mode}.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing final embedding file for mode={mode}: {emb_path}")
    embeddings = np.load(emb_path, mmap_mode="r")
    row_average = np.mean(embeddings, axis=0)
    print(f"[{mode}] final average embedding shape:", row_average.shape)
    np.save(output_dir / f"average_embeddings_{mode}.npy", row_average.astype(np.float32))

def print_dry_run(dataset: Dataset, args: EmbedArgs) -> None:
    modes = ["prompt", "image_only"] if args.embed_mode == "both" else [args.embed_mode]
    k = min(args.dry_run_examples, len(dataset))

    for i in range(k):
        ex = dataset[i]
        print("=" * 80)
        print(f"row_idx={i} prompt_id={ex['prompt_id']}")
        print("image_paths:", ex["image_paths"])
        print("prompt_text:")
        print(prompt_to_plain_text(ex["prompt"]))

        for mode in modes:
            print(f"\nMessages for mode={mode}:")
            print(
                json.dumps(
                    build_messages_for_processor(ex, mode=mode, instruction=args.instruction, args=args),
                    indent=2,
                    ensure_ascii=False,
                )
            )


def main() -> None:
    parser = HfArgumentParser(EmbedArgs)
    (args,) = parser.parse_args_into_dataclasses()

    if args.embed_mode not in {"prompt", "image_only", "both"}:
        raise ValueError("--embed_mode must be one of: prompt, image_only, both")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    xpu_count = torch.xpu.device_count() if _xpu_available() else 0
    xpu_props = torch.xpu.get_device_properties(0) if xpu_count > 0 else None
    print(
        "Launcher/GPU env:",
        "RANK=", os.environ.get("RANK"),
        "WORLD_SIZE=", os.environ.get("WORLD_SIZE"),
        "PMI_RANK=", os.environ.get("PMI_RANK"),
        "PMI_SIZE=", os.environ.get("PMI_SIZE"),
        "PMIX_RANK=", os.environ.get("PMIX_RANK"),
        "PMIX_SIZE=", os.environ.get("PMIX_SIZE"),
        "PALS_RANKID=", os.environ.get("PALS_RANKID"),
        "PALS_APSIZE=", os.environ.get("PALS_APSIZE"),
        "PALS_LOCAL_RANKID=", os.environ.get("PALS_LOCAL_RANKID"),
        "PALS_LOCAL_SIZE=", os.environ.get("PALS_LOCAL_SIZE"),
        "ZE_FLAT_DEVICE_HIERARCHY=", os.environ.get("ZE_FLAT_DEVICE_HIERARCHY"),
        "ZE_AFFINITY_MASK=", os.environ.get("ZE_AFFINITY_MASK"),
        "xpu_count=", xpu_count,
        "props=", xpu_props,
        flush=True,
    )

    configure_pil_image_loading(args)
    run_started_at = time.time()
    # Allow small launcher clock/order differences while ignoring stale shard sentinels.
    fresh_done_min_mtime = run_started_at - 300
    rank, world_size, local_rank = get_shard_info(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = ["prompt", "image_only"] if args.embed_mode == "both" else [args.embed_mode]

    final_missing = [
        mode for mode in modes
        if args.force_recompute or not (output_dir / f"embeddings_{mode}.npy").exists()
    ]

    if len(final_missing) > 0:
        torch.set_grad_enabled(False)

        dataset = prepare_datasets(args)
        if len(dataset) == 0:
            raise ValueError("No rows left after preprocessing/filtering.")

        dataset = add_global_row_indices(dataset, args.num_proc)

        if args.dry_run_examples > 0:
            print_dry_run(dataset, args)
            return

        shard_dataset = shard_dataset_for_rank(dataset, rank=rank, world_size=world_size)
        print(
            f"Embedding rank={rank}/{world_size}, local_rank={local_rank}, "
            f"full_rows={len(dataset)}, shard_rows={len(shard_dataset)}",
            flush=True,
        )

        device = resolve_device(args.device, local_rank=local_rank)
        set_torch_device(device)
        maybe_adjust_attention_for_device(args, device)
        print(f"Using device: {device}", flush=True)

        print(f"Loading processor: {args.model_name_or_path}")
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

        model_kwargs = make_model_kwargs(args)
        print(f"Loading embedding model: {args.model_name_or_path}")
        print(f"Model kwargs: {model_kwargs}")
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path,
            **model_kwargs,
        )
        model.to(device)
        if args.force_no_cache:
            disable_model_cache(model)
        model.eval()

        total_skipped_bad_images = 0
        for mode in final_missing:
            result = embed_one_mode(
                shard_dataset, model, processor, args, mode, device, rank=rank, world_size=world_size
            )
            total_skipped_bad_images += int(result.get("skipped_bad_images", 0))

        print(f"Rank {rank}: skipped bad-image rows across requested mode(s): {total_skipped_bad_images}")

        # Only rank 0 creates final merged outputs. Other ranks stop after writing shards.
        if world_size > 1:
            if rank == 0 and args.merge_shards:
                for mode in final_missing:
                    merge_mode_shards(args, mode, world_size=world_size, min_mtime=fresh_done_min_mtime)
            elif rank != 0:
                print(f"Rank {rank}: shard finished. Final merge is handled by rank 0.")
                return

    # Final average files are only created once final merged/single-process outputs exist.
    if rank == 0:
        for mode in modes:
            save_average_embedding(output_dir, mode)


if __name__ == "__main__":
    main()
