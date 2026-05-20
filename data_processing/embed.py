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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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


def build_messages_for_processor(
    ex: Dict[str, Any],
    mode: str,
    instruction: str,
) -> List[Dict[str, Any]]:
    """
    Build Qwen chat messages with actual image paths.
    This does not use the official Qwen3-VL-Embedding repo.
    """
    image_paths = list(ex["image_paths"])
    messages: List[Dict[str, Any]] = []

    if instruction is not None and instruction.strip():
        messages.append({"role": "system", "content": [{"type": "text", "text": instruction.strip()}]})

    if mode == "image_only":
        # Truly image-only unless the user explicitly passed --instruction.
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": _abs_image_path(p)} for p in image_paths],
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
                content_out.append({"type": "image", "image": _abs_image_path(img_path)})
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


def resolve_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
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


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_model_kwargs(args: EmbedArgs) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation.lower() != "none":
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


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
            outputs = backbone(**inputs, return_dict=True)
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                return outputs.last_hidden_state
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                return outputs.hidden_states[-1]
        except TypeError:
            pass

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


@torch.no_grad()
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
) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / f"embeddings_{mode}.npy"
    tmp_emb_path = out_dir / f".embeddings_{mode}.full.tmp.npy"
    meta_path = out_dir / f"metadata_{mode}.jsonl"
    skip_path = out_dir / f"skipped_{mode}.jsonl"

    n = len(dataset)
    mmap = None
    written = 0
    skipped_bad_images = 0

    tmp_emb_path.unlink(missing_ok=True)

    with meta_path.open("w", encoding="utf-8") as meta_f, skip_path.open("w", encoding="utf-8") as skip_f:
        for idx_range in batched_indices(n, args.batch_size):
            source_indices = [int(i) for i in idx_range]
            examples = [dataset[i] for i in source_indices]
            messages_batch = [
                build_messages_for_processor(ex, mode=mode, instruction=args.instruction)
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
                    one_messages = [build_messages_for_processor(ex, mode=mode, instruction=args.instruction)]
                    try:
                        one = embed_conversations(model, processor, one_messages, args, device)
                        emb_list.append(one[0])
                        valid_examples.append((source_row_idx, ex))
                    except Exception as one_error:
                        if not _is_bad_image_error(one_error):
                            raise RuntimeError(
                                f"Embedding failed for row_idx={source_row_idx}, "
                                f"prompt_id={ex['prompt_id']}, images={ex['image_paths']}. "
                                "This did not look like a bad-image decode/path error, so it was not skipped."
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
        raise RuntimeError(
            f"No embeddings were written for mode={mode}. "
            f"Skipped {skipped_bad_images} bad-image row(s). See: {skip_path}"
        )

    if written < n:
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

    print(f"Saved embeddings to: {emb_path}")
    print(f"Saved metadata to:   {meta_path}")
    print(f"[{mode}] skipped bad-image rows: {skipped_bad_images}")
    if skipped_bad_images > 0:
        print(f"Saved skipped-row report to: {skip_path}")

    return skipped_bad_images


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
                    build_messages_for_processor(ex, mode=mode, instruction=args.instruction),
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

    emb_path = args.output_dir / f"embeddings_{args.embed_mode}.npy"
    if not emb_path.exists():
        torch.set_grad_enabled(False)

        dataset = prepare_datasets(args)
        if len(dataset) == 0:
            raise ValueError("No rows left after preprocessing/filtering.")

        if args.dry_run_examples > 0:
            print_dry_run(dataset, args)
            return

        device = resolve_device(args.device)

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
        model.eval()

        modes = ["prompt", "image_only"] if args.embed_mode == "both" else [args.embed_mode]
        total_skipped_bad_images = 0
        for mode in modes:
            total_skipped_bad_images += embed_one_mode(dataset, model, processor, args, mode, device)

        print(f"Total skipped bad-image rows across requested mode(s): {total_skipped_bad_images}")
    
    else:
        embeddings = np.load(emb_path)
        row_average = np.mean(embeddings, axis=0)
        print("Final embeddings shape:", row_average.shape)
        np.save(f"{args.output_dir}/average_embeddings_{args.embed_mode}.npy", row_average)


if __name__ == "__main__":
    main()
