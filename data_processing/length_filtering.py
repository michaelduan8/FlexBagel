import os
import json
from pathlib import Path

import numpy as np
import math
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor

# Important: run this before importing torch/model code.
# This script only uses the processor, so it should stay CPU-side.

# DATASET_ORIGINAL_NAME = "/mnt/rexvqa/train_vqa_data_filtered"
DATASET_ORIGINAL_NAME = "/mnt/quilt/train"
# DATASET_ORIGINAL_NAME = "/mnt/pubmedvision/filtered_train"
# DATASET_ORIGINAL_NAME = "/mnt/surg390k/total_train_normalized"

INPUT_DATASET = f"{DATASET_ORIGINAL_NAME}.jsonl"
OUTPUT_DATASET = f"{DATASET_ORIGINAL_NAME}_w_length.jsonl"
REMOVED_DATASET = f"{DATASET_ORIGINAL_NAME}_removed_long_examples.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_ROOT = None
SAMPLE_SIZE = 1000
SEED = 2026
# Filtering options.
# You can use one or both. If both are set, the stricter one is used.
MAX_VLM_LENGTH = None
# MAX_VLM_LENGTH = 8192
MAX_VLM_LENGTH_PERCENTILE = 99.5
# MAX_VLM_LENGTH_PERCENTILE = None
KEEP_LENGTH_COLUMNS = True

BAD_LENGTH = int(1e12)
NUM_PROC = 16

# Avoid tokenizer fork warning/noise in multiprocessing.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_original_dataset(input_path_or_name: str):
    if input_path_or_name.endswith(".jsonl") or input_path_or_name.endswith(".json"):
        return load_dataset("json", data_files=input_path_or_name, split="train")
    elif input_path_or_name.endswith(".parquet"):
        return load_dataset("parquet", data_files=input_path_or_name, split="train")
    else:
        return load_dataset(input_path_or_name, split="train")


def resolve_image_path(img_item, image_root=None, input_parent=None):
    if isinstance(img_item, dict):
        img_path = img_item.get("path", None)
    else:
        img_path = img_item

    if img_path is None:
        raise ValueError(f"Cannot resolve image path from {img_item}")

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


def build_messages_like_training(example):
    """
    Mirrors your training preprocess_dataset(...).convert_row(...).
    """
    assert "images" in example, "Example must contain `images`."
    assert "conversation" in example, "Example must contain `conversation`."
    # assert "id" in example, "Example must contain `id`."
    if "id" not in example:
        example["id"] = "UNKNOWN"

    conversation = example["conversation"]
    images = example["images"]

    if conversation[-1]["role"] != "assistant":
        raise ValueError(f"Last turn is not assistant for id={example['id']}")

    prompt = []

    for turn in conversation[:-1]:
        role = turn["role"]
        content_text = turn["content"]
        img_loc = turn.get("img_loc", None)

        if role == "user" and img_loc is not None:
            text_part = [{"type": "text", "text": content_text}]
            image_parts = [{"type": "image"} for _ in images]

            if img_loc == "before":
                content = image_parts + text_part
            elif img_loc == "after":
                content = text_part + image_parts
            else:
                raise ValueError(
                    f"Unsupported img_loc={img_loc} for id={example['id']}. "
                    "Expected before/after/None."
                )
        else:
            content = [{"type": "text", "text": content_text}]

        prompt.append({"role": role, "content": content})

    completion = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": conversation[-1]["content"]}],
        }
    ]

    return prompt + completion


def open_images(example, image_root=None, input_parent=None):
    pil_images = []

    for img_item in example["images"]:
        img_path = resolve_image_path(
            img_item,
            image_root=image_root,
            input_parent=input_parent,
        )

        img = Image.open(img_path).convert("RGB")
        pil_images.append(img)

    return pil_images


def print_length_stats(lengths, title):
    lengths = np.asarray(lengths, dtype=np.int64)
    valid = lengths[lengths < BAD_LENGTH]

    print(f"\n=== {title} ===")
    print(f"total examples: {len(lengths)}")
    print(f"valid examples: {len(valid)}")
    print(f"bad examples:   {len(lengths) - len(valid)}")

    if len(valid) == 0:
        print("No valid lengths.")
        return

    print(f"mean:  {valid.mean():.1f}")
    print(f"p50:   {np.percentile(valid, 50):.1f}")
    print(f"p90:   {np.percentile(valid, 90):.1f}")
    print(f"p95:   {np.percentile(valid, 95):.1f}")
    print(f"p99:   {np.percentile(valid, 99):.1f}")
    print(f"p99.5: {np.percentile(valid, 99.5):.1f}")
    print(f"max:   {valid.max()}")


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize_qwen(height, width, factor, min_pixels, max_pixels):
    """
    Qwen2-VL / Qwen2.5-VL-style smart resize.

    This mirrors the logic used to decide the visual grid size.
    The visual token count depends on resized H/W, not on image content.
    """
    if height < factor or width < factor:
        raise ValueError(
            f"Image too small: height={height}, width={width}, factor={factor}"
        )

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Extreme aspect ratio: height={height}, width={width}"
        )

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return int(h_bar), int(w_bar)


def get_qwen_image_token_count_from_path(img_item):
    """
    Fast image-token estimate for Qwen2/Qwen2.5-VL.

    It only reads image metadata/size instead of fully preprocessing pixels.
    """
    img_path = resolve_image_path(
        img_item,
        image_root=IMAGE_ROOT,
        input_parent=input_parent,
    )

    with Image.open(img_path) as img:
        width, height = img.size

    image_processor = processor.image_processor

    patch_size = getattr(image_processor, "patch_size", 14)
    merge_size = getattr(image_processor, "merge_size", 2)

    # Qwen processors usually expose these.
    min_pixels = getattr(image_processor, "min_pixels", 56 * 56)
    max_pixels = getattr(image_processor, "max_pixels", 28 * 28 * 1280)

    factor = patch_size * merge_size

    resized_h, resized_w = smart_resize_qwen(
        height=height,
        width=width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    grid_t = 1

    # Qwen processor expands one <|image_pad|> into:
    # image_grid_thw.prod() // merge_size**2 tokens.
    num_image_tokens = (grid_t * grid_h * grid_w) // (merge_size ** 2)

    return int(num_image_tokens)


def count_image_placeholders_in_messages(messages):
    count = 0
    for msg in messages:
        for part in msg["content"]:
            if part["type"] == "image":
                count += 1
    return count


def compute_vlm_length_fast_qwen(example):
    try:
        messages = build_messages_like_training(example)

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Token length with one placeholder token per image.
        text_inputs = processor.tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=False,
        )
        base_text_len = len(text_inputs["input_ids"])

        num_placeholders = count_image_placeholders_in_messages(messages)

        # In your current data format, example["images"] are used as the image list.
        # Usually num_placeholders == len(example["images"]).
        if num_placeholders != len(example["images"]):
            raise ValueError(
                f"Placeholder/image mismatch for id={example.get('id', None)}: "
                f"num_placeholders={num_placeholders}, num_images={len(example['images'])}"
            )

        image_token_counts = [
            get_qwen_image_token_count_from_path(img_item)
            for img_item in example["images"]
        ]

        # base_text_len already counts each image placeholder as 1 token.
        # Replace each placeholder with its expanded visual-token count.
        full_len = base_text_len + sum(t - 1 for t in image_token_counts)

        return {
            "vlm_length": int(full_len),
            "vlm_text_length": int(base_text_len),
            "vlm_image_token_estimate": int(sum(image_token_counts)),
            "vlm_num_images": len(example["images"]),
            "vlm_measurement_error": "",
        }

    except Exception as e:
        ex_id = example.get("id", "UNKNOWN")
        print(f"[ERROR] Failed to measure id={ex_id}: {repr(e)}")

        return {
            "vlm_length": BAD_LENGTH,
            "vlm_text_length": BAD_LENGTH,
            "vlm_image_token_estimate": BAD_LENGTH,
            "vlm_num_images": len(example.get("images", [])),
            "vlm_measurement_error": repr(e),
        }

input_path = Path(INPUT_DATASET)
input_parent = input_path.parent if input_path.exists() else None

print(f"Loading processor: {MODEL_NAME}")
processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)

print(f"Loading dataset: {INPUT_DATASET}")
dataset = load_original_dataset(INPUT_DATASET)

print(f"Original dataset size: {len(dataset)}")

if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(dataset):
    print(f"Subsampling {SAMPLE_SIZE} examples before measurement...")
    dataset = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
    print(f"Subsampled dataset size: {len(dataset)}")

print("Measuring VLM lengths with fast Qwen image-size method...")

dataset = dataset.map(
    compute_vlm_length_fast_qwen,
    num_proc=NUM_PROC,
    desc="Fast measuring VLM sequence lengths",
)

lengths = np.asarray(dataset["vlm_length"], dtype=np.int64)

print_length_stats(
    lengths,
    "Fast-estimated full VLM sequence length: text + image tokens",
)

valid_lengths = lengths[lengths < BAD_LENGTH]

if len(valid_lengths) == 0:
    raise RuntimeError("No valid examples after length measurement.")

thresholds = []

if MAX_VLM_LENGTH is not None:
    thresholds.append(int(MAX_VLM_LENGTH))

if MAX_VLM_LENGTH_PERCENTILE is not None:
    if not (0 < MAX_VLM_LENGTH_PERCENTILE <= 100):
        raise ValueError("MAX_VLM_LENGTH_PERCENTILE must be in (0, 100].")

    percentile_cutoff = int(np.percentile(valid_lengths, MAX_VLM_LENGTH_PERCENTILE))
    thresholds.append(percentile_cutoff)

    print(
        f"Percentile cutoff p{MAX_VLM_LENGTH_PERCENTILE}: "
        f"{percentile_cutoff}"
    )

if len(thresholds) == 0:
    raise ValueError(
        "No filtering threshold was provided. "
        "Set MAX_VLM_LENGTH or MAX_VLM_LENGTH_PERCENTILE."
    )

cutoff = min(thresholds)

print(f"Final VLM length cutoff: {cutoff}")

before = len(dataset)

kept = dataset.filter(
    lambda ex: ex["vlm_length"] <= cutoff,
    desc=f"Filtering vlm_length <= {cutoff}",
)

removed = dataset.filter(
    lambda ex: ex["vlm_length"] > cutoff,
    desc="Collecting removed examples",
)

after = len(kept)

print(
    f"\nFiltered dataset: {before} -> {after} "
    f"removed={before - after} "
    f"removed_pct={(before - after) / max(before, 1) * 100:.2f}%"
)

print_length_stats(kept["vlm_length"], "Kept examples")
print_length_stats(removed["vlm_length"], "Removed examples")

length_columns = [
    "vlm_length",
    "vlm_text_length",
    "vlm_image_token_estimate",
    "vlm_num_images",
    "vlm_measurement_error",
]

kept_to_save = kept
removed_to_save = removed

if not KEEP_LENGTH_COLUMNS:
    kept_to_save = kept_to_save.remove_columns(
        [c for c in length_columns if c in kept_to_save.column_names]
    )
    removed_to_save = removed_to_save.remove_columns(
        [c for c in length_columns if c in removed_to_save.column_names]
    )

print(f"Saving filtered dataset to: {OUTPUT_DATASET}")

if OUTPUT_DATASET.endswith(".parquet"):
    kept_to_save.to_parquet(OUTPUT_DATASET)
else:
    kept_to_save.to_json(
        OUTPUT_DATASET,
        orient="records",
        lines=True,
        force_ascii=False,
    )

if REMOVED_DATASET is not None:
    print(f"Saving removed examples to: {REMOVED_DATASET}")

    if REMOVED_DATASET.endswith(".parquet"):
        removed_to_save.to_parquet(REMOVED_DATASET)
    else:
        removed_to_save.to_json(
            REMOVED_DATASET,
            orient="records",
            lines=True,
            force_ascii=False,
        )

print("Done.")