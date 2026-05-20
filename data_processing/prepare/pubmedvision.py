from __future__ import annotations

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Any

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets


def normalize_image_path(image: str, raw_data_dir: Path) -> str:
    # for i in range(1, 20):  # 1 through 19
    # f"images_{i}", 
    target_path = os.path.join(raw_data_dir, image.removeprefix("../"))
    if os.path.exists(target_path):
        return target_path
    return "non_existent_image.jpg"

# def row_has_existing_image(row: dict[str, Any], raw_data_dir: Path) -> bool:
#     for img_path in row["image"]:
#         if not os.path.exists(normalize_image_path(img_path, raw_data_dir)):
#             return False
#     return True

def row_has_existing_image(row: dict[str, Any], raw_data_dir: Path) -> bool:
    return all(image_path != "non_existent_image.jpg" for image_path in row["images"])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a JSON file and convert each row to one JSONL line."
    )
    parser.add_argument("--input1", required=True, help="Path to input JSON file.")
    parser.add_argument("--input2", required=True, help="Path to input JSON file.")
    parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    parser.add_argument(
        "--raw_data_dir",
        required=True,
        help="Local directory that contains raw data assets.",
    )
    return parser.parse_args()


def map_row(row: pd.Series, idx: int, raw_data_dir: Path) -> dict[str, Any]:
    """
    Per row, we remap the conversation and image metadata
    """
    normalized_images = [
        normalize_image_path(img_path, raw_data_dir) for img_path in row["image"]
    ]
    normalized_conversation = []
    for conversation in row["conversations"]:
        role = "user" if conversation["from"] == "human" else "assistant"
        content = conversation["value"]
        img_loc = "before" if role == "user" else None
        normalized_conversation.append({
            "role": role,
            "content": content,
            "img_loc": img_loc,
        })

    assert len(normalized_conversation) == 2, (
        f"Expected exactly 2 turns in conversation, but found "
        f"{len(normalized_conversation)} in row with id {row['id']}."
    )

    raw_images = row["image"]

    return {
        "id": row["id"],
        "orig_images": raw_images,
        "images": normalized_images,
        "conversation": normalized_conversation,
    }


def main() -> None:
    args = parse_args()

    input1 = Path(args.input1)
    input2 = Path(args.input2)
    output = Path(args.output)
    raw_data_dir = Path(args.raw_data_dir)

    dataset = load_dataset("json", data_files=str(input1), split="train")
    dataset2 = load_dataset("json", data_files=str(input2), split="train")
    dataset = concatenate_datasets([dataset, dataset2])
    total = len(dataset)
    print(f"Loaded {total} rows from {input1} and {input2}.")

    # Filter rows where all images exist
    dataset = dataset.map(
        map_row,
        fn_kwargs={"raw_data_dir": raw_data_dir},
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=12,
    )
    mapped_dataset = dataset.filter(
        row_has_existing_image,
        fn_kwargs={"raw_data_dir": raw_data_dir},
        num_proc=12,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    mapped_dataset.to_json(str(output), orient="records", lines=True)

    print(
        f"Converted {len(mapped_dataset)} rows from {input} "
        f"to {output} (raw_data_dir={raw_data_dir})."
    )


if __name__ == "__main__":
    main()

