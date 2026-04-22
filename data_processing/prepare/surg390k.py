from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


def normalize_image_path(image: str, raw_data_dir: Path) -> str:
    normalized_image = os.path.join(raw_data_dir, image.split("finetune_data/")[-1])
    if "EndoVis-VQLA" in normalized_image:
        normalized_image = normalized_image.replace("EndoVis-VQLA", "EndoVis_part")
    return normalized_image


def row_has_existing_image(row: dict[str, Any], raw_data_dir: Path) -> bool:
    return os.path.exists(normalize_image_path(row["image"], raw_data_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a JSON file and convert each row to one JSONL line."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--raw_data_dir",
        required=True,
        help="Local directory that contains raw data assets.",
    )
    return parser.parse_args()


def map_row(row: dict[str, Any], idx: int, raw_data_dir: Path) -> dict[str, Any]:
    """
    Per row, we remap the conversation and image metadata
    """
    image = row["image"]
    conversation = row["conversations"]

    # normalize image to local directory
    normalized_image = normalize_image_path(image, raw_data_dir)

    out = {}
    out["id"] = f"surg390k-{idx:06d}"

    out["images"] = [normalized_image]

    normalized_conversation = []
    for turn in conversation:
        role = "user" if turn["from"] == "human" else "assistant"
        content = turn["value"]

        parsed_turn = {
            "role": role,
            "content": content,
        }
        if role == "user":
            parsed_turn["img_loc"] = "before"
        else:
            parsed_turn["img_loc"] = None
            
        normalized_conversation.append(parsed_turn)

    assert len(normalized_conversation) == 2, f"Expected exactly 2 turns in conversation, but found {len(normalized_conversation)} in row with id {out['id']}."

    out["conversation"] = normalized_conversation

    # Save remaining keys as metadata if they exist
    # metadata = {
    #     key: value
    #     for key, value in row.items()
    #     if key not in {"image", "conversations"}
    # }
    # if metadata:
    #     out["metadata"] = metadata
    # else:
    #     out["metadata"] = None

    return out


def main() -> None:
    args = parse_args()

    input = Path(args.input)
    output = Path(args.output)
    raw_data_dir = Path(args.raw_data_dir)

    dataset = load_dataset("json", data_files=str(input), split="train")
    dataset = dataset.filter(
        row_has_existing_image,
        fn_kwargs={"raw_data_dir": raw_data_dir},
        num_proc=12,
    )
    mapped_dataset = dataset.map(
        map_row,
        fn_kwargs={"raw_data_dir": raw_data_dir},
        with_indices=True,
        remove_columns=dataset.column_names,
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
