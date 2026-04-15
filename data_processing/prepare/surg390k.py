from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from data_processing.utils import load_json, write

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


def map_row(idx: int, row: dict[str, Any], raw_data_dir: Path) -> dict[str, Any]:
    """
    Per row, we remap the conversation and image metadata
    """
    out = {}
    out["id"] = f"surg390k-{idx:06d}"
    
    image = row["image"]
    conversation = row["conversations"]

    # normalize image to local directory
    normalized_image = os.path.join(raw_data_dir, image.split("finetune_data/")[-1])
    assert os.path.exists(normalized_image), f"Image file {normalized_image} does not exist."

    out["images"] = [normalized_image]

    normalized_conversation = []
    for turn in conversation:
        role = "user" if turn["from"] == "human" else "assistant"
        content = turn["value"]

        normalized_conversation.append({"role": role, "content": content})

    out["conversation"] = normalized_conversation

    # Save remaining keys as metadata if they exist
    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"image", "conversations"}
    }
    if metadata:
        out["metadata"] = metadata

    return out


def main() -> None:
    args = parse_args()

    input = Path(args.input)
    output = Path(args.output)
    raw_data_dir = Path(args.raw_data_dir)

    metadata = load_json(input)
    mapped_records = [map_row(idx, row, raw_data_dir) for idx, row in enumerate(metadata)]
    write(output, mapped_records)

    print(
        f"Converted {len(mapped_records)} rows from {input} "
        f"to {output} (raw_data_dir={raw_data_dir})."
    )


if __name__ == "__main__":
    main()
