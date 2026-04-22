from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a Hugging Face dataset and convert each row to one JSONL line."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset path, e.g. org/dataset_name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train).",
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


def map_row(row, raw_data_dir: Path) -> dict[str, Any]:
    """
    Per row, we remap the conversation and image metadata
    """
    id = row["id"]
    image = os.path.join(raw_data_dir, row["image"])
    
    row["images"] = [image]

    conversation = row["conversations"]

    parsed = []
    counter = 0
    for turn in conversation:
        role = "user" if turn["from"] == "human" else "assistant"
        content = turn["value"]

        img_loc = None
        if "<image>" in content:
            assert role == "user", f"Expected image tag to only appear in user turns, but found in assistant turn in row with id {id}."
            counter += 1
            if content.startswith("<image>"):
                img_loc = "before"
            elif content.endswith("<image>"):
                img_loc = "after"
            else:
                raise ValueError(f"Unexpected image tag location in conversation content: {content}")

            content = content.replace("<image>", "").strip()
        
        parsed_turn = {
            "role": role,
            "content": content,
            "img_loc": img_loc,
        }

        parsed.append(parsed_turn)
    
    assert counter == 1, f"Expected exactly one image tag in conversation content, but found {counter} in row with id {id}."

    row["conversation"] = parsed

    return row


def main() -> None:
    args = parse_args()

    dataset_path = args.dataset
    split = args.split
    output = Path(args.output)
    raw_data_dir = Path(args.raw_data_dir)

    #dataset = load_dataset(dataset_path, split=split)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.filter(
        lambda row, raw_data_dir: os.path.exists(os.path.join(raw_data_dir, row["image"])),
        fn_kwargs={"raw_data_dir": raw_data_dir},
        num_proc=12,
    )
    mapped_dataset = dataset.map(
        map_row, 
        fn_kwargs={"raw_data_dir": raw_data_dir},
        remove_columns=dataset.column_names, 
        num_proc=12
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    mapped_dataset.to_json(str(output), orient="records", lines=True)

    print(
        f"Converted {len(mapped_dataset)} rows from dataset {dataset_path}:{split} "
        f"to {output} (raw_data_dir={raw_data_dir})."
    )


if __name__ == "__main__":
    main()
