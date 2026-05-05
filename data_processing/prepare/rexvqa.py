from __future__ import annotations

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Any

from tqdm import tqdm


def normalize_image_path(image: str, raw_data_dir: Path) -> str:
    normalized_image = os.path.join(raw_data_dir, image.removeprefix("../"))
    return normalized_image


def row_has_existing_image(row: pd.Series, raw_data_dir: Path) -> bool:
    for img_path in row["ImagePath"]:
        if not os.path.exists(normalize_image_path(img_path, raw_data_dir)):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a JSON file and convert each row to one JSONL line."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file.")
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
        normalize_image_path(img_path, raw_data_dir) for img_path in row["ImagePath"]
    ]

    options = "\n\n".join(row["options"])
    query = {
        "role": "user",
        "content": f"{row['question']}\n\n{options}\n\nAnswer with only the letter corresponding to the correct answer choice.",
        "img_loc": "before",
    }
    answer = {
        "role": "assistant",
        "content": row["correct_answer"],
        "img_loc": None,
    }
    normalized_conversation = [query, answer]

    assert len(normalized_conversation) == 2, (
        f"Expected exactly 2 turns in conversation, but found "
        f"{len(normalized_conversation)} in row with id {row['id']}."
    )

    return {
        "id": row["id"],
        "images": normalized_images,
        "conversation": normalized_conversation,
    }


def main() -> None:
    args = parse_args()

    input = Path(args.input)
    output = Path(args.output)
    raw_data_dir = Path(args.raw_data_dir)

    df = pd.read_json(input, orient="index")
    df = df.reset_index().rename(columns={"index": "id"})
    total = len(df)
    print(f"Loaded {total} rows from {input}.")

    # Filter rows where all images exist
    mask = []
    for _, row in tqdm(df.iterrows(), total=total, desc="Filtering"):
        mask.append(row_has_existing_image(row, raw_data_dir))

    df = df[mask].reset_index(drop=True)
    print(f"Filter complete: {len(df)} kept, {total - len(df)} skipped.\n")

    # Map each row to the output schema
    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Mapping"):
        records.append(map_row(row, idx, raw_data_dir))

    result_df = pd.DataFrame(records)

    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_json(output, orient="records", lines=True)
    print(f"Saved {len(result_df)} rows to {output} (raw_data_dir={raw_data_dir}).")


if __name__ == "__main__":
    main()

