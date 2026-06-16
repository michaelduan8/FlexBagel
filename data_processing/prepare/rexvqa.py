from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import re
import pandas as pd
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm


OPTION_PREFIX_PATTERN = re.compile(r"^([A-Z])\.\s+(.*)$")


def normalize_image_path(image: str, raw_data_dir: Path) -> str:
    normalized_image = os.path.join(raw_data_dir, image.removeprefix("../"))
    return normalized_image


def row_has_existing_image(row: dict[str, Any], raw_data_dir: Path) -> bool:
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes to use for filtering and mapping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible option shuffling.",
    )
    return parser.parse_args()


def shuffle_options(
    options: list[str], correct_answer: str, row_id: Any, seed: int | None
) -> tuple[list[str], str]:
    parsed_options: list[tuple[str, str]] = []
    for option in options:
        match = OPTION_PREFIX_PATTERN.match(option.strip())
        if match is None:
            raise ValueError(f"Unexpected option format in row {row_id}: {option!r}")
        parsed_options.append((match.group(1), match.group(2)))

    original_correct_answer = correct_answer.strip().upper().removesuffix(".")
    correct_option_text = None
    for option_letter, option_text in parsed_options:
        if option_letter == original_correct_answer:
            correct_option_text = option_text
            break

    if correct_option_text is None:
        raise ValueError(
            f"Correct answer {correct_answer!r} does not match any option in row {row_id}."
        )

    shuffled_option_texts = [option_text for _, option_text in parsed_options]
    rng = random.Random(f"{seed}:{row_id}" if seed is not None else None)
    rng.shuffle(shuffled_option_texts)

    shuffled_options = []
    updated_correct_answer = None
    for index, option_text in enumerate(shuffled_option_texts):
        option_letter = chr(ord("A") + index)
        shuffled_options.append(f"{option_letter}. {option_text}")
        if option_text == correct_option_text:
            updated_correct_answer = option_letter

    if updated_correct_answer is None:
        raise ValueError(f"Failed to remap correct answer for row {row_id}.")

    return shuffled_options, updated_correct_answer


def map_row(
    row: dict[str, Any], raw_data_dir: Path, seed: int | None
) -> dict[str, Any]:
    """
    Per row, we remap the conversation and image metadata
    """
    normalized_images = [
        normalize_image_path(img_path, raw_data_dir) for img_path in row["ImagePath"]
    ]

    shuffled_options, updated_correct_answer = shuffle_options(
        row["options"], row["correct_answer"], row["id"], seed
    )
    options = "\n\n".join(shuffled_options)
    query = {
        "role": "user",
        "content": f"{row['question']}\n\n{options}\n\nAnswer with only the letter corresponding to the correct answer choice.",
        "img_loc": "before",
    }
    answer = {
        "role": "assistant",
        "content": updated_correct_answer,
        "img_loc": None,
    }
    normalized_conversation = [query, answer]

    assert len(normalized_conversation) == 2, (
        f"Expected exactly 2 turns in conversation, but found "
        f"{len(normalized_conversation)} in row with id {row['id']}."
    )

    return {
        "id": row["id"],
        "orig_images": row["ImagePath"],
        "images": normalized_images,
        "conversation": normalized_conversation,
    }


def main() -> None:
    args = parse_args()

    input = Path(args.input)
    output = Path(args.output)
    raw_data_dir = Path(args.raw_data_dir)
    num_workers = max(1, args.num_workers)
    seed = args.seed

    df = pd.read_json(input, orient="index")
    df = df.reset_index().rename(columns={"index": "id"})
    rows = df.to_dict(orient="records")
    total = len(df)
    print(f"Loaded {total} rows from {input}.")

    # Filter rows where all images exist
    filter_fn = partial(row_has_existing_image, raw_data_dir=raw_data_dir)
    with mp.Pool(processes=num_workers) as pool:
        mask = list(
            tqdm(
                pool.imap(filter_fn, rows, chunksize=64),
                total=total,
                desc="Filtering",
            )
        )

    filtered_rows = [row for row, keep in zip(rows, mask) if keep]
    print(
        f"Filter complete: {len(filtered_rows)} kept, "
        f"{total - len(filtered_rows)} skipped.\n"
    )

    # Map each row to the output schema
    map_fn = partial(map_row, raw_data_dir=raw_data_dir, seed=seed)
    with mp.Pool(processes=num_workers) as pool:
        records = list(
            tqdm(
                pool.imap(map_fn, filtered_rows, chunksize=64),
                total=len(filtered_rows),
                desc="Mapping",
            )
        )

    result_df = pd.DataFrame(records)

    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_json(output, orient="records", lines=True)
    print(f"Saved {len(result_df)} rows to {output} (raw_data_dir={raw_data_dir}).")


if __name__ == "__main__":
    main()

