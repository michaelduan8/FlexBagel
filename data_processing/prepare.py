"""
Convert a HuggingFace dataset to a JSONL file with a standardized format.

Usage example:
    python hf_to_jsonl.py \
        --dataset "squad" \
        --output_path "output.jsonl" \
        --user_content_template "Context: {context}\n\nQuestion: {question}" \
        --assistant_content_template "{answers[text][0]}" \
        --split "train" \
        --max_chars 4000
"""

import re
import json
import argparse
from pathlib import Path
from datasets import load_dataset


def normalize_dataset_name(name: str) -> str:
    """Normalize a HuggingFace dataset name for use as a prompt_id prefix."""
    # Replace slashes, spaces, and other non-alphanumeric chars with underscores
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return normalized.strip("_").lower()


def format_template(template: str, row: dict) -> str:
    """
    Format a template string using values from a dataset row.
    Supports standard Python str.format_map() syntax: {column_name}
    """
    return template.format_map(row)


def convert_dataset_to_jsonl(
    dataset_name: str,
    output_path: str,
    user_content_template: str,
    assistant_content_template: str,
    split: str = "train",
    subset: str | None = None,
    max_chars: int | None = None,
):
    """
    Load a HuggingFace dataset and write it as a JSONL file.

    Args:
        dataset_name:               HuggingFace dataset identifier (e.g. "squad", "openai/gsm8k").
        output_path:                Path to the output .jsonl file.
        user_content_template:      Python format string using dataset column names for the user turn.
        assistant_content_template: Python format string using dataset column names for the assistant turn.
        split:                      Dataset split to use (default: "train").
        subset:                     Optional dataset subset/config name.
        max_chars:                  Optional max combined character length of prompt content +
                                    completion. Rows exceeding this are skipped.
    """
    print(f"Loading dataset '{dataset_name}' (split='{split}', subset={subset!r})...")
    ds = load_dataset(dataset_name, subset, split=split)

    id_prefix = normalize_dataset_name(dataset_name)

    skipped = 0
    records = []

    print(f"Processing {len(ds)} rows...")
    for index, row in enumerate(ds):
        row = dict(row)

        try:
            user_content = format_template(user_content_template, row)
            completion = format_template(assistant_content_template, row)
        except (KeyError, IndexError, TypeError) as e:
            print(f"  [row {index}] Skipping — template formatting error: {e}")
            skipped += 1
            continue

        # Optional pre-filter: skip rows that exceed the character limit
        if max_chars is not None and (len(user_content) + len(completion)) > max_chars:
            skipped += 1
            continue

        records.append({
            "prompt_id": f"{id_prefix}_{index}",
            "prompt": [{"role": "user", "content": user_content}],
            "completion": completion,
        })

    # Ensure output directory exists, then write all records at once
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")

    print(f"Done. Written: {len(records)} rows | Skipped: {skipped} rows → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace dataset to a JSONL file."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="HuggingFace dataset name (e.g. 'squad', 'openai/gsm8k')",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path for the output .jsonl file",
    )
    parser.add_argument(
        "--user_content_template",
        required=True,
        help=(
            "Python format string for the user message content. "
            "Reference dataset columns with {column_name}. "
            "Example: 'Context: {context}\\n\\nQuestion: {question}'"
        ),
    )
    parser.add_argument(
        "--assistant_content_template",
        required=True,
        help=(
            "Python format string for the completion string. "
            "Example: '{answer}'"
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: 'train')",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional dataset subset/config name (e.g. 'main' for gsm8k)",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=None,
        help=(
            "Optional max combined character count of user content + completion. "
            "Rows exceeding this are excluded from the output."
        ),
    )

    args = parser.parse_args()

    convert_dataset_to_jsonl(
        dataset_name=args.dataset,
        output_path=args.output_path,
        user_content_template=args.user_content_template,
        assistant_content_template=args.assistant_content_template,
        split=args.split,
        subset=args.subset,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()