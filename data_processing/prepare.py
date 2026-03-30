"""
Convert a HuggingFace dataset to a JSONL file with a standardized format.

Usage example:
    python hf_to_jsonl.py \
        --dataset "squad" \
        --output_path "output.jsonl" \
        --user_content_template "Context: {context}\n\nQuestion: {question}" \
        --assistant_content_template "{messages[1][\"content\"]}" \
        --split "train" \
        --num_samples 10000 \
        --max_chars 4000
"""

import re
import json
import string
import argparse
import ast
import random
from pathlib import Path
from datasets import load_dataset

from tqdm import tqdm


def normalize_dataset_name(name: str) -> str:
    """Normalize a HuggingFace dataset name for use as a prompt_id prefix."""
    # Replace slashes, spaces, and other non-alphanumeric chars with underscores
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return normalized.strip("_").lower()


def format_template(template: str, row: dict) -> str:
    """
    Format a template string using values from a dataset row.
    Supports nested path access in replacement fields, including:
      - {column_name}
      - {answers[text][0]}
      - {messages[1]["content"]}
    """
    return DeepTemplateFormatter().format(template, **row)


def row_matches_filters(row: dict, column_filters: dict[str, object]) -> bool:
    """
    Check whether a row satisfies all column filters.

    Filter keys can use the same nested accessor syntax as templates, e.g.:
      - prompt_language
      - metadata.lang
      - messages[2]["role"]

    Filter values:
      - scalar: require exact equality
      - list: row value must be one of the listed values
    """
    if not column_filters:
        return True

    formatter = DeepTemplateFormatter()
    for field_path, expected in column_filters.items():
        try:
            actual = formatter._resolve_field_path(field_path, row)
        except (KeyError, IndexError, TypeError):
            return False

        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False

    return True


class DeepTemplateFormatter(string.Formatter):
    """Formatter that supports deeply nested field access with quoted keys."""

    _INDEX_PATTERN = re.compile(r"^-?\d+$")

    def get_field(self, field_name, args, kwargs):
        value = self._resolve_field_path(field_name, kwargs)
        return value, field_name

    def _resolve_field_path(self, field_name: str, context: dict):
        root, accessors = self._parse_field_name(field_name)

        if root not in context:
            raise KeyError(root)

        value = context[root]
        for accessor_type, accessor in accessors:
            if accessor_type == "attr":
                if isinstance(value, dict) and accessor in value:
                    value = value[accessor]
                else:
                    value = getattr(value, accessor)
                continue

            value = value[accessor]

        return value

    def _parse_field_name(self, field_name: str):
        i = 0
        n = len(field_name)

        # Parse the root key before any .attr or [index] access.
        while i < n and field_name[i] not in ".[":
            i += 1

        root = field_name[:i].strip()
        if not root:
            raise KeyError("Empty template field")

        accessors = []
        while i < n:
            ch = field_name[i]
            if ch == ".":
                i += 1
                start = i
                while i < n and field_name[i] not in ".[":
                    i += 1
                attr = field_name[start:i].strip()
                if not attr:
                    raise KeyError(f"Invalid attribute accessor in '{field_name}'")
                accessors.append(("attr", attr))
                continue

            if ch == "[":
                end = field_name.find("]", i + 1)
                if end == -1:
                    raise KeyError(f"Missing closing bracket in '{field_name}'")

                token = field_name[i + 1:end].strip()
                if not token:
                    raise KeyError(f"Empty bracket accessor in '{field_name}'")

                accessors.append(("item", self._parse_index_token(token)))
                i = end + 1
                continue

            raise KeyError(f"Invalid accessor syntax in '{field_name}'")

        return root, accessors

    def _parse_index_token(self, token: str):
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
            try:
                return ast.literal_eval(token)
            except (ValueError, SyntaxError) as exc:
                raise KeyError(f"Invalid quoted accessor token: {token}") from exc

        if self._INDEX_PATTERN.fullmatch(token):
            return int(token)

        # Preserve compatibility with standard format syntax like [text].
        return token


def convert_dataset_to_jsonl(
    dataset_name: str,
    output_path: str,
    user_content_template: str,
    assistant_content_template: str,
    split: str = "train",
    subset: str | None = None,
    num_samples: int | None = None,
    max_chars: int | None = None,
    column_filters: dict[str, object] | None = None,
    seed: int | None = None,
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
        num_samples:                Optional number of output samples to write after processing
                                    the full dataset and randomly sampling valid rows.
        max_chars:                  Optional max combined character length of prompt content +
                                    completion. Rows exceeding this are skipped.
        column_filters:             Optional dict of field path -> expected value filters.
                                    All filters must match for a row to be included.
        seed:                       Optional random seed used when --num_samples performs
                                    random downsampling.
    """
    print(f"Loading dataset '{dataset_name}' (split='{split}', subset={subset!r})...")
    ds = load_dataset(dataset_name, subset, split=split, num_proc=6)

    id_prefix = normalize_dataset_name(dataset_name)

    skipped = 0
    records = []

    print(f"Processing {len(ds)} rows...")
    for index, row in tqdm(enumerate(ds)):
        row = dict(row)
        if index % 10000 == 0:
            print(row)
        # assert len(row["messages"]) == 3
        # assert row["messages"][0]["role"] == "system" and row["messages"][0]["content"] == "" and row["messages"][1]["role"] == "user" and row["messages"][2]["role"] == "assistant"
        # assert "content" in row["messages"][2] and "reasoning_content" in row["messages"][2]

        # Optional row filter
        if column_filters and not row_matches_filters(row, column_filters):
            skipped += 1
            continue

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

    if num_samples is not None and len(records) > num_samples:
        rng = random.Random(seed)
        records = rng.sample(records, num_samples)

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
            "Reference dataset columns with {column_name} and nested paths like "
            "{messages[1][\"content\"]}. "
            "Example: 'Context: {context}\\n\\nQuestion: {question}'"
        ),
    )
    parser.add_argument(
        "--assistant_content_template",
        required=True,
        help=(
            "Python format string for the completion string. "
            "Examples: '{answer}', '{answers[text][0]}', '{messages[1][\"content\"]}'"
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
        "--num_samples",
        type=int,
        default=None,
        help=(
            "Optional number of output samples to write. "
            "Processes the full dataset first, then randomly samples this many valid rows."
        ),
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
    parser.add_argument(
        "--column_filters",
        type=str,
        default=None,
        help=(
            "Optional JSON object mapping field paths to required values. "
            "Rows must match all filters to be included. "
            "Example: '{\"prompt_language\":\"en\",\"answer_language\":\"en\"}'. "
            "You can also use nested paths like '{\"messages[2]['role']\":\"assistant\"}'. "
            "If a value is a list, any listed value is accepted."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic row sampling when --num_samples is set.",
    )

    args = parser.parse_args()

    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num_samples must be a positive integer")

    column_filters = None
    if args.column_filters is not None:
        try:
            parsed_filters = json.loads(args.column_filters)
        except json.JSONDecodeError as e:
            parser.error(f"--column_filters must be valid JSON: {e}")
        if not isinstance(parsed_filters, dict):
            parser.error("--column_filters must be a JSON object (dictionary)")
        if any(not isinstance(k, str) for k in parsed_filters.keys()):
            parser.error("--column_filters keys must be strings")
        column_filters = parsed_filters

    convert_dataset_to_jsonl(
        dataset_name=args.dataset,
        output_path=args.output_path,
        user_content_template=args.user_content_template,
        assistant_content_template=args.assistant_content_template,
        split=args.split,
        subset=args.subset,
        num_samples=args.num_samples,
        max_chars=args.max_chars,
        column_filters=column_filters,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
