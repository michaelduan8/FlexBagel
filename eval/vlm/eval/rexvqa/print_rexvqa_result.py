#!/usr/bin/env python3
"""
Compute accuracy from a JSON file containing a list of prediction records.

Each item in the JSON list should be a dict with at least:
    - "answer":    the predicted answer
    - "answer_gt": the ground-truth answer

Usage:
    python compute_accuracy.py /path/to/file.json
"""

import argparse
import json
import sys


def compute_accuracy(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: expected the JSON file to contain a list of dicts.", file=sys.stderr)
        sys.exit(1)

    total = 0
    correct = 0
    missing_fields = 0

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Warning: item {idx} is not a dict, skipping.", file=sys.stderr)
            continue

        if "answer" not in item or "answer_gt" not in item:
            missing_fields += 1
            print(
                f"Warning: item {idx} (id={item.get('id', 'unknown')}) "
                f"missing 'answer' or 'answer_gt' field, skipping.",
                file=sys.stderr,
            )
            continue

        total += 1
        pred = str(item["answer"]).strip()
        gt = str(item["answer_gt"]).strip()

        if pred == gt:
            correct += 1

    if total == 0:
        print("No valid records found to score.")
        return

    accuracy = correct / total
    print(f"Total scored examples: {total}")
    if missing_fields:
        print(f"Skipped (missing fields): {missing_fields}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy (answer == answer_gt) from a JSON file (list of dicts)."
    )
    parser.add_argument("json_path", help="Path to the input .json file")
    args = parser.parse_args()

    compute_accuracy(args.json_path)


if __name__ == "__main__":
    main()