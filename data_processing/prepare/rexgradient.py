from __future__ import annotations

import argparse
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
    subject_id = id.split('_')[0][1:]

    accession_number = row["AccessionNumber"]
    study_id = row["StudyInstanceUid"]

    images = glob.glob(os.path.join(raw_data_dir, subject_id, accession_number, 'studies', study_id, 'series', '*', 'instances', '*'))
    images = list(images)

    row["images"] = images

    patient_data = f"Sex: {row['PatientSex']}\n\nAge: {row['PatientAge']}\n\nDescription: {row['StudyDescription']}\n\nIndication: {row['Indication']}"
    report = f"Findings: {row['Findings']}\n\nImpression: {row['Impression']}"

    conversation = [
        {"role": "user", "content": f"Please generate a radiology report using the above images collected from a patient study.\n\nPatient Study Details:\n\n{patient_data}"},
        {"role": "assistant", "content": report},
    ]

    row["conversation"] = conversation

    return row


def main() -> None:
    args = parse_args()

    dataset_path = args.dataset
    split = args.split
    output = Path(args.output)
    raw_data_dir = Path(args.raw_data_dir)

    dataset = load_dataset(dataset_path, split=split)
    mapped_dataset = dataset.map(map_row, fn_kwargs={"raw_data_dir": raw_data_dir})

    output.parent.mkdir(parents=True, exist_ok=True)
    mapped_dataset.to_json(str(output), orient="records", lines=True)

    print(
        f"Converted {len(mapped_dataset)} rows from dataset {dataset_path}:{split} "
        f"to {output} (raw_data_dir={raw_data_dir})."
    )


if __name__ == "__main__":
    main()
