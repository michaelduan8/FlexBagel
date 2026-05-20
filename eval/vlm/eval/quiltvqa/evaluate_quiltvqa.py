"""
Batched multimodal inference with data-parallel sharding across GPUs.
Each worker process owns one GPU and one shard of the test data.
"""

import argparse
import json
import os
import re
import string
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import List

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class ResultEntry:
    id:        str
    images:    List[str]
    question:  str
    answer:    str
    answer_gt: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched multimodal inference with vLLM.")
    parser.add_argument("--model_id",          type=str, required=True,  help="Path to model checkpoint")
    parser.add_argument("--dataset",           type=str, required=True,  help="Path to test data JSON or JSONL")
    parser.add_argument("--raw_data_dir",       type=str, required=True, help="Directory containing raw data (e.g. images)")
    parser.add_argument("--result_folder",      type=str, required=True,  help="Directory to save results")
    parser.add_argument("--batch_size",         type=int, default=64,   help="Items per batch per worker")
    parser.add_argument("--num_gpus",           type=int, default=1,     help="Number of GPUs (data-parallel workers)")
    return parser.parse_args()


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(model_id: str) -> LLM:
    return LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 10},
    )


def get_sampling_params() -> SamplingParams:
    return SamplingParams(temperature=0, max_tokens=1024)


# ── Data helpers ───────────────────────────────────────────────────────────────

def build_conversation(item, raw_data_dir) -> dict:
    """Load the image for one item and format it into vLLM chat format."""
    question = item["question"]
    image = item["image"]

    image_path = os.path.join(raw_data_dir, image)
    assert os.path.exists(image_path), f"Image not found: {image_path}"

    return [{
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": question}
        ],
    }]

# ── Inference ──────────────────────────────────────────────────────────────────

def run_worker(
    rank:            int,
    model_id:        str,
    shard:           list[dict],
    raw_data_dir:    str,
    batch_size:      int,
    result_queue:    mp.Queue,
) -> None:
    """Worker function: owns one GPU, processes one data shard."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    llm             = load_model(model_id)
    sampling_params = get_sampling_params()
    results         = []
    num_batches     = (len(shard) + batch_size - 1) // batch_size

    for batch_idx in tqdm(
        range(num_batches),
        desc=f"GPU {rank}",
        position=rank,
        leave=True,
    ):
        start  = batch_idx * batch_size
        batch_items = shard[start : start + batch_size]
        batch = [build_conversation(item, raw_data_dir) for item in batch_items]

        outputs = llm.chat(batch, sampling_params)
        results.extend(outputs)

    result_queue.put((rank, results))  # include rank so ordering is correct


def run_data_parallel(
    model_id:   str,
    test_data:  list[dict],
    raw_data_dir: str,
    batch_size: int,
    num_gpus:   int,
) -> list[ResultEntry]:
    """Shard data across `num_gpus` workers and collect results."""
    shards = [test_data[i::num_gpus] for i in range(num_gpus)]
    queue  = mp.Queue()

    procs = [
        mp.Process(
            target=run_worker,
            args=(rank, model_id, shards[rank], raw_data_dir, batch_size, queue),
        )
        for rank in range(num_gpus)
    ]

    for p in procs: p.start()
    all_results = [queue.get() for _ in procs]   # one (rank, results) tuple per worker
    for p in procs: p.join()

    # Interleave to restore original order (shard[i::num_gpus] → inverse)
    n       = sum(len(r) for _, r in all_results)
    ordered = [None] * n
    for rank, worker_results in all_results:      # rank is now correct regardless of queue order
        for local_idx, entry in enumerate(worker_results):
            ordered[local_idx * num_gpus + rank] = entry

    return ordered


# ── I/O & metrics ──────────────────────────────────────────────────────────────

def save_results(results: list[ResultEntry], folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "test_result.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    return path


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    test_metadata = [dict(item) for item in dataset]

    print(f"Loaded {len(test_metadata)} test items")
    print(f"First: {test_metadata[0]}  |  Last: {test_metadata[-1]}")
    print(f"Model: {args.model_id}  |  GPUs: {args.num_gpus}")

    results = run_data_parallel(
        model_id=args.model_id,
        test_data=test_metadata,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
    )

    # TODO: add outputs to test_metadata
    for meta, res in zip(test_metadata, results):
        output = res.outputs[0].text.strip()
        meta["pred"] = output

    result_path = save_results(test_metadata, args.result_folder)
    print(f"Done. {len(test_metadata)} results saved to {result_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn")   # required for CUDA + multiprocessing
    main()