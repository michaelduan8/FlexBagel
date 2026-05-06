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
    parser.add_argument("--test_data_path",     type=str, required=True,  help="Path to test data JSONL")
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
        max_model_len=8192,
        limit_mm_per_prompt={"image": 25},
    )


def get_sampling_params() -> SamplingParams:
    return SamplingParams(temperature=0, max_tokens=1)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_test_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        items = []
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                line = json.loads(line)
                if len(line["images"]) > 10:
                    print(len(line["images"]))
                    print(line["conversation"])
                    print(f"Skipping item with more than 10 images (id={line['id']})")
                    continue

                items.append(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        return items


def resolve_image(image_path: str) -> Image.Image:
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    return Image.open(image_path)


def build_conversation(item: dict) -> tuple[list[dict], str]:
    """Format a single item into vLLM chat format.
    Returns (conversation, ground_truth_response)."""
    turns = item["conversation"]
    if len(turns) != 2:
        raise ValueError(f"Expected 2 turns, got {len(turns)} (id={item['id']})")

    images = [
        {"type": "image_pil", "image_pil": resolve_image(p)}
        for p in item["images"]
    ]
    conversation = [{
        "role": "user",
        "content": images + [{"type": "text", "text": turns[0]["content"]}],
    }]
    return conversation, turns[1]["content"]


def prepare_batch(items: list[dict]) -> tuple[list[list[dict]], list[dict]]:
    """Build conversations and metadata for a batch."""
    conversations, metadata = [], []
    for item in items:
        conversation, gt_response = build_conversation(item)
        conversations.append(conversation)
        metadata.append({
            "id":        item["id"],
            "images":    item["images"],
            "question":  conversation[0]["content"][-1]["text"],
            "answer_gt": gt_response,
        })
    return conversations, metadata


# ── Inference ──────────────────────────────────────────────────────────────────

def run_worker(
    rank:            int,
    model_id:        str,
    shard:           list[dict],
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
        batch  = shard[start : start + batch_size]

        conversations, metadata = prepare_batch(batch)
        outputs = llm.chat(conversations, sampling_params)

        for output, meta in zip(outputs, metadata):
            results.append(ResultEntry(
                id=meta["id"],
                images=meta["images"],
                question=meta["question"],
                answer=output.outputs[0].text.strip(),
                answer_gt=meta["answer_gt"],
            ))

        del conversations, metadata, outputs

    result_queue.put(results)


def run_data_parallel(
    model_id:   str,
    test_data:  list[dict],
    batch_size: int,
    num_gpus:   int,
) -> list[ResultEntry]:
    """Shard data across `num_gpus` workers and collect results."""
    shards = [test_data[i::num_gpus] for i in range(num_gpus)]
    queue  = mp.Queue()

    procs = [
        mp.Process(
            target=run_worker,
            args=(rank, model_id, shards[rank], batch_size, queue),
        )
        for rank in range(num_gpus)
    ]

    for p in procs: p.start()
    all_results = [queue.get() for _ in procs]   # one list per worker
    for p in procs: p.join()

    # Interleave to restore original order (shard[i::num_gpus] → inverse)
    n       = sum(len(r) for r in all_results)
    ordered = [None] * n
    for rank, worker_results in enumerate(all_results):
        for local_idx, entry in enumerate(worker_results):
            ordered[local_idx * num_gpus + rank] = entry
    return ordered


# ── I/O & metrics ──────────────────────────────────────────────────────────────

def save_results(results: list[ResultEntry], folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "test_result.json")
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=4)
    return path


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def compute_accuracy(results: list[ResultEntry]) -> dict:
    if not results:
        return {"correct": 0, "total": 0, "accuracy": 0.0}
    correct = sum(1 for r in results if normalize(r.answer) == normalize(r.answer_gt))
    total   = len(results)
    return {"correct": correct, "total": total, "accuracy": correct / total}


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    test_data = load_test_data(args.test_data_path)

    print(f"Loaded {len(test_data)} test items")
    print(f"First: {test_data[0]}  |  Last: {test_data[-1]}")
    print(f"Model: {args.model_id}  |  GPUs: {args.num_gpus}")

    results = run_data_parallel(
        model_id=args.model_id,
        test_data=test_data,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
    )

    result_path = save_results(results, args.result_folder)
    print(f"Done. {len(results)} results saved to {result_path}")

    metrics = compute_accuracy(results)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")


if __name__ == "__main__":
    mp.set_start_method("spawn")   # required for CUDA + multiprocessing
    main()