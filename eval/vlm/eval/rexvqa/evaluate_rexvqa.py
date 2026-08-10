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
from typing import Any, List

from PIL import Image
from tqdm import tqdm

from modeling.flex_qwen2_5_vl_moe.monitor import MoeRoutingMonitor


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
    parser = argparse.ArgumentParser(description="Batched multimodal inference for REXVQA.")
    parser.add_argument("--model_id", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data JSONL")
    parser.add_argument("--result_folder", type=str, required=True, help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=64, help="Items per batch per worker")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs (data-parallel workers)")
    parser.add_argument(
        "--inference_backend",
        type=str,
        choices=("vllm", "transformers"),
        default="vllm",
        help="Model loading and generation backend",
    )
    parser.add_argument(
        "--transformers_batch_size",
        type=int,
        default=None,
        help="Per-worker micro-batch size used only when --inference_backend=transformers; defaults to the full worker batch",
    )
    parser.add_argument(
        "--norm_topk_prob",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable norm_topk_prob on both text and vision MoE configs.",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=None,
        help="Override num_experts_per_tok on both text and vision configs when provided.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt prepended to every conversation. Pass an empty string ('') to suppress the template default.",
    )
    parser.add_argument(
        "--router_debug",
        action="store_true",
        help="Enable router debug mode.",
    )
    return parser.parse_args()


# ── Model ──────────────────────────────────────────────────────────────────────

def load_vllm_model(model_id: str):
    from vllm import LLM

    return LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 25},
    )


def get_vllm_sampling_params():
    from vllm import SamplingParams

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


def build_conversation(item: dict, system_prompt: str | None = None) -> tuple[list[dict], str]:
    """Format a single item into vLLM chat format.
    Returns (conversation, ground_truth_response)."""
    turns = item["conversation"]
    if len(turns) != 2:
        raise ValueError(f"Expected 2 turns, got {len(turns)} (id={item['id']})")

    images = [
        {"type": "image_pil", "image_pil": resolve_image(p)}
        for p in item["images"]
    ]
    conversation = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({
        "role": "user",
        "content": images + [{"type": "text", "text": turns[0]["content"]}],
    })
    return conversation, turns[1]["content"]


def prepare_batch(
    items: list[dict],
    system_prompt: str | None = None,
) -> tuple[list[list[dict]], list[dict]]:
    """Build conversations and metadata for a batch."""
    conversations, metadata = [], []
    for item in items:
        conversation, gt_response = build_conversation(item, system_prompt)
        conversations.append(conversation)
        metadata.append({
            "id":        item["id"],
            "images":    item["images"],
            "question":  turns_to_question(conversation),
            "answer_gt": gt_response,
        })
    return conversations, metadata


def turns_to_question(conversation: list[dict]) -> str:
    last_turn = conversation[-1]
    content = last_turn["content"]
    return content[-1]["text"]


def _extract_pil_images_from_turns(turns: list[dict]) -> list[Image.Image]:
    images = []
    for turn in turns:
        content = turn.get("content", [])
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_pil":
                    img = block["image_pil"]
                    images.append(img.convert("RGB") if img.mode != "RGB" else img)
    return images


def _turns_to_processor_messages(turns: list[dict]) -> list[dict]:
    messages = []
    for turn in turns:
        content = turn["content"]
        if isinstance(content, str):
            messages.append({"role": turn["role"], "content": content})
            continue

        new_content = []
        for block in content:
            if block["type"] == "image_pil":
                new_content.append({"type": "image"})
            else:
                new_content.append(block)
        messages.append({"role": turn["role"], "content": new_content})
    return messages


def get_vision_info(turns_batch: list[list[dict]]) -> tuple[Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        return None, None

    try:
        result = process_vision_info(turns_batch)
        return result[0], result[1]
    except (TypeError, ValueError):
        return None, None


def preprocess_transformers_batch(conversations: list[list[dict]], processor: "AutoProcessor") -> Any:
    assert processor.tokenizer.padding_side == "left", (
        f"Expected left padding, got '{processor.tokenizer.padding_side}'. "
        "Set processor.tokenizer.padding_side = 'left' in load_processor()."
    )

    messages_batch = [_turns_to_processor_messages(turns) for turns in conversations]
    image_inputs, video_inputs = get_vision_info(conversations)
    fallback_images_flat = [
        img
        for turns in conversations
        for img in _extract_pil_images_from_turns(turns)
    ]
    prompts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_batch
    ]

    processor_kwargs: dict[str, Any] = {
        "text": prompts,
        "padding": True,
        "return_tensors": "pt",
        "images": image_inputs if image_inputs else fallback_images_flat,
    }
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs

    return processor(**processor_kwargs)


def run_transformers_generate(
    conversations: list[list[dict]],
    metadata: list[dict],
    processor: "AutoProcessor",
    model: Any,
    micro_batch_size: int | None,
) -> list[ResultEntry]:
    import torch

    results = []
    batch_size = micro_batch_size or len(conversations)

    for start in range(0, len(conversations), batch_size):
        conv_chunk = conversations[start : start + batch_size]
        meta_chunk = metadata[start : start + batch_size]

        model_inputs = preprocess_transformers_batch(conv_chunk, processor)
        model_inputs = {
            key: value.to("cuda") if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        padded_prompt_len = model_inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=1,
                pad_token_id=processor.tokenizer.pad_token_id
                if processor.tokenizer.pad_token_id is not None
                else processor.tokenizer.eos_token_id,
            )

        decoded = [
            processor.tokenizer.decode(
                generated_ids[i, padded_prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i in range(len(conv_chunk))
        ]

        for meta, text in zip(meta_chunk, decoded):
            results.append(ResultEntry(
                id=meta["id"],
                images=meta["images"],
                question=meta["question"],
                answer=text.strip(),
                answer_gt=meta["answer_gt"],
            ))

    return results


def load_processor(model_id: str) -> "AutoProcessor":
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    return processor


def register_local_transformers_architectures() -> None:
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText
        from modeling.flex_qwen2_5_vl_moe import (
            Flex_Qwen2_5_VLMoeConfig,
            Flex_Qwen2_5_VLMoeForConditionalGeneration,
        )
    except ImportError:
        return

    try:
        AutoConfig.register("flex_qwen2_5_vl_moe", Flex_Qwen2_5_VLMoeConfig)
    except ValueError:
        pass

    try:
        AutoModelForImageTextToText.register(
            Flex_Qwen2_5_VLMoeConfig,
            Flex_Qwen2_5_VLMoeForConditionalGeneration,
        )
    except ValueError:
        pass


def load_transformers_model(
    model_id: str,
    norm_topk_prob: bool,
    num_experts_per_tok: int | None,
    router_debug: bool = False,
) -> Any:
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    from transformers.utils import is_flash_attn_2_available

    register_local_transformers_architectures()

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if getattr(config, "text_config", None) is not None:
        config.text_config.norm_topk_prob = norm_topk_prob
        if num_experts_per_tok is not None:
            config.text_config.num_experts_per_tok = num_experts_per_tok
        config.text_config.output_router_logits = False
    if getattr(config, "vision_config", None) is not None:
        config.vision_config.norm_topk_prob = norm_topk_prob
        if num_experts_per_tok is not None:
            config.vision_config.num_experts_per_tok = num_experts_per_tok
        config.vision_config.output_router_logits = False

    use_bf16 = torch.cuda.is_bf16_supported()
    model_kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
        "trust_remote_code": True,
    }
    if is_flash_attn_2_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    routing_monitor = MoeRoutingMonitor(
        model,
        skip_first_call_per_layer=False,
        track_transition_matrix=True
    ).register() if router_debug else None
    return model.eval().to("cuda"), routing_monitor


# ── Inference ──────────────────────────────────────────────────────────────────

def run_worker(
    rank: int,
    model_id: str,
    shard: list[dict],
    batch_size: int,
    inference_backend: str,
    transformers_batch_size: int | None,
    router_debug: bool,
    norm_topk_prob: bool,
    num_experts_per_tok: int | None,
    system_prompt: str | None,
    result_queue: mp.Queue,
) -> None:
    """Worker function: owns one GPU, processes one data shard."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if inference_backend == "vllm":
        model = load_vllm_model(model_id)
        sampling_params = get_vllm_sampling_params()
        processor = None
        routing_monitor = None
    else:
        model, routing_monitor = load_transformers_model(
            model_id,
            norm_topk_prob,
            num_experts_per_tok,
            router_debug=router_debug,
        )
        processor = load_processor(model_id)
        sampling_params = None

    results = []
    num_batches = (len(shard) + batch_size - 1) // batch_size

    for batch_idx in tqdm(
        range(num_batches),
        desc=f"GPU {rank}",
        position=rank,
        leave=True,
    ):
        start  = batch_idx * batch_size
        batch  = shard[start : start + batch_size]

        conversations, metadata = prepare_batch(batch, system_prompt)

        if inference_backend == "vllm":
            outputs = model.chat(conversations, sampling_params)

            for output, meta in zip(outputs, metadata):
                results.append(ResultEntry(
                    id=meta["id"],
                    images=meta["images"],
                    question=meta["question"],
                    answer=output.outputs[0].text.strip(),
                    answer_gt=meta["answer_gt"],
                ))
            del outputs
        else:
            results.extend(
                run_transformers_generate(
                    conversations,
                    metadata,
                    processor,
                    model,
                    transformers_batch_size,
                )
            )

        del conversations, metadata

    if router_debug and routing_monitor is not None:
        routing_monitor.print_grouped_summary()
        routing_monitor.print_top_transitions(grouped=True, top_n=10)
        routing_monitor.remove()

    result_queue.put((rank, results))


def run_data_parallel(
    model_id: str,
    test_data: list[dict],
    batch_size: int,
    num_gpus: int,
    inference_backend: str,
    transformers_batch_size: int | None,
    router_debug: bool,
    norm_topk_prob: bool,
    num_experts_per_tok: int | None,
    system_prompt: str | None,
) -> list[ResultEntry]:
    """Shard data across `num_gpus` workers and collect results."""
    shards = [test_data[i::num_gpus] for i in range(num_gpus)]
    queue = mp.Queue()

    procs = [
        mp.Process(
            target=run_worker,
            args=(
                rank,
                model_id,
                shards[rank],
                batch_size,
                inference_backend,
                transformers_batch_size,
                router_debug,
                norm_topk_prob,
                num_experts_per_tok,
                system_prompt,
                queue,
            ),
        )
        for rank in range(num_gpus)
    ]

    for p in procs:
        p.start()
    all_results = {}
    for _ in procs:
        rank, worker_results = queue.get()
        all_results[rank] = worker_results
    for p in procs:
        p.join()

    # Interleave to restore original order (shard[i::num_gpus] → inverse)
    n = sum(len(r) for r in all_results.values())
    ordered = [None] * n
    for rank in range(num_gpus):
        worker_results = all_results.get(rank, [])
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
    if args.router_debug:
        print("Limiting it to 100 items for router debug mode")
        test_data = test_data[:100]

    print(f"Loaded {len(test_data)} test items")
    print(f"First: {test_data[0]}  |  Last: {test_data[-1]}")
    print(
        f"Model: {args.model_id}  |  Backend: {args.inference_backend}  |  GPUs: {args.num_gpus}"
    )

    results = run_data_parallel(
        model_id=args.model_id,
        test_data=test_data,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        inference_backend=args.inference_backend,
        transformers_batch_size=args.transformers_batch_size,
        router_debug=args.router_debug,
        norm_topk_prob=args.norm_topk_prob,
        num_experts_per_tok=args.num_experts_per_tok,
        system_prompt=args.system_prompt,
    )

    result_path = save_results(results, args.result_folder)
    print(f"Done. {len(results)} results saved to {result_path}")

    metrics = compute_accuracy(results)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")


if __name__ == "__main__":
    mp.set_start_method("spawn")   # required for CUDA + multiprocessing
    main()