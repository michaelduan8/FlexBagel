# alrope/surg390k_qwen2_5-3b-vl  —  data-parallel multi-GPU edition
#
# Architecture (ported from the batched multimodal inference reference):
#   - num_gpus persistent worker processes, each owning one GPU and a slice
#     of CPU cores (via sched_setaffinity).
#   - prepared_fewshot_examples is pickled ONCE at worker-spawn time (not
#     repeated per task) to avoid re-serialising images on every batch.
#   - Main process splits each large batch into sub-batches, dispatches raw
#     items through per-worker queues, collects results at original indices.
#
# Inference backends:
#   - vllm (default): uses llm.chat with image_pil content blocks.
#   - transformers: uses HuggingFace AutoModelForImageTextToText with
#     optional micro-batching (--transformers_batch_size).

import argparse
import json
import math
import multiprocessing as mp
import os
import random
from typing import Any

from PIL import Image
from tqdm import tqdm

from modeling.flex_qwen2_5_vl_moe.monitor import MoeRoutingMonitor


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",         type=str,                           help="id for the eval session")
    parser.add_argument("--model_id",       type=str,                           help="path to your model checkpoint")
    parser.add_argument("--test_data_path", type=str,                           help="path to your test data")
    parser.add_argument("--raw_data_dir",   type=str,                           help="path to the directory where raw data is stored")
    parser.add_argument("--split_str",      type=str,  default=None,            help="string to split the image path on for relative path extraction")
    parser.add_argument("--result_folder",  type=str,                           help="path to the directory where results will be saved")
    parser.add_argument("--batch_size",     type=int,  default=256,             help="items per large batch (split across GPUs)")
    parser.add_argument("--num_gpus",       type=int,  default=1,               help="number of GPUs")
    parser.add_argument("--num_fewshot",    type=int,  default=5,               help="number of held-out test examples to use as few-shot context")
    parser.add_argument("--fewshot_seed",   type=int,  default=0,               help="random seed used to select held-out few-shot examples")
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
        help="Per-worker micro-batch size used only when --inference_backend=transformers; "
             "defaults to the full worker sub-batch",
    )
    parser.add_argument(
        "--norm_topk_prob",
        action="store_true",
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
        help="System prompt prepended to every conversation.  Pass an empty string ('') to "
             "suppress the model's default system prompt (e.g. 'You are a helpful assistant.') "
             "injected by apply_chat_template.  Leave unset to use the template default.",
    )
    parser.add_argument(
        "--transformers_text_only_eval",
        action="store_true",
        help="Ignore image inputs during transformers preprocessing and run eval with text-only prompts.",
    )
    parser.add_argument(
        "--router_debug",
        action="store_true",
        help="Enable router debug mode.",
    )
    return parser.parse_args()


# ── Image helpers ──────────────────────────────────────────────────────────────

def normalize_images(image_path: str, raw_data_dir: str, split_str: str = None):
    if split_str:
        image_path = image_path.split(split_str)[-1]
    image_path = os.path.join(raw_data_dir, image_path)
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    return Image.open(image_path), image_path


# ── Few-shot grouping ──────────────────────────────────────────────────────────

def parse_main_tags(item: dict) -> list[str]:
    raw = item.get("main_tag", "")
    if not raw:
        return []
    if isinstance(raw, list):
        return [t.strip() for t in raw if str(t).strip()]
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def get_group_key(item: dict) -> str:
    tags = parse_main_tags(item)
    return ",".join(tags) if tags else ""


def select_grouped_fewshot_examples(
    test_items: list[dict], num_fewshot: int, seed: int
) -> tuple[dict, list[dict], dict]:
    if num_fewshot <= 0:
        return {}, test_items, {}

    grouped = {}
    for item in test_items:
        grouped.setdefault(get_group_key(item), []).append(item)

    use_single = not any(k for k in grouped)
    if use_single:
        grouped = {"__all__": list(test_items)}

    rng = random.Random(seed)
    fewshot_items, eval_items, labels = {}, [], {}

    for key, items in grouped.items():
        if len(items) <= num_fewshot:
            display = "all" if key == "__all__" else (key or "<empty>")
            raise ValueError(
                f"Group '{display}' needs >{num_fewshot} items for few-shot hold-out, "
                f"but only has {len(items)}."
            )
        shuffled = list(items)
        rng.shuffle(shuffled)
        fewshot_items[key] = shuffled[:num_fewshot]
        eval_items.extend(shuffled[num_fewshot:])
        labels[key] = "all" if key == "__all__" else (key or "<empty>")

    return fewshot_items, eval_items, labels


def prepare_fewshot_examples(
    grouped_fewshot_items: dict, raw_data_dir: str, split_str: str
) -> dict[str, list[dict]]:
    """Load images for each few-shot example once; result is passed to workers at spawn time."""
    prepared = {}
    for key, items in grouped_fewshot_items.items():
        prepared[key] = []
        for item in items:
            image, norm_path = normalize_images(item["image"], raw_data_dir, split_str)
            conv = item["conversations"]
            assert len(conv) == 2, (
                f"Expected 2 turns, got {len(conv)} (id={item.get('id', '?')})"
            )
            prepared[key].append({
                "id":         item.get("id", ""),
                "image":      image,
                "image_path": norm_path,
                "question":   conv[0]["value"],
                "answer":     conv[1]["value"],
                "main_tag":   item.get("main_tag", ""),
            })
    return prepared


# ── Conversation builder ───────────────────────────────────────────────────────

def build_conversation(
    item: dict,
    image: Image.Image,
    fewshot_examples: list[dict],
    system_prompt: str | None = None,
) -> tuple[list[dict], str]:
    """
    Build a multi-turn conversation with few-shot examples prepended.
    Returns (turns, ground_truth_answer).

    `system_prompt`:
      - None  → omit the system turn entirely; apply_chat_template will inject
                 its own default (e.g. "You are a helpful assistant.").
      - ""    → insert an explicit empty system turn, which suppresses the
                 template default.  Use this when the model was fine-tuned
                 without any system prompt.
      - str   → insert that string as the system turn.

    Each turn uses {"type": "image_pil", "image_pil": ...} content blocks so
    it is compatible with both the vLLM llm.chat interface and the transformers
    preprocessing path (which extracts PIL images from these blocks).
    """
    conv = item["conversations"]
    assert len(conv) == 2, f"Expected 2 turns, got {len(conv)} (id={item.get('id', '?')})"

    turns = []

    # Inject explicit system turn when caller has specified one (including "").
    # Leaving system_prompt=None lets apply_chat_template use its own default.
    if system_prompt is not None:
        turns.append({"role": "system", "content": system_prompt})

    for ex in fewshot_examples:
        turns.append({
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": ex["image"]},
                {"type": "text",      "text":      ex["question"]},
            ],
        })
        turns.append({"role": "assistant", "content": ex["answer"]})

    turns.append({
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": image},
            {"type": "text",      "text":      conv[0]["value"]},
        ],
    })
    return turns, conv[1]["value"]


# ── Batch preprocessing (runs inside each worker on its pinned cores) ──────────

def process_batch(
    batch_items:               list[dict],
    batch_start_idx:           int,
    prepared_fewshot_examples: dict[str, list[dict]],
    run_id:                    str,
    raw_data_dir:              str,
    split_str:                 str | None,
    system_prompt:             str | None = None,
) -> tuple[list[list[dict]], list[dict]]:
    conversations, metadata = [], []

    for i, item in enumerate(batch_items):
        image, norm_path = normalize_images(item["image"], raw_data_dir, split_str)
        group_key = get_group_key(item)

        if not prepared_fewshot_examples:
            fewshot = []
        elif "__all__" in prepared_fewshot_examples:
            fewshot = prepared_fewshot_examples["__all__"]
        else:
            fewshot = prepared_fewshot_examples.get(group_key, [])

        conversation, gt_response = build_conversation(item, image, fewshot, system_prompt)
        conversations.append(conversation)

        metadata.append({
            "id":       run_id + "_" + str(batch_start_idx + i),
            "image":    norm_path,
            "question": item["conversations"][0]["value"],
            "answer_gt": gt_response,
            "fewshot_examples": [
                {"question": ex["question"], "answer": ex["answer"]}
                for ex in fewshot
            ],
            "main_tag": item.get("main_tag", ""),
            "sub_task": item.get("sub_task", ""),
            **( {"sub_tag": item["sub_tag"]} if "sub_tag" in item else {} ),
            **( {"object":  item["object"]}  if "object"  in item else {} ),
        })

    return conversations, metadata


# ── Transformers helpers ───────────────────────────────────────────────────────

def _extract_pil_images_from_turns(turns: list[dict]) -> list[Image.Image]:
    """
    Walk all turns and collect every PIL image embedded as
    {"type": "image_pil", "image_pil": <PIL.Image>} content blocks,
    preserving the left-to-right, top-to-bottom order the processor expects.
    """
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
    """
    Convert build_conversation-style turns (using {"type": "image_pil", ...}
    blocks) into the standard HuggingFace chat format expected by
    processor.apply_chat_template:
      {"type": "image"} placeholder  — the processor will fill in pixel values
      {"type": "text",  "text": ...}
    Assistant turns (plain string content) are passed through unchanged.
    """
    messages = []
    for turn in turns:
        content = turn["content"]
        if isinstance(content, str):
            # Assistant turn — already a plain string
            messages.append({"role": turn["role"], "content": content})
        else:
            new_content = []
            for block in content:
                if block["type"] == "image_pil":
                    new_content.append({"type": "image"})
                else:
                    new_content.append(block)
            messages.append({"role": turn["role"], "content": new_content})
    return messages


def get_vision_info(turns_batch: list[list[dict]]) -> tuple[Any, Any]:
    """
    Extract image/video inputs via qwen_vl_utils when available.
    Receives the original turns (with image_pil blocks) so process_vision_info
    has access to real pixel data.
    Falls back to (None, None); the caller then uses pre-loaded PIL images.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        return None, None

    try:
        result = process_vision_info(turns_batch)
        return result[0], result[1]
    except (ValueError, TypeError):
        return None, None


def preprocess_transformers_batch(
    conversations: list[list[dict]],
    processor: "AutoProcessor",
    text_only_eval: bool = False,
) -> Any:
    """
    Tokenise a batch of multi-turn conversations (with few-shot prefixes) for
    the transformers backend.

    Each element of `conversations` is the list of turns produced by
    build_conversation. When `text_only_eval` is enabled, image blocks are
    dropped before apply_chat_template and no image/video tensors are passed to
    the processor.

    Returns model_inputs.  Callers must derive per-sequence prompt lengths from
    attention_mask.sum(dim=1) rather than input_ids.shape[1], because
    Qwen2.5-VL uses dynamic resolution and different images produce different
    numbers of vision tokens, so a single scalar length is incorrect for batches
    where sequences have different prompt lengths.
    """
    assert processor.tokenizer.padding_side == "left", (
        f"Expected left padding, got '{processor.tokenizer.padding_side}'. "
        "Set processor.tokenizer.padding_side = 'left' in load_processor()."
    )
    if text_only_eval:
        messages_batch = []
        for turns in conversations:
            text_only_turns = []
            for turn in turns:
                content = turn["content"]
                if isinstance(content, str):
                    text_only_turns.append(turn)
                    continue

                text_blocks = [
                    block
                    for block in content
                    if block.get("type") != "image_pil"
                ]
                text_only_turns.append({"role": turn["role"], "content": text_blocks})
            messages_batch.append(text_only_turns)
    else:
        messages_batch = [_turns_to_processor_messages(turns) for turns in conversations]

        image_inputs, video_inputs = get_vision_info(conversations)
        fallback_images_flat = [
            img
            for turns in conversations
            for img in _extract_pil_images_from_turns(turns)
        ]

    # FIX 4: apply_chat_template per conversation to avoid older processor builds
    # silently misinterpreting list[list[dict]] as one multi-turn conversation.
    # Each msg is already a list[dict]; do NOT wrap it in another list or
    # apply_chat_template returns a list[str] instead of a str, which causes
    # AttributeError: 'list' object has no attribute 'replace' downstream.
    #
    # The trailing '\n' in '<|im_start|>assistant\n' is stripped from the prompt
    # string before tokenization.  During SFT training the tokenizer saw the
    # generation boundary as a contiguous string
    # (...assistant\n<first_answer_chars>), so BPE could merge '\n' with the
    # opening characters of the answer into a single token (e.g. '\nU', '\n[').
    # At inference, apply_chat_template places '\n' at the very end of the
    # string, forcing it to tokenize as a standalone token the model never saw
    # in that position during training.  The result is a short prefix artifact
    # (e.g. 'Ul\nUltrasound Sensing', '[\n[0.53, ...]').
    #
    # Stripping the trailing '\n' here lets the model emit it as its first
    # generated token (restoring the training distribution); response.strip()
    # later removes any leading whitespace from the decoded output.
    prompts = [
        processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        for msg in messages_batch
    ]

    processor_kwargs: dict = {
        "text":           prompts,
        "padding":        True,
        "return_tensors": "pt",
    }
    if not text_only_eval:
        processor_kwargs["images"] = image_inputs if image_inputs else fallback_images_flat
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs

    return processor(**processor_kwargs)


def run_transformers_generate(
    conversations:    list[list[dict]],
    metadata:         list[dict],
    processor:        "AutoProcessor",
    model:            Any,
    micro_batch_size: int | None,
    text_only_eval:   bool = False,
) -> list[dict]:
    import torch

    results    = []
    batch_size = micro_batch_size or len(conversations)

    for start in range(0, len(conversations), batch_size):
        conv_chunk = conversations[start : start + batch_size]
        meta_chunk = metadata[start : start + batch_size]

        model_inputs = preprocess_transformers_batch(
            conv_chunk,
            processor,
            text_only_eval=text_only_eval,
        )
        model_inputs = {
            k: v.to("cuda") if hasattr(v, "to") else v
            for k, v in model_inputs.items()
        }

        # The padded prompt length is uniform across the batch — this is the
        # correct offset at which generated tokens begin in every row of
        # generated_ids. input_lengths (unpadded) is intentionally NOT used
        # for slicing: with left-padding the pad tokens sit at the start of
        # each row, so generated_ids[i] is:
        #   [<pad>...<pad> | real_prompt_tokens | generated_tokens]
        # and model.generate() returns shape [B, padded_prompt_len + new_tokens],
        # making padded_prompt_len the correct universal cut point.
        padded_prompt_len = model_inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                do_sample=True,
                temperature=0.9,
                top_p=0.5,
                max_new_tokens=1024,
                pad_token_id=processor.tokenizer.pad_token_id
                    if processor.tokenizer.pad_token_id is not None
                    else processor.tokenizer.eos_token_id,
            )

        # Slice off the prompt prefix uniformly; skip_special_tokens handles
        # any pad tokens that appear if a sequence hit max_new_tokens before
        # the others and was padded on the right in generated_ids.
        decoded = [
            processor.tokenizer.decode(
                generated_ids[i, padded_prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i in range(len(conv_chunk))
        ]

        for meta, text in zip(meta_chunk, decoded):
            response = text.strip()
            results.append({
                "id":       meta["id"],
                "image":    meta["image"],
                "question": meta["question"],
                "answer":   response,
                "answer_gt": meta["answer_gt"],
                "fewshot_examples": meta["fewshot_examples"],
                "main_tag": meta["main_tag"],
                "sub_task": meta["sub_task"],
                **( {"sub_tag": meta["sub_tag"]} if "sub_tag" in meta else {} ),
                **( {"object":  meta["object"]}  if "object"  in meta else {} ),
            })

    return results


# ── Model loading ──────────────────────────────────────────────────────────────

def load_processor(model_id: str) -> "AutoProcessor":
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    return processor


def print_transformers_prompt_examples(
    model_id: str,
    test_data: list[dict],
    raw_data_dir: str,
    split_str: str | None,
    system_prompt: str | None,
    prepared_fewshot_examples: dict[str, list[dict]],
    text_only_eval: bool,
    limit: int = 5,
) -> None:
    sample_items = test_data[:limit]
    if not sample_items:
        return

    processor = load_processor(model_id)
    conversations, _ = process_batch(
        sample_items,
        0,
        prepared_fewshot_examples,
        run_id="prompt_preview",
        raw_data_dir=raw_data_dir,
        split_str=split_str,
        system_prompt=system_prompt,
    )
    if text_only_eval:
        prompt_messages = []
        for turns in conversations:
            prompt_messages.append([
                {"role": turn["role"], "content": turn["content"]}
                if isinstance(turn["content"], str)
                else {
                    "role": turn["role"],
                    "content": [
                        block
                        for block in turn["content"]
                        if block.get("type") != "image_pil"
                    ],
                }
                for turn in turns
            ])
    else:
        prompt_messages = [_turns_to_processor_messages(turns) for turns in conversations]

    prompts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in prompt_messages
    ]

    print(f"Previewing {len(prompts)} templated transformers prompts before worker startup:")
    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n=== Templated prompt {idx}/{len(prompts)} ===")
        print(prompt)


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
        
    use_bf16      = torch.cuda.is_bf16_supported()
    model_kwargs  = {
        "config":            config,
        "torch_dtype":       torch.bfloat16 if use_bf16 else torch.float16,
        "trust_remote_code": True,
    }
    if is_flash_attn_2_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    routing_monitor = MoeRoutingMonitor(
        model,
        skip_first_call_per_layer=False,
    ).register() if router_debug else None
    # lm_head = model.lm_head.weight
    # embed = model.model.language_model.embed_tokens.weight

    # print(f"Same object (tied):     {lm_head.data_ptr() == embed.data_ptr()}")
    # print(f"Values exactly equal:   {torch.equal(lm_head, embed)}")
    # print(f"Max absolute diff:      {(lm_head.float() - embed.float()).abs().max().item()}")
    # print(f"Mean absolute diff:     {(lm_head.float() - embed.float()).abs().mean().item()}")
    # quit()
    return model.eval().to("cuda"), routing_monitor


# ── Worker ─────────────────────────────────────────────────────────────────────

def run_worker(
    rank:                      int,
    model_id:                  str,
    inference_backend:         str,
    transformers_batch_size:   int | None,
    transformers_text_only_eval: bool,
    router_debug:              bool,
    norm_topk_prob:            bool,
    num_experts_per_tok:       int | None,
    cores:                     list[int],
    run_id:                    str,
    raw_data_dir:              str,
    split_str:                 str | None,
    system_prompt:             str | None,
    prepared_fewshot_examples: dict[str, list[dict]],  # pickled once at spawn time
    task_queue:                mp.Queue,
    result_queue:              mp.Queue,
    ready_queue:               mp.Queue,
) -> None:
    """
    GPU worker: pins itself to `cores`, loads the model, then loops —
    preprocessing each sub-batch on its pinned CPU cores before running
    inference on its GPU.
    """
    # Set env vars before any CUDA import
    os.environ["CUDA_VISIBLE_DEVICES"]         = str(rank)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    os.sched_setaffinity(0, set(cores))

    if inference_backend == "vllm":
        from vllm import LLM, SamplingParams
        model           = LLM(
            model=model_id,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=32768,
            limit_mm_per_prompt={"image": 10},
        )
        
        sampling_params = SamplingParams(temperature=0.9, top_p=0.5, max_tokens=1024, seed=0)
        processor       = None
    else:
        processor              = load_processor(model_id)
        model, routing_monitor = load_transformers_model(
            model_id,
            norm_topk_prob,
            num_experts_per_tok,
            router_debug=router_debug
        )
        sampling_params        = None

    ready_queue.put(rank)

    while True:
        task = task_queue.get()
        if task is None:   # sentinel → shut down
            if router_debug and routing_monitor is not None:
                routing_monitor.print_grouped_summary()
                routing_monitor.print_top_transitions(grouped=True, top_n=10)
                routing_stats = routing_monitor.grouped_summary()
                routing_monitor.remove()
            break

        global_start, batch_items = task
        conversations, metadata = process_batch(
            batch_items, global_start,
            prepared_fewshot_examples,
            run_id, raw_data_dir, split_str,
            system_prompt,
        )

        if inference_backend == "vllm":
            outputs = model.chat(conversations, sampling_params=sampling_params)
            results = []
            for output, meta in zip(outputs, metadata):
                response = output.outputs[0].text.strip()
                results.append({
                    "id":       meta["id"],
                    "image":    meta["image"],
                    "question": meta["question"],
                    "answer":   response,
                    "answer_gt": meta["answer_gt"],
                    "fewshot_examples": meta["fewshot_examples"],
                    "main_tag": meta["main_tag"],
                    "sub_task": meta["sub_task"],
                    **( {"sub_tag": meta["sub_tag"]} if "sub_tag" in meta else {} ),
                    **( {"object":  meta["object"]}  if "object"  in meta else {} ),
                })
        else:
            results = run_transformers_generate(
                conversations, metadata,
                processor, model,
                transformers_batch_size,
                transformers_text_only_eval,
            )

        result_queue.put((global_start, results))


# ── Orchestration ──────────────────────────────────────────────────────────────

def run_inference(
    model_id:                  str,
    inference_backend:         str,
    transformers_batch_size:   int | None,
    transformers_text_only_eval: bool,
    router_debug:              bool,
    norm_topk_prob:            bool,
    num_experts_per_tok:       int | None,
    test_data:                 list[dict],
    batch_size:                int,
    num_gpus:                  int,
    run_id:                    str,
    raw_data_dir:              str,
    split_str:                 str | None,
    system_prompt:             str | None,
    prepared_fewshot_examples: dict[str, list[dict]],
) -> list[dict]:
    """
    1. Divide CPU cores evenly across num_gpus workers.
    2. Spawn workers; each pins itself, loads the model, signals ready.
    3. Loop over large batches: split into raw sub-batches, dispatch.
       Workers preprocess on their pinned cores, then run inference.
    4. Collect results back at original indices.
    """
    ctx = mp.get_context("spawn")

    allowed_cores    = sorted(os.sched_getaffinity(0))
    cores_per_worker = max(1, len(allowed_cores) // num_gpus)
    worker_cores = [
        allowed_cores[rank * cores_per_worker : (rank + 1) * cores_per_worker]
        for rank in range(num_gpus)
    ]
    print(f"Allowed cores: {allowed_cores}")
    print(f"CPU affinity: {num_gpus} workers × {cores_per_worker} cores each")

    task_queues  = [ctx.Queue() for _ in range(num_gpus)]
    result_queue = ctx.Queue()
    ready_queue  = ctx.Queue()

    procs = [
        ctx.Process(
            target=run_worker,
            args=(
                rank, model_id,
                inference_backend, transformers_batch_size,
                transformers_text_only_eval,
                router_debug,
                norm_topk_prob, num_experts_per_tok,
                worker_cores[rank],
                run_id, raw_data_dir, split_str,
                system_prompt,
                prepared_fewshot_examples,          # pickled once per worker at spawn
                task_queues[rank], result_queue, ready_queue,
            ),
        )
        for rank in range(num_gpus)
    ]
    for p in procs:
        p.start()

    print(f"Waiting for {num_gpus} workers to load models…")
    ready_ranks = sorted([ready_queue.get() for _ in range(num_gpus)])
    print(f"All workers ready: {ready_ranks}")

    n           = len(test_data)
    ordered     = [None] * n
    num_batches = math.ceil(n / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Batches", unit="batch"):
        batch_start = batch_idx * batch_size
        large_batch = test_data[batch_start : batch_start + batch_size]
        m           = len(large_batch)
        chunk_size  = math.ceil(m / num_gpus)

        # Dispatch raw sub-batches to workers
        active_workers = []
        for rank in range(num_gpus):
            chunk = large_batch[rank * chunk_size : (rank + 1) * chunk_size]
            if not chunk:
                break
            task_queues[rank].put((batch_start + rank * chunk_size, chunk))
            active_workers.append(rank)

        # Collect results in arrival order; place at correct global indices
        for _ in active_workers:
            global_start, results = result_queue.get()
            for local_idx, entry in enumerate(results):
                ordered[global_start + local_idx] = entry

    # Shut down workers
    for q in task_queues:
        q.put(None)
    for p in procs:
        p.join()

    return ordered


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    with open(args.test_data_path, "r", encoding="utf-8") as f:
        test = json.load(f)
        if args.router_debug:
            print("Limiting it to 100 items for router debug mode")
            test = test[:100] # TEMP: limit to 100 items for quick testing; remove this line for full eval

    print(f"Loaded {len(test)} test items")
    print(
        f"Model: {args.model_id}  |  Backend: {args.inference_backend}  |  "
        f"GPUs: {args.num_gpus}  |  Batch size: {args.batch_size}"
    )

    os.makedirs(args.result_folder, exist_ok=True)
    result_path  = os.path.join(args.result_folder, "test_result.json")
    fewshot_path = os.path.join(args.result_folder, "fewshot_examples.json")

    # ── Few-shot selection ─────────────────────────────────────────────────────
    grouped_fewshot_items, test, group_labels = select_grouped_fewshot_examples(
        test, args.num_fewshot, args.fewshot_seed
    )
    # Load fewshot images once here so workers receive them at spawn time
    prepared_fewshot_examples = prepare_fewshot_examples(
        grouped_fewshot_items, args.raw_data_dir, args.split_str
    )

    with open(fewshot_path, "w") as f:
        json.dump(
            {
                group_labels[key]: [
                    {
                        "id":       item.get("id", ""),
                        "image":    item.get("image", ""),
                        "main_tag": item.get("main_tag", ""),
                        "question": item["conversations"][0]["value"],
                        "answer_gt": item["conversations"][1]["value"],
                    }
                    for item in items
                ]
                for key, items in grouped_fewshot_items.items()
            },
            f, indent=4,
        )

    if grouped_fewshot_items:
        for key, items in grouped_fewshot_items.items():
            print(f"Held out {len(items)} few-shot examples for group '{group_labels[key]}'")
    print(f"Few-shot cache saved at {fewshot_path}")
    print(f"Running evaluation on {len(test)} remaining items")

    if args.inference_backend == "transformers":
        print_transformers_prompt_examples(
            model_id=args.model_id,
            test_data=test,
            raw_data_dir=args.raw_data_dir,
            split_str=args.split_str,
            system_prompt=args.system_prompt,
            prepared_fewshot_examples=prepared_fewshot_examples,
            text_only_eval=args.transformers_text_only_eval,
        )

    # ── Data-parallel inference ────────────────────────────────────────────────
    results = run_inference(
        model_id=args.model_id,
        inference_backend=args.inference_backend,
        transformers_batch_size=args.transformers_batch_size,
        transformers_text_only_eval=args.transformers_text_only_eval,
        router_debug=args.router_debug,
        norm_topk_prob=args.norm_topk_prob,
        num_experts_per_tok=args.num_experts_per_tok,
        test_data=test,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        run_id=args.run_id,
        raw_data_dir=args.raw_data_dir,
        split_str=args.split_str,
        system_prompt=args.system_prompt,
        prepared_fewshot_examples=prepared_fewshot_examples,
    )

    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Done. {len(results)} results saved to {result_path}")


if __name__ == "__main__":
    main()