#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import pickle
import random
import struct

from typing import Any

import torch

from datasets import load_dataset
from PIL import Image as PILImage
from PIL import ImageFile
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from modeling.flex_qwen2_5_vl_moe import (
	Flex_Qwen2_5_VLMoeConfig,
	Flex_Qwen2_5_VLMoeForConditionalGeneration,
)
from modeling.routing_embeddings.hidden_state_monitor import FfnHiddenStateMonitor


ImageFile.LOAD_TRUNCATED_IMAGES = True

_ORIG_GETEXIF = PILImage.Image.getexif


def safe_getexif(self):
	try:
		return _ORIG_GETEXIF(self)
	except (SyntaxError, OSError, ValueError, struct.error) as error:
		filename = getattr(self, "filename", None)
		print(f"[WARN] Ignoring broken EXIF for image: {filename}, error={repr(error)}")
		return {}


PILImage.Image.getexif = safe_getexif


def register_local_architectures() -> None:
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


def preprocess_dataset(dataset):
	"""Borrow the prompt/image shaping from mm_tune.py for inference-time use."""

	def convert_row(item, idx):
		image_key = "images"
		assert image_key in item and "conversation" in item

		prompt_id = item["id"] if "id" in item else idx
		conversation = item["conversation"]
		assert conversation[-1]["role"] == "assistant"

		images = item[image_key]
		prompt = []
		used_image = False

		for turn in conversation[:-1]:
			content = [{"type": "text", "text": turn["content"]}]
			if (
				turn["role"] == "user"
				and turn.get("img_loc") is not None
				and images
				and not used_image
			):
				image_content = [{"type": "image"} for _ in images]
				if turn["img_loc"] == "before":
					content = image_content + content
				else:
					content = content + image_content
				used_image = True

			prompt.append({"role": turn["role"], "content": content})

		return {
			"prompt_id": prompt_id,
			"prompt": prompt,
			"completion": [{
				"role": "assistant",
				"content": [{"type": "text", "text": conversation[-1]["content"]}],
			}],
			"images": [{"bytes": None, "path": img_path} for img_path in images] if images else [],
		}

	return dataset.map(
		convert_row,
		remove_columns=dataset.column_names,
		num_proc=12,
		with_indices=True,
	)


def dataset_to_examples(dataset) -> list[dict[str, Any]]:
	examples = []
	for row in dataset:
		examples.append(
			{
				"prompt_id": row["prompt_id"],
				"prompt": row["prompt"],
				"image_paths": [image["path"] for image in row["images"]] if row["images"] else [],
			}
		)
	return examples


def load_and_sample_examples(dataset_path: str, query_amount: int, seed: int) -> list[dict[str, Any]]:
	dataset = load_dataset("json", data_files=dataset_path, split="train")
	dataset = preprocess_dataset(dataset)

	if len(dataset) == 0:
		raise ValueError(f"No rows available after preprocessing dataset: {dataset_path}")
	if query_amount <= 0:
		raise ValueError("query_amount must be positive")

	sample_size = min(query_amount, len(dataset))
	if sample_size < query_amount:
		print(
			f"[WARN] Requested {query_amount} queries but dataset only has {len(dataset)} rows; "
			f"using {sample_size}."
		)

	dataset = dataset.shuffle(seed=seed).select(range(sample_size))
	return dataset_to_examples(dataset)


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
	if dtype_name == "auto":
		if device.type == "cuda":
			return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
		return torch.float32
	if dtype_name == "bf16":
		return torch.bfloat16
	if dtype_name == "fp16":
		return torch.float16
	if dtype_name == "fp32":
		return torch.float32
	raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_processor(model_name: str, device: torch.device, dtype_name: str):
	register_local_architectures()
	processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
	tokenizer = getattr(processor, "tokenizer", None)
	if tokenizer is not None and tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	torch_dtype = resolve_torch_dtype(dtype_name, device)
	model = AutoModelForImageTextToText.from_pretrained(
		model_name,
		torch_dtype=torch_dtype,
		device_map=None,
	)
	model.eval()
	model.to(device)
	return model, processor


def load_example_images(example: dict[str, Any]) -> list[PILImage.Image] | None:
	image_paths = example["image_paths"]
	if not image_paths:
		return None

	images = []
	for image_path in image_paths:
		with PILImage.open(image_path) as image:
			images.append(image.convert("RGB"))
	return images


def prepare_inputs(example: dict[str, Any], processor, device: torch.device) -> dict[str, torch.Tensor]:
	text = processor.apply_chat_template(
		example["prompt"],
		tokenize=False,
		add_generation_prompt=True,
	)
	images = load_example_images(example)
	inputs = processor(text=text, images=images, return_tensors="pt")
	return {name: value.to(device) for name, value in inputs.items()}


def squeeze_hidden_state(hidden_state: torch.Tensor) -> torch.Tensor:
	if hidden_state.ndim == 1:
		return hidden_state.to(torch.float32).cpu()
	if hidden_state.ndim == 2:
		return hidden_state.mean(dim=0).to(torch.float32).cpu()
	raise ValueError(f"Expected pooled hidden state to be 1D or 2D, got shape={tuple(hidden_state.shape)}")


def merge_layer_statistics(
	total_sums: dict[str, torch.Tensor],
	total_counts: dict[str, int],
	partial_sums: dict[str, torch.Tensor],
	partial_counts: dict[str, int],
) -> None:
	for layer_name, layer_sum in partial_sums.items():
		if layer_name not in total_sums:
			total_sums[layer_name] = layer_sum.clone()
			total_counts[layer_name] = partial_counts[layer_name]
		else:
			total_sums[layer_name] += layer_sum
			total_counts[layer_name] += partial_counts[layer_name]


@torch.inference_mode()
def extract_average_hidden_states(
	model,
	processor,
	examples,
	device: torch.device,
	return_counts: bool = False,
	progress_desc: str | None = None,
	progress_position: int = 0,
	progress_leave: bool = True,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, int]]:
	monitor = FfnHiddenStateMonitor(
		model,
		capture="first",
		pool="last_token",
		store_device="cpu",
		store_dtype=torch.float32,
	).register()

	layer_sums: dict[str, torch.Tensor] = {}
	layer_counts: dict[str, int] = {}

	try:
		example_iterator = tqdm(
			examples,
			desc=progress_desc or "Examples",
			total=len(examples),
			position=progress_position,
			leave=progress_leave,
		)
		for example in example_iterator:
			inputs = prepare_inputs(example, processor, device)
			monitor.start_query()
			try:
				model.generate(
					**inputs,
					max_new_tokens=1,
					do_sample=False,
					use_cache=True,
				)
			finally:
				states = monitor.end_query(label=example["prompt_id"])

			if not states:
				raise RuntimeError(
					f"No FFN hidden states were captured for prompt_id={example['prompt_id']}"
				)

			for layer_name, hidden_state in states.items():
				if isinstance(hidden_state, list):
					raise ValueError(
						f"Unexpected list-valued hidden state for {layer_name}; capture should be 'first'."
					)
				vector = squeeze_hidden_state(hidden_state)
				if layer_name not in layer_sums:
					layer_sums[layer_name] = vector.clone()
					layer_counts[layer_name] = 1
				else:
					layer_sums[layer_name] += vector
					layer_counts[layer_name] += 1
	finally:
		monitor.remove()

	if not layer_sums:
		raise RuntimeError("No layer hidden states were accumulated.")
	if return_counts:
		return layer_sums, layer_counts

	averaged_states = {
		layer_name: (layer_sum / layer_counts[layer_name]).numpy()
		for layer_name, layer_sum in layer_sums.items()
	}
	return averaged_states


def split_examples(examples: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
	shards = [[] for _ in range(num_shards)]
	for idx, example in enumerate(examples):
		shards[idx % num_shards].append(example)
	return [shard for shard in shards if shard]


def worker_extract_hidden_states(
	worker_rank: int,
	model_name: str,
	examples: list[dict[str, Any]],
	dtype_name: str,
	result_queue,
) -> None:
	device = torch.device(f"cuda:{worker_rank}")
	torch.cuda.set_device(device)
	model = None
	try:
		model, processor = load_model_and_processor(model_name, device, dtype_name)
		layer_sums, layer_counts = extract_average_hidden_states(
			model=model,
			processor=processor,
			examples=examples,
			device=device,
			return_counts=True,
			progress_desc=f"GPU {worker_rank}",
			progress_position=worker_rank,
			progress_leave=True,
		)
		result_queue.put(
			{
				"ok": True,
				"rank": worker_rank,
				"layer_sums": {name: tensor.numpy() for name, tensor in layer_sums.items()},
				"layer_counts": layer_counts,
			}
		)
	except Exception as error:
		result_queue.put(
			{
				"ok": False,
				"rank": worker_rank,
				"error": repr(error),
			}
		)
	finally:
		if model is not None:
			del model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()


def extract_average_hidden_states_multi_gpu(
	model_name: str,
	examples: list[dict[str, Any]],
	num_gpus: int,
	dtype_name: str,
) -> dict[str, torch.Tensor]:
	shards = split_examples(examples, num_gpus)
	ctx = mp.get_context("spawn")
	result_queue = ctx.Queue()
	processes = []

	for worker_rank, shard in enumerate(shards):
		process = ctx.Process(
			target=worker_extract_hidden_states,
			args=(worker_rank, model_name, shard, dtype_name, result_queue),
		)
		process.start()
		processes.append(process)

	total_sums: dict[str, torch.Tensor] = {}
	total_counts: dict[str, int] = {}
	worker_errors = []

	for _ in processes:
		result = result_queue.get()
		if not result["ok"]:
			worker_errors.append(f"worker {result['rank']}: {result['error']}")
			continue

		partial_sums = {
			name: torch.from_numpy(array)
			for name, array in result["layer_sums"].items()
		}
		merge_layer_statistics(total_sums, total_counts, partial_sums, result["layer_counts"])

	for process in processes:
		process.join()

	if worker_errors:
		raise RuntimeError("Multi-GPU extraction failed: " + "; ".join(worker_errors))
	if not total_sums:
		raise RuntimeError("Multi-GPU extraction completed without any layer statistics.")

	return {
		layer_name: (layer_sum / total_counts[layer_name]).numpy()
		for layer_name, layer_sum in total_sums.items()
	}


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Sample prompts from a JSONL conversation dataset, run one-token inference, "
			"capture FFN input hidden states, and save per-layer averages as a pickle."
		)
	)
	parser.add_argument("--model", required=True, help="Model name or local checkpoint path.")
	parser.add_argument("--dataset", required=True, help="Path to the input JSONL dataset.")
	parser.add_argument("--query-amount", type=int, required=True, help="Number of prompts to sample.")
	parser.add_argument("--output-path", required=True, help="Path to the output pickle file.")
	parser.add_argument("--seed", type=int, default=2025, help="Random seed for sampling.")
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device for inference.",
	)
	parser.add_argument(
		"--dtype",
		choices=["auto", "bf16", "fp16", "fp32"],
		default="auto",
		help="Torch dtype for model loading.",
	)
	parser.add_argument(
		"--num-gpus",
		type=int,
		default=1,
		help="Number of GPUs to use via simple prompt-level data parallelism.",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()

	random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device(args.device)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA device requested but no CUDA device is available.")
	if args.num_gpus <= 0:
		raise ValueError("num_gpus must be positive")
	if args.num_gpus > 1 and device.type != "cuda":
		raise ValueError("Multi-GPU data parallelism requires --device cuda.")
	if args.num_gpus > torch.cuda.device_count():
		raise ValueError(
			f"Requested {args.num_gpus} GPUs but only {torch.cuda.device_count()} are available."
		)

	sampled_examples = load_and_sample_examples(args.dataset, args.query_amount, args.seed)
	print(f"Loaded and sampled {len(sampled_examples)} prompts from {args.dataset}")

	if args.num_gpus == 1:
		model, processor = load_model_and_processor(args.model, device, args.dtype)
		averaged_states = extract_average_hidden_states(
			model=model,
			processor=processor,
			examples=sampled_examples,
			device=device,
			progress_desc="Examples",
		)
	else:
		print(f"Running prompt-level data parallel extraction across {args.num_gpus} GPUs")
		averaged_states = extract_average_hidden_states_multi_gpu(
			model_name=args.model,
			examples=sampled_examples,
			num_gpus=args.num_gpus,
			dtype_name=args.dtype,
		)

	output_dir = os.path.dirname(args.output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	with open(args.output_path, "wb") as handle:
		pickle.dump(averaged_states, handle)

	print(
		f"Saved averaged FFN hidden states for {len(averaged_states)} layers to {args.output_path}"
	)


if __name__ == "__main__":
	main()
