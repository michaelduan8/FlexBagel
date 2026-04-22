import io
import numpy as np
import os
import pathlib
import random
import sys
import torch
import torch.distributed as dist
#dist.init_process_group(backend="nccl")

import wandb
import json

from dataclasses import dataclass, field
from datasets import load_dataset, Dataset, concatenate_datasets, Image, Sequence
from datetime import datetime
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image as PILImage
from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText, HfArgumentParser, TrainerCallback
from trl import SFTTrainer, SFTConfig

@dataclass
class SFTArgs:
    run_id: str = field(metadata={"help": "ID for the training run"})
    model: str = field(metadata={"help": "Model to use for training"})
    datasets: list[str] = field(metadata={"help": "Dataset(s) to use for training"})
    run_seed: int = field(default=2025, metadata={"help": "Random seed for training"})
    run_output_dir: str = field(default="./checkpoints", metadata={"help": "Directory to save training runs"})
    sample_size: list[int] = field(
        default=None,
        metadata={"help": "Sampling control. Pass one value for global sampling after datasets are merged "
                          "(same behavior as before), or pass one value per dataset to sample each dataset "
                          "individually before merging."}
    )
    eval_n_epochs: float = field(default=2.0, metadata={"help": "Evaluate every N epochs"})
    save_n_epochs: float = field(default=1.0, metadata={"help": "Save a checkpoint every N epochs"})
    filter_by_id: list[str] = field(
        default=None,
        metadata={"help": "Only keep rows whose prompt_id contains at least one of these substrings. If None, no filtering is applied."}
    )
    skip_eval: bool = field(default=False, metadata={"help": "Skip all evaluation. Also skips train/val split — all data is used for training. Useful when using DeepSpeed Stage 1/2."})
    train_expert_idx: int = field(
        default=None,
        metadata={"help": "If set, freeze all weights except this expert's FFN and router row. "
                          "If use_lora is also set, LoRA is applied only to this expert's modules."}
    )
    router_tuning_only: bool = field(
        default=False,
        metadata={"help": "If set, freeze all weights except MoE router gate weights across all experts. "
                          "When enabled, train_expert_idx is ignored and forced to None."}
    )
    unfreeze_attn: bool = field(
        default=False,
        metadata={"help": "If set, unfreeze attention layers after expert/router freezing and before LoRA setup."}
    )
    unfreeze_embed: bool = field(
        default=False,
        metadata={"help": "If set, unfreeze embedding layer parameters after expert/router freezing and before LoRA setup."}
    )

    # LoRA parameters
    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA training"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank (dimension of low-rank matrices)"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha (scaling factor)"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability"})
    lora_target_modules: list[str] = field(
        default=None,
        metadata={"help": "Target modules for LoRA. If None, uses default for the model architecture."}
    )
    merge_and_save: bool = field(default=False, metadata={"help": "Merge LoRA weights into base model and save full model"})


def register_local_architectures():
    print("Registering local architectures...")


def preprocess_dataset(dataset):
    """Convert dataset to conversational prompt-completion format for SFTTrainer."""

    def load_bytes(item):
        assert "images" in item, "Item must contain 'images' key"
        loaded = []
        estimated_total_image_bytes = 0
        widths = []
        heights = []
        sizes = []

        for img_path in item["images"]:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                with PILImage.open(io.BytesIO(img_bytes)) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)
                    size_mb = len(img_bytes) / (1024 * 1024)
                    sizes.append(size_mb)

                loaded.append(img_bytes)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                loaded.append(None)

        item["image_bytes"] = loaded
        item["image_widths"] = widths
        item["image_heights"] = heights
        item["image_sizes"] = sizes
        return item

    def convert_row(item):
        assert "images" in item and "conversation" in item and "id" in item

        # Convert bytes → PIL here, no multiprocessing so no pickling issues
        # images = []
        # for img_bytes in item["image_bytes"]:
        #     try:
        #         image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        #         images.append(image)
        #     except Exception as e:
        #         print(f"Error converting image: {e}")
        #rgb_bytes = []
        # image_widths = []
        # image_heights = []
        # img_sizes = []
        # try:
        #     for img_bytes in item["image_bytes"]:
        #         img = PILImage.open(io.BytesIO(img_bytes))
        #         if img.mode != "RGB":
        #             # Only re-encode if we actually need to convert the mode
        #             img = img.convert("RGB")
        #             buf = io.BytesIO()
        #             img.save(buf, format="PNG")
        #             # rgb_bytes.append(buf.getvalue())
        #             rgb_bytes.append(buf.getvalue())
        #         else:
        #             rgb_bytes.append(img_bytes)
        #         # image_widths.append(img.width)
        #         # image_heights.append(img.height)
        #         # img_sizes.append(len(rgb_bytes[-1]) / (1024 * 1024))
        # except Exception as e:
        #     print(f"Error processing image bytes for item {item['id']}: {e}")

        id = item["id"]
        conversation = item["conversation"]
        assert conversation[-1]["role"] == "assistant"
        assert len(conversation) <= 3

        images = item["images"]

        prompt = []
        # TODO: not sure how to handle multiturn data
        for turn in conversation[:-1]:
            if turn["role"] == "user" and turn["img_loc"] is not None:
                content = [{"type": "text", "text": turn["content"]}]
                #content = [{"type": "image"} for _ in rgb_bytes] + content if turn["img_loc"] == "before" else content + [{"type": "image"} for _ in rgb_bytes]
                content = [{"type": "image"} for _ in images] + content if turn["img_loc"] == "before" else content + [{"type": "image"} for _ in images]
            else:
                content = [{"type": "text", "text": turn["content"]}]
            prompt.append({"role": turn["role"], "content": content})

        return {
            "prompt_id": id,
            "prompt": prompt,
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": conversation[-1]["content"]}]}],
            # TODO: could be ok to not have images?
            # "images": [{"bytes": b, "path": None} for b in rgb_bytes] if rgb_bytes else None,
            "images": [{"bytes": None, "path": img_path} for img_path in images] if images else None,
        }

    #dataset = dataset.map(load_bytes, num_proc=24)
    dataset = dataset.map(convert_row, remove_columns=dataset.column_names, num_proc=12).filter(lambda x: x["images"] is not None)
    dataset = dataset.cast_column("images", Sequence(Image()))

    # widths_kept = [w for row in dataset["image_widths"] for w in row]
    # heights_kept = [h for row in dataset["image_heights"] for h in row]
    # sizes_kept = [s for row in dataset["image_sizes"] for s in row]
    # if len(widths_kept) > 0 and len(heights_kept) > 0:
    #     avg_width_kept = float(np.mean(widths_kept))
    #     std_width_kept = float(np.std(widths_kept))
    #     avg_height_kept = float(np.mean(heights_kept))
    #     std_height_kept = float(np.std(heights_kept))
    #     avg_size_kept = float(np.mean(sizes_kept))
    #     std_size_kept = float(np.std(sizes_kept))
    #     print(
    #         "Image dimension stats (kept images): "
    #         f"n={len(widths_kept)}, "
    #         f"width_avg={avg_width_kept:.1f}px, width_std={std_width_kept:.1f}px, "
    #         f"height_avg={avg_height_kept:.1f}px, height_std={std_height_kept:.1f}px, "
    #         f"size_avg={avg_size_kept:.2f}MB, size_std={std_size_kept:.2f}MB"
    #     )

    return dataset


def prepare_datasets(datasets, seed, sample_size=None, filter_by_id=None, skip_eval=False):
    if sample_size is not None:
        if isinstance(sample_size, int):
            sample_sizes = [sample_size]
        else:
            sample_sizes = list(sample_size)

        if len(sample_sizes) not in (1, len(datasets)):
            raise ValueError(
                f"sample_size must be either a single value or a list with one value per dataset "
                f"(got {len(sample_sizes)} values for {len(datasets)} datasets)."
            )
        if any(size is not None and size < 0 for size in sample_sizes):
            raise ValueError("sample_size values must be non-negative.")
    else:
        sample_sizes = None

    loaded_dataset = []
    for idx, dataset_name in enumerate(datasets):
        print(f"Loading dataset: {dataset_name}")

        if "jsonl" in dataset_name:
            dataset = load_dataset("json", data_files=dataset_name, split="train")
        elif "parquet" in dataset_name:
            dataset = load_dataset("parquet", data_files=dataset_name, split="train")
        else:
            dataset = load_dataset(dataset_name, split="train")

        # Remove rows with too many images:
        # print("Filtering out rows with more than 1 images...")
        # pre = len(dataset)
        # dataset = dataset.filter(
        #     lambda x: len(x["image"]) <= 1
        # )
        # print(f"Filtered: {pre} -> {len(dataset)} rows")

        # Per-dataset sampling: one sample size per dataset path.
        if sample_sizes is not None and len(sample_sizes) == len(datasets) and len(sample_sizes) > 1:
            dataset_sample_size = sample_sizes[idx]
            if dataset_sample_size is not None and dataset_sample_size < len(dataset):
                print(f"Sampling {dataset_sample_size} examples from dataset[{idx}]...")
                dataset = dataset.shuffle(seed=seed).select(range(dataset_sample_size))
            
            print(f"Sampled dataset[{idx}] size: {len(dataset)}")

        loaded_dataset.append(preprocess_dataset(dataset))
    
    loaded_dataset = concatenate_datasets(loaded_dataset)

    # Apply id substring filter before any further processing
    # if filter_by_id is not None:``
    #     pre = len(loaded_dataset)
    #     loaded_dataset = loaded_dataset.filter(
    #         lambda x: any(s in x["prompt_id"] for s in filter_by_id)
    #     )
    #     print(f"Filtered: {pre} -> {len(loaded_dataset)} rows")

    print("Shuffling Dataset...")
    loaded_dataset = loaded_dataset.shuffle(seed=seed) 

    print(f"Total loaded dataset size: {len(loaded_dataset)}")

    # Global sampling behavior (legacy): apply one sample size after merge.
    if sample_sizes is not None and len(sample_sizes) == 1:
        global_sample_size = sample_sizes[0]
        if global_sample_size is not None and global_sample_size < len(loaded_dataset):
            print(f"Sampling {global_sample_size} examples from merged dataset...")
            loaded_dataset = loaded_dataset.select(range(global_sample_size))
            print(f"Sampled dataset size: {len(loaded_dataset)}")

    # Skip train/val split if evaluation is disabled — use all data for training
    if skip_eval:
        print("skip_eval=True: skipping train/val split, using full dataset for training.")
        return loaded_dataset, None

    dataset_split = loaded_dataset.train_test_split(test_size=0.1, seed=seed)
    train_dataset = dataset_split["train"]
    test_dataset = dataset_split["test"]

    return train_dataset, test_dataset


class WandbLoggingCallback(TrainerCallback):
    """A custom callback to update wandb config and log data examples."""
    def __init__(self, stats): #, train_examples, test_examples=None
        self.stats = stats
        # self.train_examples = train_examples
        # self.test_examples = test_examples

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            wandb.config.update(self.stats)

            # def format_messages(ex):
            #     formatted_str = ""
            #     for msg in ex["prompt"] + ex["completion"]:
            #         formatted_str += f"**{msg['role'].capitalize()}**: {msg['content']}\n\n"
            #     return formatted_str

            # log_data = {
            #     "train_examples": "\n\n---\n\n".join([format_messages(ex) for ex in self.train_examples])
            # }

            # if self.test_examples is not None:
            #     log_data["test_examples"] = "\n\n---\n\n".join([format_messages(ex) for ex in self.test_examples])

            # wandb.log(log_data)


# def get_default_lora_target_modules(model_name: str) -> list[str]:
    # """Get default LoRA target modules based on model architecture."""
    # model_name_lower = model_name.lower()

    # if "llama" in model_name_lower or "mistral" in model_name_lower:
    #     return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # elif "gpt2" in model_name_lower:
    #     return ["c_attn", "c_proj", "c_fc"]
    # elif "opt" in model_name_lower:
    #     return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    # elif "bloom" in model_name_lower:
    #     return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    # elif "falcon" in model_name_lower:
    #     return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    # elif "qwen" in model_name_lower:
    #     return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # else:
    #     return ["q_proj", "v_proj"]


# def print_trainable_parameters(model):
    # """Print the number of trainable parameters in the model."""
    # trainable_params = 0
    # all_param = 0
    # for _, param in model.named_parameters():
    #     all_param += param.numel()
    #     if param.requires_grad:
    #         trainable_params += param.numel()

    # trainable_pct = 100 * trainable_params / all_param
    # print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {trainable_pct:.4f}")
    # return trainable_params, all_param, trainable_pct


# def freeze_all_except_expert(model, expert_idx: int, use_lora: bool = False):
    # """
    # Freeze all parameters except the target expert's FFN and its router row.

    # If use_lora=True, all base weights are frozen here and LoRA adapters will be
    # inserted on top of the target expert's modules afterward — so we don't need to
    # unfreeze the expert FFN weights themselves (the LoRA adapters are trainable by
    # default).

    # If use_lora=False, the target expert's FFN weights are unfrozen for full
    # fine-tuning.

    # The router gate row for expert_idx is kept trainable. Gradient masking for the
    # router is handled by ExpertSFTTrainer.training_step, which zeroes out all
    # router gradient rows except expert_idx after each backward pass, before the
    # optimizer step. This is cleaner than a backward hook because it avoids polluting
    # Adam's momentum/variance buffers for the frozen rows.

    # Args:
    #     model: The MoE model to modify.
    #     expert_idx: Index of the expert to train.
    #     use_lora: Whether LoRA will be applied after this call.
    # """
    # for name, param in model.named_parameters():
    #     if not use_lora and f".mlp.experts.{expert_idx}." in name:
    #         param.requires_grad = True
    #         print(f"[trainable] {name}")
    #     elif ".mlp.gate.weight" in name:
    #         # Unfreeze the whole gate; ExpertSFTTrainer masks gradients for all
    #         # rows except expert_idx after each backward pass.
    #         param.requires_grad = True
    #         print(f"[trainable-router] {name}")
    #     else:
    #         param.requires_grad = False


# def freeze_all_except_router(model):
    # """Freeze all parameters except MoE router gate weights."""
    # for name, param in model.named_parameters():
    #     if ".mlp.gate.weight" in name:
    #         param.requires_grad = True
    #         print(f"[trainable-router] {name}")
    #     else:
    #         param.requires_grad = False


# def unfreeze_attention_layers(model):
    # """Unfreeze attention projection parameters by name heuristic."""
    # trainable_count = 0
    # attn_name_markers = (
    #     ".self_attn.",
    #     ".attn.",
    #     ".attention.",
    # )
    # attn_proj_suffixes = (
    #     ".q_proj.",
    #     ".k_proj.",
    #     ".v_proj.",
    #     ".o_proj.",
    #     ".c_attn.",
    #     ".c_proj.",
    #     ".query_key_value.",
    #     ".out_proj.",
    # )

    # for name, param in model.named_parameters():
    #     if any(marker in name for marker in attn_name_markers) or any(suffix in name for suffix in attn_proj_suffixes):
    #         if not param.requires_grad:
    #             param.requires_grad = True
    #         trainable_count += 1
    #         print(f"[trainable-attn] {name}")

    # print(f"Unfroze attention parameters: {trainable_count}")


# def unfreeze_embedding_layers(model):
    # """Unfreeze token embedding parameters by name heuristic."""
    # trainable_count = 0
    # embed_name_markers = (
    #     ".embed_tokens.",
    #     ".word_embeddings.",
    #     ".wte.",
    #     ".tok_embeddings.",
    # )

    # for name, param in model.named_parameters():
    #     if any(marker in name for marker in embed_name_markers):
    #         if not param.requires_grad:
    #             param.requires_grad = True
    #         trainable_count += 1
    #         print(f"[trainable-embed] {name}")

    # print(f"Unfroze embedding parameters: {trainable_count}")


# class ExpertSFTTrainer(SFTTrainer):
    # """
    # SFTTrainer subclass that zeroes out router gate gradients for all experts
    # except the target one, after backward but before the optimizer step.

    # This is preferable to a backward hook because the gradient zeroing happens
    # at a well-defined point in the training loop, and Adam's momentum/variance
    # buffers for the frozen rows are never updated with non-zero values.
    # """
    # def __init__(self, *args, expert_idx: int = None, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.expert_idx = expert_idx

    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     loss = super().training_step(model, inputs, num_items_in_batch)

    #     if self.expert_idx is not None:
    #         for name, param in model.named_parameters():
    #             if ".mlp.gate.weight" in name and param.grad is not None:
    #                 with torch.no_grad():
    #                     mask = torch.zeros_like(param.grad)
    #                     mask[self.expert_idx] = 1.0
    #                     param.grad *= mask

    #     return loss



def get_structured_paths(args):
    """
    Create structured paths within the mounted volumes for organized storage.

    This function maps the configuration to specific directory paths that allow
    multiple models, datasets, and experiments to coexist without conflicts.
    """
    checkpoint_path = (
        pathlib.Path(args.run_output_dir) / args.run_id
    )

    return {
        # "dataset_cache": dataset_cache_path,
        "checkpoints": checkpoint_path,
    }


def main():
    parser = HfArgumentParser([SFTConfig, SFTArgs])
    sft_config, sft_args = parser.parse_args_into_dataclasses()

    print("Parsed SFTConfig:", sft_config)
    print("SFTArgs:", sft_args)

    if sft_args.router_tuning_only and sft_args.train_expert_idx is not None:
        raise ValueError("router_tuning_only is incompatible with train_expert_idx because router tuning affects the routing gate of all experts, not just a single expert.")

    if sft_args.router_tuning_only and sft_args.use_lora:
        raise ValueError("router_tuning_only is incompatible with use_lora because LoRA introduces additional trainable parameters.")

    # load_dotenv()

    run_id = sft_args.run_id
    model_name = sft_args.model
    datasets = sft_args.datasets
    run_seed = sft_args.run_seed
    run_output_dir = sft_args.run_output_dir

    paths = get_structured_paths(sft_args)

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    register_local_architectures()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_bf16_supported()
    print(f"Using {'bfloat16' if use_bf16 else 'float16'}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation="flash_attention_2",
        device_map=None,  # device_map conflicts with DeepSpeed, use auto if single gpu
    )

    # ------------------------------------------------------------------
    # Expert freezing — must happen before LoRA so that PEFT sees the
    # correct requires_grad state when deciding which modules to wrap.
    # ------------------------------------------------------------------
    # if sft_args.router_tuning_only:
    #     print("Freezing all weights except router gate weights.")
    #     freeze_all_except_router(model)
    #     model.enable_input_require_grads() # Need this to still build computational graph?
    #     print_trainable_parameters(model)
    # elif sft_args.train_expert_idx is not None:
    #     print(f"Freezing all weights except expert {sft_args.train_expert_idx} "
    #           f"({'LoRA adapters' if sft_args.use_lora else 'full fine-tune'}).")
    #     freeze_all_except_expert(
    #         model,
    #         expert_idx=sft_args.train_expert_idx,
    #         use_lora=sft_args.use_lora,
    #     )
    #     model.enable_input_require_grads() # Need this to still build computational graph?

    # if sft_args.unfreeze_attn:
    #     print("unfreeze_attn=True: unfreezing attention layers.")
    #     unfreeze_attention_layers(model)

    # if sft_args.unfreeze_embed:
    #     print("unfreeze_embed=True: unfreezing embedding layers.")
    #     unfreeze_embedding_layers(model)

    # ------------------------------------------------------------------
    # LoRA setup
    # ------------------------------------------------------------------
    lora_stats = {}
    if sft_args.use_lora:
        if sft_args.train_expert_idx is not None:
            # Scope LoRA to just the target expert's submodules via substring match.
            # PEFT matches target_modules as substrings of the full module path, so
            # "experts.1" will match e.g. "model.layers.5.mlp.experts.1.gate_proj".
            target_modules = [f"experts.{sft_args.train_expert_idx}"]
            print(f"LoRA scoped to expert {sft_args.train_expert_idx}: target_modules={target_modules}")
        else:
            target_modules = sft_args.lora_target_modules or get_default_lora_target_modules(model_name)
            print(f"Applying LoRA with r={sft_args.lora_r}, alpha={sft_args.lora_alpha}")
            print(f"LoRA target modules: {target_modules}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=sft_args.lora_r,
            lora_alpha=sft_args.lora_alpha,
            lora_dropout=sft_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=None,
        )

        model = get_peft_model(model, lora_config)
        trainable_params, all_params, trainable_pct = print_trainable_parameters(model)
        lora_stats = {
            "lora_r": sft_args.lora_r,
            "lora_alpha": sft_args.lora_alpha,
            "lora_dropout": sft_args.lora_dropout,
            "lora_target_modules": target_modules,
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_pct": trainable_pct,
        }
    elif sft_args.train_expert_idx is not None:
        # Full fine-tune of expert only — just print the parameter count
        print_trainable_parameters(model)

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------
    local_rank = sft_config.local_rank
    if local_rank == 0:
        train_dataset, test_dataset = prepare_datasets(
            datasets,
            seed=run_seed,
            sample_size=sft_args.sample_size,
            filter_by_id=sft_args.filter_by_id,
            skip_eval=sft_args.skip_eval,
        )

    dist.barrier()

    if local_rank != 0:
        train_dataset, test_dataset = prepare_datasets(
            datasets,
            seed=run_seed,
            sample_size=sft_args.sample_size,
            filter_by_id=sft_args.filter_by_id,
            skip_eval=sft_args.skip_eval,
        )

    # train_stats, _ = get_dataset_stats(train_dataset, tokenizer, "Train")
    # test_stats = {}
    # if test_dataset is not None:
    #     test_stats, _ = get_dataset_stats(test_dataset, tokenizer, "Test")

    # Sample some examples to log to wandb
    # NUM_EXAMPLES_TO_LOG = 5
    # train_examples_to_log = train_dataset.shuffle(seed=run_seed).select(range(NUM_EXAMPLES_TO_LOG))
    # test_examples_to_log = (
    #     test_dataset.shuffle(seed=run_seed).select(range(NUM_EXAMPLES_TO_LOG))
    #     if test_dataset is not None else None
    # )

    # ------------------------------------------------------------------
    # SFTConfig
    # ------------------------------------------------------------------
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # lora_suffix = "_lora" if sft_args.use_lora else ""
    # router_suffix = "_router" if sft_args.router_tuning_only else ""
    # expert_suffix = f"_expert{sft_args.train_expert_idx}" if sft_args.train_expert_idx is not None else ""
    # run_output_dir = os.path.join(
    #     run_output_dir,
    #     f"{run_id}{lora_suffix}{router_suffix}{expert_suffix}_{model_name.replace('/', '_')}_{timestamp}"
    # )
    # run_name = f"{run_id}{lora_suffix}{router_suffix}{expert_suffix}_{timestamp}"

    checkpoint_path = paths["checkpoints"]
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    per_device_train_batch_size = sft_config.per_device_train_batch_size
    gradient_accumulation_steps = sft_config.gradient_accumulation_steps

    steps_per_epoch = len(train_dataset) / (per_device_train_batch_size * gradient_accumulation_steps * num_gpus)
    save_steps = max(1, int(steps_per_epoch * sft_args.save_n_epochs))
    print(f"Save steps: {save_steps} (every {sft_args.save_n_epochs} epochs)")

    sft_config.output_dir = str(checkpoint_path)
    sft_config.save_strategy = "steps"
    sft_config.save_steps = save_steps
    sft_config.bf16 = use_bf16
    sft_config.fp16 = not use_bf16
    sft_config.report_to = "wandb"
    sft_config.run_name = run_id
    sft_config.completion_only_loss = True
    sft_config.seed = run_seed
    sft_config.data_seed = run_seed

    if sft_args.skip_eval:
        sft_config.eval_strategy = "no"
        sft_config.do_eval = False
        print("skip_eval=True: evaluation disabled.")
    else:
        eval_steps = max(1, int(steps_per_epoch * sft_args.eval_n_epochs))
        print(f"Eval steps: {eval_steps} (every {sft_args.eval_n_epochs} epochs)")
        sft_config.eval_strategy = "steps"
        sft_config.eval_steps = eval_steps
        sft_config.do_eval = True

    print("Final SFTConfig:", sft_config)

    wandb_stats = {
        "sample_size": sft_args.sample_size,
        "filter_by_id": sft_args.filter_by_id,
        "train_dataset_size": len(train_dataset),
        "test_dataset_size": len(test_dataset) if test_dataset is not None else 0,
        # **train_stats,
        # **test_stats,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "initial_gpu_memory_gb": torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0,
        "use_lora": sft_args.use_lora,
        # "router_tuning_only": sft_args.router_tuning_only,
        # "unfreeze_attn": sft_args.unfreeze_attn,
        # "unfreeze_embed": sft_args.unfreeze_embed,
        **lora_stats,
    }

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    if local_rank == 0:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,  # None when skip_eval=True
            callbacks=[WandbLoggingCallback(wandb_stats)]
        )

    dist.barrier()

    if local_rank != 0:
        # Let the remainder of the workers load from HF Cache
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,  # None when skip_eval=True
            callbacks=[WandbLoggingCallback(wandb_stats)],
            # expert_idx=sft_args.train_expert_idx,
        )

    # dataset = trainer.train_dataset
    # print(dataset[0])
    # quit()

    if not sft_args.skip_eval:
        trainer.evaluate()  # Evaluate once before training

    print("Starting training...")
    trainer.train()
    final_output_dir = os.path.join(str(checkpoint_path), "final")

    if sft_args.use_lora and sft_args.merge_and_save:
        print("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        print(f"Merged model saved to {final_output_dir}")
    else:
        trainer.save_model(output_dir=final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        if sft_args.use_lora:
            print(f"LoRA adapter saved to {final_output_dir}")

    if not sft_args.skip_eval:
        trainer.evaluate()  # Final evaluation after training

    wandb.finish()
    
if __name__ == "__main__":
    main()
