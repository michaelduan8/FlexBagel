import io
import numpy as np
import os
import pathlib
import random
import re
import shutil
import sys
import torch

import wandb
import json

from dataclasses import dataclass, field
from datasets import load_dataset, Dataset, concatenate_datasets, Image, Sequence, Value
from datetime import datetime
from dotenv import load_dotenv
# from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image as PILImage
from PIL import ImageFile
import struct
from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText, HfArgumentParser, TrainerCallback, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import SFTTrainer, SFTConfig
from modeling.flex_qwen2_5_vl_moe import Flex_Qwen2_5_VLMoeConfig, Flex_Qwen2_5_VLMoeForConditionalGeneration
from torch.optim import AdamW
from safetensors.torch import load_file, save_file

ImageFile.LOAD_TRUNCATED_IMAGES = True

_ORIG_GETEXIF = PILImage.Image.getexif

def safe_getexif(self):
    try:
        return _ORIG_GETEXIF(self)
    except (SyntaxError, OSError, ValueError, struct.error) as e:
        # Broken EXIF/TIFF metadata. Ignore EXIF orientation instead of crashing.
        filename = getattr(self, "filename", None)
        print(f"[WARN] Ignoring broken EXIF for image: {filename}, error={repr(e)}")
        return {}

PILImage.Image.getexif = safe_getexif

@dataclass
class SFTArgs:
    run_id: str = field(metadata={"help": "ID for the training run"})
    model: str = field(metadata={"help": "Model to use for training"})
    datasets: list[str] = field(metadata={"help": "Dataset(s) to use for training"})
    run_seed: int = field(default=2025, metadata={"help": "Random seed for training"})
    run_output_dir: str = field(default="./checkpoints", metadata={"help": "Directory to save training runs"})
    auto_resume: bool = field(
        default=True,
        metadata={"help": "Automatically resume from the latest checkpoint in run_output_dir/run_id if it exists."}
    )
    resume_checkpoint_path: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from. Overrides auto_resume."}
    )
    fix_resume_checkpoint_keys: bool = field(
        default=True,
        metadata={"help": "Before resuming, create/use a runtime-key checkpoint with Qwen2.5-VL parameter keys remapped to the current model wrapper."}
    )
    runtime_key_checkpoint_suffix: str = field(
        default="-runtime-key",
        metadata={"help": "Suffix for the derived checkpoint directory used after checkpoint key remapping."}
    )
    delete_intermediate_checkpoints: bool = field(
        default=True,
        metadata={"help": "Delete checkpoint-* directories after successful training completion."}
    )
    sample_size: list[int] = field(
        default=None,
        metadata={"help": "Sampling control. Pass one value for global sampling after datasets are merged "
                          "(same behavior as before), or pass one value per dataset to sample each dataset "
                          "individually before merging."}
    )
    eval_n_epochs: float = field(default=2.0, metadata={"help": "Evaluate every N epochs"})
    save_n_epochs: float = field(default=1.0, metadata={"help": "Save a checkpoint every N epochs"})
    lr_vision: float = field(default=None, metadata={"help": "Learning rate for vision tower"})
    lr_llm: float = field(default=None, metadata={"help": "Learning rate for LLM decoder"})
    lr_connector: float = field(default=None, metadata={"help": "Learning rate for VL connector MLP"})
    filter_by_id: list[str] = field(
        default=None,
        metadata={"help": "Only keep rows whose prompt_id contains at least one of these substrings. If None, no filtering is applied."}
    )
    skip_eval: bool = field(default=False, metadata={"help": "Skip all evaluation. Also skips train/val split — all data is used for training. Useful when using DeepSpeed Stage 1/2."})
    num_experts_per_tok: int = field(
        default=2,
        metadata={"help": "Number of experts to activate per token in MoE layers. Setting >1 can improve performance but increases memory usage and may require higher DeepSpeed stages."}
    )
    norm_topk_prob: bool = field(
        default=False,
        metadata={"help": "If set, normalize the topk probability of the router gate."}
    )
    output_router_logits: bool = field(
        default=False,
        metadata={"help": "If set, output the router logits."}
    )
    router_aux_loss_coef: float = field(
        default=0.0,
        metadata={"help": "Auxiliary loss coefficient for the MoE router."}
    )
    router_depth_aux_loss_coef: float = field(
        default=0.0,
        metadata={"help": "Auxiliary loss coefficient for the MoE router depth."}
    )
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
    unfreeze_non_ffn: bool = field(
        default=False,
        metadata={"help": "If set, unfreeze all parameters that do not belong to FFN/MLP layers after expert/router freezing and before LoRA setup."}
    )
    freeze_lm_decoder: bool = field(
        default=False,
        metadata={"help": "If set, freeze the language-model decoder parameters so they receive no gradient updates. "
                          "This also freezes lm_head when present."}
    )
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "If set, freeze the vision tower parameters so they receive no gradient updates."}
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


def build_vlm_optimizer(
    model,
    sft_config,
    lr_vision=None,
    lr_llm=None,
    lr_connector=None,
):
    """
    Different LR for:
      - vision tower:        visual.*
      - LLM decoder:         language_model.*
      - VL connector MLP:    visual.merger.mlp.*

    Other optimizer args come from sft_config.
    """

    # Use sft_config.learning_rate as default if not specified
    lr_vision = sft_config.learning_rate if lr_vision is None else lr_vision
    lr_llm = sft_config.learning_rate if lr_llm is None else lr_llm
    lr_connector = sft_config.learning_rate if lr_connector is None else lr_connector

    vision_params = []
    llm_params = []
    connector_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check connector first because it is inside visual
        if "visual.merger.mlp." in name:
            connector_params.append(param)

        elif "visual." in name:
            vision_params.append(param)

        elif "language_model." in name:
            llm_params.append(param)

        else:
            other_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": vision_params,
            "lr": lr_vision,
            "weight_decay": sft_config.weight_decay,
            "name": "vision_tower",
        },
        {
            "params": llm_params,
            "lr": lr_llm,
            "weight_decay": sft_config.weight_decay,
            "name": "llm_decoder",
        },
        {
            "params": connector_params,
            "lr": lr_connector,
            "weight_decay": sft_config.weight_decay,
            "name": "vl_connector_mlp",
        },
    ]

    if len(other_params) > 0:
        optimizer_grouped_parameters.append(
            {
                "params": other_params,
                "lr": lr_llm,
                "weight_decay": sft_config.weight_decay,
                "name": "other",
            }
        )

    optimizer = AdamW(
        optimizer_grouped_parameters,
        betas=(sft_config.adam_beta1, sft_config.adam_beta2),
        eps=sft_config.adam_epsilon,
    )

    return optimizer


def measure_vlm_lengths(dataset, processor, n=200):
    lengths = []
    text_only_lengths = []
    image_token_estimates = []

    n = min(n, len(dataset))

    for ex in dataset.select(range(n)):
        messages = ex["prompt"] + ex["completion"]

        # Text with image placeholders
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Load PIL images from dataset
        images = ex["images"]

        # Full VLM processing: text + images
        inputs = processor(
            text=text,
            images=images,
            return_tensors="pt",
        )

        full_len = inputs["input_ids"].shape[-1]

        # Text-only length, for comparison
        text_inputs = processor.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        text_len = text_inputs["input_ids"].shape[-1]

        lengths.append(full_len)
        text_only_lengths.append(text_len)
        image_token_estimates.append(full_len - text_len)

    lengths = np.array(lengths)
    text_only_lengths = np.array(text_only_lengths)
    image_token_estimates = np.array(image_token_estimates)

    print("=== Full VLM sequence length: text + image tokens ===")
    print(f"mean: {lengths.mean():.1f}")
    print(f"p50:  {np.percentile(lengths, 50):.1f}")
    print(f"p90:  {np.percentile(lengths, 90):.1f}")
    print(f"p95:  {np.percentile(lengths, 95):.1f}")
    print(f"p99:  {np.percentile(lengths, 99):.1f}")
    print(f"max:  {lengths.max()}")

    print("\n=== Text-only length ===")
    print(f"mean: {text_only_lengths.mean():.1f}")
    print(f"p95:  {np.percentile(text_only_lengths, 95):.1f}")
    print(f"max:  {text_only_lengths.max()}")

    print("\n=== Estimated image-token contribution ===")
    print(f"mean: {image_token_estimates.mean():.1f}")
    print(f"p50:  {np.percentile(image_token_estimates, 50):.1f}")
    print(f"p90:  {np.percentile(image_token_estimates, 90):.1f}")
    print(f"p95:  {np.percentile(image_token_estimates, 95):.1f}")
    print(f"max:  {image_token_estimates.max()}")


def register_local_architectures():
    print("Registering local architectures...")

    # Register configs to AutoConfig
    AutoConfig.register("flex_qwen2_5_vl_moe", Flex_Qwen2_5_VLMoeConfig)

    # Register models to AutoModelForCausalLM
    AutoModelForImageTextToText.register(Flex_Qwen2_5_VLMoeConfig, Flex_Qwen2_5_VLMoeForConditionalGeneration)


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

    def convert_row(item, idx):
        image_key = "images"
        assert image_key in item and "conversation" in item
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

        if "prompt_id" in item:
            id = item["prompt_id"]
        elif "id" in item:
            id = item["id"]
        else:
            id = idx

        conversation = item["conversation"]
        assert conversation[-1]["role"] == "assistant"
        # assert len(conversation) <= 3

        images = item[image_key]

        prompt = []
        used_image = False

        for turn in conversation[:-1]:
            content = [{"type": "text", "text": turn["content"]}]

            # Minimal multi-turn support:
            # If a user turn has img_loc, attach the top-level images only once.
            # This avoids duplicating image placeholders across multiple user turns.
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
            "prompt_id": id,
            "prompt": prompt,
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": conversation[-1]["content"]}]}],
            # TODO: could be ok to not have images?
            # "images": [{"bytes": b, "path": None} for b in rgb_bytes] if rgb_bytes else None,
            "images": [{"bytes": None, "path": img_path} for img_path in images] if images else None,
        }

    #dataset = dataset.map(load_bytes, num_proc=24)
    dataset = dataset.map(convert_row, remove_columns=dataset.column_names, num_proc=12, with_indices=True).filter(lambda x: x["images"] is not None)
    dataset = dataset.cast_column("images", Sequence(Image())).cast_column("prompt_id", Value("string"))

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


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_pct = 100 * trainable_params / all_param
    print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {trainable_pct:.4f}")
    return trainable_params, all_param, trainable_pct


def freeze_all_except_expert(model, expert_idx: int, use_lora: bool = False, verbose: bool = True):
    """
    Freeze all parameters except the target expert and router gates in BOTH:
      1. decoder MoE blocks:      model.language_model.layers.*.mlp
      2. vision tower MoE blocks: model.visual.blocks.*.mlp

    If use_lora=True:
        Base expert weights stay frozen. LoRA adapters should be inserted after
        this call and will be trainable by default.

    If use_lora=False:
        The target expert FFN weights are unfrozen for full fine-tuning.

    Router gates:
        The full gate.weight is unfrozen because PyTorch cannot set
        requires_grad=True for only one row. Your trainer should mask all router
        gradient rows except expert_idx before optimizer.step().
    """

    def is_sparse_moe_block(module):
        return (
            hasattr(module, "experts")
            and hasattr(module, "gate")
            and hasattr(module.gate, "weight")
        )

    def get_scope(module_name: str) -> str:
        parts = module_name.split(".")

        # Works when called on Flex_Qwen2_5_VLMoeForConditionalGeneration
        # or on the inner Flex_Qwen2_5_VLMoeModel.
        if "visual" in parts and "blocks" in parts:
            return "vision"
        if "language_model" in parts and "layers" in parts:
            return "decoder"

        # Works if you pass model.visual or model.language_model directly.
        if len(parts) > 0 and parts[0] == "blocks":
            return "vision"
        if len(parts) > 0 and parts[0] == "layers":
            return "decoder"

        return "other"

    # 1. Freeze everything first.
    for _, param in model.named_parameters():
        param.requires_grad = False

    num_decoder_moe = 0
    num_vision_moe = 0
    num_other_moe = 0
    trainable_names = []

    # 2. Unfreeze target expert + router gate in every SparseMoeBlock.
    for module_name, module in model.named_modules():
        if not is_sparse_moe_block(module):
            continue

        scope = get_scope(module_name)

        if expert_idx < 0 or expert_idx >= len(module.experts):
            raise ValueError(
                f"expert_idx={expert_idx} is invalid for {module_name}; "
                f"this block has {len(module.experts)} experts."
            )

        if scope == "decoder":
            num_decoder_moe += 1
        elif scope == "vision":
            num_vision_moe += 1
        else:
            num_other_moe += 1

        # Full fine-tuning case: unfreeze the target expert FFN.
        # LoRA case: keep base expert frozen; LoRA adapters are added later.
        if not use_lora:
            target_expert = module.experts[expert_idx]
            for pname, param in target_expert.named_parameters():
                param.requires_grad = True
                full_name = f"{module_name}.experts.{expert_idx}.{pname}".lstrip(".")
                trainable_names.append(full_name)
                if verbose:
                    print(f"[trainable-{scope}-expert] {full_name}")

        # Router: unfreeze whole gate.weight.
        # Your trainer masks rows except expert_idx after backward.
        for pname, param in module.gate.named_parameters():
            param.requires_grad = True
            full_name = f"{module_name}.gate.{pname}".lstrip(".")
            trainable_names.append(full_name)
            if verbose:
                print(f"[trainable-{scope}-router] {full_name}")

    if num_decoder_moe + num_vision_moe + num_other_moe == 0:
        raise RuntimeError(
            "No SparseMoeBlock found. Check that you passed the full VLM model, "
            "or that the model actually has num_experts > 0."
        )

    if verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(
            "[freeze_all_except_expert] "
            f"decoder MoE blocks={num_decoder_moe}, "
            f"vision MoE blocks={num_vision_moe}, "
            f"other MoE blocks={num_other_moe}"
        )
        print(
            "[freeze_all_except_expert] "
            f"trainable params={trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.4f}%)"
        )

    return trainable_names


def freeze_all_except_router(model):
    """Freeze all parameters except MoE router gate weights."""
    for name, param in model.named_parameters():
        if ".mlp.gate.weight" in name:
            param.requires_grad = True
            print(f"[trainable-router] {name}")
        else:
            param.requires_grad = False


def freeze_lm_decoder_parameters(model, freeze_lm_head: bool = True, verbose: bool = True):
    """
    Freeze the language-model decoder path so it receives no optimizer updates.

    This is intentionally a post-processing freeze: call it after other freezing/
    unfreezing logic and after LoRA injection, so it also disables any trainable
    adapters inserted under the language_model path.
    """
    frozen_names = []

    for name, param in model.named_parameters():
        is_lm_decoder = (
            name.startswith("language_model.")
            or name.startswith("model.language_model.")
            or ".language_model." in name
        )
        is_lm_head = freeze_lm_head and (
            name == "lm_head.weight"
            or name.startswith("lm_head.")
            or name.endswith(".lm_head.weight")
            or ".lm_head." in name
        )

        if is_lm_decoder or is_lm_head:
            if param.requires_grad:
                frozen_names.append(name)
            param.requires_grad = False

    if verbose:
        print(
            f"[freeze_lm_decoder] frozen trainable params/tensors under language_model"
            f"{' and lm_head' if freeze_lm_head else ''}: {len(frozen_names)}"
        )
        for name in frozen_names[:50]:
            print(f"[frozen-lm-decoder] {name}")
        if len(frozen_names) > 50:
            print(f"[frozen-lm-decoder] ... {len(frozen_names) - 50} more")

    return frozen_names

def unfreeze_non_ffn_parameters(model, verbose: bool = True):
    """
    Unfreeze all parameters except FFN/MLP weights under decoder layers or
    vision blocks.

    This is intended as a post-processing step after a selective freeze so that
    attention, embeddings, norms, patch merger weights, and other non-FFN
    weights can be trained together while decoder/vision feed-forward blocks
    remain frozen.
    """
    layer_prefixes = (
        "model.language_model.layers.",
        "language_model.layers.",
        "model.layers.",
        "layers.",
        "model.visual.blocks.",
        "visual.blocks.",
        "blocks.",
    )
    ffn_name_markers = (
        ".mlp.",
        ".ffn.",
        "feed_forward",
        "feedforward",
    )

    def is_decoder_or_vision_ffn(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in layer_prefixes) and any(
            marker in name for marker in ffn_name_markers
        )

    unfrozen_names = []

    for name, param in model.named_parameters():
        if is_decoder_or_vision_ffn(name):
            continue

        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_names.append(name)
            if verbose:
                print(f"[trainable-non-ffn] {name}")

    if verbose:
        print(f"[unfreeze_non_ffn] unfroze params/tensors outside FFN/MLP paths: {len(unfrozen_names)}")

    return unfrozen_names


def freeze_vision_tower_parameters(model, verbose: bool = True):
    """
    Freeze the vision tower path so it receives no optimizer updates.

    This is intentionally a post-processing freeze: call it after other freezing/
    unfreezing logic and after LoRA injection, so it also disables any trainable
    adapters inserted under the vision tower path.
    """
    frozen_names = []

    for name, param in model.named_parameters():
        is_vision_tower = (
            name.startswith("visual.")
            or name.startswith("model.visual.")
            or ".visual." in name
        )

        if is_vision_tower:
            if param.requires_grad:
                frozen_names.append(name)
            param.requires_grad = False

    if verbose:
        print(
            "[freeze_vision_tower] frozen trainable params/tensors under visual: "
            f"{len(frozen_names)}"
        )
        for name in frozen_names[:50]:
            print(f"[frozen-vision-tower] {name}")
        if len(frozen_names) > 50:
            print(f"[frozen-vision-tower] ... {len(frozen_names) - 50} more")

    return frozen_names


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


class ExpertSFTTrainer(SFTTrainer):
    """
    SFTTrainer subclass that zeroes out router gate gradients for all experts
    except the target one, after backward but before the optimizer step.

    This is preferable to a backward hook because the gradient zeroing happens
    at a well-defined point in the training loop, and Adam's momentum/variance
    buffers for the frozen rows are never updated with non-zero values.
    """
    def __init__(self, *args, expert_idx: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_idx = expert_idx

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.expert_idx is not None:
            for name, param in model.named_parameters():
                if ".mlp.gate.weight" in name and param.grad is not None:
                    with torch.no_grad():
                        mask = torch.zeros_like(param.grad)
                        mask[self.expert_idx] = 1.0
                        param.grad *= mask

        return loss



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


def _checkpoint_step(checkpoint_dir: pathlib.Path) -> int:
    """Return the numeric step from a checkpoint-* directory name, or -1 if invalid."""
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_dir.name)
    return int(match.group(1)) if match else -1


def get_latest_checkpoint(checkpoint_path: pathlib.Path) -> str | None:
    """Find the latest Hugging Face Trainer checkpoint under checkpoint_path."""
    if not checkpoint_path.exists():
        return None

    checkpoints = [
        path for path in checkpoint_path.iterdir()
        if path.is_dir() and _checkpoint_step(path) >= 0
    ]
    if not checkpoints:
        return None

    latest = max(checkpoints, key=_checkpoint_step)
    return str(latest)


def resolve_resume_checkpoint(sft_args, checkpoint_path: pathlib.Path) -> str | None:
    """Resolve the checkpoint path used by trainer.train(resume_from_checkpoint=...)."""
    if sft_args.resume_checkpoint_path is not None:
        resume_path = pathlib.Path(sft_args.resume_checkpoint_path).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_checkpoint_path does not exist: {resume_path}")
        print(f"Resuming from explicit checkpoint: {resume_path}")
        return str(resume_path)

    if sft_args.auto_resume:
        latest_checkpoint = get_latest_checkpoint(checkpoint_path)
        if latest_checkpoint is not None:
            print(f"Auto-resume enabled. Resuming from latest checkpoint: {latest_checkpoint}")
            return latest_checkpoint
        print(f"Auto-resume enabled, but no checkpoint-* directory found in {checkpoint_path}. Starting from scratch.")

    return None



def remap_qwen25vl_runtime_key(key: str) -> str:
    """
    Remap legacy Qwen2.5-VL checkpoint keys to the names expected by the
    currently loaded wrapper model.

    This is needed for Trainer resume because resume_from_checkpoint loads the
    raw checkpoint state dict directly; it does not go through from_pretrained's
    checkpoint conversion mapping.
    """
    # legacy vision tower:
    # visual.blocks... -> model.visual.blocks...
    if key.startswith("visual."):
        return "model.visual." + key[len("visual."):]

    # legacy language model:
    # model.layers...       -> model.language_model.layers...
    # model.embed_tokens... -> model.language_model.embed_tokens...
    # model.norm...         -> model.language_model.norm...
    if key.startswith("model.") and not (
        key.startswith("model.visual.")
        or key.startswith("model.language_model.")
    ):
        return "model.language_model." + key[len("model."):]

    # lm_head and already-correct keys stay unchanged.
    return key


def get_runtime_key_checkpoint_path(
    resume_checkpoint: str | None,
    suffix: str = "-runtime-key",
) -> str | None:
    """
    Return the checkpoint path that trainer.train should use.

    If the checkpoint is a sharded safetensors model, use a derived directory
    so the original checkpoint remains untouched. Otherwise return the original
    path unchanged.
    """
    if resume_checkpoint is None:
        return None

    src_ckpt = pathlib.Path(resume_checkpoint).expanduser()
    if src_ckpt.name.endswith(suffix):
        return str(src_ckpt)

    if not (src_ckpt / "model.safetensors.index.json").exists():
        print(
            f"[checkpoint-key-fix] {src_ckpt} has no model.safetensors.index.json; "
            "using it unchanged."
        )
        return str(src_ckpt)

    return str(src_ckpt.with_name(src_ckpt.name + suffix))


def _rewrite_safetensor_shard_keys_if_needed(shard_path: pathlib.Path):
    """Rewrite one shard in place if it still contains legacy keys."""
    sd = load_file(str(shard_path), device="cpu")
    sd2 = {}
    changed = False

    for old_key, tensor in sd.items():
        new_key = remap_qwen25vl_runtime_key(old_key)
        if new_key != old_key:
            changed = True
        if new_key in sd2:
            raise RuntimeError(
                f"Key collision inside {shard_path.name}: {old_key} -> {new_key}"
            )
        sd2[new_key] = tensor

    if not changed:
        return False

    backup_path = shard_path.with_name(f"{shard_path.name}.legacy_key_backup")
    if backup_path.exists():
        raise FileExistsError(
            f"{backup_path} already exists, but {shard_path} still appears to need remapping. "
            "Please inspect the checkpoint; a previous run may have stopped midway."
        )

    shard_path.rename(backup_path)
    save_file(sd2, str(shard_path), metadata={"format": "pt"})
    print(f"[checkpoint-key-fix] Rewrote {shard_path.name}")
    print(f"[checkpoint-key-fix] Backup:  {backup_path.name}")
    return True


def _remap_checkpoint_index_and_shards(ckpt: pathlib.Path):
    """Apply Fixing Code 1, but make it safe to call repeatedly."""
    index_path = ckpt / "model.safetensors.index.json"
    if not index_path.exists():
        return

    with open(index_path, "r") as f:
        index = json.load(f)

    old_weight_map = index["weight_map"]
    new_weight_map = {}

    for old_key, shard_name in old_weight_map.items():
        new_key = remap_qwen25vl_runtime_key(old_key)
        if new_key in new_weight_map and new_weight_map[new_key] != shard_name:
            raise RuntimeError(f"Key collision after remap: {old_key} -> {new_key}")
        new_weight_map[new_key] = shard_name

    if new_weight_map != old_weight_map:
        backup_index_path = ckpt / "model.safetensors.index.json.legacy_key_backup"
        if not backup_index_path.exists():
            shutil.copy2(index_path, backup_index_path)
        index["weight_map"] = new_weight_map
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"[checkpoint-key-fix] Updated index key names: {index_path}")
    else:
        print("[checkpoint-key-fix] Index already uses runtime key names.")

    # Use all shard names now referenced by the updated index. This also lets us
    # fix a checkpoint whose index was updated but whose shards were not.
    shard_names = sorted(set(new_weight_map.values()))
    for shard_name in shard_names:
        _rewrite_safetensor_shard_keys_if_needed(ckpt / shard_name)


def _ensure_lm_head_weight(ckpt: pathlib.Path):
    """Apply Fixing Code 2: synthesize lm_head.weight from embed_tokens if needed."""
    index_path = ckpt / "model.safetensors.index.json"
    if not index_path.exists():
        return

    embed_key = "model.language_model.embed_tokens.weight"
    lm_head_key = "lm_head.weight"

    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    if lm_head_key in weight_map:
        print(f"[checkpoint-key-fix] {lm_head_key} already exists in index: {weight_map[lm_head_key]}")
        return

    if embed_key not in weight_map:
        raise KeyError(f"Cannot synthesize {lm_head_key}; missing {embed_key} in index")

    shard_name = weight_map[embed_key]
    shard_path = ckpt / shard_name
    backup_path = ckpt / f"{shard_name}.pre_lm_head_backup"

    sd = load_file(str(shard_path), device="cpu")
    if embed_key not in sd:
        raise KeyError(f"{embed_key} not found inside {shard_name}")

    if lm_head_key not in sd:
        if backup_path.exists():
            raise FileExistsError(
                f"{backup_path} already exists, but {lm_head_key} is still missing from {shard_name}. "
                "Please inspect the checkpoint; a previous run may have stopped midway."
            )
        sd[lm_head_key] = sd[embed_key].clone()
        shard_path.rename(backup_path)
        save_file(sd, str(shard_path), metadata={"format": "pt"})
        print(f"[checkpoint-key-fix] Added {lm_head_key} to {shard_name}")
        print(f"[checkpoint-key-fix] Backup: {backup_path.name}")

        added_bytes = sd[lm_head_key].numel() * sd[lm_head_key].element_size()
        if isinstance(index.get("metadata", {}).get("total_size"), int):
            index["metadata"]["total_size"] += added_bytes
    else:
        print(f"[checkpoint-key-fix] {lm_head_key} already exists inside {shard_name}; only fixing index.")

    weight_map[lm_head_key] = shard_name
    index["weight_map"] = weight_map
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[checkpoint-key-fix] Updated index: {lm_head_key} -> {shard_name}")


def prepare_qwen25vl_runtime_key_checkpoint(
    resume_checkpoint: str | None,
    suffix: str = "-runtime-key",
) -> str | None:
    """
    Create/use a resume checkpoint whose tensor keys match the runtime model.

    The returned path should be passed to trainer.train(resume_from_checkpoint=...).
    The original checkpoint is not modified unless the input path already ends
    with suffix.
    """
    if resume_checkpoint is None:
        return None

    src_ckpt = pathlib.Path(resume_checkpoint).expanduser()
    if not src_ckpt.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {src_ckpt}")

    if not (src_ckpt / "model.safetensors.index.json").exists():
        print(
            f"[checkpoint-key-fix] {src_ckpt} has no model.safetensors.index.json; "
            "skipping key fix."
        )
        return str(src_ckpt)

    if src_ckpt.name.endswith(suffix):
        dst_ckpt = src_ckpt
    else:
        dst_ckpt = src_ckpt.with_name(src_ckpt.name + suffix)
        if not dst_ckpt.exists():
            print(f"[checkpoint-key-fix] Copying checkpoint for runtime-key resume:")
            print(f"[checkpoint-key-fix]   src: {src_ckpt}")
            print(f"[checkpoint-key-fix]   dst: {dst_ckpt}")
            shutil.copytree(src_ckpt, dst_ckpt)
        else:
            print(f"[checkpoint-key-fix] Using existing runtime-key checkpoint: {dst_ckpt}")

    _remap_checkpoint_index_and_shards(dst_ckpt)
    _ensure_lm_head_weight(dst_ckpt)

    marker_path = dst_ckpt / ".qwen25vl_runtime_key_fix_done"
    marker_path.write_text(
        json.dumps(
            {
                "source_checkpoint": str(src_ckpt),
                "runtime_checkpoint": str(dst_ckpt),
                "fixed_at": datetime.now().isoformat(),
            },
            indent=2,
        )
    )

    return str(dst_ckpt)

def delete_intermediate_checkpoints(checkpoint_path: pathlib.Path, final_output_dir: str):
    """Delete checkpoint-* directories after training has completed successfully."""
    final_path = pathlib.Path(final_output_dir).resolve()
    deleted = []

    for path in checkpoint_path.iterdir():
        if not path.is_dir() or _checkpoint_step(path) < 0:
            continue
        if path.resolve() == final_path:
            continue
        print(f"Deleting intermediate checkpoint: {path}")
        shutil.rmtree(path)
        deleted.append(str(path))

    print(f"Deleted {len(deleted)} intermediate checkpoint(s).")


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
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    # train_dataset, test_dataset = prepare_datasets(
    #     datasets,
    #     seed=run_seed,
    #     sample_size=sft_args.sample_size,
    #     filter_by_id=sft_args.filter_by_id,
    #     skip_eval=sft_args.skip_eval,
    # )

    # measure_vlm_lengths(train_dataset, processor, n=500)
    # assert False

    use_bf16 = torch.cuda.is_bf16_supported()
    print(f"Using {'bfloat16' if use_bf16 else 'float16'}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation="flash_attention_2",
        # attn_implementation={"": "sdpa"},
        device_map=None,  # device_map conflicts with DeepSpeed, use auto if single gpu
    )

    # Set num_experts_per_tok to activate for training
    model.config.text_config.num_experts_per_tok = sft_args.num_experts_per_tok
    model.config.text_config.norm_topk_prob = sft_args.norm_topk_prob
    model.config.text_config.output_router_logits = sft_args.output_router_logits
    model.config.text_config.router_aux_loss_coef = sft_args.router_aux_loss_coef
    model.config.text_config.router_depth_aux_loss_coef = sft_args.router_depth_aux_loss_coef

    model.config.vision_config.num_experts_per_tok = sft_args.num_experts_per_tok
    model.config.vision_config.norm_topk_prob = sft_args.norm_topk_prob
    model.config.vision_config.output_router_logits = sft_args.output_router_logits
    model.config.vision_config.router_aux_loss_coef = sft_args.router_aux_loss_coef
    model.config.vision_config.router_depth_aux_loss_coef = sft_args.router_depth_aux_loss_coef

    print("Loaded model class:", type(model))
    print("Config class:", type(model.config))
    print("Model type:", model.config.model_type)
    print("Conversion mapping:", getattr(model, "_checkpoint_conversion_mapping", None))

    names = [name for name, _ in model.named_parameters()]
    print("has wrapped language_model:", any(n.startswith("model.language_model.layers.0") for n in names))
    print("has raw model.layers:", any(n.startswith("model.layers.0") for n in names))
    print("has wrapped visual:", any(n.startswith("model.visual.blocks.0") for n in names))
    print("has raw visual:", any(n.startswith("visual.blocks.0") for n in names))

    # ------------------------------------------------------------------
    # Expert freezing — must happen before LoRA so that PEFT sees the
    # correct requires_grad state when deciding which modules to wrap.
    # ------------------------------------------------------------------
    if sft_args.router_tuning_only:
        print("Freezing all weights except router gate weights.")
        freeze_all_except_router(model)
        model.enable_input_require_grads() # Need this to still build computational graph?
        print_trainable_parameters(model)
    elif sft_args.train_expert_idx is not None:
        print(f"Freezing all weights except expert {sft_args.train_expert_idx} "
              f"({'LoRA adapters' if sft_args.use_lora else 'full fine-tune'}).")
        print_trainable_parameters(model)
        freeze_all_except_expert(
            model,
            expert_idx=sft_args.train_expert_idx,
            use_lora=sft_args.use_lora,
        )
        print("After freezing expert:")
        print_trainable_parameters(model)
        model.enable_input_require_grads() # Need this to still build computational graph?

    if sft_args.unfreeze_attn:
        raise NotImplementedError("unfreeze_attn is not implemented for mm_tune.py")
        print("unfreeze_attn=True: unfreezing attention layers.")
        unfreeze_attention_layers(model)

    if sft_args.unfreeze_embed:
        raise NotImplementedError("unfreeze_embed is not implemented for mm_tune.py")
        print("unfreeze_embed=True: unfreezing embedding layers.")
        unfreeze_embedding_layers(model)

    if sft_args.unfreeze_non_ffn:
        print("unfreeze_non_ffn=True: unfreezing all non-FFN/non-MLP parameters.")
        unfreeze_non_ffn_parameters(model)
        print("After unfreeze_non_ffn:")
        print_trainable_parameters(model)

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

    if sft_args.freeze_lm_decoder:
        print("freeze_lm_decoder=True: freezing language-model decoder parameters.")
        freeze_lm_decoder_parameters(model, freeze_lm_head=True)
        print("After freeze_lm_decoder:")
        print_trainable_parameters(model)

    if sft_args.freeze_vision_tower:
        print("freeze_vision_tower=True: freezing vision tower parameters.")
        freeze_vision_tower_parameters(model)
        print("After freeze_vision_tower:")
        print_trainable_parameters(model)

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------
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
    raw_resume_checkpoint = resolve_resume_checkpoint(sft_args, checkpoint_path)
    if sft_args.fix_resume_checkpoint_keys:
        resume_checkpoint = get_runtime_key_checkpoint_path(
            raw_resume_checkpoint,
            suffix=sft_args.runtime_key_checkpoint_suffix,
        )
    else:
        resume_checkpoint = raw_resume_checkpoint

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

    # VLM-specific safety
    sft_config.remove_unused_columns = False
    sft_config.max_length = None
    if sft_config.gradient_checkpointing:
        sft_config.gradient_checkpointing_kwargs = {"use_reentrant": False}

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
        "unfreeze_non_ffn": sft_args.unfreeze_non_ffn,
        "freeze_lm_decoder": sft_args.freeze_lm_decoder,
        "freeze_vision_tower": sft_args.freeze_vision_tower,
        **lora_stats,
    }

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    optimizer = build_vlm_optimizer(
        model,
        sft_config,
        lr_vision=sft_args.lr_vision,
        lr_llm=sft_args.lr_llm,
        lr_connector=sft_args.lr_connector
    )

    trainer = ExpertSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=processor,
        optimizers=(optimizer, None),  # let Trainer create scheduler only if you handle it separately
        callbacks=[WandbLoggingCallback(wandb_stats)],
        expert_idx=sft_args.train_expert_idx,
    )

    # Materialize the fixed resume checkpoint after Trainer/Accelerate exists, so
    # only rank 0 rewrites files and all other ranks wait before loading them.
    if raw_resume_checkpoint is not None and sft_args.fix_resume_checkpoint_keys:
        if trainer.is_world_process_zero():
            fixed_checkpoint = prepare_qwen25vl_runtime_key_checkpoint(
                raw_resume_checkpoint,
                suffix=sft_args.runtime_key_checkpoint_suffix,
            )
            if fixed_checkpoint != resume_checkpoint:
                raise RuntimeError(
                    f"Internal checkpoint path mismatch: expected {resume_checkpoint}, got {fixed_checkpoint}"
                )
        trainer.accelerator.wait_for_everyone()
        if not pathlib.Path(resume_checkpoint).exists():
            raise FileNotFoundError(f"Fixed resume checkpoint was not created: {resume_checkpoint}")

    batch = next(iter(trainer.get_train_dataloader()))

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape), v.dtype)
        else:
            print(k, type(v))

    assert batch["input_ids"].ndim == 2, batch["input_ids"].shape
    assert batch["attention_mask"].ndim == 2, batch["attention_mask"].shape
    assert "pixel_values" in batch, batch.keys()
    assert "image_grid_thw" in batch, batch.keys()
    # dataset = trainer.train_dataset
    # print(dataset[0])
    # quit()

    if not sft_args.skip_eval:
        trainer.evaluate()  # Evaluate once before training

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_output_dir = os.path.join(str(checkpoint_path), "final")


    # Wait for all ranks to finish training
    trainer.accelerator.wait_for_everyone()

    # Save only from rank 0 / main process
    if trainer.is_world_process_zero():
        if sft_args.use_lora and sft_args.merge_and_save:
            print("Merging LoRA weights into base model...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(final_output_dir)
            tokenizer.save_pretrained(final_output_dir)
            processor.save_pretrained(final_output_dir)
            print(f"Merged model saved to {final_output_dir}")
        else:
            trainer.save_model(output_dir=final_output_dir)
            tokenizer.save_pretrained(final_output_dir)
            processor.save_pretrained(final_output_dir)
            if sft_args.use_lora:
                print(f"LoRA adapter saved to {final_output_dir}")

    trainer.accelerator.wait_for_everyone()

    if not sft_args.skip_eval and trainer.is_world_process_zero():
        trainer.evaluate()  # Final evaluation after training

    if sft_args.delete_intermediate_checkpoints and trainer.is_world_process_zero():
        print("Training completed successfully. Deleting intermediate checkpoints...")
        delete_intermediate_checkpoints(checkpoint_path, final_output_dir)

    trainer.accelerator.wait_for_everyone()
    wandb.finish()
    
if __name__ == "__main__":
    main()
