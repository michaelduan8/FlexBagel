import io
import numpy as np
import os
import pathlib
import random
import sys
import torch

import wandb
import json

from dataclasses import dataclass, field
from datasets import load_dataset, Dataset, concatenate_datasets, Image, Sequence
from datetime import datetime
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image as PILImage
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText, HfArgumentParser, TrainerCallback
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint
import shutil

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
    auto_resume: bool = field(
        default=True,
        metadata={"help": "Automatically resume from the latest checkpoint in run_output_dir/run_id if it exists."}
    )
    resume_from_checkpoint: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from. Overrides auto_resume."}
    )
    delete_intermediate_checkpoints: bool = field(
        default=True,
        metadata={"help": "Delete checkpoint-* directories after successful training completion."}
    )


def delete_intermediate_checkpoints(checkpoint_path, keep_final=True):
    """
    Delete Hugging Face intermediate checkpoint-* directories after training completes.

    Keeps:
      - final/ directory
      - non-checkpoint files in checkpoint_path

    Deletes:
      - checkpoint-100/
      - checkpoint-200/
      - etc.
    """
    checkpoint_path = pathlib.Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"No checkpoint path found: {checkpoint_path}")
        return

    deleted = 0
    for child in checkpoint_path.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            print(f"Deleting intermediate checkpoint: {child}")
            shutil.rmtree(child)
            deleted += 1

    print(f"Deleted {deleted} intermediate checkpoint directories.")


def register_local_architectures():
    print("Registering local architectures...")


def preprocess_dataset(dataset):
    """Convert raw rows into one full-conversation example per row.

    This keeps the whole multi-turn dialogue as one training sample. The custom
    collator below is responsible for masking labels so that loss is computed
    only on assistant turns and never on system/user/prompt tokens.

    Returned columns:
      - prompt_id: original row id
      - messages: full conversation in chat-template format
      - images: images referenced by user turns, in placeholder order
    """

    def _maybe_parse_conversation(conversation):
        if isinstance(conversation, str):
            return json.loads(conversation)
        return conversation

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def _turn_image_paths(turn, row_images):
        row_images = _as_list(row_images)

        if turn.get("images") is not None:
            return _as_list(turn.get("images"))
        if turn.get("image") is not None:
            return _as_list(turn.get("image"))

        indices = None
        for key in ("image_indices", "image_idxs", "img_indices", "img_idxs"):
            if turn.get(key) is not None:
                indices = _as_list(turn.get(key))
                break
        if indices is not None:
            return [row_images[int(idx)] for idx in indices]

        # Backward-compatible behavior from your old script: if img_loc is set,
        # attach all row-level images to that user turn.
        if turn.get("img_loc") is not None:
            return row_images

        return []

    def _message_from_turn(turn, row_images, image_accumulator):
        role = turn["role"]
        text = turn.get("content", "")

        if role == "user" and turn.get("img_loc") is not None:
            turn_images = _turn_image_paths(turn, row_images)
            image_blocks = [{"type": "image"} for _ in turn_images]
            text_blocks = [{"type": "text", "text": text}]
            image_accumulator.extend(turn_images)

            if turn.get("img_loc") == "after":
                content = text_blocks + image_blocks
            else:
                content = image_blocks + text_blocks
        else:
            content = [{"type": "text", "text": text}]

        return {"role": role, "content": content}

    def convert_row(item, idx):
        assert "conversation" in item
        if "id" in item:
            row_id = item["id"]
        else:
            row_id = f"row_{idx}"
        row_images = _as_list(item.get("images"))
        conversation = _maybe_parse_conversation(item["conversation"])

        messages = []
        used_images = []
        assistant_count = 0

        for turn_idx, turn in enumerate(conversation):
            role = turn.get("role")
            if role in ("user", "system"):
                messages.append(_message_from_turn(turn, row_images, used_images))
            elif role == "assistant":
                assistant_count += 1
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": turn.get("content", "")}],
                })
            else:
                raise ValueError(f"Unsupported role {role!r} in row {row_id} turn {turn_idx}")

        # Drop rows with no assistant turn because they cannot contribute SFT loss.
        if assistant_count == 0:
            return {"prompt_id": row_id, "messages": None, "images": []}

        return {
            "prompt_id": row_id,
            "messages": messages,
            "images": [{"bytes": None, "path": img_path} for img_path in used_images],
        }

    dataset = dataset.map(
        convert_row,
        remove_columns=dataset.column_names,
        num_proc=12,
        with_indices=True,
        desc="Converting rows to full-conversation SFT examples",
    )
    dataset = dataset.filter(lambda x: x["messages"] is not None)
    dataset = dataset.cast_column("images", Sequence(Image()))
    return dataset


class FullConversationCompletionOnlyCollator:
    """Data collator for one full conversation per row with assistant-only loss.

    This is the full-conversation analogue of DataCollatorForCompletionOnlyLM:
      - tokenize/render the complete chat once;
      - set labels to -100 everywhere;
      - unmask labels only for assistant message spans.

    Compared with expanding into one prompt/completion pair per assistant turn,
    this is more token-efficient because the shared history is encoded once.
    """

    def __init__(self, processor, tokenizer=None, max_length=None, label_pad_token_id=-100):
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else getattr(processor, "tokenizer", processor)
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _text_from_content(content):
        if isinstance(content, str):
            return content
        pieces = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                pieces.append(block.get("text", ""))
        return "".join(pieces)

    def _render(self, messages):
        # The chat template string contains the model-specific role headers and
        # image placeholders, but no generation prompt because we train existing
        # assistant messages, not ask the model to start a new one.
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _assistant_char_spans(self, messages, full_text):
        """Return character spans corresponding to assistant turns.

        We get spans by rendering prefixes of the same conversation. This avoids
        hard-coding response templates like '<|assistant|>' that differ across
        Qwen/Llama/other chat templates.
        """
        spans = []
        search_pos = 0

        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            before = self._render(messages[:i]) if i > 0 else ""
            upto = self._render(messages[: i + 1])

            if full_text.startswith(before) and full_text.startswith(upto):
                start, end = len(before), len(upto)
            else:
                # Fallback for templates where rendering a prefix is not exactly
                # a prefix of rendering the full conversation. Search for the
                # assistant content inside the full text.
                assistant_text = self._text_from_content(msg.get("content", ""))
                start = full_text.find(assistant_text, search_pos)
                if start < 0:
                    continue
                end = start + len(assistant_text)

            if end > start:
                spans.append((start, end))
                search_pos = end

        return spans

    @staticmethod
    def _overlaps(offset, span):
        token_start, token_end = offset
        span_start, span_end = span
        return token_end > span_start and token_start < span_end

    def __call__(self, features):
        messages_batch = [f["messages"] for f in features]
        texts = [self._render(messages) for messages in messages_batch]
        images_batch = [f.get("images", []) for f in features]

        processor_kwargs = dict(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if self.max_length is not None:
            processor_kwargs["max_length"] = self.max_length
        if any(len(imgs) > 0 for imgs in images_batch):
            processor_kwargs["images"] = images_batch

        batch = self.processor(**processor_kwargs)

        labels = batch["input_ids"].clone()
        labels[:] = self.label_pad_token_id

        for batch_idx, (messages, full_text) in enumerate(zip(messages_batch, texts)):
            spans = self._assistant_char_spans(messages, full_text)
            if not spans:
                continue

            tok = self.tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            offsets = tok["offset_mapping"]
            input_ids = tok["input_ids"]

            # Account for left padding if the tokenizer uses it.
            seq_len = int(batch["attention_mask"][batch_idx].sum().item())
            pad_offset = labels.shape[1] - seq_len if self.tokenizer.padding_side == "left" else 0

            for token_idx, offset in enumerate(offsets):
                if token_idx + pad_offset >= labels.shape[1]:
                    break
                if any(self._overlaps(offset, span) for span in spans):
                    labels[batch_idx, token_idx + pad_offset] = input_ids[token_idx]

        batch["labels"] = labels
        return batch

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

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
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
    train_dataset, test_dataset = prepare_datasets(
        datasets,
        seed=run_seed,
        sample_size=sft_args.sample_size,
        filter_by_id=sft_args.filter_by_id,
        skip_eval=sft_args.skip_eval,
    )

    checkpoint_path = paths["checkpoints"]
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = None

    if sft_args.resume_from_checkpoint is not None:
        resume_checkpoint = sft_args.resume_from_checkpoint
    elif sft_args.auto_resume:
        resume_checkpoint = get_last_checkpoint(str(checkpoint_path))

    if local_rank in (-1, 0):
        if resume_checkpoint is not None:
            print(f"Auto-resuming from checkpoint: {resume_checkpoint}")
        else:
            print("No existing checkpoint found. Starting from scratch.")

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
    # We provide labels ourselves in FullConversationCompletionOnlyCollator.
    # Do not let SFTTrainer assume a single prompt/completion boundary.
    sft_config.completion_only_loss = False
    sft_config.seed = run_seed
    sft_config.data_seed = run_seed
    # Keep raw dataset columns (messages/images) until the custom collator sees them.
    sft_config.remove_unused_columns = False
    dataset_kwargs = dict(getattr(sft_config, "dataset_kwargs", None) or {})
    dataset_kwargs["skip_prepare_dataset"] = True
    sft_config.dataset_kwargs = dataset_kwargs

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
    # Full-conversation assistant-only masking collator
    # ------------------------------------------------------------------
    max_length = getattr(sft_config, "max_length", None)
    if max_length is None:
        max_length = getattr(sft_config, "max_seq_length", None)
    data_collator = FullConversationCompletionOnlyCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    print("Checking data collator...")
    batch = data_collator([train_dataset[0]])
    print(batch["input_ids"].shape)
    print(batch["labels"].shape)
    print((batch["labels"] != -100).sum())
    batch = data_collator([train_dataset[1]])
    print(batch["input_ids"].shape)
    print(batch["labels"].shape)
    print((batch["labels"] != -100).sum())

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # None when skip_eval=True
        data_collator=data_collator,
        callbacks=[WandbLoggingCallback(wandb_stats)]
    )

    # dataset = trainer.train_dataset
    # print(dataset[0])
    # quit()

    if not sft_args.skip_eval:
        trainer.evaluate()  # Evaluate once before training

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
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

    if sft_args.delete_intermediate_checkpoints:
        if local_rank in (-1, 0):
            print("Training completed successfully. Deleting intermediate checkpoints...")
            delete_intermediate_checkpoints(checkpoint_path)

    wandb.finish()
    
if __name__ == "__main__":
    main()
