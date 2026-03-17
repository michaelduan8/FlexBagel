#!/usr/bin/env python3
"""
Model parameter averaging ("Model soups") for Hugging Face Transformers models.

Examples
--------
Uniform average of local HF model directories:
    python hf_model_soup.py \
        --models /path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3 \
        --output_dir /path/to/averaged_model

Weighted average:
    python hf_model_soup.py \
        --models /path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3 \
        --weights 0.2 0.3 0.5 \
        --output_dir /path/to/averaged_model

Using a Hub base model class/config but local fine-tuned checkpoints:
    python hf_model_soup.py \
        --models ./ft_run1 ./ft_run2 ./ft_run3 \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --model_class causal-lm \
        --output_dir ./averaged_llama

Notes
-----
- All models must share the same architecture and parameter names/shapes.
- This script averages floating-point tensors only.
- Non-floating tensors are copied from the first model.
- Input model paths can be local HF directories or model IDs on the Hub.
- The averaged model is saved with save_pretrained().
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

MODEL_CLASS_MAP = {
    "auto": AutoModel,
    "causal-lm": AutoModelForCausalLM,
    "seq2seq-lm": AutoModelForSeq2SeqLM,
    "masked-lm": AutoModelForMaskedLM,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average Hugging Face Transformers model checkpoints."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="HF model directories or Hub model IDs to average.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional averaging weights. Must match number of models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save averaged HF model.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help=(
            "Optional base model/config source used to instantiate the final averaged model. "
            "If omitted, the first entry in --models is used."
        ),
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="causal-lm",
        choices=list(MODEL_CLASS_MAP.keys()),
        help=(
            "Which HF auto model class to use. "
            "'causal-lm' is typical for Llama/Qwen/Mistral style models."
        ),
    )
    parser.add_argument(
        "--strict_keys",
        action="store_true",
        help="Require all models to have exactly the same state_dict keys.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Dtype used when loading source models.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load models onto. Usually cpu.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code=True for custom HF model repos.",
    )
    parser.add_argument(
        "--copy_tokenizer",
        action="store_true",
        help="Also save tokenizer from base_model/first model into output_dir.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push averaged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_repo",
        type=str,
        default=None,
        help="Repo name like username/model-name",
    )

    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Create private repo on HF",
    )
    return parser.parse_args()


def normalize_weights(weights: List[float]) -> List[float]:
    total = sum(weights)
    if total == 0:
        raise ValueError("Sum of weights cannot be zero.")
    return [w / total for w in weights]


def parse_torch_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def get_model_cls(name: str):
    return MODEL_CLASS_MAP[name]


def load_hf_model(model_name_or_path: str, model_cls, torch_dtype, device: str, trust_remote_code: bool):
    kwargs = {
        "trust_remote_code": trust_remote_code,
    }

    # transformers supports low_cpu_mem_usage on many models; useful for large models
    # but safe to omit if unsupported in older versions
    kwargs["low_cpu_mem_usage"] = True

    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype

    model = model_cls.from_pretrained(model_name_or_path, **kwargs)

    if device != "cpu":
        model = model.to(device)

    model.eval()
    return model


def compare_keys(
    state_dicts: List[Dict[str, torch.Tensor]],
    strict_keys: bool = False,
) -> List[str]:
    key_sets = [set(sd.keys()) for sd in state_dicts]

    if strict_keys:
        ref = key_sets[0]
        for i, ks in enumerate(key_sets[1:], start=1):
            if ks != ref:
                missing = sorted(ref - ks)
                extra = sorted(ks - ref)
                raise ValueError(
                    f"Model {i} keys do not match reference.\n"
                    f"Missing keys: {missing[:20]}\n"
                    f"Extra keys: {extra[:20]}"
                )
        return sorted(ref)

    common_keys = set.intersection(*key_sets)
    if not common_keys:
        raise ValueError("No common keys found across models.")
    return sorted(common_keys)


def average_state_dicts(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
    strict_keys: bool = False,
) -> Tuple[OrderedDict, List[str]]:
    keys = compare_keys(state_dicts, strict_keys=strict_keys)
    avg_state = OrderedDict()
    skipped = []

    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        ref_tensor = tensors[0]

        same_shape = all(t.shape == ref_tensor.shape for t in tensors)
        if not same_shape:
            skipped.append(key)
            continue

        # floating / complex weights should be averaged; everything else copied
        if torch.is_floating_point(ref_tensor):
            # Accumulate in float32 for numerical stability, then cast back
            acc = torch.zeros_like(ref_tensor, dtype=torch.float32, device="cpu")
            for w, t in zip(weights, tensors):
                acc.add_(t.detach().to(device="cpu", dtype=torch.float32), alpha=w)
            avg_state[key] = acc.to(dtype=ref_tensor.dtype)
        else:
            avg_state[key] = ref_tensor.detach().cpu().clone()

    return avg_state, skipped


def build_empty_model(base_model: str, model_cls, trust_remote_code: bool):
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    model = model_cls.from_config(config, trust_remote_code=trust_remote_code)
    return model


def maybe_copy_tokenizer(base_model: str, output_dir: str, trust_remote_code: bool):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer saved.")
    except Exception as e:
        print(f"Warning: tokenizer was not saved: {e}")


def main() -> None:
    args = parse_args()

    if len(args.models) < 2:
        raise ValueError("Please provide at least 2 models to average.")

    if args.weights is not None:
        if len(args.weights) != len(args.models):
            raise ValueError("Number of weights must match number of models.")
        weights = normalize_weights(args.weights)
    else:
        weights = [1.0 / len(args.models)] * len(args.models)

    model_cls = get_model_cls(args.model_class)
    torch_dtype = parse_torch_dtype(args.torch_dtype)
    base_model = args.base_model or args.models[0]

    print("Loading HF models...")
    state_dicts = []
    for path in args.models:
        model = load_hf_model(
            model_name_or_path=path,
            model_cls=model_cls,
            torch_dtype=torch_dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        sd = model.state_dict()
        state_dicts.append(sd)
        print(f"  Loaded: {path} ({len(sd)} tensors)")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("Averaging parameters...")
    avg_state_dict, skipped = average_state_dicts(
        state_dicts=state_dicts,
        weights=weights,
        strict_keys=args.strict_keys,
    )

    print("Building output model from config...")
    out_model = build_empty_model(
        base_model=base_model,
        model_cls=model_cls,
        trust_remote_code=args.trust_remote_code,
    )

    missing, unexpected = out_model.load_state_dict(avg_state_dict, strict=False)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Saving HF model...")
    out_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,   # saves safetensors
    )

    if args.copy_tokenizer:
        maybe_copy_tokenizer(
            base_model=base_model,
            output_dir=args.output_dir,
            trust_remote_code=args.trust_remote_code,
        )

    print(f"Saved to {args.output_dir}")

    if args.push_to_hub:
        if args.hub_repo is None:
            raise ValueError("--hub_repo required when using --push_to_hub")

        from huggingface_hub import create_repo
        from transformers import AutoTokenizer

        print("Creating repo on HF Hub...")

        create_repo(
            args.hub_repo,
            private=args.hub_private,
            exist_ok=True,
        )

        print("Pushing model to Hub...")

        out_model.push_to_hub(args.hub_repo)

        if args.copy_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.push_to_hub(args.hub_repo)

        print("Push complete.")

    if skipped:
        print(f"Skipped {len(skipped)} keys due to incompatible shapes.")
        for k in skipped[:20]:
            print(f"  - {k}")
        if len(skipped) > 20:
            print("  ...")

    if missing:
        print(f"Warning: {len(missing)} missing keys when loading averaged state_dict.")
        for k in missing[:20]:
            print(f"  missing: {k}")
        if len(missing) > 20:
            print("  ...")

    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys when loading averaged state_dict.")
        for k in unexpected[:20]:
            print(f"  unexpected: {k}")
        if len(unexpected) > 20:
            print("  ...")


if __name__ == "__main__":
    main()