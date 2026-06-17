#!/usr/bin/env python3
"""
Model parameter averaging ("model soup") for the custom Flex-Qwen2.5-VL MoE model.

Expected repo layout
--------------------
Run this from the repository root, or put this file under modeling/upcycle/ and run
it either as a module or as a script:

    python -m modeling.upcycle.vlmoe_model_soup \
        --models /path/to/ckpt1 /path/to/ckpt2 \
        --output_dir /path/to/averaged_vlmoe \
        --torch_dtype bfloat16

or:

    python modeling/upcycle/vlmoe_model_soup.py \
        --models /path/to/ckpt1 /path/to/ckpt2 \
        --output_dir /path/to/averaged_vlmoe \
        --torch_dtype bfloat16

The script assumes your custom model code is importable as:

    modeling.flex_qwen2_5_vl_moe.configuration_flex_qwen2_5_vl_moe
    modeling.flex_qwen2_5_vl_moe.modeling_flex_qwen2_5_vl_moe

Notes
-----
- All input checkpoints must have the same architecture and tensor shapes.
- Floating-point tensors are averaged.
- Non-floating tensors are copied from the first model.
- The output model is saved with save_pretrained(..., safe_serialization=True).
- By default, the processor is copied from --base_model or the first input model.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer

# -----------------------------------------------------------------------------
# Import custom VLMoE classes.
#
# When this file lives at modeling/upcycle/vlmoe_model_soup.py, running it
# directly makes Python put modeling/upcycle/ on sys.path, not the repo root.
# Add the repo root so absolute imports like `modeling.flex_qwen2_5_vl_moe...`
# work in both direct-script and `python -m ...` execution modes.
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
if len(THIS_FILE.parents) >= 3:
    REPO_ROOT = THIS_FILE.parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

try:
    from modeling.flex_qwen2_5_vl_moe.configuration_flex_qwen2_5_vl_moe import (
        Flex_Qwen2_5_VLMoeConfig,
    )
    from modeling.flex_qwen2_5_vl_moe.modeling_flex_qwen2_5_vl_moe import (
        Flex_Qwen2_5_VLMoeForConditionalGeneration,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import the custom Flex-Qwen2.5-VL MoE classes.\n"
        "Expected imports:\n"
        "  from modeling.flex_qwen2_5_vl_moe.configuration_flex_qwen2_5_vl_moe "
        "import Flex_Qwen2_5_VLMoeConfig\n"
        "  from modeling.flex_qwen2_5_vl_moe.modeling_flex_qwen2_5_vl_moe "
        "import Flex_Qwen2_5_VLMoeForConditionalGeneration\n\n"
        "Run from the repo root, for example:\n"
        "  python -m modeling.upcycle.vlmoe_model_soup ...\n"
        "or make sure the repo root is on PYTHONPATH."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average custom Flex-Qwen2.5-VL MoE Hugging Face checkpoints."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Local HF checkpoint directories or Hub model IDs to average.",
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
            "Optional config/processor source used to instantiate and save the final model. "
            "If omitted, the first entry in --models is used."
        ),
    )
    parser.add_argument(
        "--strict_keys",
        action="store_true",
        help="Require all models to have exactly the same state_dict keys.",
    )
    parser.add_argument(
        "--strict_shapes",
        action="store_true",
        help="Fail on any common key whose tensor shape differs across models.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help=(
            "Dtype used when loading source models. Use bfloat16/float16 to save memory. "
            "If auto, Transformers chooses the dtype from the checkpoint/config."
        ),
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="same",
        choices=["same", "float32", "float16", "bfloat16"],
        help=(
            "Dtype of the saved floating-point parameters. "
            "'same' uses the dtype of the averaged state_dict tensors."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load each model onto before copying tensors to CPU. Usually cpu.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Passed to from_pretrained()/AutoProcessor when loading checkpoints/processors.",
    )
    parser.add_argument(
        "--skip_processor",
        action="store_true",
        help="Do not copy AutoProcessor/AutoTokenizer into output_dir.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push averaged model to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_repo",
        type=str,
        default=None,
        help="Repo name like username/model-name.",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Create private repo on HF Hub.",
    )
    return parser.parse_args()


def normalize_weights(weights: List[float]) -> List[float]:
    total = sum(weights)
    if total == 0:
        raise ValueError("Sum of weights cannot be zero.")
    return [w / total for w in weights]


def parse_torch_dtype(dtype_str: str):
    if dtype_str in {"auto", "same"}:
        return dtype_str
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def load_vlmoe_state_dict_cpu(
    model_name_or_path: str,
    torch_dtype,
    device: str,
    trust_remote_code: bool,
) -> OrderedDict[str, torch.Tensor]:
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype

    model = Flex_Qwen2_5_VLMoeForConditionalGeneration.from_pretrained(
        model_name_or_path,
        **kwargs,
    )
    if device != "cpu":
        model = model.to(device)
    model.eval()

    # Clone to CPU so deleting the model really frees GPU/model storage.
    sd_cpu: OrderedDict[str, torch.Tensor] = OrderedDict()
    with torch.no_grad():
        for key, tensor in model.state_dict().items():
            sd_cpu[key] = tensor.detach().cpu().clone()

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return sd_cpu


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
    strict_shapes: bool = False,
) -> Tuple[OrderedDict[str, torch.Tensor], List[str]]:
    keys = compare_keys(state_dicts, strict_keys=strict_keys)
    avg_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    skipped: List[str] = []

    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        ref_tensor = tensors[0]

        same_shape = all(t.shape == ref_tensor.shape for t in tensors)
        if not same_shape:
            msg = f"Shape mismatch for key {key}: {[tuple(t.shape) for t in tensors]}"
            if strict_shapes:
                raise ValueError(msg)
            skipped.append(key)
            continue

        if torch.is_floating_point(ref_tensor):
            # Accumulate in float32 for numerical stability, then cast back to
            # the first checkpoint's dtype unless --output_dtype changes it later.
            acc = torch.zeros_like(ref_tensor, dtype=torch.float32, device="cpu")
            for w, t in zip(weights, tensors):
                acc.add_(t.detach().to(device="cpu", dtype=torch.float32), alpha=w)
            avg_state[key] = acc.to(dtype=ref_tensor.dtype)
        else:
            avg_state[key] = ref_tensor.detach().cpu().clone()

    return avg_state, skipped


def infer_first_floating_dtype(state_dict: Dict[str, torch.Tensor]) -> torch.dtype:
    for tensor in state_dict.values():
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return torch.float32


def cast_floating_state_dict_(state_dict: OrderedDict[str, torch.Tensor], dtype: torch.dtype) -> None:
    for key, tensor in list(state_dict.items()):
        if torch.is_floating_point(tensor) and tensor.dtype != dtype:
            state_dict[key] = tensor.to(dtype=dtype)


def build_empty_vlmoe_model(base_model: str, trust_remote_code: bool):
    config = Flex_Qwen2_5_VLMoeConfig.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
    )
    return Flex_Qwen2_5_VLMoeForConditionalGeneration(config)


def maybe_copy_processor(base_model: str, output_dir: str, trust_remote_code: bool) -> None:
    """Copy VL processor if available; fall back to tokenizer for text-only assets."""
    try:
        processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
        )
        processor.save_pretrained(output_dir)
        print("Processor saved.")
        return
    except Exception as e:
        print(f"Warning: processor was not saved via AutoProcessor: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer saved.")
    except Exception as e:
        print(f"Warning: tokenizer was not saved either: {e}")


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

    torch_dtype = parse_torch_dtype(args.torch_dtype)
    output_dtype_arg = parse_torch_dtype(args.output_dtype)
    base_model = args.base_model or args.models[0]

    print("Loading custom VLMoE model state_dicts...")
    state_dicts: List[OrderedDict[str, torch.Tensor]] = []
    for path in args.models:
        sd = load_vlmoe_state_dict_cpu(
            model_name_or_path=path,
            torch_dtype=torch_dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        state_dicts.append(sd)
        print(f"  Loaded: {path} ({len(sd)} tensors)")

    print("Averaging parameters...")
    avg_state_dict, skipped = average_state_dicts(
        state_dicts=state_dicts,
        weights=weights,
        strict_keys=args.strict_keys,
        strict_shapes=args.strict_shapes,
    )

    # Free source state dicts before instantiating the output model.
    del state_dicts
    gc.collect()

    if output_dtype_arg == "same":
        save_dtype = infer_first_floating_dtype(avg_state_dict)
    else:
        save_dtype = output_dtype_arg
        cast_floating_state_dict_(avg_state_dict, save_dtype)

    print(f"Building output VLMoE model from config: {base_model}")
    out_model = build_empty_vlmoe_model(
        base_model=base_model,
        trust_remote_code=args.trust_remote_code,
    )
    out_model = out_model.to(dtype=save_dtype)

    missing, unexpected = out_model.load_state_dict(avg_state_dict, strict=False)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Saving averaged VLMoE model to {args.output_dir} with dtype={save_dtype}...")
    out_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
    )

    if not args.skip_processor:
        maybe_copy_processor(
            base_model=base_model,
            output_dir=args.output_dir,
            trust_remote_code=args.trust_remote_code,
        )

    print(f"Saved to {args.output_dir}")

    if args.push_to_hub:
        if args.hub_repo is None:
            raise ValueError("--hub_repo required when using --push_to_hub")

        from huggingface_hub import create_repo

        print("Creating repo on HF Hub...")
        create_repo(
            args.hub_repo,
            private=args.hub_private,
            exist_ok=True,
        )

        print("Pushing model to Hub...")
        out_model.push_to_hub(args.hub_repo)

        if not args.skip_processor:
            try:
                processor = AutoProcessor.from_pretrained(args.output_dir, trust_remote_code=args.trust_remote_code)
                processor.push_to_hub(args.hub_repo)
            except Exception as e:
                print(f"Warning: processor push failed: {e}")

        print("Push complete.")

    if skipped:
        print(f"Skipped {len(skipped)} keys due to incompatible shapes.")
        for key in skipped[:20]:
            print(f"  - {key}")
        if len(skipped) > 20:
            print("  ...")

    if missing:
        print(f"Warning: {len(missing)} missing keys when loading averaged state_dict.")
        for key in missing[:20]:
            print(f"  missing: {key}")
        if len(missing) > 20:
            print("  ...")

    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys when loading averaged state_dict.")
        for key in unexpected[:20]:
            print(f"  unexpected: {key}")
        if len(unexpected) > 20:
            print("  ...")


if __name__ == "__main__":
    main()
