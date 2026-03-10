import argparse
import logging
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from modeling.flex_qwen2_moe import FlexQwen2MoeConfig, FlexQwen2MoeForCausalLM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpertSource:
    model_idx: int
    expert_idx: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge independently trained Flex Qwen2 MoE checkpoints into a larger MoE checkpoint."
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        help="Input MoE checkpoint paths/HF IDs.",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=True,
        help="Output path to save merged MoE checkpoint.",
    )
    parser.add_argument(
        "--shared-expert-index",
        type=int,
        default=0,
        help="Expert index treated as shared across all models (default: 0).",
    )
    parser.add_argument(
        "--shared-from-model",
        type=int,
        default=0,
        help="Model index used as source for merged expert 0 (default: 0).",
    )
    parser.add_argument(
        "--base-model-index",
        type=int,
        default=0,
        help="Model index used for non-expert weights and tokenizer (default: 0).",
    )
    parser.add_argument(
        "--experts-per-model",
        nargs="*",
        default=None,
        help=(
            "Optional per-model expert selections to append after shared expert. "
            "Length must equal number of models. Each item is comma-separated indices, e.g. "
            "'1' '1' for two 2-expert models. If omitted, defaults to all experts except --shared-expert-index."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer source path/HF ID. Defaults to model at --base-model-index.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Loading dtype for source checkpoints.",
    )
    parser.add_argument(
        "--strict-shared-check",
        action="store_true",
        help="Fail if shared/non-expert tensors differ across models. Default is warn-only.",
    )
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def is_sparse_layer(layer_idx: int, config: FlexQwen2MoeConfig) -> bool:
    if layer_idx in config.mlp_only_layers:
        return False
    return config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0


def parse_experts_per_model(
    specs: list[str] | None,
    num_models: int,
    num_experts: int,
    shared_expert_idx: int,
) -> list[list[int]]:
    if specs is None:
        defaults = [i for i in range(num_experts) if i != shared_expert_idx]
        return [defaults[:] for _ in range(num_models)]

    if len(specs) != num_models:
        raise ValueError(
            f"--experts-per-model length ({len(specs)}) must equal number of models ({num_models})."
        )

    parsed: list[list[int]] = []
    for model_idx, raw in enumerate(specs):
        raw = raw.strip()
        if raw == "":
            parsed.append([])
            continue

        indices = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate expert index in --experts-per-model[{model_idx}]={raw!r}")

        for idx in indices:
            if idx < 0 or idx >= num_experts:
                raise ValueError(
                    f"Invalid expert index {idx} in --experts-per-model[{model_idx}] (num_experts={num_experts})."
                )
            if idx == shared_expert_idx:
                raise ValueError(
                    f"--experts-per-model[{model_idx}] includes shared expert index {shared_expert_idx}; "
                    "shared expert is always mapped to merged expert 0 only."
                )

        parsed.append(indices)

    return parsed


def _assert_or_warn_equal(
    key: str,
    state_dicts: list[dict[str, torch.Tensor]],
    strict: bool,
):
    ref = state_dicts[0].get(key)
    if ref is None:
        return

    for i, sd in enumerate(state_dicts[1:], 1):
        cur = sd.get(key)
        if cur is None:
            continue
        if not torch.equal(ref, cur):
            msg = f"Tensor mismatch for shared key '{key}' between model 0 and model {i}. Using model 0."
            if strict:
                raise ValueError(msg)
            log.warning(msg)


def load_moe(path: str, dtype: torch.dtype):
    log.info(f"Loading MoE model: {path}")
    model = FlexQwen2MoeForCausalLM.from_pretrained(path, torch_dtype=dtype)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = model.config
    del model
    return sd, cfg


def main():
    args = parse_args()
    if len(args.models) < 1:
        raise ValueError("Need at least one model.")

    if not (0 <= args.base_model_index < len(args.models)):
        raise ValueError("--base-model-index out of range.")
    if not (0 <= args.shared_from_model < len(args.models)):
        raise ValueError("--shared-from-model out of range.")

    dtype = parse_dtype(args.dtype)

    # 1) Load all source MoE checkpoints
    source_sds: list[dict[str, torch.Tensor]] = []
    source_cfgs: list[FlexQwen2MoeConfig] = []
    for p in args.models:
        sd, cfg = load_moe(p, dtype=dtype)
        source_sds.append(sd)
        source_cfgs.append(cfg)

    base_cfg = source_cfgs[args.base_model_index]

    # 2) Validate architecture compatibility
    fields_to_match = [
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "decoder_sparse_step",
        "mlp_only_layers",
        "vocab_size",
    ]
    for i, cfg in enumerate(source_cfgs):
        for field in fields_to_match:
            if getattr(cfg, field) != getattr(base_cfg, field):
                raise ValueError(
                    f"Incompatible config at model {i}: field '{field}' differs "
                    f"({getattr(cfg, field)} vs base {getattr(base_cfg, field)})."
                )

    num_source_experts = int(base_cfg.num_experts)
    if not (0 <= args.shared_expert_index < num_source_experts):
        raise ValueError(
            f"--shared-expert-index must be in [0, {num_source_experts - 1}], got {args.shared_expert_index}."
        )

    selected_per_model = parse_experts_per_model(
        specs=args.experts_per_model,
        num_models=len(args.models),
        num_experts=num_source_experts,
        shared_expert_idx=args.shared_expert_index,
    )

    # 3) Build merged expert mapping
    # merged expert 0 always maps to the shared expert from --shared-from-model
    merged_sources: list[ExpertSource] = [
        ExpertSource(model_idx=args.shared_from_model, expert_idx=args.shared_expert_index)
    ]

    for model_idx, expert_ids in enumerate(selected_per_model):
        for expert_idx in expert_ids:
            merged_sources.append(ExpertSource(model_idx=model_idx, expert_idx=expert_idx))

    merged_num_experts = len(merged_sources)
    if merged_num_experts < 1:
        raise ValueError("Merged model must have at least one expert.")

    log.info("Merged expert mapping (merged_idx -> src_model:src_expert):")
    for merged_idx, src in enumerate(merged_sources):
        log.info(f"  {merged_idx} -> {src.model_idx}:{src.expert_idx}")

    # 4) Create target config/model skeleton
    target_cfg_dict = base_cfg.to_dict()
    target_cfg_dict["num_experts"] = merged_num_experts
    target_cfg = FlexQwen2MoeConfig(**target_cfg_dict)

    log.info(f"Instantiating target model with num_experts={merged_num_experts}")
    target_model = FlexQwen2MoeForCausalLM(target_cfg)
    target_sd = target_model.state_dict()

    base_sd = source_sds[args.base_model_index]

    # 5) Copy weights into merged state dict
    for key in list(target_sd.keys()):
        if not key.startswith("model.layers."):
            if key in base_sd:
                _assert_or_warn_equal(key, source_sds, strict=args.strict_shared_check)
                target_sd[key] = base_sd[key].clone()
            continue

        parts = key.split(".")
        layer_idx = int(parts[2])
        sparse = is_sparse_layer(layer_idx, target_cfg)

        # Attention and norms are shared
        if ".self_attn." in key or "layernorm" in key:
            if key in base_sd:
                _assert_or_warn_equal(key, source_sds, strict=args.strict_shared_check)
                target_sd[key] = base_sd[key].clone()
            continue

        # Dense MLP layers are shared
        if ".mlp." in key and not sparse:
            if key in base_sd:
                _assert_or_warn_equal(key, source_sds, strict=args.strict_shared_check)
                target_sd[key] = base_sd[key].clone()
            continue

        # Sparse layer router gate
        if ".mlp.gate.weight" in key and sparse:
            gate_rows = []
            for src in merged_sources:
                src_gate = source_sds[src.model_idx][key]
                gate_rows.append(src_gate[src.expert_idx].clone())
            stacked = torch.stack(gate_rows, dim=0).to(dtype=target_sd[key].dtype)
            if stacked.shape != target_sd[key].shape:
                raise ValueError(
                    f"Gate shape mismatch at {key}: built {tuple(stacked.shape)} vs target {tuple(target_sd[key].shape)}"
                )
            target_sd[key] = stacked
            continue

        # Sparse layer expert FFNs
        if ".mlp.experts." in key and sparse:
            suffix = key.split(".mlp.experts.", 1)[1]
            merged_expert_idx = int(suffix.split(".", 1)[0])

            src = merged_sources[merged_expert_idx]
            src_key = key.replace(
                f".mlp.experts.{merged_expert_idx}.",
                f".mlp.experts.{src.expert_idx}.",
            )
            src_tensor = source_sds[src.model_idx].get(src_key)
            if src_tensor is None:
                raise KeyError(
                    f"Missing source tensor for {key} from model {src.model_idx} key {src_key}"
                )
            target_sd[key] = src_tensor.clone()
            continue

        # Unhandled keys remain as initialized params

    missing, unexpected = target_model.load_state_dict(target_sd, strict=True)
    if missing:
        log.warning(f"Missing keys: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected}")

    log.info(f"Saving merged model to {args.target}")
    target_model = target_model.to(dtype)
    target_model.save_pretrained(args.target, safe_serialization=True)
    target_cfg.save_pretrained(args.target)

    tokenizer_source = args.tokenizer if args.tokenizer is not None else args.models[args.base_model_index]
    log.info(f"Saving tokenizer from {tokenizer_source} to {args.target}")
    tok = AutoTokenizer.from_pretrained(tokenizer_source)
    tok.save_pretrained(args.target)

    log.info("Done.")


if __name__ == "__main__":
    main()
