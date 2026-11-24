import argparse
import copy
import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

# Adjust this import if your file name/module path is different.
# It should point to the file that defines Flex_Qwen2_5_VLMoeForConditionalGeneration.
from modeling.flex_qwen2_5_vl_moe import (
    Flex_Qwen2_5_VLMoeConfig,
    Flex_Qwen2_5_VLMoeForConditionalGeneration,
)
from modeling.flex_qwen2_5_vl_moe.configuration_flex_qwen2_5_vl_moe import validate_gate_row_to_expert

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpertSource:
    model_idx: int
    expert_idx: int


@dataclass(frozen=True)
class SparseKeyInfo:
    component: str  # "text" or "vision"
    kind: str       # "router", "expert", or "shared_expert"
    layer_idx: int
    merged_expert_idx: int | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge independently trained Flex_Qwen2_5_VLMoE checkpoints into "
            "one larger VLMoE checkpoint. This version supports both text MoE "
            "under model.language_model.layers.* and vision MoE under model.visual.blocks.*."
        )
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        help="Input VLMoE checkpoint paths/HF IDs.",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=True,
        help="Output path to save merged VLMoE checkpoint.",
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
        help="Model index used for non-expert weights, config defaults, and tokenizer/processor (default: 0).",
    )
    parser.add_argument(
        "--experts-per-model",
        nargs="*",
        default=None,
        help=(
            "Optional per-model expert selections to append after the shared expert. "
            "Length must equal number of models. Each item is comma-separated indices, e.g. "
            "'1' '1' for two 2-expert models. If omitted, defaults to all experts except "
            "--shared-expert-index."
        ),
    )
    parser.add_argument(
        "--as-shared-expert",
        action="store_true",
        help=(
            "Make the --shared-expert-index expert (from --shared-from-model) an always-on shared expert "
            "(mlp.shared_expert.*) instead of routed expert 0. Remaining selected experts become routed "
            "experts mlp.experts.0..R-1, and gate row 0 keeps the shared expert's logit so it is weighted by "
            "its softmax prob over all R+1 logits. Sets use_shared_expert=True and num_experts=R in the target config."
        ),
    )
    parser.add_argument(
        "--many-to-one-public",
        action="store_true",
        help=(
            "Keep every source model's router row for the shared (--shared-expert-index) expert instead of only "
            "--shared-from-model's: all of them route to merged expert 0 (the shared expert's FFN, which must be "
            "bit-identical across sources), and each model's other rows route to its own merged experts. Gate rows "
            "are ordered per model [shared_1, experts_1..., shared_2, experts_2..., ...] and the mapping is written "
            "to config.{text,vision}_config.gate_row_to_expert (see modeling/flex_qwen2_5_vl_moe/GATE_ROW_TO_EXPERT.md). "
            "With --merge-non-ffn-weights, non-FFN tensors are averaged over the source models, once each. "
            "Not compatible with --as-shared-expert."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer/processor source path/HF ID. Defaults to model at --base-model-index.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Loading/saving dtype for checkpoints.",
    )
    parser.add_argument(
        "--strict-shared-check",
        action="store_true",
        help="Fail if shared/non-expert tensors differ across models. Default is warn-only.",
    )
    parser.add_argument(
        "--merge-non-ffn-weights",
        action="store_true",
        help=(
            "Average floating-point shared non-FFN tensors across the selected expert sources "
            "that are being merged. If one source model contributes multiple selected experts, "
            "its shared weights are counted once per selected expert. Dense/shared FFN tensors are still copied from "
            "--base-model-index."
        ),
    )
    parser.add_argument(
        "--non-ffn-seeds",
        nargs="+",
        default=None,
        help=(
            "Optional checkpoint paths/HF IDs to use as the averaging pool for shared non-FFN tensors. "
            "When omitted, shared non-FFN tensors are averaged across the selected expert sources from --models."
        ),
    )
    parser.add_argument(
        "--merge-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge text decoder experts under model.language_model.layers.* (default: true).",
    )
    parser.add_argument(
        "--merge-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge vision tower experts under model.visual.blocks.* (default: true).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained/AutoTokenizer/AutoProcessor.",
    )
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def cfg_value(cfg: Any, field: str) -> Any:
    if isinstance(cfg, dict):
        return cfg[field]
    return getattr(cfg, field)


def is_text_sparse_layer(layer_idx: int, config: Flex_Qwen2_5_VLMoeConfig) -> bool:
    text_cfg = config.text_config
    mlp_only_layers = set(getattr(text_cfg, "mlp_only_layers", []))
    if layer_idx in mlp_only_layers:
        return False
    return int(text_cfg.num_experts) > 0 and (layer_idx + 1) % int(text_cfg.decoder_sparse_step) == 0


def is_vision_sparse_block(block_idx: int, config: Flex_Qwen2_5_VLMoeConfig) -> bool:
    # In the uploaded architecture, every vision block uses SparseMoeBlock when
    # vision_config.num_experts > 0. There is no sparse-step condition for vision.
    return int(config.vision_config.num_experts) > 0


def get_sparse_key_info(
    key: str,
    config: Flex_Qwen2_5_VLMoeConfig,
    merge_text: bool,
    merge_vision: bool,
) -> SparseKeyInfo | None:
    if merge_text and key.startswith("model.language_model.layers."):
        parts = key.split(".")
        # model.language_model.layers.{idx}.mlp...
        layer_idx = int(parts[3])
        if is_text_sparse_layer(layer_idx, config):
            if ".mlp.gate.weight" in key:
                return SparseKeyInfo(component="text", kind="router", layer_idx=layer_idx)
            if ".mlp.shared_expert." in key:
                return SparseKeyInfo(component="text", kind="shared_expert", layer_idx=layer_idx)
            if ".mlp.experts." in key:
                suffix = key.split(".mlp.experts.", 1)[1]
                merged_expert_idx = int(suffix.split(".", 1)[0])
                return SparseKeyInfo(
                    component="text",
                    kind="expert",
                    layer_idx=layer_idx,
                    merged_expert_idx=merged_expert_idx,
                )

    if merge_vision and key.startswith("model.visual.blocks."):
        parts = key.split(".")
        # model.visual.blocks.{idx}.mlp...
        block_idx = int(parts[3])
        if is_vision_sparse_block(block_idx, config):
            if ".mlp.gate.weight" in key:
                return SparseKeyInfo(component="vision", kind="router", layer_idx=block_idx)
            if ".mlp.shared_expert." in key:
                return SparseKeyInfo(component="vision", kind="shared_expert", layer_idx=block_idx)
            if ".mlp.experts." in key:
                suffix = key.split(".mlp.experts.", 1)[1]
                merged_expert_idx = int(suffix.split(".", 1)[0])
                return SparseKeyInfo(
                    component="vision",
                    kind="expert",
                    layer_idx=block_idx,
                    merged_expert_idx=merged_expert_idx,
                )

    return None


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
                    f"Invalid expert index {idx} in --experts-per-model[{model_idx}] "
                    f"(num_experts={num_experts})."
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
        if ref.shape != cur.shape or not torch.equal(ref, cur):
            msg = f"Tensor mismatch for shared key '{key}' between model 0 and model {i}. Using model 0."
            if strict:
                raise ValueError(msg)
            log.warning(msg)


def get_selected_model_indices(merged_sources: list[ExpertSource]) -> list[int]:
    return [src.model_idx for src in merged_sources]


def build_many_to_one_gate_rows(
    merged_sources: list[ExpertSource],
    num_models: int,
    shared_expert_idx: int,
) -> tuple[list[ExpertSource], list[int]]:
    """Gate rows [shared_m, experts_m...] per model m; shared rows route to merged expert 0, others to their own."""
    row_sources, row_to_expert = [], []
    for model_idx in range(num_models):
        row_sources.append(ExpertSource(model_idx=model_idx, expert_idx=shared_expert_idx))
        row_to_expert.append(0)
        for merged_idx, src in enumerate(merged_sources[1:], start=1):
            if src.model_idx == model_idx:
                row_sources.append(src)
                row_to_expert.append(merged_idx)
    return row_sources, row_to_expert


def check_shared_expert_identical(
    source_sds: list[dict[str, torch.Tensor]],
    shared_expert_idx: int,
    reference_model_idx: int,
) -> int:
    """The shared expert is copied once, so its FFN must be bit-identical across all source models."""
    marker = f".mlp.experts.{shared_expert_idx}."
    keys = [k for k in source_sds[reference_model_idx] if marker in k]
    if not keys:
        raise KeyError(f"No '{marker}' tensors in source model {reference_model_idx}.")
    for model_idx, sd in enumerate(source_sds):
        for key in keys:
            if key not in sd or not torch.equal(sd[key], source_sds[reference_model_idx][key]):
                raise ValueError(
                    f"--many-to-one-public: shared expert tensor {key!r} differs between model {model_idx} and "
                    f"model {reference_model_idx}; the shared expert must be identical (frozen) across sources."
                )
    return len(keys)


def is_ffn_key(key: str) -> bool:
    if ".mlp." not in key:
        return False

    return key.startswith("model.language_model.layers.") or key.startswith("model.visual.blocks.")


def merge_shared_weight(
    key: str,
    target_tensor: torch.Tensor,
    source_sds: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    source_tensors: list[torch.Tensor] = []
    for model_idx, sd in enumerate(source_sds):
        src_tensor = sd.get(key)
        if src_tensor is None:
            raise KeyError(f"Missing shared tensor for key {key!r} in source model {model_idx}")
        if src_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"Shared tensor shape mismatch for {key}: source model {model_idx} has {tuple(src_tensor.shape)} "
                f"vs target {tuple(target_tensor.shape)}"
            )
        source_tensors.append(src_tensor)

    ref_tensor = source_tensors[0]
    if not torch.is_floating_point(ref_tensor):
        return ref_tensor.clone().to(dtype=target_tensor.dtype)

    merged_tensor = torch.zeros_like(ref_tensor, dtype=torch.float32)
    for src_tensor in source_tensors:
        merged_tensor.add_(src_tensor.to(dtype=torch.float32))

    merged_tensor.div_(len(source_tensors))
    return merged_tensor.to(dtype=target_tensor.dtype)


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


def load_vlmoe(path: str, dtype: torch.dtype, trust_remote_code: bool):
    register_local_architectures()
    log.info(f"Loading VLMoE model: {path}")
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    cfg = copy.deepcopy(model.config)
    del model
    return sd, cfg


def require_same_attr(cfgs: list[Any], path: str):
    parts = path.split(".")

    def get(root):
        cur = root
        for p in parts:
            cur = cfg_value(cur, p)
        return cur

    ref = get(cfgs[0])
    for i, cfg in enumerate(cfgs[1:], 1):
        cur = get(cfg)
        if cur != ref:
            raise ValueError(f"Incompatible config at model {i}: '{path}' differs ({cur!r} vs base {ref!r}).")


def validate_compatible_configs(
    cfgs: list[Flex_Qwen2_5_VLMoeConfig],
    merge_text: bool,
    merge_vision: bool,
    require_matching_num_experts: bool = True,
):
    # Do not include num_experts here; it is intentionally changed in the target.
    text_fields = [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "decoder_sparse_step",
        "mlp_only_layers",
        "vocab_size",
        "hidden_act",
        "rms_norm_eps",
        "layer_types",
        "num_experts_per_tok",
        "norm_topk_prob",
    ]
    vision_fields = [
        "hidden_size",
        "intermediate_size",
        "depth",
        "num_heads",
        "patch_size",
        "temporal_patch_size",
        "in_channels",
        "out_hidden_size",
        "spatial_merge_size",
        "fullatt_block_indexes",
        "window_size",
        "hidden_act",
        "num_experts_per_tok",
        "norm_topk_prob",
    ]
    top_fields = [
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
    ]

    for field in top_fields:
        if hasattr(cfgs[0], field):
            require_same_attr(cfgs, field)
    if merge_text:
        for field in text_fields:
            if hasattr(cfgs[0].text_config, field):
                require_same_attr(cfgs, f"text_config.{field}")
    if merge_vision:
        for field in vision_fields:
            if hasattr(cfgs[0].vision_config, field):
                require_same_attr(cfgs, f"vision_config.{field}")

    if require_matching_num_experts:
        # Source expert counts must match across source checkpoints for each merged component.
        if merge_text:
            require_same_attr(cfgs, "text_config.num_experts")
        if merge_vision:
            require_same_attr(cfgs, "vision_config.num_experts")


def infer_num_source_experts(
    base_cfg: Flex_Qwen2_5_VLMoeConfig,
    merge_text: bool,
    merge_vision: bool,
) -> int:
    counts: list[tuple[str, int]] = []
    if merge_text:
        counts.append(("text", int(base_cfg.text_config.num_experts)))
    if merge_vision:
        counts.append(("vision", int(base_cfg.vision_config.num_experts)))

    counts = [(name, n) for name, n in counts if n > 0]
    if not counts:
        raise ValueError("No active experts found in selected components. Check --merge-text/--merge-vision and config.*.num_experts.")

    uniq = {n for _, n in counts}
    if len(uniq) != 1:
        details = ", ".join(f"{name}={n}" for name, n in counts)
        raise ValueError(
            "This script uses one expert mapping for all selected components, so their source num_experts "
            f"must match. Got {details}. Run once with --no-merge-text or --no-merge-vision, "
            "or extend the script to use separate mappings."
        )
    return counts[0][1]


def set_target_num_experts(
    cfg: Flex_Qwen2_5_VLMoeConfig,
    merged_num_experts: int,
    merge_text: bool,
    merge_vision: bool,
    use_shared_expert: bool = False,
):
    # merged_num_experts counts routed experts only when use_shared_expert is set.
    if merge_text and int(cfg.text_config.num_experts) > 0:
        cfg.text_config.num_experts = merged_num_experts
        if use_shared_expert:
            cfg.text_config.use_shared_expert = True
            cfg.text_config.shared_expert_intermediate_size = cfg.text_config.intermediate_size
        if int(cfg.text_config.num_experts_per_tok) > merged_num_experts:
            raise ValueError(
                f"text_config.num_experts_per_tok={cfg.text_config.num_experts_per_tok} exceeds merged_num_experts={merged_num_experts}."
            )
    if merge_vision and int(cfg.vision_config.num_experts) > 0:
        cfg.vision_config.num_experts = merged_num_experts
        if use_shared_expert:
            cfg.vision_config.use_shared_expert = True
            cfg.vision_config.shared_expert_intermediate_size = cfg.vision_config.intermediate_size
        if int(cfg.vision_config.num_experts_per_tok) > merged_num_experts:
            raise ValueError(
                f"vision_config.num_experts_per_tok={cfg.vision_config.num_experts_per_tok} exceeds merged_num_experts={merged_num_experts}."
            )
    if hasattr(cfg, "num_experts"):
        # Harmless compatibility for config variants that mirror num_experts at top level.
        cfg.num_experts = merged_num_experts


def copy_router_weight(
    key: str,
    target_tensor: torch.Tensor,
    source_sds: list[dict[str, torch.Tensor]],
    merged_sources: list[ExpertSource],
) -> torch.Tensor:
    gate_rows = []
    for src in merged_sources:
        src_gate = source_sds[src.model_idx].get(key)
        if src_gate is None:
            raise KeyError(f"Missing router tensor for key {key!r} in source model {src.model_idx}")
        gate_rows.append(src_gate[src.expert_idx].clone())

    stacked = torch.stack(gate_rows, dim=0).to(dtype=target_tensor.dtype)
    if stacked.shape != target_tensor.shape:
        raise ValueError(f"Gate shape mismatch at {key}: built {tuple(stacked.shape)} vs target {tuple(target_tensor.shape)}")
    return stacked


def copy_expert_weight(
    key: str,
    target_tensor: torch.Tensor,
    source_sds: list[dict[str, torch.Tensor]],
    merged_sources: list[ExpertSource],
    merged_expert_idx: int,
) -> torch.Tensor:
    src = merged_sources[merged_expert_idx]
    src_key = key.replace(
        f".mlp.experts.{merged_expert_idx}.",
        f".mlp.experts.{src.expert_idx}.",
    )
    src_tensor = source_sds[src.model_idx].get(src_key)
    if src_tensor is None:
        raise KeyError(f"Missing source tensor for {key!r} from model {src.model_idx}, key {src_key!r}")
    if src_tensor.shape != target_tensor.shape:
        raise ValueError(
            f"Expert tensor shape mismatch for {key}: source {src_key} has {tuple(src_tensor.shape)}, "
            f"target has {tuple(target_tensor.shape)}"
        )
    return src_tensor.clone().to(dtype=target_tensor.dtype)


def copy_shared_expert_weight(
    key: str,
    target_tensor: torch.Tensor,
    source_sds: list[dict[str, torch.Tensor]],
    shared_source: ExpertSource,
) -> torch.Tensor:
    src_key = key.replace(".mlp.shared_expert.", f".mlp.experts.{shared_source.expert_idx}.")
    src_tensor = source_sds[shared_source.model_idx].get(src_key)
    if src_tensor is None:
        raise KeyError(f"Missing source tensor for {key!r} from model {shared_source.model_idx}, key {src_key!r}")
    if src_tensor.shape != target_tensor.shape:
        raise ValueError(
            f"Shared expert tensor shape mismatch for {key}: source {src_key} has {tuple(src_tensor.shape)}, "
            f"target has {tuple(target_tensor.shape)}"
        )
    return src_tensor.clone().to(dtype=target_tensor.dtype)


def main():
    args = parse_args()
    if len(args.models) < 1:
        raise ValueError("Need at least one model.")
    if not (args.merge_text or args.merge_vision):
        raise ValueError("At least one of --merge-text or --merge-vision must be enabled.")
    if not (0 <= args.base_model_index < len(args.models)):
        raise ValueError("--base-model-index out of range.")
    if not (0 <= args.shared_from_model < len(args.models)):
        raise ValueError("--shared-from-model out of range.")

    dtype = parse_dtype(args.dtype)

    # 1) Load all source VLMoE checkpoints.
    source_sds: list[dict[str, torch.Tensor]] = []
    source_cfgs: list[Flex_Qwen2_5_VLMoeConfig] = []
    for p in args.models:
        sd, cfg = load_vlmoe(p, dtype=dtype, trust_remote_code=args.trust_remote_code)
        source_sds.append(sd)
        source_cfgs.append(cfg)

    non_ffn_source_sds = source_sds
    non_ffn_source_cfgs = source_cfgs
    if args.non_ffn_seeds is not None:
        non_ffn_source_sds = []
        non_ffn_source_cfgs = []
        for p in args.non_ffn_seeds:
            sd, cfg = load_vlmoe(p, dtype=dtype, trust_remote_code=args.trust_remote_code)
            non_ffn_source_sds.append(sd)
            non_ffn_source_cfgs.append(cfg)

    base_cfg = source_cfgs[args.base_model_index]

    # 2) Validate that non-expert architecture is compatible.
    validate_compatible_configs(source_cfgs, merge_text=args.merge_text, merge_vision=args.merge_vision)
    for model_idx, cfg in enumerate(source_cfgs):
        if getattr(cfg.text_config, "use_shared_expert", False) or getattr(cfg.vision_config, "use_shared_expert", False):
            raise ValueError(f"Source model {model_idx} already uses use_shared_expert=True; merging such checkpoints is not supported.")

    num_source_experts = infer_num_source_experts(base_cfg, merge_text=args.merge_text, merge_vision=args.merge_vision)
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

    # 3) Build merged expert mapping.
    # merged expert 0 always maps to the shared expert from --shared-from-model.
    merged_sources: list[ExpertSource] = [
        ExpertSource(model_idx=args.shared_from_model, expert_idx=args.shared_expert_index)
    ]
    for model_idx, expert_ids in enumerate(selected_per_model):
        for expert_idx in expert_ids:
            merged_sources.append(ExpertSource(model_idx=model_idx, expert_idx=expert_idx))

    # With --as-shared-expert, merged_sources[0] becomes mlp.shared_expert and the rest are routed
    # experts mlp.experts.0..R-1. Gate rows still follow merged_sources (row 0 = shared logit).
    routed_sources = merged_sources[1:] if args.as_shared_expert else merged_sources
    merged_num_experts = len(routed_sources)
    if merged_num_experts < 1:
        raise ValueError("Merged model must have at least one routed expert.")

    selected_model_indices = get_selected_model_indices(merged_sources)

    gate_row_sources = merged_sources
    gate_row_to_expert = None
    if args.many_to_one_public:
        if args.as_shared_expert:
            raise ValueError("--many-to-one-public cannot be combined with --as-shared-expert.")
        n_checked = check_shared_expert_identical(source_sds, args.shared_expert_index, args.shared_from_model)
        log.info("Shared expert %d is identical across all %d source models (%d tensors checked).",
                 args.shared_expert_index, len(source_sds), n_checked)
        gate_row_sources, gate_row_to_expert = build_many_to_one_gate_rows(
            merged_sources, len(args.models), args.shared_expert_index
        )
        # every source model contributes a router row, so average non-FFN tensors over models once each
        selected_model_indices = list(range(len(args.models)))
        log.info("Many-to-one gate rows (row -> src_model:src_expert -> merged expert):")
        for row, (src, expert) in enumerate(zip(gate_row_sources, gate_row_to_expert)):
            log.info(f"  {row} -> {src.model_idx}:{src.expert_idx} -> {expert}")

    log.info("Merged expert mapping (merged_idx -> src_model:src_expert):")
    if args.as_shared_expert:
        log.info(f"  shared -> {merged_sources[0].model_idx}:{merged_sources[0].expert_idx}")
    for merged_idx, src in enumerate(routed_sources):
        log.info(f"  {merged_idx} -> {src.model_idx}:{src.expert_idx}")
    if args.merge_non_ffn_weights:
        if args.non_ffn_seeds is None:
            log.info(
                "Averaging floating-point shared non-FFN tensors across selected expert sources from models: %s",
                selected_model_indices,
            )
        else:
            log.info(
                "Averaging floating-point shared non-FFN tensors across explicit --non-ffn-seeds: %s",
                args.non_ffn_seeds,
            )

    # 4) Create target config/model skeleton.
    target_cfg = copy.deepcopy(base_cfg)
    set_target_num_experts(
        target_cfg,
        merged_num_experts=merged_num_experts,
        merge_text=args.merge_text,
        merge_vision=args.merge_vision,
        use_shared_expert=args.as_shared_expert,
    )
    if gate_row_to_expert is not None:
        for component, enabled in (("text_config", args.merge_text), ("vision_config", args.merge_vision)):
            sub_cfg = getattr(target_cfg, component)
            if enabled and int(sub_cfg.num_experts) > 0:
                sub_cfg.gate_row_to_expert = list(gate_row_to_expert)
                validate_gate_row_to_expert(sub_cfg)

    log.info(
        "Instantiating target model with text_num_experts=%s, vision_num_experts=%s",
        getattr(target_cfg.text_config, "num_experts", None),
        getattr(target_cfg.vision_config, "num_experts", None),
    )
    target_model = Flex_Qwen2_5_VLMoeForConditionalGeneration(target_cfg)
    target_sd = target_model.state_dict()
    base_sd = source_sds[args.base_model_index]

    # 5) Copy weights into merged state dict.
    copied_router = 0
    copied_expert = 0
    copied_shared = 0
    left_initialized: list[str] = []

    for key in list(target_sd.keys()):
        sparse_info = get_sparse_key_info(
            key,
            target_cfg,
            merge_text=args.merge_text,
            merge_vision=args.merge_vision,
        )

        if sparse_info is not None and sparse_info.kind == "router":
            target_sd[key] = copy_router_weight(key, target_sd[key], source_sds, gate_row_sources)
            copied_router += 1
            continue

        if sparse_info is not None and sparse_info.kind == "expert":
            assert sparse_info.merged_expert_idx is not None
            target_sd[key] = copy_expert_weight(
                key,
                target_sd[key],
                source_sds,
                routed_sources,
                sparse_info.merged_expert_idx,
            )
            copied_expert += 1
            continue

        if sparse_info is not None and sparse_info.kind == "shared_expert":
            target_sd[key] = copy_shared_expert_weight(key, target_sd[key], source_sds, merged_sources[0])
            copied_expert += 1
            continue

        # Everything else is non-expert/shared: embeddings, attention, norms, dense MLP,
        # visual patch embed/merger, lm_head, etc. Use base model, optionally check equality.
        if key in base_sd:
            if args.merge_non_ffn_weights and not is_ffn_key(key):
                if args.non_ffn_seeds is None:
                    selected_non_ffn_sds = [source_sds[model_idx] for model_idx in selected_model_indices]
                else:
                    selected_non_ffn_sds = non_ffn_source_sds
                target_sd[key] = merge_shared_weight(key, target_sd[key], selected_non_ffn_sds)
            else:
                _assert_or_warn_equal(key, source_sds, strict=args.strict_shared_check)
                if base_sd[key].shape != target_sd[key].shape:
                    raise ValueError(
                        f"Shared tensor shape mismatch for {key}: base {tuple(base_sd[key].shape)} vs target {tuple(target_sd[key].shape)}"
                    )
                target_sd[key] = base_sd[key].clone().to(dtype=target_sd[key].dtype)
            copied_shared += 1
        else:
            # This should be rare. For correctness, report it so you know something
            # remained randomly initialized in the target skeleton.
            left_initialized.append(key)

    if left_initialized:
        msg = "Some target tensors were left as initialized because they were not found in base_sd: " + ", ".join(left_initialized[:20])
        if len(left_initialized) > 20:
            msg += f", ... ({len(left_initialized)} total)"
        log.warning(msg)

    log.info(
        "Copied %d shared tensors, %d router tensors, and %d expert tensors.",
        copied_shared,
        copied_router,
        copied_expert,
    )

    target_model.load_state_dict(target_sd, strict=True)

    log.info(f"Saving merged model to {args.target}")
    target_model = target_model.to(dtype)
    target_model.save_pretrained(args.target, safe_serialization=True)
    target_cfg.save_pretrained(args.target)

    tokenizer_source = args.tokenizer if args.tokenizer is not None else args.models[args.base_model_index]
    log.info(f"Saving tokenizer from {tokenizer_source} to {args.target}")
    tok = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
    tok.save_pretrained(args.target)

    # Qwen-VL style checkpoints often need the processor/image processor too.
    try:
        log.info(f"Saving processor from {tokenizer_source} to {args.target}")
        processor = AutoProcessor.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
        processor.save_pretrained(args.target)
    except Exception as exc:
        log.warning("Could not save AutoProcessor; tokenizer was saved. Reason: %s", exc)

    log.info("Done.")


if __name__ == "__main__":
    main()
