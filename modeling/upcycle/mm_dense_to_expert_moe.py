import argparse
import logging
import numpy as np
import pickle
import torch
from modeling.flex_qwen2_5_vl_moe import Flex_Qwen2_5_VLMoeForConditionalGeneration, Flex_Qwen2_5_VLMoeConfig
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoProcessor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upcycle dense Qwen2.5 VL HF models into a Flex_Qwen2_5_VLMoe HF model"
    )
    parser.add_argument("-m", "--models", nargs="+", required=True,
                        help="Paths/HF hub IDs for dense Qwen2.5 VL checkpoints (one per expert)")
    parser.add_argument("-t", "--target", type=str, required=True,
                        help="Output path to save the MoE model")
    parser.add_argument("-d", "--text-embeddings", nargs="+", default=[],
                        help="Paths to .npy embedding files for seeding the text router gates "
                             "(one per model, same order as --models). "
                             "Shape: [num_experts, hidden_size] per file, or [hidden_size] for a single expert.")
    # parser.add_argument("--shared-text-gate-embedding", type=str, default=None,
    #                     help="Path to a .npy embedding of shape [hidden_size] used to seed the shared text expert gate. "
    #                          "If not provided but --text-embeddings are, defaults to the mean of those embeddings.")
    parser.add_argument("-v", "--vision-embeddings", nargs="+", default=[],
                        help="Paths to .npy embedding files for seeding the vision router gates "
                             "(one per model, same order as --models). "
                             "Shape: [num_experts, hidden_size] per file, or [hidden_size] for a single expert.")
    parser.add_argument("--embeddings-dict", nargs="+", default=[],
                        help="Paths to pickled dicts mapping FFN layer names to averaged hidden states, "
                             "one file per expert in the same order as --models. This is an alternative to "
                             "--text-embeddings/--vision-embeddings and seeds each MoE router row from the "
                             "matching layer's hidden state.")
    # parser.add_argument("--shared-vision-gate-embedding", type=str, default=None,
    #                     help="Path to a .npy embedding of shape [hidden_size] used to seed the shared vision expert gate. "
    #                          "If not provided but --vision-embeddings are, defaults to the mean of those embeddings.")
    parser.add_argument("--decoder-sparse-step", type=int, default=1)
    parser.add_argument("--mlp-only-layers", nargs="*", type=int, default=[])
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    # parser.add_argument("--normalize-router-gate-and-hidden", action="store_true",
    #                     help="L2-normalize hidden states and router gate rows before routing logits.")
    parser.add_argument("--norm-topk-prob", action="store_true")
    parser.add_argument("--output-router-logits", action="store_true",
                        help="Enable returning router logits during forward pass.")
    parser.add_argument("--merge-non-moe-weights", action="store_true",
                        help="Average all non-MoE/shared weights across input models instead of taking model 0.")
    parser.add_argument("--shared-expert-init", choices=["mean", "first"], default="mean")
    parser.add_argument("--processor", type=str, default=None,
                        help="Path/HF hub ID to load the processor from. "
                             "Defaults to the first model in --models.")
    return parser.parse_args()


def build_moe_config(dense_config: Qwen2_5_VLConfig, num_experts: int, args) -> Flex_Qwen2_5_VLMoeConfig:
    d = dense_config.to_dict()
    dense_vc = d.pop("vision_config")
    dense_tc = d.pop("text_config")

    vision_config = {
        "depth": dense_vc["depth"],
        "hidden_size": dense_vc["hidden_size"],
        "hidden_act": dense_vc.get("hidden_act", "silu"),
        "intermediate_size": dense_vc["intermediate_size"],
        "num_heads": dense_vc["num_heads"],
        "in_channels": dense_vc["in_channels"],
        "patch_size": dense_vc["patch_size"],
        "spatial_merge_size": dense_vc["spatial_merge_size"],
        "temporal_patch_size": dense_vc["temporal_patch_size"],
        "tokens_per_second": dense_vc["tokens_per_second"],
        "window_size": dense_vc["window_size"],
        "out_hidden_size": dense_vc["out_hidden_size"],
        "fullatt_block_indexes": dense_vc["fullatt_block_indexes"],
        "initializer_range": dense_vc["initializer_range"],
        "moe_intermediate_size": dense_vc["intermediate_size"],
        "shared_expert_intermediate_size": dense_vc["intermediate_size"],
        "num_experts_per_tok": args.num_experts_per_tok,
        "num_experts": num_experts,
        "norm_topk_prob": args.norm_topk_prob,
    }

    text_config = {
        "vocab_size": dense_tc["vocab_size"],
        "hidden_size": dense_tc["hidden_size"],
        "intermediate_size": dense_tc["intermediate_size"],
        "num_hidden_layers": dense_tc["num_hidden_layers"],
        "num_attention_heads": dense_tc["num_attention_heads"],
        "num_key_value_heads": dense_tc["num_key_value_heads"],
        "hidden_act": dense_tc["hidden_act"],
        "max_position_embeddings": dense_tc["max_position_embeddings"],
        "initializer_range": dense_tc["initializer_range"],
        "rms_norm_eps": dense_tc["rms_norm_eps"],
        "use_cache": dense_tc["use_cache"],
        "tie_word_embeddings": dense_tc.get("tie_word_embeddings", False),
        "rope_theta": dense_tc["rope_theta"],
        "use_sliding_window": dense_tc["use_sliding_window"],
        "sliding_window": dense_tc["sliding_window"],
        "max_window_layers": dense_tc["max_window_layers"],
        "layer_types": dense_tc["layer_types"],
        "attention_dropout": dense_tc["attention_dropout"],
        "rope_scaling": dense_tc["rope_scaling"],
        "image_token_id": dense_tc.get("image_token_id", None),
        "video_token_id": dense_tc.get("video_token_id", None),
        "decoder_sparse_step": args.decoder_sparse_step,
        "moe_intermediate_size": dense_tc["intermediate_size"],
        "shared_expert_intermediate_size": dense_tc["intermediate_size"],
        "num_experts_per_tok": args.num_experts_per_tok,
        "num_experts": num_experts,
        "norm_topk_prob": args.norm_topk_prob,
        "output_router_logits": args.output_router_logits,
        "router_aux_loss_coef": 0.001,
        "mlp_only_layers": args.mlp_only_layers,
    }


    return Flex_Qwen2_5_VLMoeConfig(
        text_config=text_config,
        vision_config=vision_config,
        **d,
    )


def load_embeddings(embedding_paths: list[str], hidden_size: int) -> torch.Tensor | None:
    """
    Load and concatenate per-expert embeddings into a router weight matrix
    of shape [num_seeded_experts, hidden_size].

    Each .npy file can be:
      - shape [hidden_size]          — single embedding for that expert
      - shape [N, hidden_size]       — will be mean-pooled to [hidden_size]
    """
    if not embedding_paths:
        return None

    rows = []
    for i, path in enumerate(embedding_paths):
        emb = np.load(path).astype(np.float32)
        if emb.ndim == 2:
            log.info(f"Embedding {i} shape {emb.shape}, mean-pooling to [{hidden_size}]")
            emb = emb.mean(axis=0)
        assert emb.shape == (hidden_size,), \
            f"Embedding {i} has unexpected shape {emb.shape}, expected ({hidden_size},)"
        rows.append(torch.from_numpy(emb))

    router_weight = torch.stack(rows, dim=0)  # [num_experts, hidden_size]

    # Log cosine similarities between all pairs so the caller can sanity-check
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            sim = torch.nn.functional.cosine_similarity(rows[i].unsqueeze(0), rows[j].unsqueeze(0)).item()
            log.info(f"Cosine similarity between embedding {i} and {j}: {sim:.4f}")

    return router_weight


def seed_router_weight(
    current_weight: torch.Tensor,
    seeded_weight: torch.Tensor | None,
    router_name: str,
    num_experts: int,
) -> torch.Tensor:
    if seeded_weight is None:
        log.info(f"[router]  {router_name}  — random init")
        return current_weight

    num_seeded = seeded_weight.shape[0]
    if num_seeded > num_experts:
        raise ValueError(
            f"Got {num_seeded} {router_name} embeddings but only {num_experts} experts are available."
        )

    updated_weight = current_weight.clone()
    updated_weight[:num_seeded] = seeded_weight.to(current_weight.dtype)

    if num_seeded < num_experts:
        log.warning(
            f"Only {num_seeded} {router_name} embeddings were provided for {num_experts} experts. "
            f"Seeding experts [0:{num_seeded}] and leaving the remaining {num_experts - num_seeded} experts at random initialization."
        )
    else:
        log.info(f"[router]  {router_name}  — seeded from embeddings")

    return updated_weight


def _normalize_hidden_state_key_to_router_key(layer_name: str) -> str | None:
    if layer_name.endswith(".mlp.gate.weight"):
        router_key = layer_name
    elif layer_name.endswith(".mlp"):
        router_key = layer_name + ".gate.weight"
    else:
        log.warning(
            f"Skipping hidden-state layer key '{layer_name}': expected a key ending in '.mlp' or '.mlp.gate.weight'."
        )
        return None

    prefix_rewrites = (
        ("language_model.layers.", "model.language_model.layers."),
        ("visual.blocks.", "model.visual.blocks."),
        ("model.layers.", "model.language_model.layers."),
        ("blocks.", "model.visual.blocks."),
    )
    for old_prefix, new_prefix in prefix_rewrites:
        if router_key.startswith(old_prefix):
            return new_prefix + router_key[len(old_prefix):]

    if router_key.startswith("model.language_model.layers.") or router_key.startswith("model.visual.blocks."):
        return router_key

    log.warning(
        f"Skipping hidden-state layer key '{layer_name}': could not map it to a MoE router key. "
        "Expected a text key under language_model.layers.* or a vision key under visual.blocks.*."
    )
    return None


def load_embeddings_dicts(
    embedding_dict_paths: list[str],
    text_hidden_size: int,
    vision_hidden_size: int,
) -> list[dict[str, torch.Tensor]]:
    if not embedding_dict_paths:
        return []

    expert_router_dicts = []
    for expert_idx, path in enumerate(embedding_dict_paths):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        if not isinstance(payload, dict):
            raise TypeError(f"Embedding dict {path} must contain a dict, got {type(payload).__name__}.")

        router_dict = {}
        for layer_name, value in payload.items():
            router_key = _normalize_hidden_state_key_to_router_key(layer_name)
            if router_key is None:
                continue

            vector = torch.as_tensor(value, dtype=torch.float32)
            if vector.ndim == 2:
                vector = vector.mean(dim=0)
            elif vector.ndim != 1:
                raise ValueError(
                    f"Embedding dict {path} key '{layer_name}' has shape {tuple(vector.shape)}; expected 1D or 2D."
                )

            if router_key.startswith("model.language_model.layers."):
                expected_hidden_size = text_hidden_size
            elif router_key.startswith("model.visual.blocks."):
                expected_hidden_size = vision_hidden_size
            else:
                raise ValueError(f"Unexpected normalized router key: {router_key}")

            if vector.shape != (expected_hidden_size,):
                raise ValueError(
                    f"Embedding dict {path} key '{layer_name}' normalized to '{router_key}' has shape {tuple(vector.shape)}, "
                    f"expected ({expected_hidden_size},)."
                )

            router_dict[router_key] = vector

        log.info(
            f"Loaded embeddings dict for expert {expert_idx} from {path} with {len(router_dict)} router seeds"
        )
        expert_router_dicts.append(router_dict)

    return expert_router_dicts


def seed_router_weight_from_layer_dicts(
    current_weight: torch.Tensor,
    expert_router_dicts: list[dict[str, torch.Tensor]],
    router_name: str,
    num_experts: int,
) -> torch.Tensor:
    if not expert_router_dicts:
        log.info(f"[router]  {router_name}  — random init")
        return current_weight

    updated_weight = current_weight.clone()
    seeded_experts = []

    for expert_idx, router_dict in enumerate(expert_router_dicts):
        if expert_idx >= num_experts:
            break

        layer_seed = router_dict.get(router_name)
        if layer_seed is None:
            continue

        if layer_seed.shape != (current_weight.shape[1],):
            raise ValueError(
                f"Router seed for expert {expert_idx} key '{router_name}' has shape {tuple(layer_seed.shape)}, "
                f"expected ({current_weight.shape[1]},)."
            )

        updated_weight[expert_idx] = layer_seed.to(current_weight.dtype)
        seeded_experts.append(expert_idx)

    if not seeded_experts:
        log.info(f"[router]  {router_name}  — random init (no matching layer in embeddings dicts)")
        return current_weight

    if len(seeded_experts) < num_experts:
        log.warning(
            f"[router]  {router_name}  — seeded experts {seeded_experts}; "
            f"leaving the remaining {num_experts - len(seeded_experts)} experts at random initialization."
        )
    else:
        log.info(f"[router]  {router_name}  — seeded from per-layer embeddings dicts")

    return updated_weight


# def load_shared_gate_embedding(path: str | None, router_weight: torch.Tensor | None, hidden_size: int) -> torch.Tensor | None:
#     """
#     Load or derive the shared expert gate embedding of shape [1, hidden_size].

#     Priority:
#       1. Explicit --shared-gate-embedding path
#       2. Mean of router embeddings (if those were provided)
#       3. None → random init
#     """
#     if path is not None:
#         emb = np.load(path).astype(np.float32)
#         if emb.ndim == 2:
#             emb = emb.mean(axis=0)
#         assert emb.shape == (hidden_size,), \
#             f"Shared gate embedding has unexpected shape {emb.shape}"
#         log.info(f"Shared expert gate seeded from {path}")
#         return torch.from_numpy(emb).unsqueeze(0)  # [1, hidden_size]

#     if router_weight is not None:
#         mean_emb = router_weight.mean(dim=0, keepdim=True)  # [1, hidden_size]
#         log.info("Shared expert gate seeded from mean of router embeddings")
#         return mean_emb

#     return None


def is_sparse_layer(layer_idx: int, config) -> bool:
    if layer_idx in config.mlp_only_layers:
        return False
    return config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0


def load_dense_model(path: str, dtype=torch.bfloat16):
    log.info(f"  Loading {path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype="auto")
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    config = model.config
    del model
    return sd, config


def _get_shared_weight(key: str, state_dicts: list[dict[str, torch.Tensor]], merge_weights: bool) -> torch.Tensor | None:
    tensors = [sd[key] for sd in state_dicts if key in sd]
    if not tensors:
        return None

    if not merge_weights:
        ref = tensors[0]
        for i, tensor in enumerate(tensors[1:], 1):
            if not torch.equal(ref, tensor):
                log.warning(
                    f"Shared key '{key}' differs between expert 0 and expert {i}. "
                    "Using expert 0's value."
                )
                break
        return ref.clone()

    ref_shape = tensors[0].shape
    ref_dtype = tensors[0].dtype
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.shape != ref_shape:
            raise ValueError(
                f"Cannot average shared key '{key}': model 0 has shape {ref_shape}, "
                f"but model {i} has shape {tensor.shape}."
            )

    avg = torch.stack([tensor.float() for tensor in tensors], dim=0).mean(dim=0)
    return avg.to(ref_dtype)


def main():
    args = parse_args()
    num_experts = len(args.models)
    assert num_experts >= 1

    if args.text_embeddings:
        assert len(args.text_embeddings) <= num_experts, \
            f"Got {len(args.text_embeddings)} text embeddings but only {num_experts} models are available."

    if args.vision_embeddings:
        assert len(args.vision_embeddings) <= num_experts, \
            f"Got {len(args.vision_embeddings)} vision embeddings but only {num_experts} models are available."

    if args.embeddings_dict:
        assert len(args.embeddings_dict) <= num_experts, \
            f"Got {len(args.embeddings_dict)} embeddings dicts but only {num_experts} models are available."

    if args.embeddings_dict and (args.text_embeddings or args.vision_embeddings):
        raise ValueError(
            "--embeddings-dict is an alternative to --text-embeddings/--vision-embeddings; do not provide both."
        )

    # ------------------------------------------------------------------
    # 1. Load dense models
    # ------------------------------------------------------------------
    dense_sds = []
    dense_config = None
    for i, path in enumerate(args.models):
        log.info(f"Loading dense model {i}/{num_experts}: {path}")
        sd, cfg = load_dense_model(path)
        if dense_config is None:
            dense_config = cfg
        else:
            assert cfg.hidden_size == dense_config.hidden_size
            assert cfg.num_hidden_layers == dense_config.num_hidden_layers
        dense_sds.append(sd)

    # ------------------------------------------------------------------
    # 2. Build MoE config + skeleton
    # ------------------------------------------------------------------
    moe_config = build_moe_config(dense_config, num_experts, args)
    log.info(f"MoE config:\n{moe_config}")

    log.info("Instantiating Flex_Qwen2_5_VLMoe skeleton on CPU ...")
    moe_model = Flex_Qwen2_5_VLMoeForConditionalGeneration(moe_config)
    moe_sd = moe_model.state_dict()
    print(moe_sd.keys())

    # ------------------------------------------------------------------
    # 3. Load embeddings for gate initialization
    # ------------------------------------------------------------------
    text_router_weight = load_embeddings(args.text_embeddings, dense_config.text_config.hidden_size)
    vision_router_weight = load_embeddings(args.vision_embeddings, dense_config.vision_config.hidden_size)
    layerwise_router_seeds = load_embeddings_dicts(
        args.embeddings_dict,
        text_hidden_size=dense_config.text_config.hidden_size,
        vision_hidden_size=dense_config.vision_config.hidden_size,
    )
    # shared_gate_weight = load_shared_gate_embedding(
    #     args.shared_gate_embedding, router_weight, dense_config.hidden_size
    # )

    # ------------------------------------------------------------------
    # 4. Copy weights
    # ------------------------------------------------------------------
    for key in list(moe_sd.keys()):
        # TODO: not sure if this keying is generalizable

        # Global shared weights
        if not key.startswith("model.visual.blocks.") and not key.startswith("model.language_model.layers."):
            shared_weight = _get_shared_weight(key, dense_sds, args.merge_non_moe_weights)
            if shared_weight is not None:
                moe_sd[key] = shared_weight
            else:
                log.warning(f"Key not found in dense model: {key}")
            continue

        # Vision weights
        if key.startswith("model.visual.blocks."):
            parts = key.split(".")

            # Attention + norms — always shared
            if ".attn." in key or "norm" in key:
                shared_weight = _get_shared_weight(key, dense_sds, args.merge_non_moe_weights)
                if shared_weight is not None:
                    moe_sd[key] = shared_weight
                else:
                    log.warning(f"Shared layer key not found: {key}")
                continue

            # Sparse MoE layers, TODO: all blocks in vision encoder are sparse currently
            if ".mlp." in key:
                block_prefix = ".".join(key.split(".")[:4]) # "model.visual.blocks.N"
                if ".mlp.gate.weight" in key:
                    if layerwise_router_seeds:
                        moe_sd[key] = seed_router_weight_from_layer_dicts(
                            moe_sd[key], layerwise_router_seeds, key, num_experts
                        )
                    else:
                        moe_sd[key] = seed_router_weight(moe_sd[key], vision_router_weight, key, num_experts)
                    continue

                # Routed expert FFN
                if ".mlp.experts." in key:
                    after_experts = key.split(".mlp.experts.")[1]
                    expert_idx = int(after_experts.split(".")[0])
                    sub_key = ".".join(after_experts.split(".")[1:])
                    dense_key = f"{block_prefix}.mlp.{sub_key}"

                    if expert_idx < num_experts and dense_key in dense_sds[expert_idx]:
                        moe_sd[key] = dense_sds[expert_idx][dense_key].clone()
                        log.debug(f"[expert-{expert_idx}]  {dense_key} -> {key}")
                    else:
                        log.warning(f"Expert key not found: {dense_key} (expert {expert_idx})")
                    continue

        # Text weights
        if key.startswith("model.language_model.layers."):
            parts = key.split(".")
            layer_idx = int(parts[3])
            sparse = is_sparse_layer(layer_idx, moe_config.text_config)

            # Attention + norms — always shared
            if ".self_attn." in key or "layernorm" in key:
                shared_weight = _get_shared_weight(key, dense_sds, args.merge_non_moe_weights)
                if shared_weight is not None:
                    moe_sd[key] = shared_weight
                else:
                    log.warning(f"Shared layer key not found: {key}")
                continue

            # Dense MLP layers
            if ".mlp." in key and not sparse:
                shared_weight = _get_shared_weight(key, dense_sds, args.merge_non_moe_weights)
                if shared_weight is not None:
                    moe_sd[key] = shared_weight
                else:
                    log.warning(f"Dense MLP key not found: {key}")
                continue

            # Sparse MoE layers
            if ".mlp." in key and sparse:
                layer_prefix = ".".join(key.split(".")[:4])  # "model.language_model.layers.N"

                # Router gate — seed with embeddings if provided
                if ".mlp.gate.weight" in key:
                    if layerwise_router_seeds:
                        moe_sd[key] = seed_router_weight_from_layer_dicts(
                            moe_sd[key], layerwise_router_seeds, key, num_experts
                        )
                    else:
                        moe_sd[key] = seed_router_weight(moe_sd[key], text_router_weight, key, num_experts)
                    continue

                # # Shared expert gate — seed with embedding if provided
                # if ".mlp.shared_expert_gate.weight" in key:
                #     raise KeyError("Shared expert gate is disabled in this version. Remove this check if you want to enable it.")
                    
                #     if shared_gate_weight is not None:
                #         target_dtype = moe_sd[key].dtype
                #         moe_sd[key] = shared_gate_weight.to(target_dtype)
                #         log.info(f"[shared-expert-gate]  {key}  — seeded from embedding")
                #     else:
                #         log.info(f"[shared-expert-gate]  {key}  — random init")
                #     continue

                # # Shared expert FFN
                # if ".mlp.shared_expert." in key:
                #     raise KeyError("Shared expert FFN is disabled in this version. Remove this check if you want to enable it.")
                    
                #     dense_key = key.replace(".mlp.shared_expert.", ".mlp.")
                #     if dense_key in dense_sds[0]:
                #         if args.shared_expert_init == "mean":
                #             avg = torch.stack([sd[dense_key].float() for sd in dense_sds]).mean(0)
                #             moe_sd[key] = avg.to(dense_sds[0][dense_key].dtype)
                #         else:
                #             moe_sd[key] = dense_sds[0][dense_key].clone()
                #         log.debug(f"[shared-expert]  {dense_key} -> {key}")
                #     else:
                #         log.warning(f"Shared expert source key not found: {dense_key}")
                #     continue

                # Routed expert FFN
                if ".mlp.experts." in key:
                    after_experts = key.split(".mlp.experts.")[1]
                    expert_idx = int(after_experts.split(".")[0])
                    sub_key = ".".join(after_experts.split(".")[1:])
                    dense_key = f"{layer_prefix}.mlp.{sub_key}"

                    if expert_idx < num_experts and dense_key in dense_sds[expert_idx]:
                        moe_sd[key] = dense_sds[expert_idx][dense_key].clone()
                        log.debug(f"[expert-{expert_idx}]  {dense_key} -> {key}")
                    else:
                        log.warning(f"Expert key not found: {dense_key} (expert {expert_idx})")
                    continue

        log.warning(f"Unhandled key: {key}")

    # ------------------------------------------------------------------
    # 5. Save model
    # ------------------------------------------------------------------
    missing, unexpected = moe_model.load_state_dict(moe_sd, strict=True)
    if missing:
        log.warning(f"Missing keys: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected}")

    log.info(f"Saving to {args.target} ...")
    moe_model = moe_model.to(torch.bfloat16)
    moe_model.save_pretrained(args.target, safe_serialization=True)
    moe_config.save_pretrained(args.target)

    # ------------------------------------------------------------------
    # 6. Save processor
    # ------------------------------------------------------------------
    processor_source = args.processor if args.processor is not None else args.models[0]
    log.info(f"Loading processor from {processor_source} ...")
    processor = AutoProcessor.from_pretrained(processor_source)
    processor.save_pretrained(args.target)
    log.info(f"Processor saved to {args.target}")

    log.info("Done.")


if __name__ == "__main__":
    main()
