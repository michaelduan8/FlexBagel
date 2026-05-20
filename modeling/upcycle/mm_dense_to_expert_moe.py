import argparse
import logging
import numpy as np
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
    of shape [num_experts, hidden_size].

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


def is_sparse_layer(layer_idx: int, config: Flex_Qwen2_5_VLMoeConfig) -> bool:
    if layer_idx in config.text_config.mlp_only_layers:
        return False
    return config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0


def load_dense_model(path: str, dtype=torch.bfloat16):
    log.info(f"  Loading {path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype="auto")
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    config = model.config
    del model
    return sd, config


# def _assert_equal_across_experts(key: str, state_dicts: list):
#     ref = state_dicts[0].get(key)
#     if ref is None:
#         return
#     for i, sd in enumerate(state_dicts[1:], 1):
#         if key in sd and not torch.equal(ref, sd[key]):
#             log.warning(
#                 f"Shared key '{key}' differs between expert 0 and expert {i}. "
#                 f"Using expert 0's value."
#             )


def main():
    args = parse_args()
    num_experts = len(args.models)
    assert num_experts >= 1

    if args.embeddings:
        assert len(args.embeddings) == num_experts, \
            f"Got {len(args.embeddings)} embeddings but {num_experts} models — must match."

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
    # shared_gate_weight = load_shared_gate_embedding(
    #     args.shared_gate_embedding, router_weight, dense_config.hidden_size
    # )

    # ------------------------------------------------------------------
    # 4. Copy weights
    # ------------------------------------------------------------------
    for key in list(moe_sd.keys()):
        # TODO: not sure if this keying is generalizable

        # Global shared weights
        if not key.startswith("model.visual.blocks.") or not key.startswith("model.language_model.layers."):
            if key in dense_sds[0]:
                _assert_equal_across_experts(key, dense_sds)
                moe_sd[key] = dense_sds[0][key].clone()
            else:
                log.warning(f"Key not found in dense model: {key}")
            continue

        parts = key.split(".")
        layer_idx = int(parts[2])
        sparse = is_sparse_layer(layer_idx, moe_config)

        # Vision weights
        if key.startswith("model.visual.blocks."):
            parts = key.split(".")
            block_idx = int(parts[3])

            # Attention + norms — always shared
            if ".attn." in key or "norm" in key:
                if key in dense_sds[0]:
                    _assert_equal_across_experts(key, dense_sds)
                    moe_sd[key] = dense_sds[0][key].clone()
                else:
                    log.warning(f"Shared layer key not found: {key}")
                continue

            # Sparse MoE layers, TODO: all blocks in vision encoder are sparse currently
            if ".mlp." in key:
                block_prefix = ".".join(key.split(".")[:4]) # "model.visual.blocks.N"
                if ".mlp.gate.weight" in key:
                    if vision_router_weight is not None:
                        target_dtype = moe_sd[key].dtype
                        moe_sd[key] = vision_router_weight.to(target_dtype)
                        log.info(f"[router]  {key}  — seeded from embeddings")
                    else:
                        log.info(f"[router]  {key}  — random init")
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
            sparse = is_sparse_layer(layer_idx, moe_config)

            # Attention + norms — always shared
            if ".self_attn." in key or "layernorm" in key:
                if key in dense_sds[0]:
                    _assert_equal_across_experts(key, dense_sds)
                    moe_sd[key] = dense_sds[0][key].clone()
                else:
                    log.warning(f"Shared layer key not found: {key}")
                continue

            # Dense MLP layers
            if ".mlp." in key and not sparse:
                if key in dense_sds[0]:
                    _assert_equal_across_experts(key, dense_sds)
                    moe_sd[key] = dense_sds[0][key].clone()
                else:
                    log.warning(f"Dense MLP key not found: {key}")
                continue

            # Sparse MoE layers
            if ".mlp." in key and sparse:
                layer_prefix = ".".join(key.split(".")[:4])  # "model.language_model.layers.N"

                # Router gate — seed with embeddings if provided
                if ".mlp.gate.weight" in key:
                    if text_router_weight is not None:
                        target_dtype = moe_sd[key].dtype
                        moe_sd[key] = text_router_weight.to(target_dtype)
                        log.info(f"[router]  {key}  — seeded from embeddings")
                    else:
                        log.info(f"[router]  {key}  — random init")
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
