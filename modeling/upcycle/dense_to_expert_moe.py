import argparse
import logging
import numpy as np
import torch
from transformers import Qwen2ForCausalLM, Qwen2MoeForCausalLM, Qwen2Config, Qwen2MoeConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upcycle dense Qwen2 HF models into a Qwen2MoE HF model"
    )
    parser.add_argument("-m", "--models", nargs="+", required=True,
                        help="Paths/HF hub IDs for dense Qwen2 checkpoints (one per expert)")
    parser.add_argument("-t", "--target", type=str, required=True,
                        help="Output path to save the MoE model")
    parser.add_argument("-e", "--embeddings", nargs="+", default=[],
                        help="Paths to .npy embedding files for seeding the router gate "
                             "(one per model, same order as --models). "
                             "Shape: [num_experts, hidden_size] per file, or [hidden_size] for a single expert.")
    parser.add_argument("--shared-gate-embedding", type=str, default=None,
                        help="Path to a .npy embedding of shape [hidden_size] used to seed the shared expert gate. "
                             "If not provided but --embeddings are, defaults to the mean of those embeddings.")
    parser.add_argument("--decoder-sparse-step", type=int, default=1)
    parser.add_argument("--mlp-only-layers", nargs="*", type=int, default=[])
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--norm-topk-prob", action="store_true")
    parser.add_argument("--shared-expert-init", choices=["mean", "first"], default="mean")
    return parser.parse_args()


def build_moe_config(dense_config: Qwen2Config, num_experts: int, args) -> Qwen2MoeConfig:
    d = dense_config.to_dict()
    return Qwen2MoeConfig(
        vocab_size=d["vocab_size"],
        hidden_size=d["hidden_size"],
        intermediate_size=d["intermediate_size"],
        num_hidden_layers=d["num_hidden_layers"],
        num_attention_heads=d["num_attention_heads"],
        num_key_value_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
        hidden_act=d.get("hidden_act", "silu"),
        max_position_embeddings=d["max_position_embeddings"],
        initializer_range=d.get("initializer_range", 0.02),
        rms_norm_eps=d.get("rms_norm_eps", 1e-6),
        use_cache=d.get("use_cache", True),
        rope_theta=d.get("rope_theta", 10000.0),
        rope_scaling=d.get("rope_scaling"),
        attention_dropout=d.get("attention_dropout", 0.0),
        sliding_window=d.get("sliding_window"),
        use_sliding_window=d.get("use_sliding_window", False),
        max_window_layers=d.get("max_window_layers", 0),
        num_experts=num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=d["intermediate_size"],
        shared_expert_intermediate_size=d["intermediate_size"],
        decoder_sparse_step=args.decoder_sparse_step,
        mlp_only_layers=args.mlp_only_layers,
        norm_topk_prob=args.norm_topk_prob,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
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


def load_shared_gate_embedding(path: str | None, router_weight: torch.Tensor | None, hidden_size: int) -> torch.Tensor | None:
    """
    Load or derive the shared expert gate embedding of shape [1, hidden_size].

    Priority:
      1. Explicit --shared-gate-embedding path
      2. Mean of router embeddings (if those were provided)
      3. None → random init
    """
    if path is not None:
        emb = np.load(path).astype(np.float32)
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        assert emb.shape == (hidden_size,), \
            f"Shared gate embedding has unexpected shape {emb.shape}"
        log.info(f"Shared expert gate seeded from {path}")
        return torch.from_numpy(emb).unsqueeze(0)  # [1, hidden_size]

    if router_weight is not None:
        mean_emb = router_weight.mean(dim=0, keepdim=True)  # [1, hidden_size]
        log.info("Shared expert gate seeded from mean of router embeddings")
        return mean_emb

    return None


def is_sparse_layer(layer_idx: int, config: Qwen2MoeConfig) -> bool:
    if layer_idx in config.mlp_only_layers:
        return False
    return config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0


def load_dense_model(path: str, dtype=torch.bfloat16):
    log.info(f"  Loading {path} ...")
    model = Qwen2ForCausalLM.from_pretrained(path, torch_dtype=dtype)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    config = model.config
    del model
    return sd, config


def _assert_equal_across_experts(key: str, state_dicts: list):
    ref = state_dicts[0].get(key)
    if ref is None:
        return
    for i, sd in enumerate(state_dicts[1:], 1):
        if key in sd and not torch.equal(ref, sd[key]):
            log.warning(
                f"Shared key '{key}' differs between expert 0 and expert {i}. "
                f"Using expert 0's value."
            )


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
    # 2. Load embeddings for gate initialization
    # ------------------------------------------------------------------
    router_weight = load_embeddings(args.embeddings, dense_config.hidden_size)
    shared_gate_weight = load_shared_gate_embedding(
        args.shared_gate_embedding, router_weight, dense_config.hidden_size
    )

    # ------------------------------------------------------------------
    # 3. Build MoE config + skeleton
    # ------------------------------------------------------------------
    moe_config = build_moe_config(dense_config, num_experts, args)
    log.info(f"MoE config:\n{moe_config}")

    log.info("Instantiating Qwen2MoE skeleton on CPU ...")
    moe_model = Qwen2MoeForCausalLM(moe_config)
    moe_sd = moe_model.state_dict()

    # ------------------------------------------------------------------
    # 4. Copy weights
    # ------------------------------------------------------------------
    for key in list(moe_sd.keys()):
        # TODO: not sure if this keying is generalizable

        # Global shared weights
        if not key.startswith("model.layers."):
            if key in dense_sds[0]:
                _assert_equal_across_experts(key, dense_sds)
                moe_sd[key] = dense_sds[0][key].clone()
            else:
                log.warning(f"Key not found in dense model: {key}")
            continue

        parts = key.split(".")
        layer_idx = int(parts[2])
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
            layer_prefix = ".".join(key.split(".")[:3])  # "model.layers.N"

            # Router gate — seed with embeddings if provided
            if ".mlp.gate.weight" in key:
                if router_weight is not None:
                    target_dtype = moe_sd[key].dtype
                    moe_sd[key] = router_weight.to(target_dtype)
                    log.info(f"[router]  {key}  — seeded from embeddings")
                else:
                    log.info(f"[router]  {key}  — random init")
                continue

            # Shared expert gate — seed with embedding if provided
            if ".mlp.shared_expert_gate.weight" in key:
                if shared_gate_weight is not None:
                    target_dtype = moe_sd[key].dtype
                    moe_sd[key] = shared_gate_weight.to(target_dtype)
                    log.info(f"[shared-expert-gate]  {key}  — seeded from embedding")
                else:
                    log.info(f"[shared-expert-gate]  {key}  — random init")
                continue

            # Shared expert FFN
            if ".mlp.shared_expert." in key:
                dense_key = key.replace(".mlp.shared_expert.", ".mlp.")
                if dense_key in dense_sds[0]:
                    if args.shared_expert_init == "mean":
                        avg = torch.stack([sd[dense_key].float() for sd in dense_sds]).mean(0)
                        moe_sd[key] = avg.to(dense_sds[0][dense_key].dtype)
                    else:
                        moe_sd[key] = dense_sds[0][dense_key].clone()
                    log.debug(f"[shared-expert]  {dense_key} -> {key}")
                else:
                    log.warning(f"Shared expert source key not found: {dense_key}")
                continue

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
    # 5. Save
    # ------------------------------------------------------------------
    missing, unexpected = moe_model.load_state_dict(moe_sd, strict=True)
    if missing:
        log.warning(f"Missing keys: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected}")

    log.info(f"Saving to {args.target} ...")
    moe_model.save_pretrained(args.target)
    moe_config.save_pretrained(args.target)
    log.info("Done.")


if __name__ == "__main__":
    main()