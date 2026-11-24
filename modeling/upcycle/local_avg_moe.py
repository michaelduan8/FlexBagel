"""Local 2-expert stand-in for an N-expert BTX model, and the way back.

build:    From an N-expert Flex_Qwen2_5_VLMoe model, make collaborator k's local 2-expert model:
            expert 0 = expert k (the collaborator's own expert)
            expert 1 = the weight average of the other N-1 experts' FFNs (text and vision MoE blocks)
            gate rows = [row k, row j for j != k (in order)], gate_row_to_expert = [0, 1, ..., 1],
                        gate_row_aggregation = "sum"
          Every non-FFN tensor is copied unchanged. With "sum", expert 1's routing weight is the sum of the other
          collaborators' row probabilities, so expert 0 sees the same routing weight as in the N-expert model (only
          expert 1's FFN, an average, is an approximation). Train it with mm_tune --train_expert_idx 0
          --num_experts_per_tok 2 (only expert 0 and gate row 0 change).

assemble: Put each collaborator's trained expert 0 and gate row 0 back into slot k of the N-expert model; all other
          tensors come from the N-expert model, and the local models' non-FFN tensors must equal it.

    python modeling/upcycle/local_avg_moe.py build --btx <N-expert model> --expert k --out <dir>
    python modeling/upcycle/local_avg_moe.py assemble --btx <N-expert model> --local k=<trained local model> ... --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import re

import torch
from transformers import AutoProcessor, AutoTokenizer

from modeling.flex_qwen2_5_vl_moe import Flex_Qwen2_5_VLMoeConfig, Flex_Qwen2_5_VLMoeForConditionalGeneration

EXPERT_RE = re.compile(r"^(.*\.mlp)\.experts\.(\d+)\.(.+)$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--btx", required=True)
    b.add_argument("--expert", type=int, required=True, help="Index k of the collaborator's own expert in the BTX")
    b.add_argument("--out", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--btx", required=True)
    a.add_argument("--local", nargs="+", required=True, help="k=<trained local model path>, one per collaborator")
    a.add_argument("--out", required=True)
    return parser.parse_args()


def load(path):
    model = Flex_Qwen2_5_VLMoeForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16)
    return model, model.state_dict()


def check_plain_btx(config):
    for sub in (config.text_config, config.vision_config):
        if getattr(sub, "gate_row_to_expert", None) is not None or getattr(sub, "use_shared_expert", False):
            raise ValueError("expected a plain N-expert BTX (one gate row per expert, no shared expert)")


def local_config(config, own: int):
    cfg = copy.deepcopy(config)
    for sub in (cfg.text_config, cfg.vision_config):
        n = sub.num_experts
        if not 0 <= own < n or n < 2:
            raise ValueError(f"expert {own} out of range for {n} experts")
        sub.num_experts = 2
        sub.num_experts_per_tok = 2
        sub.norm_topk_prob = False
        sub.gate_row_to_expert = [0] + [1] * (n - 1)
        sub.gate_row_aggregation = "sum"
    return cfg


def build_state(btx_sd: dict, own: int, n: int) -> dict:
    """Local 2-expert state dict: experts.0 = expert `own`, experts.1 = mean of the others (in fp32, cast back),
    gate rows reordered to [own, others...]; every other tensor copied."""
    others = [j for j in range(n) if j != own]
    out = {}
    for key, t in btx_sd.items():
        m = EXPERT_RE.match(key)
        if m:
            prefix, e, rest = m.group(1), int(m.group(2)), m.group(3)
            if e == own:
                out[f"{prefix}.experts.0.{rest}"] = t.clone()
                stack = [btx_sd[f"{prefix}.experts.{j}.{rest}"].float() for j in others]
                out[f"{prefix}.experts.1.{rest}"] = (sum(stack) / len(stack)).to(t.dtype)
        elif key.endswith(".mlp.gate.weight"):
            out[key] = torch.cat([t[own:own + 1], t[others]], dim=0).clone()
        else:
            out[key] = t.clone()
    return out


def save(model, src, out):
    model.save_pretrained(out, safe_serialization=True)
    for name, cls in (("tokenizer", AutoTokenizer), ("processor", AutoProcessor)):
        try:
            cls.from_pretrained(src).save_pretrained(out)
        except Exception as exc:  # e.g. weight-only test checkpoints; real BTX checkpoints have both
            print(f"{name} not copied from {src}: {type(exc).__name__}")


def build(btx_path, own, out):
    btx, sd = load(btx_path)
    check_plain_btx(btx.config)
    n = btx.config.text_config.num_experts
    assert btx.config.vision_config.num_experts == n
    cfg = local_config(btx.config, own)
    local = Flex_Qwen2_5_VLMoeForConditionalGeneration(cfg).to(torch.bfloat16)
    local.load_state_dict(build_state(sd, own, n), strict=True)
    save(local, btx_path, out)
    print(f"local model for expert {own} of {n}: experts [own, avg of {[j for j in range(n) if j != own]}], "
          f"gate_row_to_expert {cfg.text_config.gate_row_to_expert} (sum) -> {out}")


def assemble(btx_path, locals_, out):
    btx, sd = load(btx_path)
    check_plain_btx(btx.config)
    n = btx.config.text_config.num_experts
    new = {k: v.clone() for k, v in sd.items()}
    for own, path in sorted(locals_.items()):
        _, lsd = load(path)
        for key, t in lsd.items():
            m = EXPERT_RE.match(key)
            if m:
                if int(m.group(2)) == 0:
                    new[f"{m.group(1)}.experts.{own}.{m.group(3)}"] = t.clone()
            elif key.endswith(".mlp.gate.weight"):
                assert t.shape[0] == n, f"{path}: {key} has {t.shape[0]} rows, expected {n}"
                new[key][own] = t[0]
            else:
                assert torch.equal(t, sd[key]), f"{path}: shared tensor {key} differs from the BTX"
        print(f"slot {own} <- expert 0 + gate row 0 of {path}")
    btx.load_state_dict(new, strict=True)
    save(btx, btx_path, out)
    print(f"assembled {len(locals_)} of {n} experts -> {out}")


def main():
    args = parse_args()
    if args.cmd == "build":
        build(args.btx, args.expert, args.out)
    else:
        locals_ = {int(spec.split("=", 1)[0]): spec.split("=", 1)[1] for spec in args.local}
        assemble(args.btx, locals_, args.out)


if __name__ == "__main__":
    main()
