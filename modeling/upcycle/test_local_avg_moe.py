"""Tests for modeling/upcycle/local_avg_moe.py (local 2-expert stand-in with averaged frozen experts, and back).

Run with pytest, or without it:  PYTHONPATH=. python -m modeling.upcycle.test_local_avg_moe
"""
import os
import tempfile

import torch

from modeling.flex_qwen2_5_vl_moe import Flex_Qwen2_5_VLMoeConfig, Flex_Qwen2_5_VLMoeForConditionalGeneration
from modeling.upcycle.local_avg_moe import assemble, build

TEXT = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            vocab_size=64, rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]})
VISION = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16)


def btx(tmp, n=3, top_k=2, seed=0):
    torch.manual_seed(seed)
    cfg = Flex_Qwen2_5_VLMoeConfig(text_config=dict(TEXT, num_experts=n, num_experts_per_tok=top_k),
                                   vision_config=dict(VISION, num_experts=n, num_experts_per_tok=top_k))
    model = Flex_Qwen2_5_VLMoeForConditionalGeneration(cfg).to(torch.bfloat16)
    path = os.path.join(tmp, "btx")
    model.save_pretrained(path)
    return path, model


def load(path, **cfg_overrides):
    cfg = Flex_Qwen2_5_VLMoeConfig.from_pretrained(path)
    for sub in (cfg.text_config, cfg.vision_config):
        for k, v in cfg_overrides.items():
            setattr(sub, k, v)
    return Flex_Qwen2_5_VLMoeForConditionalGeneration.from_pretrained(path, config=cfg, torch_dtype=torch.bfloat16).eval()


def test_build_layout():
    with tempfile.TemporaryDirectory() as tmp:
        path, ref = btx(tmp)
        build(path, 1, os.path.join(tmp, "local1"))
        local = load(os.path.join(tmp, "local1"))
    for sub in (local.config.text_config, local.config.vision_config):
        assert sub.num_experts == 2 and sub.num_experts_per_tok == 2 and sub.gate_row_to_expert == [0, 1, 1]
        assert sub.gate_row_aggregation == "sum" and not sub.norm_topk_prob
    r, l = ref.state_dict(), local.state_dict()
    for key in r:
        if ".mlp.experts.1." in key:
            assert torch.equal(l[key.replace(".experts.1.", ".experts.0.")], r[key])  # own expert
            avg = ((r[key.replace(".experts.1.", ".experts.0.")].float() + r[key.replace(".experts.1.", ".experts.2.")].float()) / 2)
            assert torch.equal(l[key], avg.to(r[key].dtype))  # averaged others
        elif key.endswith(".mlp.gate.weight"):
            assert torch.equal(l[key], r[key][[1, 0, 2]])
        elif ".mlp.experts." not in key:
            assert torch.equal(l[key], r[key])


def test_local_model_matches_btx_when_other_experts_are_identical():
    """With the averaged experts identical, the local model is exactly the BTX at top-N, no norm."""
    with tempfile.TemporaryDirectory() as tmp:
        path, ref = btx(tmp)
        sd = ref.state_dict()
        for key in sd:  # make experts 0 and 2 identical so their average equals each of them
            if ".mlp.experts.2." in key:
                sd[key] = sd[key.replace(".experts.2.", ".experts.0.")].clone()
        ref.load_state_dict(sd)
        ref.save_pretrained(path)
        build(path, 1, os.path.join(tmp, "local1"))
        local = load(os.path.join(tmp, "local1"))
        full = load(path, num_experts_per_tok=3, norm_topk_prob=False)
    ids = torch.randint(0, 64, (2, 7), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        a, b = local(input_ids=ids).logits.float(), full(input_ids=ids).logits.float()
    assert torch.allclose(a, b, atol=2e-2), float((a - b).abs().max())


def test_assemble_round_trip_and_slot_update():
    with tempfile.TemporaryDirectory() as tmp:
        path, ref = btx(tmp)
        for k in range(3):
            build(path, k, os.path.join(tmp, f"local{k}"))
        # untouched local models give back the BTX exactly
        assemble(path, {k: os.path.join(tmp, f"local{k}") for k in range(3)}, os.path.join(tmp, "same"))
        same = load(os.path.join(tmp, "same"))
        assert all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), same.state_dict().values()))
        # "train" local model 2 (expert 0 + gate row 0) and assemble: only slot 2 changes
        trained = load(os.path.join(tmp, "local2"))
        with torch.no_grad():
            for n, p in trained.named_parameters():
                if ".mlp.experts.0." in n:
                    p.add_(1.0)
                if n.endswith(".mlp.gate.weight"):
                    p[0] += 5.0
        trained.save_pretrained(os.path.join(tmp, "local2t"))
        assemble(path, {2: os.path.join(tmp, "local2t")}, os.path.join(tmp, "new"))
        new = load(os.path.join(tmp, "new"))
        r, nsd, t = ref.state_dict(), new.state_dict(), trained.state_dict()
        for key in r:
            if ".mlp.experts.2." in key:
                assert torch.equal(nsd[key], t[key.replace(".experts.2.", ".experts.0.")])
            elif key.endswith(".mlp.gate.weight"):
                assert torch.equal(nsd[key][2], t[key][0]) and torch.equal(nsd[key][:2], r[key][:2])
            else:
                assert torch.equal(nsd[key], r[key])
        # a local model whose shared (non-FFN) weights changed is rejected
        with torch.no_grad():
            next(p for n, p in trained.named_parameters() if "self_attn" in n).add_(1.0)
        trained.save_pretrained(os.path.join(tmp, "bad"))
        try:
            assemble(path, {2: os.path.join(tmp, "bad")}, os.path.join(tmp, "bad_out"))
        except AssertionError:
            pass
        else:
            raise AssertionError("local model with changed shared weights was accepted")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
