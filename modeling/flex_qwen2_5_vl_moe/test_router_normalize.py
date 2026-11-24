"""Tests for optional cosine routing (`router_normalize`, `router_temperature` on the text/vision MoE config).

Run with pytest, or without it:  PYTHONPATH=. python -m modeling.flex_qwen2_5_vl_moe.test_router_normalize
"""

import copy

import torch
import torch.nn.functional as F

from .configuration_flex_qwen2_5_vl_moe import Flex_Qwen2_5_VLMoeTextConfig
from .modeling_flex_qwen2_5_vl_moe import Flex_Qwen2_5_VLMoeSparseMoeBlock

HIDDEN = 16


def text_config(**kwargs):
    return Flex_Qwen2_5_VLMoeTextConfig(
        hidden_size=HIDDEN, intermediate_size=32, moe_intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=64, num_experts=3,
        num_experts_per_tok=3, norm_topk_prob=False, **kwargs,
    )


def make_block(seed=0, **kwargs):
    torch.manual_seed(seed)
    return Flex_Qwen2_5_VLMoeSparseMoeBlock(text_config(**kwargs)).eval()


def test_default_is_plain_linear_router():
    block = make_block()
    assert block.router_normalize is False
    x = torch.randn(2, 5, HIDDEN)
    out, logits = block(x)
    assert torch.equal(logits, block.gate(x.reshape(-1, HIDDEN)))
    off = make_block(router_normalize=False, router_temperature=100.0)  # same seed, explicitly off
    out_off, logits_off = off(x)
    assert torch.equal(out, out_off) and torch.equal(logits, logits_off)


def test_cosine_logits_and_scale_invariance():
    block = make_block(router_normalize=True, router_temperature=100.0)
    x = torch.randn(2, 5, HIDDEN)
    out, logits = block(x)
    flat = x.reshape(-1, HIDDEN)
    expected = 100.0 * F.normalize(flat, dim=-1) @ F.normalize(block.gate.weight, dim=-1).T
    assert torch.allclose(logits, expected, atol=1e-5)
    assert logits.abs().max() <= 100.0 + 1e-4
    scaled = copy.deepcopy(block)
    with torch.no_grad():
        scaled.gate.weight.mul_(500.0)  # router-row size no longer matters
    _, logits_s = scaled(x * 37.0)  # nor does hidden-state size: routing logits are unchanged
    assert torch.allclose(logits, logits_s, atol=1e-4)


def test_config_roundtrip_keeps_flags():
    cfg = text_config(router_normalize=True, router_temperature=100.0)
    cfg2 = Flex_Qwen2_5_VLMoeTextConfig.from_dict(cfg.to_dict())
    block = Flex_Qwen2_5_VLMoeSparseMoeBlock(cfg2)
    assert block.router_normalize is True and block.router_temperature == 100.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
