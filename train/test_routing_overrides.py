"""Tests for train/mm_tune.py load_model_with_routing_overrides (--num_experts_per_tok / --norm_topk_prob).

The MoE blocks copy top_k and norm_topk_prob from the config at construction, so the overrides must be applied before
the model is built; setting model.config afterwards silently trained with the checkpoint's saved values.

Run with pytest, or without it:  PYTHONPATH=. python -m train.test_routing_overrides
"""
import tempfile

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from modeling.flex_qwen2_5_vl_moe import Flex_Qwen2_5_VLMoeConfig, Flex_Qwen2_5_VLMoeForConditionalGeneration
from train.mm_tune import load_model_with_routing_overrides, register_local_architectures

register_local_architectures()
TEXT = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            vocab_size=64, rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]})
VISION = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16)


def save_moe(d, top_k=2):
    torch.manual_seed(0)
    cfg = Flex_Qwen2_5_VLMoeConfig(text_config=dict(TEXT, num_experts=3, num_experts_per_tok=top_k),
                                   vision_config=dict(VISION, num_experts=3, num_experts_per_tok=top_k))
    Flex_Qwen2_5_VLMoeForConditionalGeneration(cfg).save_pretrained(d)


def moe_blocks(model):
    return [m for m in model.modules() if hasattr(m, "top_k") and hasattr(m, "gate") and hasattr(m, "experts")]


def test_override_reaches_every_moe_block_and_changes_routing():
    ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(1))
    with tempfile.TemporaryDirectory() as d:
        save_moe(d, top_k=2)
        m2 = load_model_with_routing_overrides(d, num_experts_per_tok=2, norm_topk_prob=False).eval()
        m3 = load_model_with_routing_overrides(d, num_experts_per_tok=3, norm_topk_prob=False).eval()
    blocks = moe_blocks(m3)
    assert len(blocks) == 3  # 2 text layers + 1 vision block
    assert all(b.top_k == 3 for b in blocks) and m3.config.text_config.num_experts_per_tok == 3
    with torch.no_grad():
        assert not torch.equal(m2(input_ids=ids).logits, m3(input_ids=ids).logits)


def test_norm_topk_prob_override_applies():
    with tempfile.TemporaryDirectory() as d:
        save_moe(d, top_k=2)
        m = load_model_with_routing_overrides(d, num_experts_per_tok=2, norm_topk_prob=True)
    assert all(b.norm_topk_prob for b in moe_blocks(m))


def test_dense_model_loads_unchanged():
    torch.manual_seed(0)
    cfg = Qwen2_5_VLConfig(text_config=dict(TEXT), vision_config=dict(VISION))
    with tempfile.TemporaryDirectory() as d:
        ref = Qwen2_5_VLForConditionalGeneration(cfg)
        ref.save_pretrained(d)
        m = load_model_with_routing_overrides(d, num_experts_per_tok=3, norm_topk_prob=False)
    assert type(m).__name__ == "Qwen2_5_VLForConditionalGeneration" and not moe_blocks(m)
    assert all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), m.state_dict().values()))


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
