"""Tests for many-to-one gate rows (`gate_row_to_expert`, see GATE_ROW_TO_EXPERT.md).

Run with pytest, or without it:  PYTHONPATH=. python -m modeling.flex_qwen2_5_vl_moe.test_gate_row_to_expert
"""

import copy
import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .configuration_flex_qwen2_5_vl_moe import Flex_Qwen2_5_VLMoeConfig, Flex_Qwen2_5_VLMoeTextConfig
from .modeling_flex_qwen2_5_vl_moe import (
    Flex_Qwen2_5_VLMoeForConditionalGeneration,
    Flex_Qwen2_5_VLMoeSparseMoeBlock,
    gate_rows_to_expert_scores,
    load_balancing_loss_func,
    routed_only_router_logits,
)
from .monitor import MoeRoutingMonitor

HIDDEN = 16


def text_config(num_experts=3, top_k=2, norm=False, mapping=None, **kwargs):
    return Flex_Qwen2_5_VLMoeTextConfig(
        hidden_size=HIDDEN, intermediate_size=32, moe_intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=64, num_experts=num_experts,
        num_experts_per_tok=top_k, norm_topk_prob=norm, gate_row_to_expert=mapping, **kwargs,
    )


def make_block(seed=0, **kwargs):
    torch.manual_seed(seed)
    return Flex_Qwen2_5_VLMoeSparseMoeBlock(text_config(**kwargs)).eval()


def reference_one_row_per_expert(block, x):
    """The routing of the MoE block before gate_row_to_expert existed (default path), written out."""
    flat = x.reshape(-1, x.shape[-1])
    logits = block.gate(flat)
    weights, experts = torch.topk(F.softmax(logits, dim=1, dtype=torch.float), block.top_k, dim=-1)
    if block.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(flat.dtype)
    out = torch.zeros_like(flat)
    mask = F.one_hot(experts, num_classes=block.num_experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        e = e.item()
        idx, top_x = torch.where(mask[e])
        out.index_add_(0, top_x, block.experts[e](flat[top_x]) * weights[top_x, idx, None])
    return out.reshape(x.shape), logits


def test_default_path_is_bit_exact_to_one_row_per_expert():
    for norm in (False, True):
        block = make_block(norm=norm)
        assert block.gate_row_to_expert is None and block.gate.out_features == 3
        x = torch.randn(2, 5, HIDDEN)
        with torch.no_grad():
            out, logits = block(x)
            ref_out, ref_logits = reference_one_row_per_expert(block, x)
        assert torch.equal(out, ref_out) and torch.equal(logits, ref_logits)


def test_identity_mapping_matches_default():
    for norm in (False, True):
        default = make_block(norm=norm)
        identity = make_block(norm=norm, mapping=[0, 1, 2])
        identity.load_state_dict(default.state_dict())
        x = torch.randn(2, 5, HIDDEN)
        with torch.no_grad():
            assert torch.allclose(default(x)[0], identity(x)[0], atol=1e-6)


def test_many_to_one_topk_over_distinct_experts_with_best_row_weight():
    # rows: [pub_a, dom_1, pub_b, dom_2] -> experts [0, 1, 0, 2]
    block = make_block(mapping=[0, 1, 0, 2], top_k=2)
    assert block.gate.out_features == 4
    x = torch.randn(3, 7, HIDDEN)
    with torch.no_grad():
        out, row_logits = block(x)
        flat = x.reshape(-1, HIDDEN)
        probs = F.softmax(row_logits, dim=-1)
        scores = torch.stack([probs[:, [0, 2]].max(-1).values, probs[:, 1], probs[:, 3]], dim=-1)
        w, e = torch.topk(scores, 2, dim=-1)
        expected = torch.zeros_like(flat)
        for t in range(flat.shape[0]):
            for j in range(2):
                expected[t] += block.experts[int(e[t, j])](flat[t:t + 1])[0] * w[t, j]
    assert torch.allclose(out.reshape(-1, HIDDEN), expected, atol=1e-5)


def test_public_expert_runs_once_when_two_public_rows_win():
    block = make_block(mapping=[0, 1, 0, 2], top_k=2)
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.weight[0, 0] = 5.0  # pub_a
        block.gate.weight[2, 0] = 4.0  # pub_b: both public rows beat the domain rows
    x = torch.zeros(1, 4, HIDDEN)
    x[..., 0] = 1.0
    seen = []
    handle = block.experts[0].register_forward_hook(lambda m, i, o: seen.append(i[0].shape[0]))
    with torch.no_grad():
        out, logits = block(x)
    handle.remove()
    assert seen == [4], seen  # one call, each of the 4 tokens once
    probs = F.softmax(logits, dim=-1)
    second = 1 if probs[0, 1] >= probs[0, 3] else 2
    flat = x.reshape(-1, HIDDEN)
    with torch.no_grad():
        expected = block.experts[0](flat) * probs[:, :1] + block.experts[second](flat) * probs[:, [1, 3][second - 1], None]
    assert torch.allclose(out.reshape(-1, HIDDEN), expected, atol=1e-5)


def test_norm_topk_prob_normalizes_over_selected_experts():
    block = make_block(mapping=[0, 1, 0, 2], top_k=2, norm=True)
    x = torch.randn(1, 6, HIDDEN)
    with torch.no_grad():
        out, logits = block(x)
        flat = x.reshape(-1, HIDDEN)
        scores = gate_rows_to_expert_scores(F.softmax(logits, -1), torch.tensor([0, 1, 0, 2]), 3, "amax")
        w, e = torch.topk(scores, 2, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        expected = sum(
            torch.stack([block.experts[int(e[t, j])](flat[t:t + 1])[0] for t in range(flat.shape[0])]) * w[:, j:j + 1]
            for j in range(2)
        )
    assert torch.allclose(out.reshape(-1, HIDDEN), expected, atol=1e-5)


def test_load_balancing_loss():
    logits = (torch.randn(10, 3), torch.randn(10, 3))
    mask = torch.ones(2, 5, dtype=torch.long)
    for attention_mask in (None, mask):
        default = load_balancing_loss_func(logits, 3, 2, attention_mask)
        identity = load_balancing_loss_func(logits, 3, 2, attention_mask, gate_row_to_expert=[0, 1, 2])
        assert torch.allclose(default, identity, atol=1e-6)
    # many-to-one: dispatch from best-row top-k over experts, router prob = summed row probs
    rows = (torch.randn(10, 4),)
    probs = F.softmax(rows[0], -1)
    mapping = torch.tensor([0, 1, 0, 2])
    summed = gate_rows_to_expert_scores(probs, mapping, 3, "sum")
    _, sel = torch.topk(gate_rows_to_expert_scores(probs, mapping, 3, "amax"), 2, dim=-1)
    expected = (F.one_hot(sel, 3).float().mean(0) * summed.mean(0)).sum() * 3
    got = load_balancing_loss_func(rows, 3, 2, gate_row_to_expert=[0, 1, 0, 2])
    assert torch.allclose(got, expected, atol=1e-6)


def test_routed_only_logits_softmax_to_summed_row_probs():
    cfg = SimpleNamespace(gate_row_to_expert=[0, 1, 0, 2], num_experts=3, use_shared_expert=False)
    rows = torch.randn(6, 4)
    (collapsed,) = routed_only_router_logits((rows,), cfg)
    summed = gate_rows_to_expert_scores(F.softmax(rows, -1), torch.tensor([0, 1, 0, 2]), 3, "sum")
    assert torch.allclose(F.softmax(collapsed, -1), summed, atol=1e-6)
    cfg_default = SimpleNamespace(gate_row_to_expert=None, num_experts=3, use_shared_expert=False)
    assert routed_only_router_logits((rows,), cfg_default)[0] is rows


def test_monitor_reports_the_routed_expert():
    block = make_block(mapping=[0, 1, 0, 2], top_k=1)
    monitor = MoeRoutingMonitor(model=None, track_transition_matrix=False)
    x = torch.randn(1, 8, HIDDEN)
    with torch.no_grad():
        out = block(x)
        monitor._make_hook("model.language_model.layers.0.mlp", block)(block, (x,), out)
        scores = gate_rows_to_expert_scores(F.softmax(out[1], -1), torch.tensor([0, 1, 0, 2]), 3, "amax")
    expected_counts = torch.bincount(scores.argmax(-1), minlength=3)
    assert torch.equal(monitor.stats["model.language_model.layers.0.mlp"]["top1_counts"], expected_counts)


def test_trainer_masks_all_rows_of_the_trained_expert():
    from train.mm_tune import ExpertSFTTrainer

    mapping = [0, 1, 0, 2]
    fake = SimpleNamespace(expert_idx=0, model=SimpleNamespace(config=SimpleNamespace(
        text_config=SimpleNamespace(gate_row_to_expert=mapping), vision_config=SimpleNamespace(gate_row_to_expert=None))))
    rows = ExpertSFTTrainer._gate_rows_for_expert
    assert rows(fake, "model.language_model.layers.3.mlp.gate.weight") == [0, 2]
    assert rows(fake, "model.visual.blocks.3.mlp.gate.weight") is None  # default: mask[expert_idx] as before


def test_config_validation():
    bad = [
        dict(mapping=[0, 1, 1]),                                  # expert 2 has no row
        dict(mapping=[0, 1, 3]),                                  # out of range
        dict(mapping=[0, 1, 2], use_shared_expert=True),
        dict(mapping=[0, 1, 2], always_active_expert=0),
        dict(mapping=[0, 1, 2], top_k=4),
    ]
    for kwargs in bad:
        try:
            text_config(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid config {kwargs}")
    block = make_block(mapping=[0, 1, 0, 2])
    block.always_active_expert = 0  # e.g. an eval-time override
    try:
        block(torch.randn(1, 2, HIDDEN))
    except ValueError:
        return
    raise AssertionError("runtime always_active_expert override was accepted")


def tiny_model_config(mapping=None):
    cfg = Flex_Qwen2_5_VLMoeConfig(
        text_config=dict(
            hidden_size=HIDDEN, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, vocab_size=64, num_experts=3, num_experts_per_tok=2,
            output_router_logits=True, router_aux_loss_coef=0.1, gate_row_to_expert=mapping,
            rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
        ),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=HIDDEN,
                           num_experts=3, num_experts_per_tok=2, gate_row_to_expert=mapping),
    )
    return cfg


def test_full_model_identity_mapping_matches_default_and_roundtrips():
    torch.manual_seed(0)
    default = Flex_Qwen2_5_VLMoeForConditionalGeneration(tiny_model_config()).eval()
    identity = Flex_Qwen2_5_VLMoeForConditionalGeneration(tiny_model_config([0, 1, 2])).eval()
    identity.load_state_dict(default.state_dict())
    ids = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        a = default(input_ids=ids, labels=ids, output_router_logits=True)
        b = identity(input_ids=ids, labels=ids, output_router_logits=True)
    assert torch.allclose(a.loss, b.loss, atol=1e-5) and torch.allclose(a.aux_loss, b.aux_loss, atol=1e-6)

    many = Flex_Qwen2_5_VLMoeForConditionalGeneration(tiny_model_config([0, 1, 0, 2])).eval()
    with torch.no_grad():
        out = many(input_ids=ids, labels=ids, output_router_logits=True)
    assert torch.isfinite(out.loss) and torch.isfinite(out.aux_loss)
    assert many.model.language_model.layers[0].mlp.gate.out_features == 4
    with tempfile.TemporaryDirectory() as tmp:
        many.config.save_pretrained(tmp)
        reloaded = Flex_Qwen2_5_VLMoeConfig.from_pretrained(tmp)
    assert reloaded.text_config.gate_row_to_expert == [0, 1, 0, 2]
    assert reloaded.vision_config.gate_row_to_expert == [0, 1, 0, 2]
    assert copy.deepcopy(default.config).text_config.gate_row_to_expert is None  # default keeps the field unset


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
