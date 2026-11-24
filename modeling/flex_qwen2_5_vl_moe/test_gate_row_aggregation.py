"""Tests for `gate_row_aggregation` (opt-in summed row probabilities for many-to-one routing, GATE_ROW_TO_EXPERT.md).

Run with pytest, or without it:  PYTHONPATH=. python -m modeling.flex_qwen2_5_vl_moe.test_gate_row_aggregation
"""
import tempfile

import torch
import torch.nn.functional as F

from . import Flex_Qwen2_5_VLMoeConfig, Flex_Qwen2_5_VLMoeForConditionalGeneration
from .configuration_flex_qwen2_5_vl_moe import Flex_Qwen2_5_VLMoeTextConfig
from .modeling_flex_qwen2_5_vl_moe import (
    Flex_Qwen2_5_VLMoeSparseMoeBlock,
    load_balancing_loss_func,
    select_experts_many_to_one,
)
from .monitor import MoeRoutingMonitor
from .test_gate_row_to_expert import HIDDEN, make_block, text_config, tiny_model_config


def block_output(block, x):
    out = block(x)
    return out[0] if isinstance(out, tuple) else out


def test_unset_and_explicit_max_are_bit_identical():
    x = torch.randn(2, 5, HIDDEN)
    for norm in (False, True):
        default = make_block(mapping=[0, 1, 0, 2], num_experts=3, top_k=2, norm=norm)
        explicit = make_block(mapping=[0, 1, 0, 2], num_experts=3, top_k=2, norm=norm, gate_row_aggregation="max")
        explicit.load_state_dict(default.state_dict())
        assert default.gate_row_aggregation is None and explicit.gate_row_aggregation == "max"
        with torch.no_grad():
            assert torch.equal(block_output(default, x), block_output(explicit, x))
    probs = F.softmax(torch.randn(6, 4), dim=-1)
    mapping = torch.tensor([0, 1, 0, 2])
    for a, b in zip(select_experts_many_to_one(probs, mapping, 3, 2),
                    select_experts_many_to_one(probs, mapping, 3, 2, aggregation="max")):
        assert torch.equal(a, b)


def test_sum_selects_and_weights_by_summed_row_probs():
    probs = torch.tensor([[0.40, 0.35, 0.25], [0.50, 0.30, 0.20]])  # rows -> experts [0, 1, 1]
    weights, experts = select_experts_many_to_one(probs, torch.tensor([0, 1, 1]), 2, 1, aggregation="sum")
    # row 0: expert 1 has 0.35 + 0.25 = 0.60 > 0.40 (max would have picked expert 0); row 1: 0.50 vs 0.50
    assert experts[0].item() == 1 and torch.isclose(weights[0, 0], torch.tensor(0.60))
    weights2, experts2 = select_experts_many_to_one(probs, torch.tensor([0, 1, 1]), 2, 2, aggregation="sum")
    assert torch.allclose(weights2.sum(-1), torch.ones(2))  # all row mass is accounted for
    max_w, max_e = select_experts_many_to_one(probs, torch.tensor([0, 1, 1]), 2, 1)
    assert max_e[0].item() == 0  # the default (max) path is unchanged


def test_sum_local_block_equals_merged_block_with_shared_expert_copies():
    """2-expert block, rows [r0, r1, r2] -> [A, B, B], 'sum', top-2 == 3-expert block [A, B, B], top-3, no norm."""
    torch.manual_seed(0)
    local = Flex_Qwen2_5_VLMoeSparseMoeBlock(
        text_config(num_experts=2, top_k=2, norm=False, mapping=[0, 1, 1], gate_row_aggregation="sum")).eval()
    merged = Flex_Qwen2_5_VLMoeSparseMoeBlock(text_config(num_experts=3, top_k=3, norm=False)).eval()
    with torch.no_grad():
        merged.gate.weight.copy_(local.gate.weight)
        merged.experts[0].load_state_dict(local.experts[0].state_dict())
        merged.experts[1].load_state_dict(local.experts[1].state_dict())
        merged.experts[2].load_state_dict(local.experts[1].state_dict())
        x = torch.randn(3, 7, HIDDEN)
        assert torch.allclose(block_output(local, x), block_output(merged, x), atol=1e-6)
        # the max rule would under-weight the shared expert, so it does not match
        local.gate_row_aggregation = "max"
        assert not torch.allclose(block_output(local, x), block_output(merged, x), atol=1e-4)


def test_load_balancing_dispatch_follows_aggregation():
    logits = (torch.tensor([[0.0, -0.1, -0.2], [0.0, -2.0, -2.0]]),)  # 1 layer, 2 tokens, rows -> [0, 1, 1]
    mapping = [0, 1, 1]
    default = load_balancing_loss_func(logits, 2, 1, gate_row_to_expert=mapping)
    explicit_max = load_balancing_loss_func(logits, 2, 1, gate_row_to_expert=mapping, gate_row_aggregation="max")
    summed = load_balancing_loss_func(logits, 2, 1, gate_row_to_expert=mapping, gate_row_aggregation="sum")
    assert torch.equal(default, explicit_max)
    # token 0: max picks expert 0 (its row is best), sum picks expert 1 (two rows) -> different dispatch
    probs = F.softmax(logits[0], -1)
    router = torch.stack([probs[:, 0], probs[:, 1] + probs[:, 2]], -1).mean(0)
    expected_sum = 2 * (torch.tensor([0.5, 0.5]) * router).sum()  # dispatch: token0 -> 1, token1 -> 0
    assert torch.isclose(summed, expected_sum), (summed, expected_sum)
    assert not torch.isclose(summed, default)


def test_config_validation_and_roundtrip():
    for bad in ("mean", 1):
        try:
            text_config(mapping=[0, 1, 2], gate_row_aggregation=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"gate_row_aggregation={bad!r} accepted")
    try:
        text_config(gate_row_aggregation="sum")  # no gate_row_to_expert
    except ValueError:
        pass
    else:
        raise AssertionError("gate_row_aggregation without gate_row_to_expert accepted")
    assert Flex_Qwen2_5_VLMoeTextConfig().gate_row_aggregation is None  # default unchanged
    cfg = tiny_model_config([0, 1, 2, 1])
    cfg.text_config.gate_row_aggregation = "sum"
    cfg.vision_config.gate_row_aggregation = "sum"
    with tempfile.TemporaryDirectory() as tmp:
        cfg.save_pretrained(tmp)
        reloaded = Flex_Qwen2_5_VLMoeConfig.from_pretrained(tmp)
    assert reloaded.text_config.gate_row_aggregation == "sum" == reloaded.vision_config.gate_row_aggregation


def test_full_model_sum_runs_and_monitor_follows_setting():
    cfg = tiny_model_config([0, 1, 2, 1])
    cfg.text_config.gate_row_aggregation = "sum"
    cfg.vision_config.gate_row_aggregation = "sum"
    torch.manual_seed(0)
    model = Flex_Qwen2_5_VLMoeForConditionalGeneration(cfg).eval()
    blocks = [m for m in model.modules() if isinstance(m, Flex_Qwen2_5_VLMoeSparseMoeBlock)]
    assert blocks and all(b.gate_row_aggregation == "sum" for b in blocks)
    ids = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids, output_router_logits=True)
    assert torch.isfinite(out.loss) and torch.isfinite(out.aux_loss)
    monitor = MoeRoutingMonitor(model, skip_first_call_per_layer=False).register()
    with torch.no_grad():
        model(input_ids=ids)
    monitor.remove()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
