"""Numerical tests for vision attention routing, context, and expert histories."""

import pytest
import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from . import (
    FlexQwen2_5VLSharedKVConfig,
    FlexQwen2_5VLHeadExpertConfig,
    FlexQwen2_5VLSharedKVVisionConfig,
    FlexQwen2_5VLHeadExpertVisionConfig,
    FlexQwen2_5VLSharedKVVisionAttention,
    FlexQwen2_5VLHeadExpertVisionAttention,
    FlexQwen2_5VLSharedKVVisionModel,
    FlexQwen2_5VLHeadExpertVisionModel,
    FlexQwen2_5VLSharedKVForConditionalGeneration,
    FlexQwen2_5VLHeadExpertForConditionalGeneration,
)
from ..flex_qwen2_5_vl_moe_connector.configuration_flex_qwen2_5_vl_moe import (
    Flex_Qwen2_5_VLMoeTextConfig,
)
from ..flex_qwen2_5_vl_moe_connector.modeling_flex_qwen2_5_vl_moe import (
    Flex_Qwen2_5_VLMoeAttention,
    Flex_Qwen2_5_VLMoeVisionAttention,
    apply_rotary_pos_emb_vision,
)

VARIANTS = [
    (
        FlexQwen2_5VLSharedKVVisionConfig,
        FlexQwen2_5VLSharedKVVisionAttention,
        FlexQwen2_5VLSharedKVVisionModel,
    ),
    (
        FlexQwen2_5VLHeadExpertVisionConfig,
        FlexQwen2_5VLHeadExpertVisionAttention,
        FlexQwen2_5VLHeadExpertVisionModel,
    ),
]


def _vision_config(cls, **kwargs):
    options = dict(
        depth=2,
        hidden_size=16,
        intermediate_size=24,
        num_heads=2,
        out_hidden_size=16,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        window_size=4,
        fullatt_block_indexes=[1],
        attention_num_experts=3,
        attention_top_k=2,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
    )
    options.update(kwargs)
    return cls(**options)


def _text_config():
    # The text decoder deliberately retains its original GQA architecture.
    return Flex_Qwen2_5_VLMoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
        pad_token_id=0,
    )


def _embeddings(batch, length):
    angles = torch.randn(batch, length, 8)
    return angles.cos(), angles.sin()


def _dense_reference(module, hidden, embeddings, mask):
    """Compute EVERY expert's full attention, then route only its output rows."""
    batch, length, _ = hidden.shape
    heads, dim = module.num_heads, module.head_dim
    output = torch.zeros_like(hidden)
    if hasattr(module, "router"):
        logits, chosen = module.router(hidden).float().topk(module.top_k, dim=-1)
        routing = torch.zeros_like(module.router(hidden)).scatter(
            -1, chosen, logits.softmax(-1)
        )
        banks = [(None, module.experts, routing)]
    else:
        banks = []
        for head, (router, bank) in enumerate(zip(module.routers, module.experts)):
            logits, chosen = router(hidden).float().topk(module.top_k, dim=-1)
            routing = torch.zeros_like(router(hidden)).scatter(
                -1, chosen, logits.softmax(-1)
            )
            banks.append((head, bank, routing))
    for head, bank, routing in banks:
        count = heads if head is None else 1
        for expert_idx, expert in enumerate(bank):
            q = expert.q_proj(hidden).reshape(batch, length, count, dim)
            k = (module.k_proj if head is None else expert.k_proj)(hidden).reshape(
                batch, length, count, dim
            )
            v = (
                (module.v_proj if head is None else expert.v_proj)(hidden)
                .reshape(batch, length, count, dim)
                .transpose(1, 2)
            )
            q, k = apply_rotary_pos_emb_vision(q, k, *embeddings)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            expert_mask = (
                mask if head is None or mask.shape[1] == 1 else mask[:, head : head + 1]
            )
            scores = q @ k.transpose(-2, -1) / dim**0.5
            scores = scores.masked_fill(~expert_mask, float("-inf"))
            context = (
                (scores.softmax(-1) @ v)
                .transpose(1, 2)
                .reshape(batch, length, count * dim)
            )
            output += expert.o_proj(context) * routing[..., expert_idx, None]
    return output


@pytest.mark.parametrize("config_cls,attention_cls,model_cls", VARIANTS)
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_sparse_queries_match_dense_reference(
    config_cls, attention_cls, model_cls, backend, top_k
):
    torch.manual_seed(51)
    config = _vision_config(config_cls, attention_top_k=top_k)
    config._attn_implementation = backend
    attention = attention_cls(config, 0).eval()
    hidden = torch.randn(2, 5, 16)
    embeddings = _embeddings(2, 5)
    # Each head may have a different permitted context mask.
    mask = torch.ones(2, 2, 5, 5, dtype=torch.bool)
    mask[1, 1, :, 2] = False
    actual, _ = attention._forward(hidden, *embeddings, mask=mask)
    expected = _dense_reference(attention, hidden, embeddings, mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    routers = [attention.router] if hasattr(attention, "router") else attention.routers
    assert all(isinstance(router, nn.Linear) for router in routers)
    if top_k > 1:
        assert all(
            router.weight.grad is not None and router.weight.grad.abs().sum() > 0
            for router in routers
        )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in attention.parameters()
    )


@pytest.mark.parametrize("config_cls,attention_cls,model_cls", VARIANTS)
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_cached_queries_match_full_context_and_history_shape(
    config_cls, attention_cls, model_cls, backend
):
    torch.manual_seed(9)
    config = _vision_config(config_cls)
    config._attn_implementation = backend
    attention = attention_cls(config, 0).eval()
    hidden = torch.randn(5, 16)
    cos, sin = _embeddings(1, 5)
    embeddings = (cos[0], sin[0])
    with torch.no_grad():
        expected = attention(
            hidden, torch.tensor([0, 5]), position_embeddings=embeddings
        )
        cache = DynamicCache()
        attention(
            hidden[:3],
            torch.tensor([0, 3]),
            position_embeddings=tuple(x[:3] for x in embeddings),
            past_key_value=cache,
        )
        actual = attention(
            hidden[3:],
            torch.tensor([0, 2]),
            position_embeddings=tuple(x[3:] for x in embeddings),
            past_key_value=cache,
        )
        # Vision is bidirectional: only the final chunk has the full context.
        torch.testing.assert_close(actual, expected[3:], atol=1e-6, rtol=1e-5)
    assert cache.get_seq_length() == 5
    expected_heads = 2 * (
        3 if attention_cls is FlexQwen2_5VLHeadExpertVisionAttention else 1
    )
    assert (
        cache.layers[0].keys.shape
        == cache.layers[0].values.shape
        == (1, expected_heads, 5, 8)
    )
    cache.reorder_cache(torch.tensor([0, 0]))
    assert cache.layers[0].keys.shape == (2, expected_heads, 5, 8)


@pytest.mark.parametrize("config_cls,attention_cls,model_cls", VARIANTS)
def test_one_expert_reduces_to_ordinary_vision_attention(
    config_cls, attention_cls, model_cls
):
    torch.manual_seed(19)
    config = _vision_config(config_cls, attention_num_experts=1, attention_top_k=1)
    config._attn_implementation = "eager"
    dense = Flex_Qwen2_5_VLMoeVisionAttention(config).eval()
    moe = attention_cls(config).eval()
    with torch.no_grad():
        q_weight, k_weight, v_weight = dense.qkv.weight.chunk(3)
        q_bias, k_bias, v_bias = dense.qkv.bias.chunk(3)
        if hasattr(moe, "router"):
            moe.k_proj.weight.copy_(k_weight)
            moe.k_proj.bias.copy_(k_bias)
            moe.v_proj.weight.copy_(v_weight)
            moe.v_proj.bias.copy_(v_bias)
            moe.experts[0].q_proj.weight.copy_(q_weight)
            moe.experts[0].q_proj.bias.copy_(q_bias)
            moe.experts[0].o_proj.load_state_dict(dense.proj.state_dict())
        else:
            for head, bank in enumerate(moe.experts):
                expert = bank[0]
                rows = slice(head * 8, (head + 1) * 8)
                for name, weight, bias in (
                    ("q_proj", q_weight, q_bias),
                    ("k_proj", k_weight, k_bias),
                    ("v_proj", v_weight, v_bias),
                ):
                    getattr(expert, name).weight.copy_(weight[rows])
                    getattr(expert, name).bias.copy_(bias[rows])
                expert.o_proj.weight.copy_(dense.proj.weight[:, rows])
                expert.o_proj.bias.copy_(dense.proj.bias / moe.num_heads)
    hidden = torch.randn(7, 16)
    cos, sin = _embeddings(1, 7)
    embeddings = (cos[0], sin[0])
    boundaries = torch.tensor([0, 3, 7])
    actual = moe(hidden, boundaries, position_embeddings=embeddings)
    expected = dense(hidden, boundaries, position_embeddings=embeddings)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_head_experts_attend_to_tokens_routed_elsewhere_and_cache_unselected_alternatives():
    config = _vision_config(
        FlexQwen2_5VLHeadExpertVisionConfig, attention_num_experts=2, attention_top_k=1
    )
    config._attn_implementation = "eager"
    attention = FlexQwen2_5VLHeadExpertVisionAttention(config, 0).eval()
    hidden = torch.zeros(1, 3, 16)
    hidden[0, :, 0] = torch.tensor([-1.0, -2.0, 3.0])
    with torch.no_grad():
        for router in attention.routers:
            router.weight.zero_()
            router.weight[0, 0] = -1
            router.weight[1, 0] = 1
        for bank in attention.experts:
            for expert in bank:
                expert.q_proj.weight.zero_()
                expert.q_proj.bias.zero_()
                expert.k_proj.weight.zero_()
                expert.k_proj.bias.zero_()
    embeddings = _embeddings(1, 3)
    cache = DynamicCache()
    _, weights = attention._forward(
        hidden, *embeddings, cache=cache, output_attentions=True
    )
    # Last query chooses expert 1 but attends to the first two tokens, which
    # selected expert 0. Zero queries yield uniform bidirectional attention.
    torch.testing.assert_close(weights[0, :, 2], torch.full((2, 3), 1 / 3))
    attention._forward(hidden[:, :1], *(x[:, :1] for x in embeddings), cache=cache)
    assert cache.layers[0].keys.shape == (1, 4, 4, 8)
    for head, bank in enumerate(attention.experts):
        for expert_idx, expert in enumerate(bank):
            expected_values = expert.v_proj(hidden[:, :1])
            torch.testing.assert_close(
                cache.layers[0].values[:, head * 2 + expert_idx, -1:], expected_values
            )


def test_head_routers_normalize_separately_and_head_outputs_are_summed():
    config = _vision_config(
        FlexQwen2_5VLHeadExpertVisionConfig, attention_num_experts=2, attention_top_k=2
    )
    attention = FlexQwen2_5VLHeadExpertVisionAttention(config).eval()
    hidden = torch.zeros(1, 1, 16)
    hidden[..., 0] = 1
    with torch.no_grad():
        for head, (router, bank) in enumerate(
            zip(attention.routers, attention.experts)
        ):
            router.weight.zero_()
            router.weight[:, 0] = torch.tensor(
                [0.7, 0.3] if head == 0 else [0.4, 0.6]
            ).log()
            for expert_idx, expert in enumerate(bank):
                expert.o_proj.weight.zero_()
                expert.o_proj.bias.fill_(1 + head * 2 + expert_idx)
    actual, _ = attention._forward(hidden, torch.ones(1, 1, 8), torch.zeros(1, 1, 8))
    torch.testing.assert_close(
        actual, torch.full_like(actual, 0.7 * 1 + 0.3 * 2 + 0.4 * 3 + 0.6 * 4)
    )


@pytest.mark.parametrize(
    "config_cls,attention_cls",
    [
        (FlexQwen2_5VLSharedKVVisionConfig, FlexQwen2_5VLSharedKVVisionAttention),
        (FlexQwen2_5VLHeadExpertVisionConfig, FlexQwen2_5VLHeadExpertVisionAttention),
    ],
)
def test_fully_routed_experts_batch_windows_without_changing_gradients(
    config_cls, attention_cls
):
    torch.manual_seed(83)
    config = _vision_config(
        config_cls,
        attention_num_experts=3,
        attention_top_k=3,
    )
    config._attn_implementation = "sdpa"
    fast = attention_cls(config).train()
    slow = attention_cls(config).train()
    slow.load_state_dict(fast.state_dict())
    slow.backend = "eager"
    hidden = torch.randn(11, 16)
    cos, sin = _embeddings(1, 11)
    boundaries = torch.tensor([0, 3, 8, 11])
    fast_input = hidden.clone().requires_grad_()
    slow_input = hidden.clone().requires_grad_()
    actual = fast(fast_input, boundaries, position_embeddings=(cos[0], sin[0]))
    expected = slow(slow_input, boundaries, position_embeddings=(cos[0], sin[0]))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(fast_input.grad, slow_input.grad, atol=1e-4, rtol=1e-4)
    for (name, fast_parameter), (_, slow_parameter) in zip(
        fast.named_parameters(), slow.named_parameters()
    ):
        torch.testing.assert_close(
            fast_parameter.grad, slow_parameter.grad, atol=1e-4, rtol=1e-4, msg=name
        )


@pytest.mark.parametrize("config_cls,attention_cls,model_cls", VARIANTS)
def test_only_selected_queries_are_projected_but_all_context_kv_are_projected(
    config_cls, attention_cls, model_cls
):
    config = _vision_config(config_cls, attention_num_experts=3, attention_top_k=1)
    attention = attention_cls(config).eval()
    hidden = torch.randn(1, 5, 16)
    hidden[..., 0] = 1
    routers = [attention.router] if hasattr(attention, "router") else attention.routers
    experts = (
        attention.experts
        if hasattr(attention, "router")
        else [expert for bank in attention.experts for expert in bank]
    )
    counts = {"q": 0, "k": 0, "v": 0}
    handles = []

    def count(kind):
        def hook(module, args, output):
            counts[kind] += args[0].numel() // 16

        return hook

    with torch.no_grad():
        for router in routers:
            router.weight.zero_()
            router.weight[0, 0] = 1
        for expert in experts:
            handles.append(expert.q_proj.register_forward_hook(count("q")))
            if hasattr(expert, "k_proj"):
                handles.append(expert.k_proj.register_forward_hook(count("k")))
                handles.append(expert.v_proj.register_forward_hook(count("v")))
        if hasattr(attention, "k_proj"):
            handles.append(attention.k_proj.register_forward_hook(count("k")))
            handles.append(attention.v_proj.register_forward_hook(count("v")))
        attention._forward(hidden, *_embeddings(1, 5))
    for handle in handles:
        handle.remove()
    assert counts["q"] == 5 * len(routers)
    expected_kv = 5 if hasattr(attention, "router") else 5 * 2 * 3
    assert counts["k"] == counts["v"] == expected_kv


@pytest.mark.parametrize("config_cls,attention_cls,model_cls", VARIANTS)
def test_vision_preserves_packed_context_boundaries(
    config_cls, attention_cls, model_cls
):
    torch.manual_seed(2)
    attention = attention_cls(_vision_config(config_cls)).eval()
    hidden = torch.randn(7, 16)
    cos, sin = _embeddings(1, 7)
    embeddings = (cos[0], sin[0])
    boundaries = torch.tensor([0, 3, 7], dtype=torch.int32)
    combined = attention(hidden, boundaries, position_embeddings=embeddings)
    separate = torch.cat(
        [
            attention(
                hidden[:3],
                torch.tensor([0, 3]),
                position_embeddings=tuple(x[:3] for x in embeddings),
            ),
            attention(
                hidden[3:],
                torch.tensor([0, 4]),
                position_embeddings=tuple(x[3:] for x in embeddings),
            ),
        ]
    )
    torch.testing.assert_close(combined, separate)
    perturbed = hidden.clone()
    perturbed[3:] += 50
    changed = attention(perturbed, boundaries, position_embeddings=embeddings)
    torch.testing.assert_close(combined[:3], changed[:3])


@pytest.mark.parametrize("config_cls,attention_cls,model_cls", VARIANTS)
def test_config_validation_and_roundtrip(config_cls, attention_cls, model_cls):
    for kwargs in (
        dict(attention_num_experts=-1),
        dict(attention_top_k=0),
        dict(attention_top_k=4),
        dict(attention_num_experts=True),
        dict(attention_top_k=1.5),
        dict(hidden_size=18),
        dict(hidden_size=12),
    ):
        with pytest.raises(ValueError):
            _vision_config(config_cls, **kwargs)
    config = _vision_config(config_cls)
    restored = config_cls.from_dict(config.to_dict())
    assert restored.attention_num_experts == 3
    assert restored.attention_top_k == 2


@pytest.mark.parametrize(
    "config_cls,vision_cls,attention_cls,model_cls",
    [
        (
            FlexQwen2_5VLSharedKVConfig,
            FlexQwen2_5VLSharedKVVisionConfig,
            FlexQwen2_5VLSharedKVVisionAttention,
            FlexQwen2_5VLSharedKVForConditionalGeneration,
        ),
        (
            FlexQwen2_5VLHeadExpertConfig,
            FlexQwen2_5VLHeadExpertVisionConfig,
            FlexQwen2_5VLHeadExpertVisionAttention,
            FlexQwen2_5VLHeadExpertForConditionalGeneration,
        ),
    ],
)
def test_multimodal_forward_text_unchanged_generation_and_reload(
    config_cls, vision_cls, attention_cls, model_cls, tmp_path
):
    config = config_cls(
        text_config=_text_config(),
        vision_config=_vision_config(vision_cls),
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
        vision_end_token_id=31,
    )
    model = model_cls(config).eval()
    assert all(
        type(layer.self_attn) is Flex_Qwen2_5_VLMoeAttention
        for layer in model.language_model.layers
    )
    assert model.config.text_config.num_key_value_heads == 1
    assert all(isinstance(block.attn, attention_cls) for block in model.visual.blocks)
    assert [block.attn.layer_idx for block in model.visual.blocks] == [0, 1]
    tokens = torch.tensor([[1, 28, 29, 29, 29, 29, 31, 2]])
    with torch.no_grad():
        result = model(
            tokens,
            pixel_values=torch.randn(16, 12),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            use_cache=True,
        )
        assert result.logits.shape == (1, 8, 32)
        assert torch.isfinite(result.logits).all()
        video = model(
            torch.tensor([[1, 28, 30, 30, 31, 2]]),
            pixel_values_videos=torch.randn(8, 12),
            video_grid_thw=torch.tensor([[2, 2, 2]]),
            use_cache=False,
        )
        assert video.logits.shape == (1, 6, 32)
        assert torch.isfinite(video.logits).all()
        generated = model.generate(
            torch.tensor([[1, 2]]), max_new_tokens=2, do_sample=False, eos_token_id=None
        )
        assert generated.shape == (1, 4)
    model.save_pretrained(tmp_path)
    restored = model_cls.from_pretrained(tmp_path).eval()
    assert type(restored.config.text_config) is Flex_Qwen2_5_VLMoeTextConfig
    assert isinstance(restored.config.vision_config, vision_cls)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(torch.tensor([[1, 2]]), use_cache=False).logits,
            model(torch.tensor([[1, 2]]), use_cache=False).logits,
        )


@pytest.mark.parametrize("config_cls,attention_cls,model_cls", VARIANTS)
def test_dense_attention_option_uses_original_vision_attention(
    config_cls, attention_cls, model_cls
):
    config = _vision_config(config_cls, attention_num_experts=0)
    model = model_cls(config).eval()
    assert all(
        type(block.attn) is Flex_Qwen2_5_VLMoeVisionAttention for block in model.blocks
    )
    hidden = torch.randn(4, 16)
    cos, sin = _embeddings(1, 4)
    result = model.blocks[0].attn(
        hidden, torch.tensor([0, 4]), position_embeddings=(cos[0], sin[0])
    )
    assert result.shape == hidden.shape
