"""Reference attention MoE forward paths, with sparse query-side computation.

Only vision attention is replaced; the text decoder is inherited unchanged.
Both variants use ordinary heads. K/V always cover every context token. The
head-expert variant packs independent (head, expert) histories into the head
axis of a DynamicCache entry, updating each attention layer exactly once.

Import the classes from modeling.flex_qwen2_5_vl_moe_attention.
Construct FlexQwen2_5VLSharedKVConfig or FlexQwen2_5VLHeadExpertConfig with a
vision_config containing attention_num_experts and attention_top_k, then use
its matching ForConditionalGeneration class. Feed-forward and connector MoE
settings retain their existing meanings. Router weights are softmax-normalized
over the selected top-k logits, separately for each head in the second variant.

These reference implementations support eager and SDPA backends. Normal vision
encoding is bidirectional and does not use a cache. For standalone incremental
attention, pass a DynamicCache to a vision attention module with layer_idx set;
each call must contain one context segment. Use a separate cache per segment
when processing several independent windows/images. Such a cache describes
that attention module's K/V history, rather than an incremental vision encoder.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.cache_utils import DynamicCache
from transformers.modeling_layers import GradientCheckpointingLayer

from .configuration_attention_moe import (
    FlexQwen2_5VLSharedKVConfig,
    FlexQwen2_5VLSharedKVVisionConfig,
    FlexQwen2_5VLHeadExpertConfig,
    FlexQwen2_5VLHeadExpertVisionConfig,
)
from ..flex_qwen2_5_vl_moe_connector.modeling_flex_qwen2_5_vl_moe import (
    Flex_Qwen2_5_VLMoeVisionBlock,
    Flex_Qwen2_5_VLMoeVisionAttention,
    Flex_Qwen2_5_VLMoeVisionTransformerPretrainedModel,
    Flex_Qwen2_5_VLMoeModel,
    Flex_Qwen2_5_VLMoeForConditionalGeneration,
    Flex_Qwen2_5_VLMoePreTrainedModel,
    Flex_Qwen2_5_VLMoeTextModel,
    Flex_Qwen2_5_VLMoeVisionPatchEmbed,
    Flex_Qwen2_5_VLMoeVisionRotaryEmbedding,
    Flex_Qwen2_5_VLMoePatchMerger,
    Flex_Qwen2_5_VLMoeRMSNorm,
    Flex_Qwen2_5_VLMoeSparseMoeBlock,
    Flex_Qwen2_5_VLMoeMLP,
    rotate_half,
)


class _QOExpert(nn.Module):
    def __init__(self, hidden_size, query_size, output_bias):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, query_size, bias=True)
        self.o_proj = nn.Linear(query_size, hidden_size, bias=output_bias)


class _HeadExpert(_QOExpert):
    def __init__(self, hidden_size, head_dim, output_bias):
        super().__init__(hidden_size, head_dim, output_bias)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, head_dim, bias=True)


def _route(router, hidden_states, top_k):
    logits, indices = router(hidden_states).float().topk(top_k, dim=-1)
    return indices, logits.softmax(dim=-1).to(hidden_states.dtype)


def _rotate(states, cos, sin):
    # states: [batch, heads, sequence, dim]; cos/sin: [batch, sequence, dim]
    rotated = (
        states.float() * cos[:, None].float()
        + rotate_half(states.float()) * sin[:, None].float()
    )
    return rotated.to(states.dtype)


class _AttentionMoe(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_causal = False
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_experts = config.attention_num_experts
        self.top_k = config.attention_top_k
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.backend = config._attn_implementation or "eager"
        if self.backend not in ("eager", "sdpa"):
            raise ValueError(
                "Attention MoE reference implementations support eager and sdpa attention"
            )

    def _attend(self, query, key, value, batch, token, mask, output_attentions):
        """Attend selected query rows to complete, unfiltered context histories."""
        key_length = key.shape[-2]
        if mask is not None and mask.ndim == 4:
            mask_batch = batch if mask.shape[0] != 1 else torch.zeros_like(batch)
            mask_token = token if mask.shape[-2] != 1 else torch.zeros_like(token)
            row_mask = mask[mask_batch, :, mask_token, :key_length]
            if row_mask.shape[1] != query.shape[1]:
                if row_mask.shape[1] != 1:
                    raise ValueError(
                        "Attention masks must broadcast across attention heads"
                    )
        else:
            row_mask = torch.ones(
                (len(token), 1, key_length), dtype=torch.bool, device=query.device
            )
            if mask is not None:
                if mask.ndim != 2:
                    raise ValueError("Expected a 2D padding mask or 4D attention mask")
                mask_batch = batch if mask.shape[0] != 1 else torch.zeros_like(batch)
                row_mask &= mask[mask_batch, None, :key_length].bool()

        result = torch.zeros_like(query)
        attention_weights = (
            query.new_zeros(query.shape[0], query.shape[1], key_length)
            if output_attentions
            else None
        )
        # Batch selected queries together; never duplicate a context K/V bank
        # for every selected token. Typical packed vision segments have batch=1.
        for batch_idx in batch.unique().tolist():
            rows = (batch == batch_idx).nonzero(as_tuple=True)[0]
            q = query[rows].transpose(0, 1).unsqueeze(0)
            k = key[batch_idx : batch_idx + 1]
            v = value[batch_idx : batch_idx + 1]
            selected_mask = row_mask[rows].transpose(0, 1).unsqueeze(0)
            if self.backend == "sdpa" and not output_attentions:
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=selected_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    scale=self.scaling,
                )
            else:
                scores = (q @ k.transpose(-2, -1)) * self.scaling
                if selected_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~selected_mask, float("-inf"))
                    valid_rows = selected_mask.any(dim=-1, keepdim=True)
                else:
                    scores = scores + selected_mask
                    # HF eager masks use the dtype's finite minimum for blocked entries.
                    blocked = torch.isneginf(selected_mask) | (
                        selected_mask <= torch.finfo(selected_mask.dtype).min
                    )
                    valid_rows = (~blocked).any(dim=-1, keepdim=True)
                # Entirely masked rows yield zeros, matching SDPA.
                scores = torch.where(valid_rows, scores, torch.zeros_like(scores))
                weights = (
                    scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
                    * valid_rows
                )
                weights = F.dropout(
                    weights, p=self.attention_dropout, training=self.training
                )
                context = weights @ v
                if attention_weights is not None:
                    attention_weights = attention_weights.index_copy(
                        0, rows, weights.squeeze(0).transpose(0, 1)
                    )
            result = result.index_copy(0, rows, context.squeeze(0).transpose(0, 1))
        return result, attention_weights

    def _update_cache(self, key, value, cache, cos, sin, cache_position):
        if cache is not None:
            if self.layer_idx is None:
                raise ValueError("An attention layer index is required when caching")
            key, value = cache.update(
                key,
                value,
                self.layer_idx,
                {
                    "cos": cos,
                    "sin": sin,
                    "cache_position": cache_position,
                },
            )
        return key, value

    def _accumulate(self, output, attentions, projected, weights, rows, head=None):
        output = output.index_add(0, rows, projected)
        if attentions is not None:
            if head is None:
                attentions = attentions.index_add(0, rows, weights)
            else:
                # Keep each head's independently normalized mixture in its own slot.
                head_weights = F.pad(weights, (0, 0, head, self.num_heads - head - 1))
                attentions = attentions.index_add(0, rows, head_weights)
        return output, attentions


class _SharedKVAttention(_AttentionMoe):
    """One shared ordinary multi-head K/V bank, with whole-layer Q/O experts."""

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.experts = nn.ModuleList(
            [
                _QOExpert(self.hidden_size, self.hidden_size, output_bias=True)
                for _ in range(self.num_experts)
            ]
        )

    def _forward_all_experts(self, hidden, cos, sin, valid_keys=None):
        """Batch fully routed Q/O experts while computing shared K/V once."""
        batch_size, length, _ = hidden.shape
        query = F.linear(
            hidden,
            torch.cat([expert.q_proj.weight for expert in self.experts], dim=0),
            torch.cat([expert.q_proj.bias for expert in self.experts], dim=0),
        ).reshape(batch_size, length, self.num_experts * self.num_heads, self.head_dim)
        query = _rotate(query.transpose(1, 2), cos, sin)
        key = self.k_proj(hidden).reshape(
            batch_size, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden).reshape(
            batch_size, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = _rotate(key, cos, sin)
        key = key[:, None].expand(-1, self.num_experts, -1, -1, -1).reshape(
            batch_size, self.num_experts * self.num_heads, length, self.head_dim
        )
        value = value[:, None].expand(-1, self.num_experts, -1, -1, -1).reshape_as(key)
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_keys[:, None, None, :] if valid_keys is not None else None,
            dropout_p=0.0,
            scale=self.scaling,
        )
        routing = self.router(hidden).float().softmax(dim=-1).to(hidden.dtype)
        weighted_context = (
            context.transpose(1, 2).reshape(
                batch_size, length, self.num_experts, self.hidden_size
            ) * routing[..., None]
        ).reshape(batch_size, length, self.num_experts * self.hidden_size)
        output_weights = torch.cat([expert.o_proj.weight for expert in self.experts], dim=1)
        output_biases = torch.stack([expert.o_proj.bias for expert in self.experts])
        output = F.linear(weighted_context, output_weights)
        output = output + F.linear(routing, output_biases.transpose(0, 1))
        return output, None

    def _forward(
        self,
        hidden,
        cos,
        sin,
        mask=None,
        cache=None,
        cache_position=None,
        output_attentions=False,
    ):
        batch_size, length, _ = hidden.shape
        if (
            self.backend == "sdpa"
            and self.top_k == self.num_experts
            and cache is None
            and not output_attentions
            and length <= 256
            and (mask is None or mask.ndim == 2)
        ):
            return self._forward_all_experts(hidden, cos, sin, mask)
        key = (
            self.k_proj(hidden)
            .view(batch_size, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden)
            .view(batch_size, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = _rotate(key, cos, sin)
        key, value = self._update_cache(key, value, cache, cos, sin, cache_position)
        selected, routing = _route(self.router, hidden, self.top_k)
        output = hidden.new_zeros(batch_size * length, self.hidden_size)
        attentions = (
            hidden.new_zeros(batch_size * length, self.num_heads, key.shape[-2])
            if output_attentions
            else None
        )
        for expert_idx, expert in enumerate(self.experts):
            batch, token, slot = (selected == expert_idx).nonzero(as_tuple=True)
            if batch.numel() == 0:
                continue
            query = expert.q_proj(hidden[batch, token]).view(
                -1, self.num_heads, self.head_dim
            )
            query = (
                query.float() * cos[batch, token, None].float()
                + rotate_half(query.float()) * sin[batch, token, None].float()
            ).to(query.dtype)
            context, weights = self._attend(
                query, key, value, batch, token, mask, output_attentions
            )
            routing_weight = routing[batch, token, slot, None]
            projected = (
                expert.o_proj(context.reshape(-1, self.hidden_size)) * routing_weight
            )
            if weights is not None:
                weights = weights * routing_weight[:, :, None]
            output, attentions = self._accumulate(
                output, attentions, projected, weights, batch * length + token
            )
        output = output.view(batch_size, length, self.hidden_size)
        if attentions is not None:
            attentions = attentions.view(
                batch_size, length, self.num_heads, -1
            ).transpose(1, 2)
        return output, attentions


class _HeadExpertAttention(_AttentionMoe):
    """Independent matched Q/K/V/O alternatives and one linear router per head."""

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.routers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.num_experts, bias=False)
                for _ in range(self.num_heads)
            ]
        )
        self.experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _HeadExpert(self.hidden_size, self.head_dim, output_bias=True)
                        for _ in range(self.num_experts)
                    ]
                )
                for _ in range(self.num_heads)
            ]
        )

    def _forward_all_experts(self, hidden, cos, sin, valid_keys=None):
        """Run fully selected head experts as one batched SDPA operation."""
        batch_size, length, _ = hidden.shape
        alternatives = [expert for bank in self.experts for expert in bank]
        projections = [
            getattr(expert, name)
            for name in ("q_proj", "k_proj", "v_proj")
            for expert in alternatives
        ]
        qkv = F.linear(
            hidden,
            torch.cat([projection.weight for projection in projections], dim=0),
            torch.cat([projection.bias for projection in projections], dim=0),
        ).reshape(batch_size, length, 3, self.num_heads * self.num_experts, self.head_dim)
        query, key, value = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        query = _rotate(query, cos, sin)
        key = _rotate(key, cos, sin)
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_keys[:, None, None, :] if valid_keys is not None else None,
            dropout_p=0.0,
            scale=self.scaling,
        )
        router_weights = torch.cat([router.weight for router in self.routers], dim=0)
        routing = F.linear(hidden, router_weights).reshape(
            batch_size, length, self.num_heads, self.num_experts
        ).float().softmax(dim=-1).to(hidden.dtype)
        weighted_context = (
            context.transpose(1, 2).reshape(
                batch_size, length, self.num_heads, self.num_experts, self.head_dim
            ) * routing[..., None]
        ).reshape(batch_size, length, -1)
        output_weights = torch.cat([expert.o_proj.weight for expert in alternatives], dim=1)
        output_biases = torch.stack(
            [expert.o_proj.bias for expert in alternatives]
        ).reshape(self.num_heads, self.num_experts, self.hidden_size)
        output = F.linear(weighted_context, output_weights)
        output = output + torch.einsum("bshe,hei->bsi", routing, output_biases)
        return output, None

    def _forward(
        self,
        hidden,
        cos,
        sin,
        mask=None,
        cache=None,
        cache_position=None,
        output_attentions=False,
    ):
        batch_size, length, _ = hidden.shape
        if (
            self.backend == "sdpa"
            and self.top_k == self.num_experts
            and cache is None
            and not output_attentions
            and length <= 256
            and (mask is None or mask.ndim == 2)
        ):
            return self._forward_all_experts(hidden, cos, sin, mask)
        if cache is not None and not isinstance(cache, DynamicCache):
            raise ValueError(
                "Head-expert attention requires DynamicCache for its independent expert histories"
            )
        # Compute all alternatives' K/V, even if no query currently selects them.
        alternatives = [expert for bank in self.experts for expert in bank]
        key = torch.stack([expert.k_proj(hidden) for expert in alternatives], dim=1)
        value = torch.stack([expert.v_proj(hidden) for expert in alternatives], dim=1)
        key = _rotate(key, cos, sin)
        key, value = self._update_cache(key, value, cache, cos, sin, cache_position)
        key_length = key.shape[-2]
        key = key.view(
            batch_size, self.num_heads, self.num_experts, key_length, self.head_dim
        )
        value = value.view_as(key)
        output = hidden.new_zeros(batch_size * length, self.hidden_size)
        attentions = (
            hidden.new_zeros(batch_size * length, self.num_heads, key_length)
            if output_attentions
            else None
        )
        for head, (router, bank) in enumerate(zip(self.routers, self.experts)):
            selected, routing = _route(router, hidden, self.top_k)
            for expert_idx, expert in enumerate(bank):
                batch, token, slot = (selected == expert_idx).nonzero(as_tuple=True)
                if batch.numel() == 0:
                    continue
                query = expert.q_proj(hidden[batch, token]).unsqueeze(1)
                query = (
                    query.float() * cos[batch, token, None].float()
                    + rotate_half(query.float()) * sin[batch, token, None].float()
                ).to(query.dtype)
                context, weights = self._attend(
                    query,
                    key[:, head, expert_idx, None],
                    value[:, head, expert_idx, None],
                    batch,
                    token,
                    mask[:, head : head + 1]
                    if mask is not None
                    and mask.ndim == 4
                    and mask.shape[1] == self.num_heads
                    else mask,
                    output_attentions,
                )
                routing_weight = routing[batch, token, slot, None]
                projected = expert.o_proj(context.squeeze(1)) * routing_weight
                if weights is not None:
                    weights = weights * routing_weight[:, :, None]
                output, attentions = self._accumulate(
                    output,
                    attentions,
                    projected,
                    weights,
                    batch * length + token,
                    head=head,
                )
        output = output.view(batch_size, length, self.hidden_size)
        if attentions is not None:
            attentions = attentions.view(
                batch_size, length, self.num_heads, key_length
            ).transpose(1, 2)
        return output, attentions


class _VisionAttentionMixin:
    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        if position_embeddings is None:
            if rotary_pos_emb is None:
                raise ValueError("Vision attention requires rotary position embeddings")
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        else:
            cos, sin = position_embeddings
        outputs = []
        # Preserve the original packed windows/full-attention segments exactly.
        boundaries = cu_seqlens.tolist()
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != hidden_states.shape[0]
            or any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:]))
        ):
            raise ValueError(
                "cu_seqlens must partition the vision tokens into nonempty segments"
            )
        if past_key_value is not None and len(boundaries) != 2:
            raise ValueError(
                "Cached vision attention requires a single segment per cache"
            )
        lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
        if (
            len(lengths) > 1
            and max(lengths) <= 256
            and isinstance(self, (_SharedKVAttention, _HeadExpertAttention))
            and self.backend == "sdpa"
            and self.top_k == self.num_experts
            and past_key_value is None
        ):
            slices = [slice(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
            padded_hidden = pad_sequence([hidden_states[part] for part in slices], batch_first=True)
            padded_cos = pad_sequence([cos[part] for part in slices], batch_first=True)
            padded_sin = pad_sequence([sin[part] for part in slices], batch_first=True)
            valid_keys = torch.arange(max(lengths), device=hidden_states.device)[None, :] < torch.tensor(
                lengths, device=hidden_states.device
            )[:, None]
            output, _ = self._forward_all_experts(
                padded_hidden, padded_cos, padded_sin, valid_keys
            )
            return torch.cat([row[:length] for row, length in zip(output, lengths)], dim=0)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            output, _ = self._forward(
                hidden_states[None, start:end],
                cos[None, start:end],
                sin[None, start:end],
                cache=past_key_value,
                cache_position=cache_position,
            )
            outputs.append(output.squeeze(0))
        return torch.cat(outputs, dim=0)


class FlexQwen2_5VLSharedKVVisionAttention(_VisionAttentionMixin, _SharedKVAttention):
    """Packed vision attention with shared K/V and whole-attention Q/O experts."""


class FlexQwen2_5VLHeadExpertVisionAttention(
    _VisionAttentionMixin, _HeadExpertAttention
):
    """Packed vision attention with independently routed Q/K/V/O experts per head."""


class _VisionMoeBlockMixin:
    """Build an MoE vision block without constructing a dense attention module."""

    def __init__(self, config, attn_implementation="sdpa"):
        GradientCheckpointingLayer.__init__(self)
        self.norm1 = Flex_Qwen2_5_VLMoeRMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Flex_Qwen2_5_VLMoeRMSNorm(config.hidden_size, eps=1e-6)
        attention_class = (
            self.attention_class
            if config.attention_num_experts > 0
            else Flex_Qwen2_5_VLMoeVisionAttention
        )
        self.attn = attention_class(config=config)
        if config.num_experts > 0:
            self.mlp = Flex_Qwen2_5_VLMoeSparseMoeBlock(config, True)
        else:
            self.mlp = Flex_Qwen2_5_VLMoeMLP(config, bias=True)


class _SharedKVVisionBlock(_VisionMoeBlockMixin, Flex_Qwen2_5_VLMoeVisionBlock):
    attention_class = FlexQwen2_5VLSharedKVVisionAttention


class _HeadExpertVisionBlock(_VisionMoeBlockMixin, Flex_Qwen2_5_VLMoeVisionBlock):
    attention_class = FlexQwen2_5VLHeadExpertVisionAttention


class _AttentionMoeModelMixin:
    _supports_flash_attn = False
    _can_compile_fullgraph = False
    _no_split_modules = [
        "Flex_Qwen2_5_VLMoeDecoderLayer",
        "_SharedKVVisionBlock",
        "_HeadExpertVisionBlock",
    ]


class _VisionModelMixin:
    """Reuse the original vision forward while constructing local MoE blocks."""

    def __init__(self, config, *args, **kwargs):
        Flex_Qwen2_5_VLMoePreTrainedModel.__init__(self, config, *args, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = Flex_Qwen2_5_VLMoeVisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Flex_Qwen2_5_VLMoeVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [self.block_class(config) for _ in range(config.depth)]
        )
        for layer_idx, block in enumerate(self.blocks):
            block.attn.layer_idx = layer_idx
        self.merger = Flex_Qwen2_5_VLMoePatchMerger(config)
        self.gradient_checkpointing = False


class FlexQwen2_5VLSharedKVVisionModel(
    _VisionModelMixin,
    _AttentionMoeModelMixin,
    Flex_Qwen2_5_VLMoeVisionTransformerPretrainedModel,
):
    config_class = FlexQwen2_5VLSharedKVVisionConfig
    block_class = _SharedKVVisionBlock


class FlexQwen2_5VLHeadExpertVisionModel(
    _VisionModelMixin,
    _AttentionMoeModelMixin,
    Flex_Qwen2_5_VLMoeVisionTransformerPretrainedModel,
):
    config_class = FlexQwen2_5VLHeadExpertVisionConfig
    block_class = _HeadExpertVisionBlock


class _CompositeMoeModelMixin:
    """Keep the original multimodal forward and its unchanged text model."""

    def __init__(self, config):
        Flex_Qwen2_5_VLMoePreTrainedModel.__init__(self, config)
        self.visual = self.vision_model_class._from_config(config.vision_config)
        self.language_model = Flex_Qwen2_5_VLMoeTextModel._from_config(
            config.text_config
        )
        self.rope_deltas = None
        self.post_init()


class FlexQwen2_5VLSharedKVModel(
    _CompositeMoeModelMixin,
    _AttentionMoeModelMixin,
    Flex_Qwen2_5_VLMoeModel,
):
    config_class = FlexQwen2_5VLSharedKVConfig
    vision_model_class = FlexQwen2_5VLSharedKVVisionModel


class FlexQwen2_5VLHeadExpertModel(
    _CompositeMoeModelMixin,
    _AttentionMoeModelMixin,
    Flex_Qwen2_5_VLMoeModel,
):
    config_class = FlexQwen2_5VLHeadExpertConfig
    vision_model_class = FlexQwen2_5VLHeadExpertVisionModel


class _ConditionalMoeModelMixin:
    def __init__(self, config):
        Flex_Qwen2_5_VLMoePreTrainedModel.__init__(self, config)
        self.model = self.model_class(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()


class FlexQwen2_5VLSharedKVForConditionalGeneration(
    _ConditionalMoeModelMixin,
    _AttentionMoeModelMixin,
    Flex_Qwen2_5_VLMoeForConditionalGeneration,
):
    config_class = FlexQwen2_5VLSharedKVConfig
    model_class = FlexQwen2_5VLSharedKVModel


class FlexQwen2_5VLHeadExpertForConditionalGeneration(
    _ConditionalMoeModelMixin,
    _AttentionMoeModelMixin,
    Flex_Qwen2_5_VLMoeForConditionalGeneration,
):
    config_class = FlexQwen2_5VLHeadExpertConfig
    model_class = FlexQwen2_5VLHeadExpertModel


__all__ = [
    "FlexQwen2_5VLSharedKVVisionAttention",
    "FlexQwen2_5VLHeadExpertVisionAttention",
    "FlexQwen2_5VLSharedKVVisionModel",
    "FlexQwen2_5VLHeadExpertVisionModel",
    "FlexQwen2_5VLSharedKVModel",
    "FlexQwen2_5VLHeadExpertModel",
    "FlexQwen2_5VLSharedKVForConditionalGeneration",
    "FlexQwen2_5VLHeadExpertForConditionalGeneration",
]
