"""Configurations for vision-only attention MoE variants of the connector model."""

from ..flex_qwen2_5_vl_moe_connector.configuration_flex_qwen2_5_vl_moe import (
    Flex_Qwen2_5_VLMoeConfig,
    Flex_Qwen2_5_VLMoeTextConfig,
    Flex_Qwen2_5_VLMoeVisionConfig,
)


class _VisionAttentionMoeConfigMixin:
    def __init__(self, attention_num_experts=4, attention_top_k=2, **kwargs):
        if (
            isinstance(attention_num_experts, bool)
            or not isinstance(attention_num_experts, int)
            or attention_num_experts < 0
        ):
            raise ValueError("attention_num_experts must be a nonnegative integer")
        if (
            isinstance(attention_top_k, bool)
            or not isinstance(attention_top_k, int)
            or attention_top_k < 1
        ):
            raise ValueError("attention_top_k must be a positive integer")
        if attention_num_experts and attention_top_k > attention_num_experts:
            raise ValueError("attention_top_k must not exceed attention_num_experts")
        super().__init__(**kwargs)
        if self.num_heads < 1 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (self.hidden_size // self.num_heads) % 4:
            raise ValueError(
                "Vision head dimensions must be divisible by four for 2D rotary embeddings"
            )
        self.attention_num_experts = attention_num_experts
        self.attention_top_k = attention_top_k


class FlexQwen2_5VLSharedKVVisionConfig(
    _VisionAttentionMoeConfigMixin, Flex_Qwen2_5_VLMoeVisionConfig
):
    model_type = "flex_qwen2_5_vl_shared_kv_vision"


class FlexQwen2_5VLHeadExpertVisionConfig(
    _VisionAttentionMoeConfigMixin, Flex_Qwen2_5_VLMoeVisionConfig
):
    model_type = "flex_qwen2_5_vl_head_expert_vision"


class _VisionAttentionMoeCompositeConfig(Flex_Qwen2_5_VLMoeConfig):
    def __init__(self, text_config=None, vision_config=None, **kwargs):
        # The parent handles dictionaries but does not assign config objects.
        if text_config is not None and not isinstance(text_config, dict):
            text_config = text_config.to_dict()
        if vision_config is not None and not isinstance(vision_config, dict):
            vision_config = vision_config.to_dict()
        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)


class FlexQwen2_5VLSharedKVConfig(_VisionAttentionMoeCompositeConfig):
    model_type = "flex_qwen2_5_vl_shared_kv"
    sub_configs = {
        "text_config": Flex_Qwen2_5_VLMoeTextConfig,
        "vision_config": FlexQwen2_5VLSharedKVVisionConfig,
    }


class FlexQwen2_5VLHeadExpertConfig(_VisionAttentionMoeCompositeConfig):
    model_type = "flex_qwen2_5_vl_head_expert"
    sub_configs = {
        "text_config": Flex_Qwen2_5_VLMoeTextConfig,
        "vision_config": FlexQwen2_5VLHeadExpertVisionConfig,
    }


__all__ = [
    "FlexQwen2_5VLSharedKVConfig",
    "FlexQwen2_5VLSharedKVVisionConfig",
    "FlexQwen2_5VLHeadExpertConfig",
    "FlexQwen2_5VLHeadExpertVisionConfig",
]
