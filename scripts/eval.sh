olmes \
    --model medr1_fino1_qwen2_5-3x1_5b_router_posttune \
    --model-type hf \
    --model-args '{"max_length": 40960, "model_path": "/scratch1/duanm/flex_experts/text_only_warmup/medr1_fino1_qwen2_5-3x1_5b_router_posttune/fino1_qwen2_5-1_5b_expert_no_temp_router__scratch1_duanm_flex_experts_text_only_warmup_medr1_fino1_qwen2_5-3x1_5b__2026-03-23_15-47-57/final/", "chat_model": true}' \
    --task medmcqa:0shot_cot::zs_cot_r1 medqa:0shot_cot::zs_cot_r1 \
    --output-dir model-eval-v0/medr1_fino1_qwen2_5-3x1_5b_router_posttune \
    --random-subsample-seed 2026 \
    --limit 100 \
    --gpus 4 \
    --num-workers 4 \
    --batch-size 1

olmes \
    --model med_r1_qwen2_5-1_5b_unfreeze_attn \
    --model-type hf \
    --model-args '{"max_length": 40960, "model_path": "/scratch1/duanm/flex_experts/text_only_warmup/med_r1_qwen2_5-1_5b_unfreeze_attn/med_r1_qwen2_5-1_5b_expert_unfreeze_attn_expert1_experts_med_r1_qwen25_moe_2x1b_instruct_2026-03-23_16-20-36/final/", "chat_model": true}' \
    --task medmcqa:0shot_cot::zs_cot_r1 medqa:0shot_cot::zs_cot_r1 \
    --output-dir model-eval-v0/med_r1_qwen2_5-1_5b_unfreeze_attn \
    --random-subsample-seed 2026 \
    --limit 100 \
    --gpus 4 \
    --num-workers 4 \
    --batch-size 1

olmes \
    --model med_r1_qwen2_5-1_5b_unfreeze_embed \
    --model-type hf \
    --model-args '{"max_length": 40960, "model_path": "/scratch1/duanm/flex_experts/text_only_warmup/med_r1_qwen2_5-1_5b_unfreeze_embed/med_r1_qwen2_5-1_5b_expert_unfreeze_embed_expert1_experts_med_r1_qwen25_moe_2x1b_instruct_2026-03-23_20-35-28/final/", "chat_model": true}' \
    --task medmcqa:0shot_cot::zs_cot_r1 medqa:0shot_cot::zs_cot_r1 \
    --output-dir model-eval-v0/med_r1_qwen2_5-1_5b_unfreeze_embed \
    --random-subsample-seed 2026 \
    --limit 100 \
    --gpus 4 \
    --num-workers 4 \
    --batch-size 1
