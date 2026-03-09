#!/bin/bash
OUTPUT_DIR="/scratch1/duanm/flex_experts/"

model_store=experts
datastore=traces/text_only_warmup/
ds_config=train/ds_config/v0.json
run_seed=2026
num_gpus=3

echo Training med_r1 expert
    # if ls ${OUTPUT_DIR}/qwq_qwen25-1.5b_if_shadows/test_v0_if_qwq_qwen25_shadow_${i}_* 1>/dev/null 2>&1; then
    #     echo "Skipping inst-follow qwq v0 Shadow $i (already exists)"
    #     continue
    # fi
# TODO: setting master port for now since I'm running another deepspeed job
PYTHONPATH=. deepspeed --master_port=29501 --num_gpus=$num_gpus train/finetune.py \
    --run_id "med_r1_qwen2_5-1_5b_expert" \
    --model $model_store/med_r1_qwen25_moe_2x1b_instruct \
    --datasets "$datastore/med_r1/med_r1_traces.jsonl" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --max_length 20000 \
    --run_seed $run_seed \
    --run_output_dir "${OUTPUT_DIR}/text_only_warmup/med_r1_qwen2_5-1_5b/" \
    --save_n_epochs 1 \
    --dataset_num_proc 6 \
    --skip_eval \
    --train_expert_idx 1 \
    --deepspeed "$ds_config"

echo Training fino1 expert
PYTHONPATH=. deepspeed --master_port=29501 --num_gpus=$num_gpus train/finetune.py \
    --run_id "fino1_qwen2_5-1_5b_expert" \
    --model $model_store/fino1_qwen25_moe_2x1b_instruct \
    --datasets "$datastore/fino1_finqa/fino1_finqa_traces.jsonl" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --max_length 20000 \
    --run_seed $run_seed \
    --run_output_dir "${OUTPUT_DIR}/text_only_warmup/fino1_qwen2_5-1_5b/" \
    --save_n_epochs 1 \
    --dataset_num_proc 6 \
    --deepspeed "$ds_config" \
    --skip_eval \
    --train_expert_idx 1