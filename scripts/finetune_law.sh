#!/bin/bash
OUTPUT_DIR="/scratch1/duanm/flex_experts/"

model_store=experts
datastore=traces/text_only_warmup/
ds_config=train/ds_config/v0.json
run_seed=2026
num_gpus=4

echo Training lawinstruct expert with flexolmo
# TODO: setting master port for now since I'm running another deepspeed job
#deepspeed --master_port=29501 --num_gpus=$num_gpus 
PYTHONPATH=. deepspeed --master_port=29501 --num_gpus=$num_gpus train/finetune.py \
    --run_id "lawinstruct_qwen2_5-1_5b_expert" \
    --model $model_store/lawinstruct_qwen25_moe_2x1_5b_instruct \
    --datasets "$datastore/lawinstruct/lawinstruct_traces.jsonl" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --max_length 25000 \
    --run_seed $run_seed \
    --run_output_dir "${OUTPUT_DIR}/text_only_warmup/lawinstruct_qwen2_5-1_5b/" \
    --save_n_epochs 1 \
    --dataset_num_proc 2 \
    --skip_eval \
    --train_expert_idx 1 \
    --packing \
    --deepspeed "$ds_config"

# echo Training lawinstruct expert
PYTHONPATH=. deepspeed --master_port=29501 --num_gpus=$num_gpus train/finetune.py \
    --run_id "lawinstruct_qwen2_5-1_5b_independent" \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --datasets "$datastore/lawinstruct/lawinstruct_traces.jsonl" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --max_length 25000 \
    --run_seed $run_seed \
    --run_output_dir "${OUTPUT_DIR}/text_only_warmup/lawinstruct_qwen2_5-1_5b_independent/" \
    --save_n_epochs 1 \
    --dataset_num_proc 2 \
    --deepspeed "$ds_config" \
    --skip_eval \
    --packing