#!/bin/bash
OUTPUT_DIR="/scratch1/duanm/flex_experts/"

model_store=experts
datastore=traces/multimodal/pathgen
ds_config=train/ds_config/v0.json
run_seed=42
num_gpus=4

PYTHONPATH=. torchrun --master_port=29501 --nproc_per_node=$num_gpus train/mm_tune.py \
    --run_id "pathgen_qwen2_5-3b-vl" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets "$datastore/pathgen_train_single_turn_w_length_memory.jsonl" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 6 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing True \
    --run_seed 42 \
    --run_output_dir "/scratch1/duanm/mm_flexolmo/models" \
    --save_n_epochs 0.2 \
    --dataset_num_proc 6 \
    --skip_eval \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --delete_intermediate_checkpoints false
    # --deepspeed train/ds_config/v0.json
