#!/bin/bash
OUTPUT_DIR="/scratch1/duanm/flex_experts/"

model_store=experts
datastore=traces/text_only_warmup/
ds_config=train/ds_config/v0.json
run_seed=42
num_gpus=4

echo Training pubmed vision qwen2.5 vl instruct
# TODO: setting master port for now since I'm running another deepspeed job
# deepspeed --master_port=29501 --num_gpus=$num_gpus
# torchrun --master_port=29501 --nproc_per_node=$num_gpus 
PYTHONPATH=. torchrun --master_port=29501 --nproc_per_node=$num_gpus train/mm_tune.py \
    --run_id "pubmed_vision_qwen2_5-3b-vl_240k" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets FreedomIntelligence/PubMedVision FreedomIntelligence/PubMedVision \
    --dataset_args '{"subset": "PubMedVision_Alignment_VQA"}' '{"subset": "PubMedVision_InstructionTuning_VQA"}' \
    --sample_size 120000 120000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --max_length 8192 \
    --run_seed $run_seed \
    --run_output_dir "${OUTPUT_DIR}/pubmed_vision/" \
    --save_n_epochs 1 \
    --dataset_num_proc 6 \
    --skip_eval
    # --deepspeed "$ds_config"
