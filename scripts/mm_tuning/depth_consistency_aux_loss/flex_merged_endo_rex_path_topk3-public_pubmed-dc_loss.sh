#!/bin/bash
OUTPUT_DIR="/scratch1/duanm/flex_experts/"

model_store=experts
datastore=traces/multimodal
ds_config=train/ds_config/v0.json
run_seed=42
num_gpus=4

for sample_size in 25000; do
    echo "Running training with sample_size=$sample_size, frozen non-ffn"
    PYTHONPATH=. torchrun --master_port=29501 --nproc_per_node=$num_gpus train/mm_tune.py \
        --run_id "flex_merged_endo_rex_path_qwen2_5-3b-vl-public_pubmed-rp_topk4_$sample_size-dc_loss" \
        --model alrope/endo_rex_pathgen_combined_qwen2_5-3b-vl-flex-from_pub_untrained \
        --datasets "$datastore/pubmed_vision/pubmed_vision_it_train_w_length.jsonl" "$datastore/surg_390k/surg_390k_train_w_length.jsonl" "$datastore/rexgradient/rexgradient_train_w_length.jsonl" "$datastore/pathgen/pathgen_train_single_turn_w_length_memory.jsonl"  \
        --sample_size $sample_size $sample_size $sample_size $sample_size \
        --router_tuning_only \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 6 \
        --logging_steps 10 \
        --learning_rate 2e-5 \
        --lr_vision 5e-6 \
        --lr_llm 1e-5 \
        --lr_connector 5e-6 \
        --warmup_ratio 0.1 \
        --gradient_checkpointing True \
        --run_seed 42 \
        --run_output_dir "/scratch1/duanm/mm_flexolmo/models" \
        --save_n_epochs 1 \
        --dataset_num_proc 6 \
        --skip_eval \
        --dataloader_num_workers 6 \
        --dataloader_persistent_workers True \
        --dataloader_prefetch_factor 2 \
        --dataloader_pin_memory True \
        --delete_intermediate_checkpoints false \
        --num_experts_per_tok 4 \
        --output_router_logits \
        --router_depth_aux_loss_coef 0.01
done
    # --deepspeed train/ds_config/v0.json
