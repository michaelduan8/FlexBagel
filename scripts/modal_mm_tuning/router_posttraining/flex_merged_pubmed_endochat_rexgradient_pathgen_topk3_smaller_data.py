from pathlib import Path

import os
import modal
import shlex
import subprocess

LOCAL_FLEXBAGEL = "."

MODEL = "H200"
NUM_GPUS = 4

#/mnt/surg390k/surg_390k_train_w_length_w_path.jsonl \
# 50000 
# --unfreeze_non_ffn \
COMMAND = f"""
    cd /FlexBagel && PYTHONPATH=. torchrun --master_port=29501 --nproc_per_node={NUM_GPUS} train/mm_tune.py \
    --run_id "flex_merged_pub_endo_qwen2_5-3b-vl-flex-topk2-router_posttuning_topk3_smaller_tune_dataset_25k" \
    --model alrope/pub_endo_combined_qwen2_5-3b-vl-flex-topk2 \
    --datasets /mnt/finevision/finevision_train_100k_w_length_w_path.jsonl /mnt/pubmedvision/pubmed_vision_it_train_w_length_w_path.jsonl /mnt/surg390k/surg_390k_train_w_length_w_path.jsonl /mnt/pathgen/pathgen_train_single_turn_w_length_memory_w_path.jsonl \
    --sample_size 25000 25000 25000 25000 25000 \
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
    --run_output_dir "/output/router_posttraining/" \
    --save_n_epochs 0.2 \
    --dataset_num_proc 6 \
    --skip_eval \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --delete_intermediate_checkpoints false \
    --num_experts_per_tok 5
"""
# --unfreeze_non_ffn \

def run_cli(command: str, use_shell=False):
    if use_shell:
        return subprocess.run(command, shell=True)
    else:
        return subprocess.run(shlex.split(command))

app = modal.App("image-setup-test")
nvidia_image= modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
)
image = nvidia_image.apt_install("git", "libgl1", "libglib2.0-0") \
        .run_commands("git clone https://github.com/michaelduan8/FlexBagel.git && cd FlexBagel && git checkout main") \
        .uv_pip_install(requirements=["requirements_new.txt"], gpu=MODEL) \
        .env({"HF_HOME": "/hf-cache"}) \
        .uv_pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl", extra_options="--no-build-isolation") \
        .uv_pip_install("datasets>=2.14.6", "fsspec") \
        .uv_pip_install("trl==0.22.2", "transformers==4.57.3") \
        .add_local_dir(os.path.join(LOCAL_FLEXBAGEL, "train"), remote_path="/FlexBagel/train", copy=True) \
        .add_local_dir(os.path.join(LOCAL_FLEXBAGEL, "modeling"), remote_path="/FlexBagel/modeling", copy=True) \

vol_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vol_pubmed_data = modal.Volume.from_name("pubmedvision", create_if_missing=False)
vol_endochat_data = modal.Volume.from_name("surg390k", create_if_missing=False)
vol_pathgen_data = modal.Volume.from_name("pathgen", create_if_missing=False)
vol_finevision_data = modal.Volume.from_name("finevision", create_if_missing=False)
vol_output = modal.Volume.from_name("output", create_if_missing=True)

GPU_TYPE = f"{MODEL}:{NUM_GPUS}"
TIMEOUT_HOURS = 24
@app.function(image=image, 
    volumes={"/hf-cache": vol_hf_cache, "/mnt/pubmedvision": vol_pubmed_data, "/mnt/surg390k": vol_endochat_data, "/mnt/pathgen": vol_pathgen_data, "/mnt/finevision": vol_finevision_data, "/output": vol_output}, 
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")], 
    gpu=GPU_TYPE,
    timeout=int(TIMEOUT_HOURS * 60 * 60))
def train():
    import shlex
    import subprocess
    import wandb
    import os
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run_cli("pip list")
    run_cli(COMMAND, use_shell=True)


@app.local_entrypoint()
def main():
    train.spawn()