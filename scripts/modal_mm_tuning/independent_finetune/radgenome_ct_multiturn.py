from pathlib import Path

import os
import modal
import shlex
import subprocess

LOCAL_FLEXBAGEL = "."

MODEL = "H200"
NUM_GPUS = 4

# RadGenome-ChestCT (CT-RATE volumes) 3D CT expert, multi-turn: one row per volume (24,128 train rows), the whole
# chest CT as one 64-frame slab video (336px, 3-window RGB: wide / soft tissue / lung), all of the volume's Q/A as
# turns: short questions first (presence, abnormality, location, size), then region findings, disorder list and
# the full findings report last. --supervise_all_assistant_turns puts the loss on every answer (+ <|im_end|>).
# Data: data_processing/prepare/radgenome_chestct.py --multiturn; frames on the private Modal volume `radgenome`
# (CT-RATE terms: no redistribution of derived data). About 125M tokens/epoch, sequences up to 6.4k tokens.
COMMAND = f"""
    cd /FlexBagel && PYTHONPATH=. torchrun --master_port=29501 --nproc_per_node={NUM_GPUS} train/mm_tune.py \
    --run_id "radgenome_ct_multiturn_qwen2_5-3b-vl" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets /mnt/radgenome/radgenome_train_video_multiturn_modal.jsonl \
    --supervise_all_assistant_turns True \
    --default_video_fps 1.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 12 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing True \
    --run_seed 42 \
    --run_output_dir "/output/radgenome/" \
    --save_n_epochs 0.2 \
    --dataset_num_proc 6 \
    --skip_eval \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --delete_intermediate_checkpoints false
"""

def run_cli(command: str, use_shell=False):
    if use_shell:
        return subprocess.run(command, shell=True)
    else:
        return subprocess.run(shlex.split(command))

app = modal.App("radgenome-ct-multiturn")
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
vol_data = modal.Volume.from_name("radgenome", create_if_missing=False)
vol_output = modal.Volume.from_name("output", create_if_missing=True)

GPU_TYPE = f"{MODEL}:{NUM_GPUS}"
TIMEOUT_HOURS = 24
@app.function(image=image,
    volumes={"/hf-cache": vol_hf_cache, "/mnt/radgenome": vol_data, "/output": vol_output},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    gpu=GPU_TYPE,
    cpu=24,
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
