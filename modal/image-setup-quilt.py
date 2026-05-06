from pathlib import Path

import modal
import shlex
import subprocess

LOCAL_FLEXBAGEL = Path(__file__).parent.parent.parent / "FlexBagel"

COMMAND = """
cd /FlexBagel && PYTHONPATH=. deepspeed --master_port=29501 --num_gpus=4 train/mm_tune_mt.py \
    --run_id "quilt_qwen2_5-3b-vl-test" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets /mnt/quilt/train.jsonl \
    --sample_size 100000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --max_seq_length 4096 \
    --run_seed 42 \
    --run_output_dir "/output/quilt/" \
    --save_n_epochs 1 \
    --dataset_num_proc 6 \
    --skip_eval \
    --deepspeed "train/ds_config/v0.json"
"""

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
        .uv_pip_install(requirements=["requirements.txt"], gpu="H200") \
        .env({"HF_HOME": "/hf-cache"}) \
        .uv_pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl", extra_options="--no-build-isolation") \
        .uv_pip_install("datasets>=2.14.6", "fsspec") \
        .add_local_dir(LOCAL_FLEXBAGEL, remote_path="/FlexBagel", copy=True)

vol_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vol_data = modal.Volume.from_name("quilt", create_if_missing=False)
vol_output = modal.Volume.from_name("output", create_if_missing=True)

MODEL = "H200"
NUM_GPUS = 4
GPU_TYPE = f"{MODEL}:{NUM_GPUS}"
TIMEOUT_HOURS = 24
@app.function(image=image, 
    volumes={"/hf-cache": vol_hf_cache, "/mnt/quilt": vol_data, "/output": vol_output}, 
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
    # run_cli("git clone https://github.com/michaelduan8/FlexBagel.git")
    run_cli(COMMAND, use_shell=True)


@app.function(image=image, 
    volumes={"/hf-cache": vol_hf_cache, "/mnt/quilt": vol_data, "/output": vol_output}, 
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")], 
    gpu="H200:1",
    timeout=int(TIMEOUT_HOURS * 60 * 60))
def test():
    import torch, sys, flash_attn
    print("torch:", torch.__version__)
    print("python:", sys.version)
    print("torch cuda:", torch.version.cuda)
    print("cxx11 abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
    print("flash_attn:", flash_attn.__version__)

@app.local_entrypoint()
def main():
    train.remote()