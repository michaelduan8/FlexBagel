from pathlib import Path

import os
import modal
import shlex
import subprocess

LOCAL_FLEXBAGEL = "."

MODEL = "H200"
NUM_GPUS = 4

# MedThinkVQA expert (HF bio-nlp-umass/MedThinkVQA, Eurorad teaching cases, CC BY-NC-SA 4.0: research only). Independent
# single-turn rows from data_processing/prepare/medthinkvqa.py (official train split minus a 5% by-case val; the 720
# test cases are never used):
#   caption    one image -> its figure legend
#   findings   all case images (<=24) + clinical history -> imaging findings (figure references removed)
#   diagnosis  all case images (<=24) + clinical history -> the diagnosis text (open-ended, no options)
# Variant (env MEDTHINKVQA_VARIANT, default "all"):
#   all  57,700 rows (44,348 caption / 6,673 findings / 6,679 diagnosis), ~46M tokens/epoch, ~1.7-2 h
#   ct   17,619 rows (16,007 CT caption / 805 findings / 807 diagnosis on CT-only cases), ~10M tokens/epoch, ~30 min
# Images (~480x480 JPEG, ~300 tokens each) are on the private Modal volume `medthinkvqa`, mounted at /mnt/medthinkvqa.
# Memory: --use_liger_kernel computes the loss without materializing [tokens x vocab] logits, and --group_by_length
# (rows carry an estimated `length`) batches similar-length rows so short caption rows are not padded to long ones.
#
# Launch from the FlexBagel root (the image copies the local train/ and modeling/), always with --detach:
#   MEDTHINKVQA_VARIANT=ct modal run --detach scripts/modal_mm_tuning/independent_finetune/medthinkvqa.py
# Read locally at launch and passed to the remote function as an argument: the container re-imports this file
# without the launch environment, so a module-level read there would always fall back to "all".
VARIANT = os.environ.get("MEDTHINKVQA_VARIANT", "all")
assert VARIANT in ("all", "ct"), VARIANT


def run_id(variant: str) -> str:
    return f"medthinkvqa_{variant}_qwen2_5-3b-vl"


COMMAND_TEMPLATE = """
    cd /FlexBagel && PYTHONPATH=. torchrun --master_port=29501 --nproc_per_node={num_gpus} train/mm_tune.py \
    --run_id "{run_id}" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets /mnt/medthinkvqa/medthinkvqa_{variant}_train_modal.jsonl \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing True \
    --use_liger_kernel True \
    --group_by_length True \
    --run_seed 42 \
    --run_output_dir "/output/medthinkvqa/" \
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

app = modal.App("medthinkvqa")
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
        .uv_pip_install("liger-kernel==0.8.1") \
        .add_local_dir(os.path.join(LOCAL_FLEXBAGEL, "train"), remote_path="/FlexBagel/train", copy=True) \
        .add_local_dir(os.path.join(LOCAL_FLEXBAGEL, "modeling"), remote_path="/FlexBagel/modeling", copy=True) \

vol_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vol_data = modal.Volume.from_name("medthinkvqa", create_if_missing=False)
vol_output = modal.Volume.from_name("output", create_if_missing=True)

GPU_TYPE = f"{MODEL}:{NUM_GPUS}"
TIMEOUT_HOURS = 24
@app.function(image=image,
    volumes={"/hf-cache": vol_hf_cache, "/mnt/medthinkvqa": vol_data, "/output": vol_output},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    gpu=GPU_TYPE,
    cpu=24,
    timeout=int(TIMEOUT_HOURS * 60 * 60))
def train(variant: str):
    import shlex
    import subprocess
    import wandb
    import os
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run_cli("pip list")
    assert variant in ("all", "ct"), variant
    command = COMMAND_TEMPLATE.format(run_id=run_id(variant), variant=variant, num_gpus=NUM_GPUS)
    print(command, flush=True)
    run_cli(command, use_shell=True)


@app.local_entrypoint()
def main():
    print(f"Launching {run_id(VARIANT)} on /mnt/medthinkvqa/medthinkvqa_{VARIANT}_train_modal.jsonl")
    train.spawn(VARIANT)
