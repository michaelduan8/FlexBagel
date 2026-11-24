from pathlib import Path

import os
import modal
import shlex
import subprocess

LOCAL_FLEXBAGEL = "."

MODEL = "H200"
NUM_GPUS = 4

# Pooled ("centralized") MultiCaRe CT + MedThinkVQA (all variant) run for the silo-pair test
# (data_processing/prepare/multicare_medthinkvqa.py): 79,359 rows = MultiCaRe CT (17,636 caption / 4,023 findings)
# + MedThinkVQA all (44,348 caption / 6,673 findings / 6,679 diagnosis), ~53M tokens, ~413 steps, ~2-2.5 h.
# Compare against the single-silo experts multicare_ct_qwen2_5-3b-vl and medthinkvqa_all_qwen2_5-3b-vl.
# MultiCaRe images + the JSONL come from the `multicare` volume, MedThinkVQA images from `medthinkvqa`.
#
# Original MultiCaRe-only notes:
# MultiCaRe CT expert (HF mauro-nievoff/MultiCaRe_Dataset; PubMed Central open-access case reports, dataset CC BY-NC-SA,
# per-article CC licenses: research only). A 2D CT silo matching MedThinkVQA's format (case-report key images, several
# per patient, with clinical history). Independent single-turn rows from data_processing/prepare/multicare_ct.py
# (split by article, 5% val; no image near-duplicates of MedThinkVQA test images):
#   caption   one CT image -> its caption (sub-figure part)                                         17,636 rows
#   findings  a patient's CT images (2-24) + presenting history -> the case text's CT findings       4,023 rows
#             passage (Eurorad "Imaging Findings"-style: CT sentences + continuations + CT-figure citations;
#             non-CT modality, PET-CT, lab/treatment/course sentences dropped; median 63 words)
# 21,659 rows, ~9.6M estimated tokens/epoch, ~30 min on 4x H200 (similar to medthinkvqa ct).
# Images (18,682 JPEGs, 1.1 GB) are on the private Modal volume `multicare`, mounted at /mnt/multicare.
#
# Launch from the FlexBagel root (the image copies the local train/ and modeling/), always with --detach:
#   modal run --detach scripts/modal_mm_tuning/independent_finetune/multicare_medthinkvqa.py
RUN_ID = "multicare_medthinkvqa_qwen2_5-3b-vl"

COMMAND_TEMPLATE = """
    cd /FlexBagel && PYTHONPATH=. torchrun --master_port=29501 --nproc_per_node={num_gpus} train/mm_tune.py \
    --run_id "{run_id}" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets /mnt/multicare/multicare_medthinkvqa_train_modal.jsonl \
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
    --run_output_dir "/output/multicare/" \
    --save_n_epochs 0.2 \
    --dataset_num_proc 6 \
    --skip_eval \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --delete_intermediate_checkpoints false \
    --auto_resume False
"""

def run_cli(command: str, use_shell=False):
    if use_shell:
        return subprocess.run(command, shell=True)
    else:
        return subprocess.run(shlex.split(command))

app = modal.App("multicare-medthinkvqa")
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
vol_data = modal.Volume.from_name("multicare", create_if_missing=False)
vol_mtv = modal.Volume.from_name("medthinkvqa", create_if_missing=False)
vol_output = modal.Volume.from_name("output", create_if_missing=True)

GPU_TYPE = f"{MODEL}:{NUM_GPUS}"
TIMEOUT_HOURS = 24
@app.function(image=image,
    volumes={"/hf-cache": vol_hf_cache, "/mnt/multicare": vol_data, "/mnt/medthinkvqa": vol_mtv, "/output": vol_output},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    gpu=GPU_TYPE,
    cpu=24,
    timeout=int(TIMEOUT_HOURS * 60 * 60))
def train():
    import wandb
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run_cli("pip list")
    command = COMMAND_TEMPLATE.format(run_id=RUN_ID, num_gpus=NUM_GPUS)
    print(command, flush=True)
    run_cli(command, use_shell=True)


@app.local_entrypoint()
def main():
    print(f"Launching {RUN_ID} on /mnt/multicare/multicare_medthinkvqa_train_modal.jsonl")
    train.spawn()
