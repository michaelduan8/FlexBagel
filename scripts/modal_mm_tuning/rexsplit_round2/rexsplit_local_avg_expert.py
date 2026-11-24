"""
RexGradient split BTX, round 2, local 2-expert approximation ("local avg"): instead of retraining split k's expert
inside the full 3-expert BTX, train it inside a local 2-expert MoE built by modeling/upcycle/local_avg_moe.py build:
  expert 0 = split k's own expert (from the round-1 BTX)        -> trainable, with its gate row 0
  expert 1 = weight average of the other two experts' FFNs      -> frozen
  gate rows = [own row, the other two collaborators' rows], gate_row_to_expert = [0, 1, 1],
              gate_row_aggregation = "sum" (expert 1's weight = the sum of the other rows' probs, as in the BTX)
Everything else is frozen exactly as in rexsplit_round2_expert.py (--train_expert_idx 0 keeps only gate row 0 trainable),
with both experts active (--num_experts_per_tok 2). Afterwards, local_avg_moe.py assemble puts each split's expert 0 and
gate row 0 back into slot k of the 3-expert BTX.

The local models must be on the `output` volume at /output/rexsplit_local_avg/split{k}-expert{k-1}.

    cd FlexBagel && ROUND2_SPLIT=1 modal run --detach scripts/modal_mm_tuning/rexsplit_round2/rexsplit_local_avg_expert.py
"""
import os
import shlex
import subprocess

import modal

LOCAL_FLEXBAGEL = "."

MODEL = "H200"
NUM_GPUS = 4

SPLIT = int(os.environ.get("ROUND2_SPLIT", "0"))
assert SPLIT in (1, 2, 3), "set ROUND2_SPLIT=1, 2 or 3"
NUM_EXPERTS = 2  # local model: own expert + averaged frozen expert, both active per token
EXPERT_FOR_SPLIT = {1: 0, 2: 1, 3: 2}  # BTX expert order = split order (keep in sync with merge_round2.py)
EXPERT = EXPERT_FOR_SPLIT[SPLIT]  # slot in the 3-expert BTX (slot 0 in the local model)
BASE_MODEL = f"/output/rexsplit_local_avg/split{SPLIT}-expert{EXPERT}"
TAG = os.environ.get("ROUND2_TAG", "v1")
RUN_ID = f"rexsplit_local_avg-{TAG}-split{SPLIT}-expert{EXPERT}-qwen2_5-3b-vl-topk{NUM_EXPERTS}"

COMMAND = f"""
    cd /FlexBagel && PYTHONPATH=. torchrun --master_port=29501 --nproc_per_node={NUM_GPUS} train/mm_tune.py \
    --run_id "{RUN_ID}" \
    --model {BASE_MODEL} \
    --datasets /mnt/rexgradient_flat/train_w_length_split_{SPLIT}_flat.jsonl \
    --train_expert_idx 0 \
    --num_experts_per_tok {NUM_EXPERTS} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 6 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing True \
    --run_seed 42 \
    --run_output_dir "/output/rexsplit_local_avg_runs/" \
    --save_n_epochs 0.25 \
    --dataset_num_proc 6 \
    --skip_eval \
    --dataloader_num_workers 6 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --delete_intermediate_checkpoints false
"""


def run_cli(command: str, use_shell=False):
    if use_shell:
        return subprocess.run(command, shell=True)
    return subprocess.run(shlex.split(command))


app = modal.App(f"rexsplit-local-avg-{TAG}-split{SPLIT}")
nvidia_image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
image = nvidia_image.apt_install("git", "libgl1", "libglib2.0-0") \
        .run_commands("git clone https://github.com/michaelduan8/FlexBagel.git && cd FlexBagel && git checkout main") \
        .uv_pip_install(requirements=["requirements_new.txt"], gpu=MODEL) \
        .env({"HF_HOME": "/hf-cache", "ROUND2_SPLIT": str(SPLIT), "ROUND2_TAG": TAG}) \
        .uv_pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl", extra_options="--no-build-isolation") \
        .uv_pip_install("datasets>=2.14.6", "fsspec") \
        .uv_pip_install("trl==0.22.2", "transformers==4.57.3") \
        .add_local_dir(os.path.join(LOCAL_FLEXBAGEL, "train"), remote_path="/FlexBagel/train", copy=True) \
        .add_local_dir(os.path.join(LOCAL_FLEXBAGEL, "modeling"), remote_path="/FlexBagel/modeling", copy=True)

vol_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vol_data = modal.Volume.from_name("rexgradient_flat", create_if_missing=False)
vol_output = modal.Volume.from_name("output", create_if_missing=True)

GPU_TYPE = f"{MODEL}:{NUM_GPUS}"
TIMEOUT_HOURS = 24


@app.function(image=image,
              volumes={"/hf-cache": vol_hf_cache, "/mnt/rexgradient_flat": vol_data, "/output": vol_output},
              secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
              gpu=GPU_TYPE,
              cpu=24,
              timeout=int(TIMEOUT_HOURS * 60 * 60))
def train():
    import wandb
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run_cli("pip list")
    run_cli(COMMAND, use_shell=True)


@app.local_entrypoint()
def main():
    train.spawn()
