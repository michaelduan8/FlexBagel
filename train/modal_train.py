import pathlib
import json
from copy import deepcopy

import modal

from dataclasses import dataclass, field, fields, replace, MISSING

from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainerCallback

app = modal.App("mm-flexolmo")

DS_CONFIG_DIR = "./train/ds_config/"
TRAIN_SCRIPT_PATH = "./train/mm_tune.py"
REMOTE_DIR = "/root/"

flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)
train_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .entrypoint([])
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.55.0",
        "trl==0.19.1",
        "wandb",
        "torch==2.6.0"
    )
    .uv_pip_install("deepspeed")
    .uv_pip_install(flash_attn_release)
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/hf_cache"})  # faster model transfers
    .add_local_dir(DS_CONFIG_DIR, remote_path=f"{REMOTE_DIR}/ds_config")
    .add_local_file(TRAIN_SCRIPT_PATH, remote_path=f"{REMOTE_DIR}/mm_tune.py")
)

with train_image.imports():
    import numpy as np
    import os
    import subprocess
    import time
    import torch
    import wandb

    from datasets import load_dataset, Dataset
    from datetime import datetime


@dataclass
class SFTArgs:
    model: str = field(metadata={"help": "Model to use for training"})
    run_id: str = field(default=None, metadata={"help": "ID for a single training run"})
    datasets: list[str] = field(default=None, metadata={"help": "Dataset(s) for a single training run"})
    run_dataset_json: str = field(
        default=None,
        metadata={
            "help": (
                "Optional JSON string for multi-run mode. "
                "Format: '{\"run_id_a\": [\"dataset1\", \"dataset2\"], \"run_id_b\": [\"dataset3\"]}'. "
                "When set, runs are executed sequentially in JSON key order."
            )
        },
    )
    run_seed: int = field(default=2026, metadata={"help": "Random seed for training"})
    run_output_dir: str = field(default="./checkpoints", metadata={"help": "Directory to save training runs"})
    sample_size: list[int] = field(default=None, metadata={"help": "Number of samples to use from the dataset. If None, use all data."})
    eval_n_epochs: float = field(default=2.0, metadata={"help": "Evaluate every N epochs"})
    save_n_epochs: float = field(default=1.0, metadata={"help": "Save a checkpoint every N epochs"})
    filter_by_id: list[str] = field(
        default=None,
        metadata={"help": "Only keep rows whose prompt_id contains at least one of these substrings. If None, no filtering is applied."}
    )
    skip_eval: bool = field(default=False, metadata={"help": "Skip all evaluation. Also skips train/val split — all data is used for training. Useful when using DeepSpeed Stage 1/2."})

    # LoRA parameters
    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA training"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank (dimension of low-rank matrices)"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha (scaling factor)"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability"})
    lora_target_modules: list[str] = field(
        default=None,
        metadata={"help": "Target modules for LoRA. If None, uses default for the model architecture."}
    )
    merge_and_save: bool = field(default=False, metadata={"help": "Merge LoRA weights into base model and save full model"})


def build_run_plan(sft_args: SFTArgs) -> list[tuple[str, list[str]]]:
    """Build ordered (run_id, datasets) specs from CLI args."""
    has_multi_run = bool(sft_args.run_dataset_json)
    has_single_run = bool(sft_args.run_id) or bool(sft_args.datasets)

    if has_multi_run and has_single_run:
        raise ValueError(
            "Choose exactly one run mode: either "
            "(--run_id and --datasets) OR --run_dataset_json, not both."
        )

    if not has_multi_run and not has_single_run:
        raise ValueError(
            "Missing run configuration. Provide either "
            "(--run_id and --datasets) OR --run_dataset_json."
        )

    if has_multi_run:
        try:
            parsed = json.loads(sft_args.run_dataset_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid --run_dataset_json value: {exc}. "
                "Expected JSON object like "
                "'{\"run_id_a\": [\"dataset1\"], \"run_id_b\": [\"dataset2\", \"dataset3\"]}'."
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError("--run_dataset_json must be a JSON object mapping run_id -> list of datasets")

        run_plan = []
        for run_id, datasets in parsed.items():
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError("All run_id keys in --run_dataset_json must be non-empty strings")
            if not isinstance(datasets, list) or not datasets:
                raise ValueError(f"run_id '{run_id}' must map to a non-empty list of datasets")
            if any(not isinstance(dataset, str) or not dataset.strip() for dataset in datasets):
                raise ValueError(f"All datasets for run_id '{run_id}' must be non-empty strings")
            run_plan.append((run_id.strip(), [dataset.strip() for dataset in datasets]))
        return run_plan

    if not sft_args.run_id:
        raise ValueError("Missing --run_id for single-run mode")
    if not sft_args.datasets:
        raise ValueError("Missing --datasets for single-run mode")
    return [(sft_args.run_id, sft_args.datasets)]


# arglist enables forwarding of command line args to hf parser
@app.local_entrypoint()
def main(*arglist):
    parser = HfArgumentParser([SFTArgs])
    cli_args = list(arglist)
    # Allow callers to pass either variadic strings or a single list/tuple.
    if len(cli_args) == 1 and isinstance(cli_args[0], (list, tuple)):
        cli_args = list(cli_args[0])

    sft_args: SFTArgs
    sft_args, arglist = parser.parse_args_into_dataclasses(
        args=cli_args,
        return_remaining_strings=True,
    )

    # Keep only args not consumed by SFTArgs parser.
    arglist = list(arglist)
    if arglist:
        print(f"Unparsed/forwarded args: {arglist}")

    # Instantiate remote tuning function on Modal
    print("Submitting finetuning job to Modal (run plan will be built on remote worker)")
    print(f"Model: {sft_args.model}")

    if sft_args.use_lora:
        print(f"LoRA configuration: rank={sft_args.lora_r}, alpha={sft_args.lora_alpha}")

    # Launch the remote job. Multi-run expansion happens inside finetune().
    job_id = finetune.spawn(sft_args, arglist)
    print(f"Training job submitted: {job_id}")
    print("When training is finished, get results using 'modal volume get'")
          

MODEL = "H200"
NUM_GPUS = 8
GPU_TYPE = f"{MODEL}:{NUM_GPUS}"
TIMEOUT_HOURS = 24

cache_volume = modal.Volume.from_name(
    "sda-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "sda-checkpoints", create_if_missing=True
)

@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    volumes={
        "/hf_cache": cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")], # TODO: modal secret for wandb key needs to be created
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def finetune(sft_args, arglist):
    run_plan = build_run_plan(sft_args)
    print(f"Built run plan on remote worker with {len(run_plan)} run(s)")

    wandb.login(key=os.environ["WANDB_API_KEY"])

    for idx, (run_id, datasets) in enumerate(run_plan, start=1):
        sft_args_copy = deepcopy(sft_args)
        per_run_args = replace(sft_args_copy, run_id=run_id, datasets=datasets, run_dataset_json=None)
        print(f"[{idx}/{len(run_plan)}] Starting finetuning experiment: {run_id}")
        print(f"[{idx}/{len(run_plan)}] Datasets: {datasets}")

        run_single_finetune(per_run_args, arglist)

        print(f"[{idx}/{len(run_plan)}] Finished finetuning experiment: {run_id}")
        if idx < len(run_plan):
            print("Sleeping 60 seconds before next run to allow GPU/resource cleanup...")
            time.sleep(60)


def run_single_finetune(sft_args, arglist):
    # Forward all non-SFTArgs CLI values and append parsed SFTArgs.
    sft_args_cli = dataclass_to_cli_args(sft_args, only_non_default=True)
    script_args = list(arglist) + sft_args_cli

    # Runs a subprocess with mm_tune.py and torchrun
    env_copy = os.environ.copy()
    print(env_copy.keys())
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        # "deepspeed",
        f"--master_port=29501",
        # f"--num_gpus={NUM_GPUS}",
        f"{REMOTE_DIR}/mm_tune.py",
        *script_args
    ]
    # print(f"Executing deepspeed with: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    

def dataclass_to_cli_args(obj, only_non_default: bool = False) -> list[str]:
    """
    Convert a dataclass instance to a flat list of CLI args.

    Args:
        obj: A dataclass instance.
        only_non_default: If True, skip fields whose value matches the default.
                          Useful for verbose dataclasses.
    """
    args = []
    for f in fields(obj):
        value = getattr(obj, f.name)
        flag = f"--{f.name}"

        # --- Skip None ---
        if value is None:
            continue

        # --- Skip defaults if requested ---
        if only_non_default:
            if f.default is not MISSING and value == f.default:
                continue
            if f.default_factory is not MISSING:  # e.g. list/dict defaults
                try:
                    if value == f.default_factory():
                        continue
                except Exception:
                    pass  # if factory fails, don't skip

        # --- Bool flags ---
        if isinstance(value, bool):
            if value:
                args.append(flag)
            # False bools are omitted (assume False is the parser default)

        # --- List fields ---
        elif isinstance(value, list):
            if len(value) == 0:
                continue  # omit empty lists
            print(value)
            args.extend([flag] + [str(item) for item in value])

        # --- Scalar fields ---
        else:
            args.extend([flag, str(value)])

    return args
