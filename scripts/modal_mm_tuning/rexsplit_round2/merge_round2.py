"""
RexGradient split BTX, round 2: assemble the retrained experts into one BTX model.

Takes /output/rexsplit_round2/<run for split k>/final (k = 1, 2, 3; see rexsplit_round2_expert.py) and builds the
3-expert model
  expert e + router row e   from the run of split e + 1 (the run that retrained expert e)
  all non-FFN weights       from the split-1 run (frozen in every run)
with modeling/upcycle/merge_vl_experts.py (text decoder and vision tower MoE blocks): expert 0 is taken as the
"shared" slot from model 0 and experts 1 and 2 from models 1 and 2 (--experts-per-model "" 1 2). Before merging,
verify_run checks each run against the round-1 model: only its own expert FFN and router row may have changed.
--strict-shared-check additionally fails if non-expert tensors differ between the runs.

    cd FlexBagel && modal run --detach scripts/modal_mm_tuning/rexsplit_round2/merge_round2.py
"""
import os
import subprocess

import modal

LOCAL_FLEXBAGEL = "."
NUM_EXPERTS = 3
EXPERT_FOR_SPLIT = {1: 0, 2: 1, 3: 2}  # keep in sync with rexsplit_round2_expert.py
TAG = os.environ.get("ROUND2_TAG", "v2")  # keep in sync with rexsplit_round2_expert.py
RUNS = [f"/output/rexsplit_round2/rexsplit_round2-{TAG}-split{k}-expert{e}-qwen2_5-3b-vl-topk{NUM_EXPERTS}/final"
        for k, e in EXPERT_FOR_SPLIT.items()]
TARGET = f"/output/rexsplit_round2/rexsplit_round2-{TAG}_btx_combined_untrained"

app = modal.App("rexsplit-round2-merge")
image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10") \
        .apt_install("git", "libgl1", "libglib2.0-0") \
        .run_commands("git clone https://github.com/michaelduan8/FlexBagel.git && cd FlexBagel && git checkout main") \
        .uv_pip_install(requirements=["requirements_new.txt"]) \
        .uv_pip_install("trl==0.22.2", "transformers==4.57.3") \
        .env({"HF_HOME": "/hf-cache", "ROUND2_TAG": TAG}) \
        .add_local_dir(os.path.join(LOCAL_FLEXBAGEL, "modeling"), remote_path="/FlexBagel/modeling", copy=True)

vol_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vol_output = modal.Volume.from_name("output", create_if_missing=False)


BASE_MODEL = "micdun/rexsplit_btx_qwen2_5-3b-vl-8bits-router_hidden_state_init-untrained"


def verify_run(base_dir: str, run_dir: str, expert: int) -> dict:
    """Check that a round-2 run changed nothing but expert `expert`'s FFN and router row `expert`.

    Compares every tensor of run_dir with base_dir (the round-1 merged model), one tensor at a time. Returns counts;
    raises if a frozen tensor or another router row changed, or if the trained expert did not change at all.
    """
    import json
    import re

    import torch
    from safetensors import safe_open

    def index(d):
        path = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(path):
            return {k: os.path.join(d, f) for k, f in json.load(open(path))["weight_map"].items()}
        single = os.path.join(d, "model.safetensors")
        with safe_open(single, "pt") as f:
            return {k: single for k in f.keys()}

    def get(files, key):
        with safe_open(files[key], "pt") as f:
            return f.get_tensor(key)

    base, run = index(base_dir), index(run_dir)
    assert set(base) == set(run), f"tensor names differ: {sorted(set(base) ^ set(run))[:5]}"
    trained_expert = re.compile(rf"\.mlp\.experts\.{expert}\.")
    counts = {"frozen_equal": 0, "expert_changed": 0, "expert_unchanged": 0, "gates": 0}
    for key in sorted(base):
        a, b = get(base, key), get(run, key)
        if trained_expert.search(key):
            counts["expert_changed" if not torch.equal(a, b) else "expert_unchanged"] += 1
        elif key.endswith(".mlp.gate.weight"):
            other = [r for r in range(a.shape[0]) if r != expert]
            assert torch.equal(a[other], b[other]), f"{key}: a router row other than {expert} changed"
            counts["gates"] += 1
        else:
            assert torch.equal(a, b), f"{key}: frozen tensor changed"
            counts["frozen_equal"] += 1
    assert counts["expert_changed"] > 0, f"expert {expert} did not change in {run_dir}"
    return counts


@app.function(image=image, volumes={"/hf-cache": vol_hf_cache, "/output": vol_output},
              secrets=[modal.Secret.from_name("huggingface-secret")],
              cpu=8, memory=196608, timeout=4 * 60 * 60)
def merge():
    from huggingface_hub import snapshot_download

    for run in RUNS:
        assert os.path.exists(os.path.join(run, "config.json")), f"missing finished run {run}"
    base_dir = snapshot_download(BASE_MODEL)  # cached in /hf-cache by the training runs
    for (k, e), run in zip(EXPERT_FOR_SPLIT.items(), RUNS):
        print(f"split {k}: only expert {e} + router row {e} changed:", verify_run(base_dir, run, e), flush=True)
    command = ["python", "modeling/upcycle/merge_vl_experts.py", "-m", *RUNS, "-t", TARGET,
               "--experts-per-model", "", "1", "2", "--shared-expert-index", "0", "--shared-from-model", "0",
               "--base-model-index", "0", "--strict-shared-check", "--dtype", "bfloat16"]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd="/FlexBagel", env={**os.environ, "PYTHONPATH": "/FlexBagel"}, check=True)
    vol_output.commit()
    print("saved", TARGET, sorted(os.listdir(TARGET)))


@app.local_entrypoint()
def main():
    merge.remote()
