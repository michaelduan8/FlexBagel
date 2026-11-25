"""
Two-round FlexOlmo pipeline over several data silos, run as one chained sequence of Modal jobs.

Stages (run in this order; each one skips itself if its outputs already exist on the `output` volume, so re-running
the same command resumes after a failure or timeout):
  split    shuffle each silo's JSONL (seeded) and write a round-1 part and a round-2 part
             --data_mode halves: round 1 gets the first --round1_fraction of the rows, round 2 the rest (disjoint)
             --data_mode full:   both rounds use every row (the original round-2 recipe)
             --round2_fraction f: round 2 gets only f of the silo's rows (halves: the next f after round 1's part,
                                  still disjoint; full: the first f of the shuffle)
  round1   dense finetune of BASE_MODEL on each silo's round-1 part (one run per silo)
  hidden   per silo, average FFN-input hidden states of BASE_MODEL on --hidden_queries rows of its round-1 part
  upcycle  mm_dense_to_expert_moe.py: round-1 models -> one BTX MoE (expert e = SILOS[e], router rows seeded from the
           hidden states, non-MoE weights averaged)
  round2   per silo e, retrain only expert e's FFN and router row e inside the BTX on its round-2 part
           (mm_tune --train_expert_idx e, at --round2_topk active experts)
  merge    check each round-2 run changed only its own expert and router row, then assemble expert e + row e from
           run e with merge_vl_experts.py

Everything lives under /output/two_round/<tag>/ on the `output` volume:
  data/<silo>_round{1,2}.jsonl, split.json, round1/<run>/final, hidden/<silo>.pkl, btx_round1/,
  round2/<run>/final, btx_round2/
With --round1_tag T, the round-1 outputs (data/<silo>_round1.jsonl, split.json, round1/, hidden/, btx_round1/) are
read from (or written to) /output/two_round/T/ instead, so several round-2 variants can share one round 1. The split
stage then re-derives the same shuffle from T's split.json settings and refuses to run if they differ.

The pipeline itself runs as a CPU Modal function that calls the GPU jobs, so it keeps going after you disconnect:

    cd FlexBagel && modal run --detach scripts/modal_mm_tuning/two_round/two_round_pipeline.py --tag halves-v1
    # only some stages / other settings:
    modal run --detach scripts/modal_mm_tuning/two_round/two_round_pipeline.py --tag halves-v1 --stages round2,merge \
        --round2-topk 1
    # round 2 on 15% of the data at a lower LR, reusing halves-v1's round 1 and BTX:
    modal run --detach scripts/modal_mm_tuning/two_round/two_round_pipeline.py --tag r2frac0.15-lr5e-6 \
        --round1-tag halves-v1 --round2-fraction 0.15 --round2-lr 5e-6
Per-silo jobs inside a stage run one at a time unless --parallel is passed.
"""
import os
import subprocess

import modal

LOCAL_FLEXBAGEL = "."

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
# Expert order of the BTX = order of this dict. name -> (Modal volume, mount point, training JSONL)
SILOS = {
    # same rows and images as rexgradient_train_w_length_modal_8bit.jsonl (the old `rexgradient` volume is gone)
    "rexgradient": ("rexgradient_flat", "/mnt/rexgradient_flat",
                    "/mnt/rexgradient_flat/rexgradient_train_w_length_modal_8bit_flat.jsonl"),
    "padchest": ("rrg24", "/mnt/rrg24", "/mnt/rrg24/padchest_en_train_w_length_memory_modal.jsonl"),
    "chexpert": ("rrg24", "/mnt/rrg24", "/mnt/rrg24/chexpert_train_w_length_memory_modal.jsonl"),
}
ALL_STAGES = ["split", "round1", "hidden", "upcycle", "round2", "merge"]
ROOT = "/output/two_round"

TRAIN_GPU = "H200:4"
NUM_TRAIN_GPUS = 4
TIMEOUT_HOURS = 24

app = modal.App("two-round-pipeline")
image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10") \
        .apt_install("git", "libgl1", "libglib2.0-0") \
        .run_commands("git clone https://github.com/michaelduan8/FlexBagel.git && cd FlexBagel && git checkout main") \
        .uv_pip_install(requirements=["requirements_new.txt"], gpu="H200") \
        .env({"HF_HOME": "/hf-cache"}) \
        .uv_pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl", extra_options="--no-build-isolation") \
        .uv_pip_install("datasets>=2.14.6", "fsspec") \
        .uv_pip_install("trl==0.22.2", "transformers==4.57.3") \
        .add_local_dir(os.path.join(LOCAL_FLEXBAGEL, "train"), remote_path="/FlexBagel/train", copy=True) \
        .add_local_dir(os.path.join(LOCAL_FLEXBAGEL, "modeling"), remote_path="/FlexBagel/modeling", copy=True)

vol_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vol_output = modal.Volume.from_name("output", create_if_missing=True)
VOLUMES = {"/hf-cache": vol_hf_cache, "/output": vol_output}
for _volume, _mount, _ in SILOS.values():
    VOLUMES[_mount] = modal.Volume.from_name(_volume, create_if_missing=False)
SECRETS = [modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")]


def run(command: list[str]):
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd="/FlexBagel", env={**os.environ, "PYTHONPATH": "/FlexBagel"}, check=True)


def paths(tag: str, round1_tag: str) -> dict:
    """Round-1 outputs live under round1_tag's folder, round-2 outputs under tag's (the same folder by default)."""
    root, root1 = f"{ROOT}/{tag}", f"{ROOT}/{round1_tag}"
    p = {"root": root, "root1": root1, "split_meta": f"{root1}/split.json",
         "btx_round1": f"{root1}/btx_round1", "btx_round2": f"{root}/btx_round2"}
    for e, silo in enumerate(SILOS):
        p[silo] = {
            "data1": f"{root1}/data/{silo}_round1.jsonl",
            "data2": f"{root}/data/{silo}_round2.jsonl",
            "round1_run": f"two_round-{round1_tag}-round1-{silo}",
            "round2_run": f"two_round-{tag}-round2-{silo}-expert{e}",
            "hidden": f"{root1}/hidden/{silo}.pkl",
        }
        p[silo]["round1_final"] = f"{root1}/round1/{p[silo]['round1_run']}/final"
        p[silo]["round2_final"] = f"{root}/round2/{p[silo]['round2_run']}/final"
    return p


def done(model_dir: str) -> bool:
    return os.path.exists(os.path.join(model_dir, "config.json"))


EFFECTIVE_BATCH_PER_GPU = 48  # per_device_train_batch_size x gradient_accumulation_steps (x NUM_TRAIN_GPUS = 192)


def mm_tune(run_id: str, model: str, data: str, output_dir: str, extra: list[str], lr: float,
            warmup_ratio: float = 0.1, per_device_batch: int = 8):
    import wandb
    wandb.login(key=os.environ["WANDB_API_KEY"])
    vol_output.reload()
    assert EFFECTIVE_BATCH_PER_GPU % per_device_batch == 0, per_device_batch
    accum = EFFECTIVE_BATCH_PER_GPU // per_device_batch
    run(["torchrun", "--master_port=29501", f"--nproc_per_node={NUM_TRAIN_GPUS}", "train/mm_tune.py",
         "--run_id", run_id, "--model", model, "--datasets", data, *extra,
         "--num_train_epochs", "1", "--per_device_train_batch_size", str(per_device_batch),
         "--gradient_accumulation_steps", str(accum),
         "--logging_steps", "10", "--learning_rate", str(lr), "--warmup_ratio", str(warmup_ratio),
         "--gradient_checkpointing", "True", "--run_seed", "42", "--run_output_dir", output_dir,
         "--save_n_epochs", "0.25", "--auto_resume", "True", "--dataset_num_proc", "6", "--skip_eval",
         "--dataloader_num_workers", "6", "--dataloader_persistent_workers", "True",
         "--dataloader_prefetch_factor", "2", "--dataloader_pin_memory", "True",
         "--delete_intermediate_checkpoints", "false"])
    vol_output.commit()


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, gpu=TRAIN_GPU, cpu=24,
              timeout=int(TIMEOUT_HOURS * 60 * 60))
def train_round1(tag: str, round1_tag: str, silo: str, lr: float, extra_args: tuple = (), per_device_batch: int = 8):
    p = paths(tag, round1_tag)
    mm_tune(p[silo]["round1_run"], BASE_MODEL, p[silo]["data1"], f"{p['root1']}/round1/", list(extra_args), lr,
            per_device_batch=per_device_batch)


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, gpu=TRAIN_GPU, cpu=24,
              timeout=int(TIMEOUT_HOURS * 60 * 60))
def train_round2(tag: str, round1_tag: str, silo: str, lr: float, topk: int, warmup_ratio: float = 0.1,
                 extra_args: tuple = (), per_device_batch: int = 8):
    p = paths(tag, round1_tag)
    expert = list(SILOS).index(silo)
    mm_tune(p[silo]["round2_run"], p["btx_round1"], p[silo]["data2"], f"{ROOT}/{tag}/round2/",
            ["--train_expert_idx", str(expert), "--num_experts_per_tok", str(topk), *extra_args], lr, warmup_ratio,
            per_device_batch)


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, gpu="H200:1", cpu=8, timeout=4 * 60 * 60)
def extract_hidden(tag: str, round1_tag: str, silo: str, queries: int):
    p = paths(tag, round1_tag)[silo]
    vol_output.reload()
    run(["python", "modeling/routing_embeddings/extract_hidden_states.py", "--model", BASE_MODEL,
         "--dataset", p["data1"], "--query-amount", str(queries), "--output-path", p["hidden"], "--dtype", "bf16"])
    vol_output.commit()


@app.function(image=image, volumes=VOLUMES, timeout=2 * 60 * 60, cpu=4, memory=16384)
def split_data(tag: str, round1_tag: str, data_mode: str, round1_fraction: float, round2_fraction: float | None,
               seed: int):
    """Write each silo's round-1 and round-2 JSONLs (seeded row shuffle; see the module docstring).

    The round-1 settings are recorded in split.json next to the round-1 data. If it already exists (round 1 shared
    with another tag), the settings must match it, so the re-derived round-2 rows stay disjoint from round 1's, and the
    existing round-1 files are left alone.
    """
    import json
    import random

    vol_output.reload()
    p = paths(tag, round1_tag)
    meta = {"data_mode": data_mode, "round1_fraction": round1_fraction, "seed": seed}
    if os.path.exists(p["split_meta"]):
        existing = json.load(open(p["split_meta"]))
        assert {k: existing[k] for k in meta} == meta, f"split settings {meta} differ from {p['split_meta']}: {existing}"
    os.makedirs(f"{p['root']}/data", exist_ok=True)
    os.makedirs(f"{p['root1']}/data", exist_ok=True)
    for silo, (_, _, source) in SILOS.items():
        with open(source, encoding="utf-8") as f:
            rows = [line for line in f if line.strip()]
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        cut = len(rows) if data_mode == "full" else int(len(rows) * round1_fraction)
        start = 0 if data_mode == "full" else cut
        end = len(rows) if round2_fraction is None else start + int(len(rows) * round2_fraction)
        assert end <= len(rows), f"{silo}: round 2 wants rows {start}:{end} of {len(rows)}; lower --round2_fraction"
        parts = {p[silo]["data1"]: order[:cut], p[silo]["data2"]: order[start:end]}
        for out, idx in parts.items():
            if out == p[silo]["data1"] and os.path.exists(out):
                continue
            with open(out, "w", encoding="utf-8") as f:
                f.writelines(rows[i] if rows[i].endswith("\n") else rows[i] + "\n" for i in idx)
        print(f"{silo}: {len(rows)} rows -> round 1 {cut}, round 2 {end - start}", flush=True)
    if not os.path.exists(p["split_meta"]):
        json.dump(meta, open(p["split_meta"], "w"))
    vol_output.commit()


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, cpu=8, memory=196608, timeout=4 * 60 * 60)
def upcycle(tag: str, round1_tag: str, topk: int):
    p = paths(tag, round1_tag)
    vol_output.reload()
    run(["python", "modeling/upcycle/mm_dense_to_expert_moe.py",
         "-m", *[p[s]["round1_final"] for s in SILOS], "-t", p["btx_round1"],
         "--embeddings-dict", *[p[s]["hidden"] for s in SILOS],
         "--merge-non-moe-weights", "--num-experts-per-tok", str(topk)])
    vol_output.commit()


def verify_run(base_dir: str, run_dir: str, expert: int) -> dict:
    """Check that a round-2 run changed nothing but expert `expert`'s FFN and router row `expert` (same check as
    rexsplit_round2/merge_round2.py). Raises if a frozen tensor or another router row changed, or the expert did not."""
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

    base, trained = index(base_dir), index(run_dir)
    assert set(base) == set(trained), f"tensor names differ: {sorted(set(base) ^ set(trained))[:5]}"
    trained_expert = re.compile(rf"\.mlp\.experts\.{expert}\.")
    counts = {"frozen_equal": 0, "expert_changed": 0, "expert_unchanged": 0, "gates": 0}
    for key in sorted(base):
        a, b = get(base, key), get(trained, key)
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


@app.function(image=image, volumes=VOLUMES, cpu=8, memory=196608, timeout=4 * 60 * 60)
def merge(tag: str, round1_tag: str):
    p = paths(tag, round1_tag)
    vol_output.reload()
    runs = [p[s]["round2_final"] for s in SILOS]
    for e, (silo, run_dir) in enumerate(zip(SILOS, runs)):
        print(f"{silo}: only expert {e} + router row {e} changed:", verify_run(p["btx_round1"], run_dir, e), flush=True)
    run(["python", "modeling/upcycle/merge_vl_experts.py", "-m", *runs, "-t", p["btx_round2"],
         "--experts-per-model", "", *[str(e) for e in range(1, len(SILOS))],
         "--shared-expert-index", "0", "--shared-from-model", "0", "--base-model-index", "0",
         "--strict-shared-check", "--dtype", "bfloat16"])
    vol_output.commit()


def per_silo(fn, args: list[tuple], parallel: bool):
    """Run fn once per argument tuple, one at a time or all at once."""
    if not args:
        return
    if parallel:
        list(fn.starmap(args))
    else:
        for a in args:
            fn.remote(*a)


@app.function(image=image, volumes={"/output": vol_output}, cpu=1, timeout=int(TIMEOUT_HOURS * 60 * 60))
def pipeline(cfg: dict):
    tag, round1_tag, stages = cfg["tag"], cfg["round1_tag"], cfg["stages"]
    silos = cfg.get("silos") or list(SILOS)  # silos whose round1/hidden/round2 jobs run; split always covers all
    p = paths(tag, round1_tag)
    print("config:", cfg, flush=True)

    def todo(stage: str, outputs: list[str], check=os.path.exists) -> bool:
        vol_output.reload()
        if stage not in stages:
            return False
        if all(check(o) for o in outputs):
            print(f"[{stage}] outputs exist, skipping", flush=True)
            return False
        print(f"[{stage}] running", flush=True)
        return True

    if todo("split", [p[s][k] for s in SILOS for k in ("data1", "data2")]):
        split_data.remote(tag, round1_tag, cfg["data_mode"], cfg["round1_fraction"], cfg["round2_fraction"],
                          cfg["seed"])
    if "round1" in stages:
        per_silo(train_round1, [(tag, round1_tag, s, cfg["round1_lr"], tuple(cfg.get("round1_extra_args", ())),
                                 cfg.get("round1_per_device_batch", 8))
                                for s in silos if not done(p[s]["round1_final"])],
                 cfg["parallel"])
    if "hidden" in stages:
        vol_output.reload()
        per_silo(extract_hidden, [(tag, round1_tag, s, cfg["hidden_queries"]) for s in silos if not os.path.exists(p[s]["hidden"])],
                 cfg["parallel"])
    if todo("upcycle", [p["btx_round1"]], check=done):
        upcycle.remote(tag, round1_tag, cfg["upcycle_topk"])
    if "round2" in stages:
        vol_output.reload()
        per_silo(train_round2, [(tag, round1_tag, s, cfg["round2_lr"], cfg["round2_topk"],
                                 cfg.get("round2_warmup_ratio", 0.1), tuple(cfg.get("round2_extra_args", ())),
                                 cfg.get("round2_per_device_batch", 8))
                                for s in silos if not done(p[s]["round2_final"])], cfg["parallel"])
    if todo("merge", [p["btx_round2"]], check=done):
        merge.remote(tag, round1_tag)
    print("finished stages", stages, "->", p["root"], flush=True)


@app.local_entrypoint()
def main(tag: str, round1_tag: str = "", stages: str = ",".join(ALL_STAGES), data_mode: str = "halves",
         round1_fraction: float = 0.5, round2_fraction: float = 0.0, seed: int = 42, round1_lr: float = 2e-5, round2_lr: float = 2e-5, round2_topk: int = 3,
         upcycle_topk: int = 2, hidden_queries: int = 1000, parallel: bool = False, silos: str = "",
         round2_warmup_ratio: float = 0.1, round2_extra_args: str = "", round2_per_device_batch: int = 8,
         round1_extra_args: str = "", round1_per_device_batch: int = 8):
    stage_list = [s for s in stages.split(",") if s]
    assert set(stage_list) <= set(ALL_STAGES), f"unknown stage in {stage_list}; choose from {ALL_STAGES}"
    silo_list = [s for s in silos.split(",") if s] or list(SILOS)
    assert set(silo_list) <= set(SILOS), f"unknown silo in {silo_list}; choose from {list(SILOS)}"
    assert silo_list == list(SILOS) or not {"upcycle", "merge"} & set(stage_list), \
        "upcycle/merge need every silo; leave --silos empty for them"
    assert data_mode in ("halves", "full"), data_mode
    assert 0 < round1_fraction < 1, round1_fraction
    assert 0 <= round2_fraction <= 1, round2_fraction
    cfg = dict(tag=tag, round1_tag=round1_tag or tag, stages=stage_list, data_mode=data_mode,
               round1_fraction=round1_fraction, round2_fraction=round2_fraction or None, seed=seed,
               round1_lr=round1_lr, round2_lr=round2_lr, round2_topk=round2_topk, upcycle_topk=upcycle_topk,
               hidden_queries=hidden_queries, parallel=parallel, silos=silo_list,
               round2_warmup_ratio=round2_warmup_ratio, round2_extra_args=round2_extra_args.split(),
               round2_per_device_batch=round2_per_device_batch,
               round1_extra_args=round1_extra_args.split(), round1_per_device_batch=round1_per_device_batch)
    assert 48 % round1_per_device_batch == 0, "per-device batch must divide 48 (effective batch 192 on 4 GPUs)"
    assert 48 % round2_per_device_batch == 0, "per-device batch must divide 48 (effective batch 192 on 4 GPUs)"
    assert 0 <= round2_warmup_ratio < 1, round2_warmup_ratio
    assert not (cfg["round2_extra_args"] and "merge" in stage_list), \
        "round2 extra args may change non-expert weights, which merge's per-run check rejects; merge those runs separately"
    call = pipeline.spawn(cfg)
    print("spawned pipeline", call.object_id, cfg)
