"""Standalone conditional Fisher mass for a Flex Qwen2.5-VL MoE checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

import torch
from tqdm.auto import tqdm

from analysis.expert_analysis import (
    DEFAULT_GROUPS, GROUP_PARTITIONS, ParameterGroup,
    estimate_conditional_token_fisher, group_summary,
)
from analysis.run_expert_analysis import (
    combine_shards, count_jsonl_rows, enable_activation_checkpointing,
    load_model, load_processor, load_shard_checkpoint, named_paths,
    normalized_scores, processed_examples, release_device, resolve_device,
    resolve_devices, save_shard_checkpoint, shard_checkpoint_path,
    shard_example_count,
)


EXPERT_SPLIT_GROUPS = (
    ParameterGroup("vision_expert_0", r"^(?:model\.)?visual\.(?!merger\.).*\.mlp\.experts\.0\."),
    ParameterGroup("vision_expert_1", r"^(?:model\.)?visual\.(?!merger\.).*\.mlp\.experts\.1\."),
    ParameterGroup("language_expert_0", r"^(?:model\.(?!visual\.)|language_model\.).*\.mlp\.experts\.0\."),
    ParameterGroup("language_expert_1", r"^(?:model\.(?!visual\.)|language_model\.).*\.mlp\.experts\.1\."),
)


def fisher_groups(split_experts: bool) -> tuple[ParameterGroup, ...]:
    return DEFAULT_GROUPS + EXPERT_SPLIT_GROUPS if split_experts else DEFAULT_GROUPS


def validate_expert_split(summary: dict) -> None:
    """Require full coverage of two-expert FFNs in each tower."""
    for tower in ("vision", "language"):
        count = sum(summary[f"{tower}_expert_{index}"]["parameter_count"]
                    for index in (0, 1))
        if count == 0 or count != summary[f"{tower}_ffn"]["parameter_count"]:
            raise ValueError(f"{tower} FFN parameters are not fully covered by experts 0 and 1")
    total = sum(summary[group.name]["parameter_count"] for group in EXPERT_SPLIT_GROUPS)
    if total != summary["expertized_ffn"]["parameter_count"]:
        raise ValueError("Expertized FFN parameters are not fully covered by experts 0 and 1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Flex MoE checkpoint directory, not its source-code directory.")
    parser.add_argument("--examples", action="append", required=True, metavar="NAME=JSONL",
                        help="Named PathGen-style JSONL dataset; repeat for each dataset.")
    parser.add_argument("--processor", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Processor checkpoint (default: Qwen/Qwen2.5-VL-3B-Instruct).")
    parser.add_argument("--output", required=True, help="Output .pt file.")
    parser.add_argument("--resume-dir", help="Saved Fisher shards (default: OUTPUT.resume).")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="First N nonempty rows per dataset; 0 means all.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--device", default="auto",
                              help="auto uses all visible XPU/CUDA GPUs; cpu or one GPU selects one device.")
    device_group.add_argument("--devices", help="Explicit GPU list, for example xpu:0,xpu:1.")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Recompute transformer blocks during backward to save GPU memory.")
    parser.add_argument("--split-experts", action="store_true",
                        help="Report vision and language experts 0 and 1 separately; requires full two-expert FFN coverage.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def load_flex_model(path: str, trust_remote_code: bool, device: torch.device | None = None):
    model = load_model(path, trust_remote_code, device)
    if model.config.model_type != "flex_qwen2_5_vl_moe":
        raise ValueError(f"{path} is {model.config.model_type!r}, expected flex_qwen2_5_vl_moe")
    return model


def checkpoint_fingerprint(path: str) -> dict:
    directory = Path(path)
    if not directory.is_dir():
        return {"path": path}
    files = sorted((*directory.glob("*.safetensors"),
                    *directory.glob("*.safetensors.index.json"),
                    *directory.glob("config.json")))
    if not files:
        raise ValueError(f"No config or safetensors found in {directory}")
    return {
        "path": str(directory.resolve()),
        "files": {file.name: {"size": file.stat().st_size,
                              "mtime_ns": file.stat().st_mtime_ns}
                  for file in files},
    }


def prepare_manifest(directory: Path, args: argparse.Namespace,
                     datasets: dict[str, str], devices: list[torch.device],
                     groups: tuple[ParameterGroup, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "method": "standalone_fisher_mass",
        "model": checkpoint_fingerprint(args.model),
        "processor": args.processor,
        "examples": {
            name: {"path": str(Path(path).resolve()),
                   "size": Path(path).stat().st_size,
                   "mtime_ns": Path(path).stat().st_mtime_ns}
            for name, path in datasets.items()
        },
        "max_examples": args.max_examples,
        "seed": args.seed,
        "activation_checkpointing": args.activation_checkpointing,
        "devices": list(map(str, devices)),
        "groups": [{"name": group.name, "include": group.include,
                    "exclude": group.exclude} for group in groups],
    }
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Resume settings differ from {path}; choose a new --resume-dir")
    else:
        if list(directory.glob("*.rank*.pt")):
            raise ValueError(f"Resume shards exist without {path}")
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def add_fisher_statistics(scores: dict, summary: dict, split_experts: bool = False) -> None:
    for dataset_scores in scores.values():
        for group, row in dataset_scores.items():
            count = row["parameter_count"]
            row["tensor_count"] = summary[group]["tensor_count"]
            row["examples"] = summary[group]["examples"]
            row["mean_fisher"] = row["fisher_mass"] / count if count else None
        for group_names in GROUP_PARTITIONS.values():
            total = sum(dataset_scores[group]["fisher_mass"] for group in group_names)
            for group in group_names:
                dataset_scores[group]["fisher_mass_percent"] = (
                    100 * dataset_scores[group]["fisher_mass"] / total if total else None)
        if split_experts:
            overall = sum(dataset_scores[group]["fisher_mass"]
                          for group in GROUP_PARTITIONS["component"])
            expertized = dataset_scores["expertized_ffn"]["fisher_mass"]
            for tower in ("vision", "language"):
                tower_ffn = dataset_scores[f"{tower}_ffn"]["fisher_mass"]
                for index in (0, 1):
                    row = dataset_scores[f"{tower}_expert_{index}"]
                    row["fisher_mass_percent"] = (
                        100 * row["fisher_mass"] / overall if overall else None)
                    row["within_expertized_percent"] = (
                        100 * row["fisher_mass"] / expertized if expertized else None)
                    row["within_tower_ffn_percent"] = (
                        100 * row["fisher_mass"] / tower_ffn if tower_ffn else None)


def fisher_worker(rank: int, device_names: list[str], args: argparse.Namespace,
                  datasets: dict[str, str], counts: dict[str, int],
                  resume_dir: str, groups: tuple[ParameterGroup, ...],
                  progress_events=None) -> dict:
    device = resolve_device(device_names[rank])
    generator = torch.Generator().manual_seed(args.seed + rank)
    processor = load_processor(args.processor, args.trust_remote_code)
    result = {}
    model = None
    try:
        for name, path in datasets.items():
            expected = shard_example_count(counts[name], rank, len(device_names))
            if expected == 0:
                continue
            checkpoint = shard_checkpoint_path(Path(resume_dir), name, rank)
            saved, raw = load_shard_checkpoint(checkpoint, expected, generator)
            if saved == expected:
                result[name] = {"count": saved, "groups": normalized_scores(raw, saved)}
                print(f"worker {rank}: {name} restored {saved} examples", flush=True)
                continue
            if model is None:
                model = load_flex_model(args.model, args.trust_remote_code, device)
                if args.activation_checkpointing:
                    enable_activation_checkpointing(model)
            seen = [0]
            examples = processed_examples(
                path, processor, device, args.max_examples, rank=rank,
                world_size=len(device_names), counter=seen, skip_examples=saved)
            try:
                group_scores = estimate_conditional_token_fisher(
                    model, examples, groups, generator,
                    score_deltas={},
                    on_example_complete=(lambda: progress_events.put(name))
                    if progress_events is not None else None,
                    checkpoint_callback=lambda count, rows: save_shard_checkpoint(
                        checkpoint, count, rows, generator),
                    initial_scores=raw, initial_count=saved,
                )
            finally:
                del examples
            result[name] = {"count": saved + seen[0], "groups": group_scores}
            print(f"worker {rank}: {name} processed {seen[0]} new examples "
                  f"({saved + seen[0]} total)", flush=True)
    finally:
        del model
        release_device(device)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_examples < 0:
        raise ValueError("--max-examples must be nonnegative")
    datasets = named_paths(args.examples, "--examples")
    groups = fisher_groups(args.split_experts)
    devices = resolve_devices(args.device, args.devices)
    counts = {name: count_jsonl_rows(path, args.max_examples)
              for name, path in datasets.items()}
    if any(count == 0 for count in counts.values()):
        raise ValueError("Every dataset needs at least one nonempty JSONL row")
    resume_dir = Path(args.resume_dir or (args.output + ".resume"))
    prepare_manifest(resume_dir, args, datasets, devices, groups)
    restored = 0
    for name, count in counts.items():
        for rank in range(len(devices)):
            saved, _ = load_shard_checkpoint(
                shard_checkpoint_path(resume_dir, name, rank),
                shard_example_count(count, rank, len(devices)), torch.Generator())
            restored += saved

    # A CPU model is used only to list canonical parameters and audit group coverage.
    model = load_flex_model(args.model, args.trust_remote_code)
    parameters = {name: p for name, p in model.named_parameters(remove_duplicate=True)
                  if p.is_floating_point()}
    summary = group_summary(parameters, groups)
    if args.split_experts:
        validate_expert_split(summary)
    del model, parameters
    for group, row in summary.items():
        print(f"{group}: {row['tensor_count']} tensors, "
              f"{row['parameter_count']} coordinates; examples={row['examples']}", flush=True)

    device_names = list(map(str, devices))
    total = sum(counts.values())
    print(f"Fisher devices: {', '.join(device_names)}; {restored}/{total} examples restored", flush=True)
    with tqdm(total=total, initial=restored, desc="Fisher examples", unit="example",
              file=sys.stdout, ascii=True, dynamic_ncols=False, mininterval=30) as progress:
        if len(devices) == 1:
            shards = [fisher_worker(0, device_names, args, datasets, counts,
                                    str(resume_dir), groups, progress_events=None)]
            progress.update(total - restored)
        else:
            with get_context("spawn").Manager() as manager:
                events = manager.Queue()
                with ProcessPoolExecutor(max_workers=len(devices),
                                         mp_context=get_context("spawn")) as pool:
                    futures = [pool.submit(fisher_worker, rank, device_names, args,
                                           datasets, counts, str(resume_dir), groups, events)
                               for rank in range(len(devices))]
                    completed = restored
                    while completed < total:
                        try:
                            name = events.get(timeout=2)
                        except Empty:
                            for future in futures:
                                if future.done():
                                    future.result()
                            if all(future.done() for future in futures):
                                break
                            continue
                        completed += 1
                        progress.update(1)
                        progress.set_postfix_str(name, refresh=False)
                    shards = [future.result() for future in futures]
                    if completed != total:
                        raise RuntimeError(f"Workers reported {completed} of {total} completed examples")
    scores = combine_shards(shards, list(datasets))
    add_fisher_statistics(scores, summary, args.split_experts)
    result = {
        "method": "standalone_fisher_mass",
        "split_experts": args.split_experts,
        "model": args.model,
        "processor": args.processor,
        "groups": summary,
        "fisher_group_scores": scores,
        "examples": datasets,
        "devices": device_names,
        "max_examples": args.max_examples,
        "resume_dir": str(resume_dir),
        "activation_checkpointing": args.activation_checkpointing,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    try:
        torch.save(result, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved standalone Fisher analysis to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
