"""Command-line checkpoint diagnostics and direct Fisher group scoring.

Run from the FlexBagel root with:
    python -m analysis.run_expert_analysis --help
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from queue import Empty
from collections.abc import Mapping
from pathlib import Path
from typing import Iterator

import torch
from tqdm.auto import tqdm

from analysis.expert_analysis import (
    DEFAULT_GROUPS,
    GROUP_PARTITIONS,
    estimate_conditional_token_fisher,
    group_summary,
    sign_conflicts,
    task_vector_similarity,
)


def named_paths(values: list[str], flag: str) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"{flag} requires distinct NAME=PATH entries")
        result[name] = path
    return result


def named_coefficients(values: list[str], names: set[str]) -> dict[str, float]:
    if not values:
        return {name: 1.0 for name in names}
    raw = named_paths(values, "--coefficient")
    if set(raw) != names:
        raise ValueError("--coefficient must specify every expert exactly once")
    try:
        result = {name: float(value) for name, value in raw.items()}
    except ValueError as error:
        raise ValueError("--coefficient values must be numbers") from error
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("--coefficient values must be finite and nonnegative")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare compatible checkpoints and estimate direct Fisher group scores.")
    parser.add_argument("--base", required=True,
                        help="Base checkpoint directory accepted by Transformers from_pretrained.")
    parser.add_argument("--expert", action="append", required=True, metavar="NAME=PATH",
                        help="Expert name and checkpoint directory; repeat for each expert.")
    parser.add_argument("--method", required=True,
                        choices=("task-vectors", "sign-conflicts", "fisher"),
                        help="Run one independent analysis and write its own result file.")
    parser.add_argument("--examples", action="append", default=[], metavar="NAME=PATH",
                        help="PathGen-style JSONL conversation and image file for one expert; repeat.")
    parser.add_argument("--fisher-expert", metavar="NAME",
                        help="In Fisher mode, process only this expert while retaining all experts for averaging.")
    parser.add_argument("--resume-dir", metavar="DIR",
                        help="Fisher shard checkpoints; default: OUTPUT.resume.")
    parser.add_argument("--output", required=True,
                        help="Destination .pt file for group summaries and diagnostic scores.")
    parser.add_argument("--coefficient", action="append", default=[], metavar="NAME=WEIGHT",
                        help="Nonnegative static averaging weight; repeat for every expert. Default: 1 each.")
    parser.add_argument("--base-coefficient", type=float, default=0.0, metavar="WEIGHT",
                        help="Nonnegative base weight in the average. Default: 0.")
    parser.add_argument("--average-mode", choices=("non-ffn", "all"), default="non-ffn",
                        help="Average shared non-FFN parameters (preserving the connector) "
                             "or all parameters, including the connector. Default: non-ffn.")
    parser.add_argument("--top-percent", type=float, default=1.0,
                        help="Percent of nonzero update coordinates for sign conflicts. Default: 1.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for sampled Fisher targets. Default: 0.")
    parser.add_argument("--max-examples", type=int, default=0, metavar="N",
                        help="Process at most N JSONL rows per expert; 0 means all rows (default).")
    parser.add_argument("--processor", metavar="PATH",
                        help="Processor checkpoint directory. Default: --base.")
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--device", default="auto", metavar="DEVICE",
                              help="auto uses every visible XPU (or CUDA GPU); "
                                   "xpu:N, cuda:N, and cpu select one device. Default: auto.")
    device_group.add_argument("--devices", metavar="DEVICE,DEVICE",
                              help="Run data-parallel Fisher workers on listed GPUs, "
                                   "for example xpu:0,xpu:1. Each GPU holds one model replica.")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Recompute transformer blocks during Fisher backward to reduce GPU memory; "
                             "preserves evaluation mode.")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow Transformers to load custom checkpoint model code.")
    return parser.parse_args(argv)


def resolve_device(requested: str, *, set_current: bool = True) -> torch.device:
    if requested == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            requested = "xpu:0"
        elif torch.cuda.is_available():
            requested = "cuda:0"
        else:
            raise RuntimeError("No GPU found. Run on an Aurora GPU node or pass --device cpu.")
    device = torch.device(requested)
    if device.type == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError(f"XPU device unavailable: {device}")
        if device.index is not None and device.index >= torch.xpu.device_count():
            raise RuntimeError(f"XPU device index out of range: {device}")
        if set_current:
            torch.xpu.set_device(device)
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device unavailable: {device}")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index out of range: {device}")
        if set_current:
            torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("--device must be auto, xpu:N, cuda:N, or cpu")
    return device


def resolve_devices(single: str, multiple: str | None) -> list[torch.device]:
    if multiple is None:
        if single == "auto":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return [torch.device(f"xpu:{index}")
                        for index in range(torch.xpu.device_count())]
            if torch.cuda.is_available():
                return [torch.device(f"cuda:{index}")
                        for index in range(torch.cuda.device_count())]
            raise RuntimeError(
                "No GPU found. Run on an Aurora GPU node or pass --device cpu.")
        return [resolve_device(single, set_current=False)]
    parts = [part.strip() for part in multiple.split(",")]
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError("--devices needs at least two comma-separated GPU devices")
    if any(part == "auto" for part in parts):
        raise ValueError("--devices requires explicit GPU indices")
    devices = [resolve_device(part, set_current=False) for part in parts]
    if any(device.type not in {"xpu", "cuda"} for device in devices):
        raise ValueError("--devices accepts only XPU or CUDA GPUs")
    if len(devices) != len(set(devices)):
        raise ValueError("--devices must list distinct GPUs")
    if len({device.type for device in devices}) != 1:
        raise ValueError("--devices cannot mix XPU and CUDA")
    return devices


def release_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_model(path: str, trust_remote_code: bool, device: torch.device | None = None):
    from transformers import AutoModelForImageTextToText
    from eval.vlm.eval.rexvqa.evaluate_rexvqa import (
        register_local_transformers_architectures,
    )

    register_local_transformers_architectures()
    model = AutoModelForImageTextToText.from_pretrained(
        path, trust_remote_code=trust_remote_code, torch_dtype="auto",
        low_cpu_mem_usage=True,
    ).eval()
    return model.to(device) if device is not None else model


def enable_activation_checkpointing(model: torch.nn.Module) -> None:
    """Checkpoint blocks in eval mode; Transformers' training-only toggle is unsuitable."""
    from torch.utils.checkpoint import checkpoint

    count = 0
    for name, module in model.named_modules():
        if not re.search(r"\.(?:layers|blocks)\.\d+$", name):
            continue
        original_forward = module.forward

        def recompute_forward(*args, _forward=original_forward, **kwargs):
            if not torch.is_grad_enabled():
                return _forward(*args, **kwargs)
            return checkpoint(_forward, *args, use_reentrant=False, **kwargs)

        module.forward = recompute_forward
        count += 1
    if count == 0:
        raise ValueError("No transformer blocks found for activation checkpointing")
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "use_cache"):
            module.config.use_cache = False
    print(f"Activation checkpointing: {count} blocks (evaluation mode)", flush=True)


def load_processor(path: str, trust_remote_code: bool):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        path, trust_remote_code=trust_remote_code, use_fast=True)


def average_parameter_names(names: list[str], mode: str) -> set[str]:
    groups = {group.name: group for group in DEFAULT_GROUPS}
    connector = groups["connector"]
    ffn = groups["architectural_ffn"]
    expertized = groups["expertized_ffn"]
    if mode == "all":
        return set(names)
    eligible = {name for name in names if not connector.matches(name)}
    if mode == "non-ffn":
        return {name for name in eligible
                if not ffn.matches(name) and not expertized.matches(name)
                and not (".mlp.gate." in name or ".mlp.router." in name)}
    raise ValueError(f"Unsupported average mode: {mode}")


def iter_jsonl(path: str, max_examples: int) -> Iterator[dict]:
    if max_examples < 0:
        raise ValueError("--max-examples must be nonnegative")
    source = Path(path)
    if source.suffix.lower() != ".jsonl":
        raise ValueError(f"Expected a JSONL examples file: {source}")
    with source.open(encoding="utf-8") as handle:
        selected = 0
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number} must contain a JSON object")
            yield record
            selected += 1
            if max_examples and selected >= max_examples:
                break


def count_jsonl_rows(path: str, max_examples: int) -> int:
    """Count the nonempty rows the estimator will visit."""
    count = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
                if max_examples and count >= max_examples:
                    break
    return count



def shard_example_count(total: int, rank: int, world_size: int) -> int:
    return max(0, (total + world_size - 1 - rank) // world_size)


def shard_checkpoint_path(directory: Path, expert: str, rank: int) -> Path:
    return directory / f"{expert}.rank{rank}.pt"


def save_shard_checkpoint(path: Path, count: int, scores: dict,
                          generator: torch.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save({"count": count, "scores": scores,
                    "generator_state": generator.get_state()}, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_shard_checkpoint(path: Path, expected_count: int,
                          generator: torch.Generator) -> tuple[int, dict | None]:
    if not path.exists():
        return 0, None
    saved = torch.load(path, map_location="cpu", weights_only=True)
    count = saved["count"]
    if not isinstance(count, int) or not 0 <= count <= expected_count:
        raise ValueError(f"Invalid saved example count in {path}")
    generator.set_state(saved["generator_state"])
    return count, saved["scores"]


def normalized_scores(raw: dict, count: int) -> dict:
    if count == 0:
        raise ValueError("Cannot normalize an empty Fisher shard")
    return {group: {key: value if key == "parameter_count" else value / count
                    for key, value in row.items()}
            for group, row in raw.items()}


def prepare_resume_dir(directory: Path, args: argparse.Namespace,
                       expert_paths: Mapping[str, str],
                       example_paths: Mapping[str, str],
                       devices: list[torch.device]) -> None:
    """Reject stale shard files from a different sampling or scoring run."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 4, "base": args.base, "experts": dict(expert_paths),
        "examples": {name: {"path": str(Path(path).resolve()),
                            "size": Path(path).stat().st_size,
                            "mtime_ns": Path(path).stat().st_mtime_ns}
                     for name, path in example_paths.items()},
        "processor": args.processor or args.base,
        "max_examples": args.max_examples, "seed": args.seed,
        "activation_checkpointing": getattr(args, "activation_checkpointing", False),
        "average_mode": args.average_mode,
        "coefficients": args.coefficient,
        "base_coefficient": args.base_coefficient,
        "devices": [str(device) for device in devices],
        "groups": [{"name": group.name, "include": group.include,
                    "exclude": group.exclude} for group in DEFAULT_GROUPS],
    }
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                f"Resume settings differ from {path}; choose a new --resume-dir")
    else:
        if list(directory.glob("*.rank*.pt")):
            raise ValueError(f"Resume shards exist without {path}")
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            temporary.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)



def encode_record(record: Mapping, processor, jsonl_path: str) -> dict[str, torch.Tensor]:
    """Build a teacher-forced final assistant answer from a PathGen JSONL row."""
    from PIL import Image

    turns = record.get("conversation")
    if not isinstance(turns, list) or not turns or turns[-1].get("role") != "assistant":
        raise ValueError(f"Record {record.get('id')} needs a final assistant turn")
    image_paths = record.get("images") or []
    if not isinstance(image_paths, list):
        raise ValueError(f"Record {record.get('id')} has invalid images")
    messages = []
    images = []
    used_images = False
    for turn in turns:
        role = turn.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Record {record.get('id')} has unsupported role {role!r}")
        content = [{"type": "text", "text": str(turn.get("content", ""))}]
        if role == "user" and turn.get("img_loc") is not None and not used_images:
            location = turn["img_loc"]
            if location not in {"before", "after"}:
                raise ValueError(f"Record {record.get('id')} has invalid img_loc")
            for image_path in image_paths:
                path = Path(image_path)
                if not path.is_absolute():
                    path = Path(jsonl_path).parent / path
                with Image.open(path) as loaded:
                    images.append(loaded.convert("RGB"))
            blocks = [{"type": "image"} for _ in images]
            content = blocks + content if location == "before" else content + blocks
            used_images = True
        messages.append({"role": role, "content": content})
    if image_paths and not used_images:
        raise ValueError(f"Record {record.get('id')} has images but no user img_loc")
    if not str(turns[-1].get("content", "")):
        raise ValueError(f"Record {record.get('id')} has an empty final answer")

    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    prefix_text = processor.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True)
    kwargs = {"images": images} if images else {}
    full = processor(text=[full_text], return_tensors="pt", **kwargs)
    prefix = processor(text=[prefix_text], return_tensors="pt", **kwargs)
    input_ids = full["input_ids"]
    prefix_ids = prefix["input_ids"]
    count = prefix_ids.shape[1]
    if count < 1 or count >= input_ids.shape[1] or not torch.equal(
        input_ids[:, :count], prefix_ids
    ):
        raise ValueError(
            f"Record {record.get('id')} prefix tokens do not align with full answer")
    labels = input_ids.clone()
    labels[:, :count] = -100
    return {**dict(full), "labels": labels}


def processed_examples(path: str, processor, device: torch.device,
                       max_examples: int, *, rank: int = 0, world_size: int = 1,
                       counter: list[int] | None = None,
                       skip_examples: int = 0) -> Iterator[dict[str, torch.Tensor]]:
    shard_index = 0
    for index, record in enumerate(iter_jsonl(path, max_examples)):
        if index % world_size != rank:
            continue
        if shard_index < skip_examples:
            shard_index += 1
            continue
        shard_index += 1
        example = encode_record(record, processor, path)
        if counter is not None:
            counter[0] += 1
        yield {key: value.to(device) if isinstance(value, torch.Tensor) else value
               for key, value in example.items()}


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()}


class Deltas(Mapping):
    """Compute a parameter delta only when the Fisher estimator visits it."""

    def __init__(self, keys: list[str], base: Mapping, experts: Mapping,
                 expert_name: str, destination: str, weights: Mapping[str, float],
                 base_weight: float, average_names: set[str]):
        self.keys = keys
        self.key_set = set(keys)
        self.base = base
        self.experts = experts
        self.expert_name = expert_name
        self.destination = destination
        self.weights = weights
        self.base_weight = base_weight
        self.average_names = average_names

    def __iter__(self):
        return iter(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, key):
        if key not in self.key_set:
            raise KeyError(key)
        source = self.experts[self.expert_name][key].float()
        if self.destination == "base":
            return source - self.base[key].float()
        if key not in self.average_names:
            return torch.zeros_like(source)
        average = self.base[key].float() * self.base_weight
        for name, state in self.experts.items():
            average = average + state[key].float() * self.weights[name]
        return source - average



def load_states(base_path: str, expert_paths: Mapping[str, str],
                trust_remote_code: bool) -> tuple[dict, dict, list[str]]:
    base_model = load_model(base_path, trust_remote_code)
    parameter_names = [name for name, parameter in base_model.named_parameters()
                       if parameter.is_floating_point()]
    base = cpu_state(base_model)
    del base_model
    experts = {}
    for name, path in expert_paths.items():
        model = load_model(path, trust_remote_code)
        names = [key for key, p in model.named_parameters()
                 if p.is_floating_point()]
        if names != parameter_names:
            raise ValueError(f"{name} has different canonical parameter names")
        experts[name] = cpu_state(model)
        del model
    return base, experts, parameter_names


def fisher_shard(rank: int, devices: list[str], args: argparse.Namespace,
                 expert_paths: dict[str, str], example_paths: dict[str, str],
                 weights: dict[str, float], base_weight: float,
                 progress_events=None, resume_dir: str | None = None,
                 example_counts: dict[str, int] | None = None) -> dict:
    """One spawned process owns one GPU and its disjoint JSONL records."""
    device = resolve_device(devices[rank])
    print(f"worker {rank}: loading checkpoints on {device}", flush=True)
    base, experts, parameter_names = load_states(
        args.base, expert_paths, args.trust_remote_code)
    average_names = average_parameter_names(parameter_names, args.average_mode)
    processor = load_processor(args.processor or args.base, args.trust_remote_code)
    generator = torch.Generator().manual_seed(args.seed + rank)
    result = {}
    for name, path in example_paths.items():
        expected = (shard_example_count(example_counts[name], rank, len(devices))
                    if example_counts is not None else None)
        checkpoint = (shard_checkpoint_path(Path(resume_dir), name, rank)
                      if resume_dir is not None else None)
        saved_count, saved_scores = (
            load_shard_checkpoint(checkpoint, expected, generator)
            if checkpoint is not None else (0, None))
        if expected is not None and saved_count == expected and saved_count:
            result[name] = {"count": saved_count,
                            "groups": normalized_scores(saved_scores, saved_count)}
            print(f"worker {rank}: {name} restored {saved_count} examples", flush=True)
            continue
        model = load_model(expert_paths[name], args.trust_remote_code, device)
        if getattr(args, "activation_checkpointing", False):
            enable_activation_checkpointing(model)
        counter = [0]
        examples = processed_examples(
            example_paths[name], processor, device, args.max_examples,
            rank=rank, world_size=len(devices), counter=counter,
            skip_examples=saved_count)
        try:
            groups = estimate_conditional_token_fisher(
                model, examples, DEFAULT_GROUPS, generator=generator,
                on_example_complete=(lambda: progress_events.put(name))
                if progress_events is not None else None,
                checkpoint_callback=(
                    lambda count, scores: save_shard_checkpoint(
                        checkpoint, count, scores, generator))
                if checkpoint is not None else None,
                initial_scores=saved_scores, initial_count=saved_count,
                score_deltas={
                    metric: Deltas(parameter_names, base, experts, name, destination,
                                   weights, base_weight, average_names)
                    for metric, destination in (
                        ("base_to_expert", "base"),
                        ("expert_to_average", "average"),
                    )
                },
            )
        except ValueError as error:
            if counter[0] != 0 or str(error) != "At least one example is required":
                raise
            groups = None
        finally:
            del model, examples
            release_device(device)
        if groups is not None:
            result[name] = {"count": saved_count + counter[0], "groups": groups}
        print(f"worker {rank}: {name} processed {counter[0]} new examples "
              f"({saved_count + counter[0]} total)", flush=True)
    return result


def combine_shards(shards: list[dict], expert_names: list[str]) -> dict:
    """Combine shard means using their number of examples, not worker count."""
    result = {}
    for name in expert_names:
        contributions = [shard[name] for shard in shards if name in shard]
        total = sum(item["count"] for item in contributions)
        if total == 0:
            raise ValueError(f"No examples were processed for expert {name!r}")
        groups = {}
        for group in contributions[0]["groups"]:
            counts = {item["groups"][group]["parameter_count"]
                      for item in contributions}
            if len(counts) != 1:
                raise ValueError(f"Parameter coverage differs across workers for {group}")
            row = {"parameter_count": counts.pop(), "example_count": total}
            metrics = set(contributions[0]["groups"][group]) - {"parameter_count", "example_count"}
            if any(set(item["groups"][group]) - {"parameter_count", "example_count"} != metrics
                   for item in contributions):
                raise ValueError(f"Fisher metrics differ across workers for {group}")
            for metric in metrics:
                row[metric] = sum(item["count"] * item["groups"][group][metric]
                                  for item in contributions) / total
            groups[group] = row
        result[name] = groups
    return result

def add_score_statistics(scores: dict, summary: dict) -> None:
    for expert_scores in scores.values():
        for group, row in expert_scores.items():
            count = row["parameter_count"]
            mass = row["fisher_mass"]
            row["examples"] = summary[group]["examples"]
            row["tensor_count"] = summary[group]["tensor_count"]
            row["mean_fisher"] = mass / count if count else None
            for metric in ("base_to_expert", "expert_to_average"):
                if metric not in row:
                    continue
                row[metric + "_per_parameter"] = row[metric] / count if count else None
                row[metric + "_fisher_rms"] = (
                    math.sqrt(row[metric] / mass) if mass else None)
        for group_names in GROUP_PARTITIONS.values():
            for metric in ("base_to_expert", "expert_to_average"):
                if metric not in expert_scores[group_names[0]]:
                    continue
                total = sum(expert_scores[group][metric] for group in group_names)
                for group in group_names:
                    expert_scores[group][metric + "_percent"] = (
                        100 * expert_scores[group][metric] / total if total else None)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experts_paths = named_paths(args.expert, "--expert")
    example_paths = named_paths(args.examples, "--examples")
    if args.method == "fisher":
        if args.fisher_expert is not None:
            if args.fisher_expert not in experts_paths:
                raise ValueError("--fisher-expert must name one of the supplied experts")
            if args.fisher_expert not in example_paths:
                raise ValueError("--fisher-expert requires its matching --examples")
            example_paths = {args.fisher_expert: example_paths[args.fisher_expert]}
        elif set(example_paths) != set(experts_paths):
            raise ValueError("Fisher requires --examples for every expert")
    elif example_paths:
        raise ValueError("--examples is only used by --method fisher")
    elif args.fisher_expert is not None:
        raise ValueError("--fisher-expert is only used by --method fisher")
    if not 0 < args.top_percent <= 100:
        raise ValueError("--top-percent must be in (0, 100]")
    if args.max_examples < 0:
        raise ValueError("--max-examples must be nonnegative")

    coefficients = named_coefficients(args.coefficient, set(experts_paths))
    if not math.isfinite(args.base_coefficient) or args.base_coefficient < 0:
        raise ValueError("--base-coefficient must be finite and nonnegative")
    total = sum(coefficients.values()) + args.base_coefficient
    if total <= 0:
        raise ValueError("Average weights must have positive total")
    weights = {name: value / total for name, value in coefficients.items()}
    base_weight = args.base_coefficient / total

    devices = resolve_devices(args.device, args.devices) if args.method == "fisher" else []
    example_counts = {}
    resume_dir = None
    restored = 0
    if args.method == "fisher":
        print(f"Fisher devices: {', '.join(map(str, devices))}", flush=True)
        example_counts = {name: count_jsonl_rows(path, args.max_examples)
                          for name, path in example_paths.items()}
        if any(count == 0 for count in example_counts.values()):
            raise ValueError("Every expert needs at least one nonempty JSONL row")
        resume_dir = Path(args.resume_dir or (args.output + ".resume"))
        prepare_resume_dir(resume_dir, args, experts_paths, example_paths, devices)
        for name, count in example_counts.items():
            for rank in range(len(devices)):
                checkpoint = shard_checkpoint_path(resume_dir, name, rank)
                saved, _ = load_shard_checkpoint(
                    checkpoint, shard_example_count(count, rank, len(devices)),
                    torch.Generator())
                restored += saved
        print(f"Fisher progress: {restored}/{sum(example_counts.values())} "
              "examples already saved", flush=True)

    base, experts, parameter_names = load_states(
        args.base, experts_paths, args.trust_remote_code)
    parameters = {name: base[name] for name in parameter_names}
    summary = group_summary(parameters)
    for group, row in summary.items():
        print(f"{group}: {row['tensor_count']} tensors, "
              f"{row['parameter_count']} coordinates; examples={row['examples']}")
    result = {"method": args.method, "groups": summary}
    if args.method == "task-vectors":
        result["task_vectors"] = task_vector_similarity(
            base, experts, parameter_names=parameter_names)
    elif args.method == "sign-conflicts":
        result["sign_conflicts"] = sign_conflicts(
            base, experts, top_percent=args.top_percent,
            parameter_names=parameter_names)
        result["top_percent"] = args.top_percent
    else:
        average_names = average_parameter_names(parameter_names, args.average_mode)
        total_examples = sum(example_counts.values())
        device_names = [str(device) for device in devices]
        if len(devices) == 1:
            device = resolve_device(device_names[0])
            processor = load_processor(args.processor or args.base, args.trust_remote_code)
            generator = torch.Generator().manual_seed(args.seed)
            shard = {}
            with tqdm(total=total_examples, initial=restored, desc="Fisher examples",
                      unit="example", file=sys.stdout, ascii=True,
                      dynamic_ncols=False, mininterval=30) as progress:
                for name, path in example_paths.items():
                    checkpoint = shard_checkpoint_path(resume_dir, name, 0)
                    saved_count, saved_scores = load_shard_checkpoint(
                        checkpoint, example_counts[name], generator)
                    if saved_count == example_counts[name]:
                        shard[name] = {
                            "count": saved_count,
                            "groups": normalized_scores(saved_scores, saved_count),
                        }
                        print(f"{name}: restored {saved_count} examples", flush=True)
                        continue
                    model = load_model(experts_paths[name], args.trust_remote_code, device)
                    if args.activation_checkpointing:
                        enable_activation_checkpointing(model)
                    counter = [0]
                    examples = processed_examples(
                        example_paths[name], processor, device, args.max_examples,
                        counter=counter, skip_examples=saved_count)
                    try:
                        groups = estimate_conditional_token_fisher(
                            model, examples, DEFAULT_GROUPS, generator=generator,
                            on_example_complete=progress.update,
                            checkpoint_callback=(
                                lambda count, scores: save_shard_checkpoint(
                                    checkpoint, count, scores, generator)),
                            initial_scores=saved_scores, initial_count=saved_count,
                            score_deltas={
                                metric: Deltas(parameter_names, base, experts, name,
                                               destination, weights, base_weight,
                                               average_names)
                                for metric, destination in (
                                    ("base_to_expert", "base"),
                                    ("expert_to_average", "average"),
                                )
                            },
                        )
                        shard[name] = {"count": saved_count + counter[0],
                                       "groups": groups}
                    finally:
                        del model, examples
                        release_device(device)
            scores = combine_shards([shard], list(example_paths))
        else:
            # Each spawned worker owns one device and writes its own resume shards.
            del base, experts, parameters
            with get_context("spawn").Manager() as manager:
                progress_events = manager.Queue()
                with tqdm(total=total_examples, initial=restored,
                          desc="Fisher examples", unit="example",
                          file=sys.stdout, ascii=True, dynamic_ncols=False,
                          mininterval=30) as progress:
                    with ProcessPoolExecutor(max_workers=len(devices),
                                             mp_context=get_context("spawn")) as pool:
                        futures = [
                            pool.submit(fisher_shard, rank, device_names, args,
                                        experts_paths, example_paths, weights,
                                        base_weight, progress_events,
                                        str(resume_dir), example_counts)
                            for rank in range(len(devices))
                        ]
                        completed = restored
                        while completed < total_examples:
                            try:
                                expert_name = progress_events.get(timeout=2)
                            except Empty:
                                for future in futures:
                                    if future.done():
                                        future.result()
                                if all(future.done() for future in futures):
                                    break
                                continue
                            completed += 1
                            progress.update(1)
                            progress.set_postfix_str(expert_name, refresh=False)
                        shards = [future.result() for future in futures]
                        if completed != total_examples:
                            raise RuntimeError(
                                f"Workers reported {completed} of {total_examples} "
                                "completed Fisher examples")
            scores = combine_shards(shards, list(example_paths))
        for name, rows in scores.items():
            count = next(iter(rows.values()))["example_count"]
            print(f"{name}: {count} Fisher examples across {len(devices)} device(s)")
        add_score_statistics(scores, summary)
        result.update({
            "fisher_group_scores": scores,
            "coefficients": coefficients,
            "base_coefficient": args.base_coefficient,
            "average_mode": args.average_mode,
            "averaged_parameter_count": len(average_names),
            "devices": device_names,
            "max_examples": args.max_examples,
            "resume_dir": str(resume_dir),
            "activation_checkpointing": args.activation_checkpointing,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    try:
        torch.save(result, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {args.method} analysis to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
