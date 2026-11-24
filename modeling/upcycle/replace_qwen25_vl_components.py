#!/usr/bin/env python3
"""Copy selected Qwen2.5-VL block weights from a donor into a target checkpoint.

Example:
    python modeling/upcycle/replace_qwen25_vl_components.py \
        --target /path/to/target --donor /path/to/donor \
        --output /path/to/hybrid \
        --components vision_ffn language_attention

Inputs may be local Hugging Face safetensors checkpoints or Hub model IDs.
The target supplies all unselected weights and config/processor files. Vision merger and layer norms
are outside the four selectable components.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import re
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file

COMPONENTS = ("vision_ffn", "vision_attention", "language_ffn", "language_attention")
PATTERNS = {
    "vision_ffn": re.compile(r"^(?:model\.)?visual\.blocks\.\d+\.mlp\."),
    "vision_attention": re.compile(r"^(?:model\.)?visual\.blocks\.\d+\.attn\."),
    "language_ffn": re.compile(r"^(?:model\.language_model|model|language_model\.model)\.layers\.\d+\.mlp\."),
    "language_attention": re.compile(r"^(?:model\.language_model|model|language_model\.model)\.layers\.\d+\.self_attn\."),
}


def component_for(key: str) -> str | None:
    return next((name for name, pattern in PATTERNS.items() if pattern.match(key)), None)


def checkpoint_files(directory: Path) -> dict[str, Path]:
    """Map tensor names to safetensors files using the HF index if present."""
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map in {index_path}")
        result = {}
        for key, filename in weight_map.items():
            if not isinstance(key, str) or not isinstance(filename, str):
                raise ValueError(f"Invalid index entry in {index_path}")
            if Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise ValueError(f"Unsafe shard name {filename!r}")
            shard = directory / filename
            if not shard.is_file():
                raise FileNotFoundError(shard)
            result[key] = shard
        return result
    single = directory / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"No safetensors checkpoint in {directory}")
    with safe_open(str(single), framework="pt", device="cpu") as handle:
        return {key: single for key in handle.keys()}


def validate_selection(target_files, donor_files, components):
    selected = {key: path for key, path in target_files.items()
                if component_for(key) in components}
    donor_selected = {key for key in donor_files if component_for(key) in components}
    for component in components:
        if not any(component_for(key) == component for key in selected):
            raise ValueError(f"Target has no tensors for {component}")
        if not any(component_for(key) == component for key in donor_selected):
            raise ValueError(f"Donor has no tensors for {component}")
    if selected.keys() != donor_selected:
        missing = sorted(selected.keys() - donor_selected)
        extra = sorted(donor_selected - selected.keys())
        raise ValueError(f"Selected tensor names differ: missing in donor={missing[:8]} "
                         f"({len(missing)} total); extra in donor={extra[:8]} ({len(extra)} total)")
    with ExitStack() as stack:
        handles = {path: stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
                   for path in set(selected.values()) | {donor_files[key] for key in selected}}
        for key, target_path in selected.items():
            target_handle, donor_handle = handles[target_path], handles[donor_files[key]]
            if key not in target_handle.keys() or key not in donor_handle.keys():
                raise ValueError(f"Index references a missing tensor: {key}")
            target_slice, donor_slice = target_handle.get_slice(key), donor_handle.get_slice(key)
            if (target_slice.get_shape() != donor_slice.get_shape()
                    or target_slice.get_dtype() != donor_slice.get_dtype()):
                raise ValueError(f"Shape or dtype mismatch for {key}: "
                                 f"target={target_slice.get_shape()}/{target_slice.get_dtype()}, "
                                 f"donor={donor_slice.get_shape()}/{donor_slice.get_dtype()}")
    return selected


def resolve_checkpoint(source: str | Path) -> Path:
    path = Path(source).expanduser()
    if path.exists():
        return path.resolve()
    if path.is_absolute() or str(source).startswith(("./", "../")):
        raise FileNotFoundError(path)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(source)))


def replace_components(target: str | Path, donor: str | Path, output: Path, components: list[str]) -> Counter:
    target, donor, output = resolve_checkpoint(target), resolve_checkpoint(donor), output.resolve()
    if target == donor:
        raise ValueError("Target and donor must be different directories")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if target in output.parents or donor in output.parents:
        raise ValueError("Output cannot be inside either input checkpoint")
    for directory in (target, donor):
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(directory / "config.json")
    target_files, donor_files = checkpoint_files(target), checkpoint_files(donor)
    selected = validate_selection(target_files, donor_files, components)
    counts = Counter(component_for(key) for key in selected)
    by_shard = {}
    for key, path in selected.items():
        by_shard.setdefault(path.relative_to(target), []).append(key)

    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        shutil.copytree(target, work, dirs_exist_ok=True)
        for relative_path, keys in by_shard.items():
            output_shard = work / relative_path
            with ExitStack() as stack:
                target_handle = stack.enter_context(safe_open(str(target / relative_path), framework="pt", device="cpu"))
                donor_handles = {path: stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
                                 for path in {donor_files[key] for key in keys}}
                tensors = {key: target_handle.get_tensor(key) for key in target_handle.keys()}
                for key in keys:
                    tensors[key] = donor_handles[donor_files[key]].get_tensor(key)
                metadata = target_handle.metadata()
                new_shard = output_shard.with_name(output_shard.name + ".new")
                save_file(tensors, str(new_shard), metadata=metadata)
            new_shard.replace(output_shard)
        work.rename(output)
    except BaseException:
        shutil.rmtree(work)
        raise
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Base HF checkpoint directory or Hub ID")
    parser.add_argument("--donor", required=True, help="HF checkpoint directory or Hub ID supplying weights")
    parser.add_argument("--output", type=Path, required=True, help="New local HF checkpoint directory")
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, required=True)
    args = parser.parse_args()
    counts = replace_components(args.target, args.donor, args.output, args.components)
    print(f"Saved {args.output}")
    for component in dict.fromkeys(args.components):
        print(f"  {component}: {counts[component]} tensors")


if __name__ == "__main__":
    main()