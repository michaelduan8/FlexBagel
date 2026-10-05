"""Run the existing checkpoint methods separately for each transformer layer.

Run from the FlexBagel root with ``python -m analysis.run_layer_checkpoint_analysis``.
Only parameters inside vision blocks or language decoder layers are included.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

from analysis.expert_analysis import (
    ParameterGroup, group_summary, sign_conflicts, task_vector_similarity,
)
from analysis.run_expert_analysis import load_states, named_paths


LAYER_KEY = re.compile(
    r"^(?:(?:model\.)?visual\.(?:blocks|layers)|"
    r"(?:model\.)?(?:language_model\.)?layers)\.(\d+)\."
)


def layer_groups(parameter_names: list[str]) -> tuple[ParameterGroup, ...]:
    """Build one nonoverlapping group per vision or language transformer block."""
    found: set[tuple[str, int]] = set()
    for name in parameter_names:
        match = LAYER_KEY.match(name)
        if match:
            tower = "vision" if ".visual." in f".{name}" else "language"
            found.add((tower, int(match.group(1))))
    if not found:
        raise ValueError("No vision blocks or language layers found in checkpoint")
    groups = []
    for tower, index in sorted(found):
        if tower == "vision":
            prefix = rf"^(?:model\.)?visual\.(?:blocks|layers)\.{index}\."
        else:
            prefix = rf"^(?:model\.)?(?:language_model\.)?layers\.{index}\."
        groups.append(ParameterGroup(f"{tower}_layer_{index}", prefix))
    return tuple(groups)


def write_tsv(path: Path, method: str, scores: dict, expert_names: list[str]) -> None:
    """Write a compact, readable view alongside the complete PyTorch result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if method == "task-vectors":
            writer.writerow(("layer", "parameter_count", "expert", "update_l2",
                             "update_rms", "relative_l2", "pair", "cosine"))
            for layer, row in scores.items():
                for expert in expert_names:
                    values = row["experts"][expert]
                    writer.writerow((layer, row["parameter_count"], expert,
                                     values["update_l2"], values["update_rms"],
                                     values["relative_l2"], "", ""))
                for pair, values in row["pairs"].items():
                    writer.writerow((layer, row["parameter_count"], "", "", "", "",
                                     "/".join(pair), values["cosine"]))
        else:
            writer.writerow(("layer", "parameter_count", "pair", "selected_a",
                             "selected_b", "overlap_count", "jaccard",
                             "opposite_sign_count", "opposite_sign_fraction"))
            for layer, row in scores.items():
                for (a, b), values in row["pairs"].items():
                    writer.writerow((layer, row["parameter_count"], f"{a}/{b}",
                                     row["selected_counts"][a],
                                     row["selected_counts"][b],
                                     values["overlap_count"], values["jaccard"],
                                     values["opposite_sign_count"],
                                     values["opposite_sign_fraction"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--expert", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--method", choices=("all", "task-vectors", "sign-conflicts"),
                        default="all")
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.top_percent <= 100:
        parser.error("--top-percent must be in (0, 100]")
    expert_paths = named_paths(args.expert, "--expert")
    base, experts, parameter_names = load_states(
        args.base, expert_paths, args.trust_remote_code)
    groups = layer_groups(parameter_names)
    summary = group_summary({name: base[name] for name in parameter_names}, groups)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ("task-vectors", "sign-conflicts") if args.method == "all" else (args.method,)
    for method in methods:
        if method == "task-vectors":
            scores = task_vector_similarity(
                base, experts, groups=groups, parameter_names=parameter_names)
            key = "task_vectors"
        else:
            scores = sign_conflicts(
                base, experts, top_percent=args.top_percent, groups=groups,
                parameter_names=parameter_names)
            key = "sign_conflicts"
        result = {"method": method, "scope": "per_layer", "base": args.base,
                  "experts": expert_paths, "groups": summary, key: scores}
        if method == "sign-conflicts":
            result["top_percent"] = args.top_percent
        torch.save(result, output_dir / f"{method}-per-layer.pt")
        write_tsv(output_dir / f"{method}-per-layer.tsv", method, scores,
                  list(experts))
        print(f"Wrote {method} for {len(groups)} layers to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
