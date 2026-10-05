"""Compare update magnitudes on jointly selected top-update coordinates by layer.

Selection exactly follows method 2: each specialist independently selects its
largest nonzero updates within each layer, with ties broken by parameter name
and flattened coordinate. Pairwise scores use only the intersection of those
selections. The base-to-specialist updates are measured in float64 on CPU.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from analysis.run_expert_analysis import load_states, named_paths
from analysis.run_layer_checkpoint_analysis import layer_groups


def delta(expert: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    return expert.detach().to(device="cpu", dtype=torch.float64) - base.detach().to(
        device="cpu", dtype=torch.float64)


def layer_difference(base: dict, experts: dict, keys: list[str],
                     top_percent: float) -> dict:
    """Compute pairwise update distance on each pair's top-update intersection."""
    if not 0 < top_percent <= 100:
        raise ValueError("top_percent must be in (0, 100]")
    names = list(experts)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    keys = sorted(keys)
    nonzero = {name: sum(int(torch.count_nonzero(delta(experts[name][key], base[key])))
                         for key in keys) for name in names}
    selected_counts = {name: math.ceil(count * top_percent / 100)
                       for name, count in nonzero.items()}
    cutoffs = {}
    tied_remaining = {}
    for name in names:
        k = selected_counts[name]
        if not k:
            continue
        top = torch.empty(0, dtype=torch.float64)
        for key in keys:
            magnitudes = delta(experts[name][key], base[key]).abs().reshape(-1)
            positive = magnitudes[magnitudes > 0]
            if positive.numel() > k:
                positive = torch.topk(positive, k).values
            top = torch.cat((top, positive))
            if top.numel() > k:
                top = torch.topk(top, k).values
        cutoff = top.min()
        cutoffs[name] = cutoff
        tied_remaining[name] = k - int((top > cutoff).sum())

    sums = {pair: {"overlap_count": 0, "abs_difference": 0.0,
                   "squared_difference": 0.0, "mean_squared_update": 0.0,
                   "abs_magnitude_gap": 0.0, "mean_abs_update": 0.0}
            for pair in pairs}
    for key in keys:
        changes = {name: delta(experts[name][key], base[key]).reshape(-1)
                   for name in names}
        selected = {}
        for name, values in changes.items():
            mask = torch.zeros(values.numel(), dtype=torch.bool)
            if selected_counts[name]:
                magnitudes = values.abs()
                mask = magnitudes > cutoffs[name]
                if tied_remaining[name]:
                    tied = torch.nonzero(magnitudes == cutoffs[name], as_tuple=True)[0]
                    chosen = tied[:tied_remaining[name]]
                    mask[chosen] = True
                    tied_remaining[name] -= chosen.numel()
            selected[name] = mask
        for pair, row in sums.items():
            a, b = pair
            shared = selected[a] & selected[b]
            count = int(shared.sum())
            if not count:
                continue
            va, vb = changes[a][shared], changes[b][shared]
            row["overlap_count"] += count
            row["abs_difference"] += (va - vb).abs().sum().item()
            row["squared_difference"] += (va - vb).square().sum().item()
            row["mean_squared_update"] += ((va.square() + vb.square()) / 2).sum().item()
            row["abs_magnitude_gap"] += (va.abs() - vb.abs()).abs().sum().item()
            row["mean_abs_update"] += ((va.abs() + vb.abs()) / 2).sum().item()
    if any(tied_remaining.values()):
        raise RuntimeError("Top-update tie selection did not complete")

    result = {}
    for pair, row in sums.items():
        count = row["overlap_count"]
        magnitude = row["mean_squared_update"]
        mean_abs = row["mean_abs_update"]
        result[pair] = {
            "overlap_count": count,
            "mean_abs_difference": row["abs_difference"] / count if count else None,
            "rms_difference": math.sqrt(row["squared_difference"] / count)
            if count else None,
            "relative_rms_difference": math.sqrt(row["squared_difference"] / magnitude)
            if magnitude else None,
            "mean_abs_magnitude_gap": row["abs_magnitude_gap"] / count
            if count else None,
            "relative_magnitude_gap": row["abs_magnitude_gap"] / mean_abs
            if mean_abs else None,
        }
    return {"parameter_count": sum(base[key].numel() for key in keys),
            "nonzero_counts": nonzero, "selected_counts": selected_counts,
            "pairs": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--expert", action="append", required=True, metavar="NAME=PATH")
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
    rows = {}
    for group in groups:
        keys = [name for name in parameter_names if group.matches(name)]
        rows[group.name] = layer_difference(base, experts, keys, args.top_percent)
        print(f"Completed {group.name}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"method": "update-difference", "scope": "per_layer",
                "base": args.base, "experts": expert_paths,
                "top_percent": args.top_percent, "layers": rows},
               out / "update-difference-per-layer.pt")
    with (out / "update-difference-per-layer.tsv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("layer", "pair", "parameter_count", "selected_a",
                         "selected_b", "overlap_count", "mean_abs_difference",
                         "rms_difference", "relative_rms_difference",
                         "mean_abs_magnitude_gap", "relative_magnitude_gap"))
        for layer, row in rows.items():
            for (a, b), values in row["pairs"].items():
                writer.writerow((layer, f"{a}/{b}", row["parameter_count"],
                                 row["selected_counts"][a],
                                 row["selected_counts"][b],
                                 *(values[name] for name in (
                                     "overlap_count", "mean_abs_difference",
                                     "rms_difference", "relative_rms_difference",
                                     "mean_abs_magnitude_gap", "relative_magnitude_gap"))))
    print(f"Wrote {len(rows)} layers to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
