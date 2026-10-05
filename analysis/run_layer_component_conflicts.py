"""Per-layer top-update conflict counts and magnitudes by model component.

Vision patch embedding and language token embedding are reported as standalone
stages because they are not part of numbered transformer blocks.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

from analysis.expert_analysis import ParameterGroup, sign_conflicts
from analysis.run_expert_analysis import load_states, named_paths
from analysis.run_layer_update_difference import layer_difference


def component_groups(parameter_names: list[str]) -> list[tuple[str, list[str]]]:
    groups = []
    specifications = (
        ("vision", "ffn", r"^(?:model\.)?visual\.blocks\.(\d+)\.mlp\."),
        ("vision", "attention", r"^(?:model\.)?visual\.blocks\.(\d+)\.attn\."),
        ("language", "ffn", r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\."),
        ("language", "attention", r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.(?:self_attn|attn)\."),
    )
    for tower, component, pattern in specifications:
        compiled = re.compile(pattern)
        indices = sorted({int(match.group(1)) for name in parameter_names
                          if (match := compiled.match(name))})
        for index in indices:
            keys = [name for name in parameter_names if
                    (match := compiled.match(name)) and int(match.group(1)) == index]
            groups.append((f"{tower}_{component}_{index}", keys))
    embeddings = (
        ("vision_embedding", r"^(?:model\.)?visual\.patch_embed\."),
        ("language_embedding", r"^(?:model\.)?(?:language_model\.)?embed_tokens\."),
    )
    for name, pattern in embeddings:
        keys = [key for key in parameter_names if re.search(pattern, key)]
        if not keys:
            raise ValueError(f"No parameters found for {name}")
        groups.append((name, keys))
    for tower in ("vision", "language"):
        for component in ("ffn", "attention"):
            if not any(name.startswith(f"{tower}_{component}_") for name, _ in groups):
                raise ValueError(f"No {tower} {component} layer parameters found")
    return groups


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
    groups = component_groups(parameter_names)
    rows = {}
    for name, keys in groups:
        distance = layer_difference(base, experts, keys, args.top_percent)
        conflicts = sign_conflicts(
            base, experts, top_percent=args.top_percent,
            groups=(ParameterGroup(name, r"^"),), parameter_names=keys)[name]
        assert distance["selected_counts"] == conflicts["selected_counts"]
        for pair, values in distance["pairs"].items():
            assert values["overlap_count"] == conflicts["pairs"][pair]["overlap_count"]
        rows[name] = {"parameter_count": distance["parameter_count"],
                      "selected_counts": distance["selected_counts"],
                      "pairs": {pair: {**values, **conflicts["pairs"][pair]}
                                for pair, values in distance["pairs"].items()}}
        print(f"Completed {name}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"method": "component-conflicts", "scope": "per_layer",
                "base": args.base, "experts": expert_paths,
                "top_percent": args.top_percent, "groups": rows},
               out / "component-conflicts-per-layer.pt")
    with (out / "component-conflicts-per-layer.tsv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("group", "pair", "parameter_count", "selected_a",
                         "selected_b", "overlap_count", "opposite_sign_count",
                         "opposite_sign_fraction", "relative_rms_difference",
                         "rms_difference", "relative_magnitude_gap"))
        for group, row in rows.items():
            for (a, b), values in row["pairs"].items():
                writer.writerow((group, f"{a}/{b}", row["parameter_count"],
                                 row["selected_counts"][a],
                                 row["selected_counts"][b],
                                 *(values[key] for key in (
                                     "overlap_count", "opposite_sign_count",
                                     "opposite_sign_fraction", "relative_rms_difference",
                                     "rms_difference", "relative_magnitude_gap"))))
    print(f"Wrote {len(rows)} component groups to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
