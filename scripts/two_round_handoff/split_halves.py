"""Regenerate the two-round round-1 / round-2 data halves, without Modal.

Same logic as `split_data` in scripts/modal_mm_tuning/two_round/two_round_pipeline.py: drop blank lines, shuffle the
row indices with random.Random(seed), round 1 gets the first int(n * round1_fraction) rows, round 2 the rest (or, with
--round2-fraction f, the next int(n * f) rows). With the same source JSONL (same rows, same order) and settings, the
output is byte-identical to the Modal runs (tag rpc-halves-v1: seed 42, round1_fraction 0.5).

    python scripts/two_round_handoff/split_halves.py --source <silo train JSONL> --out-prefix <dir>/<silo>
    # writes <dir>/<silo>_round1.jsonl and <dir>/<silo>_round2.jsonl

Image paths are copied unchanged; rewrite them for your cluster afterwards if needed (the order must not change before
splitting, so rewrite after, not before).
"""
import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--round1-fraction", type=float, default=0.5)
    ap.add_argument("--round2-fraction", type=float, default=None)
    args = ap.parse_args()

    with open(args.source, encoding="utf-8") as f:
        rows = [line for line in f if line.strip()]
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    cut = int(len(rows) * args.round1_fraction)
    end = len(rows) if args.round2_fraction is None else cut + int(len(rows) * args.round2_fraction)
    assert end <= len(rows), "round2 fraction too large"
    for suffix, idx in (("round1", order[:cut]), ("round2", order[cut:end])):
        out = f"{args.out_prefix}_{suffix}.jsonl"
        assert not os.path.exists(out), f"refusing to overwrite {out}"
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(rows[i] if rows[i].endswith("\n") else rows[i] + "\n" for i in idx)
        print(f"{out}: {len(idx)} rows")


if __name__ == "__main__":
    main()
