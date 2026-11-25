"""Make the round-2 starting model: a copy of a round-1 BTX with cosine routing and unit-length router rows.

Cosine routing (config `router_normalize: true`, `router_temperature: tau`, text and vision) scores tokens with
tau * cos(hidden, router row), so row length does not affect routing. The rows are still rescaled to unit length so
that LR-sized optimizer steps can actually turn them (rows of size 100-500 barely move). Writes a new folder only.

    python scripts/two_round_handoff/build_cosine_btx.py --src <round-1 BTX dir> --dst <new dir> --tau 100
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tau", type=float, default=100.0)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=False)
    index = json.load(open(f"{args.src}/model.safetensors.index.json"))
    n = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(f"{args.src}/{shard}")
        for k in list(tensors):
            if re.search(r"\.mlp\.gate\.weight$", k):
                w = tensors[k]
                tensors[k] = torch.nn.functional.normalize(w.float(), dim=-1).to(w.dtype).contiguous()
                n += 1
        save_file(tensors, f"{args.dst}/{shard}", metadata={"format": "pt"})
    for f in os.listdir(args.src):
        if not f.endswith(".safetensors") and f != "config.json":
            shutil.copy(f"{args.src}/{f}", f"{args.dst}/{f}")
    cfg = json.load(open(f"{args.src}/config.json"))
    for sub in ("text_config", "vision_config"):
        cfg[sub]["router_normalize"] = True
        cfg[sub]["router_temperature"] = args.tau
    json.dump(cfg, open(f"{args.dst}/config.json", "w"), indent=2)
    print(f"normalized {n} router gates; cosine routing tau={args.tau} -> {args.dst}")


if __name__ == "__main__":
    main()
