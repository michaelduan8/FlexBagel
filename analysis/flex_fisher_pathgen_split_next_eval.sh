#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${FLEXBAGEL_PYTHON:?Use the next-eval PBS wrapper}"
"$FLEXBAGEL_PYTHON" -m analysis.run_flex_fisher \
  --model alrope/pathgen_qwen2_5-3b-vl-flex-topk2 \
  --processor alrope/pathgen_qwen2_5-3b-vl-flex-topk2 \
  --examples pathgen=/flare/MatSciAI/xinxil/data/pathgen/train_flatten_w_length_memory.jsonl \
  --output /flare/MatSciAI/xinxil/output/analysis/flex-fisher/three-models/expert-split/pathgen-fisher-mass.pt \
  --resume-dir /flare/MatSciAI/xinxil/output/analysis/flex-fisher/three-models/expert-split/pathgen-fisher-mass.resume \
  --max-examples 256 \
  --activation-checkpointing \
  --split-experts \
  --trust-remote-code
