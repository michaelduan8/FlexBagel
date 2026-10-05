#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${FLEXBAGEL_PYTHON:?Use the next-eval PBS wrapper}"
"$FLEXBAGEL_PYTHON" -m analysis.run_flex_fisher \
  --model alrope/endochat_qwen2_5-3b-vl-flex-topk2 \
  --processor alrope/endochat_qwen2_5-3b-vl-flex-topk2 \
  --examples endo=/flare/MatSciAI/xinxil/data/surge396k/total_train_filtered.jsonl \
  --output /flare/MatSciAI/xinxil/output/analysis/flex-fisher/three-models/endo-fisher-mass.pt \
  --resume-dir /flare/MatSciAI/xinxil/output/analysis/flex-fisher/three-models/endo-fisher-mass.resume \
  --max-examples 256 \
  --activation-checkpointing \
  --trust-remote-code
