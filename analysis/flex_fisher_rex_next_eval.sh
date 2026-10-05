#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${FLEXBAGEL_PYTHON:?Use the next-eval PBS wrapper}"
"$FLEXBAGEL_PYTHON" -m analysis.run_flex_fisher \
  --model /flare/MatSciAI/xinxil/output/FlexBagel/rexvgradient_qwen2_5-3b-vl-flex-topk2-8bits/final \
  --processor /flare/MatSciAI/xinxil/output/FlexBagel/rexvgradient_qwen2_5-3b-vl-flex-topk2-8bits/final \
  --examples rex=/flare/MatSciAI/xinxil/data/rex/train_w_length.jsonl \
  --output /flare/MatSciAI/xinxil/output/analysis/flex-fisher/three-models/rex-fisher-mass.pt \
  --resume-dir /flare/MatSciAI/xinxil/output/analysis/flex-fisher/three-models/rex-fisher-mass.resume \
  --max-examples 256 \
  --activation-checkpointing \
  --trust-remote-code
