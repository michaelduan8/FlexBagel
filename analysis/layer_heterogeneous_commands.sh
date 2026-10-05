#!/usr/bin/env bash
# Per-layer versions of methods 1 and 2 for the heterogeneous Endo/ReX/PathGen set.
set -euo pipefail
cd "$(dirname "$0")/.."

"${FLEXBAGEL_PYTHON:-python}" -m analysis.run_layer_checkpoint_analysis \
  --base Qwen/Qwen2.5-VL-3B-Instruct \
  --expert endo=alrope/surg390k_qwen2_5-3b-vl-corrected \
  --expert rex=/flare/MatSciAI/xinxil/output/FlexBagel/rexvgradient_qwen2_5-3b-vl-8bits/final \
  --expert pathgen=alrope/pathgen_qwen2_5-3b-vl-retry \
  --method "${1:-all}" \
  --output-dir "${ANALYSIS_OUTPUT_DIR:-/flare/MatSciAI/xinxil/output/analysis/per-layer-heterogeneous}"
