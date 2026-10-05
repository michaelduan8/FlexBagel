#!/usr/bin/env bash
# Magnitude version of per-layer method 2 for Endo/ReX/PathGen.
set -euo pipefail
cd "$(dirname "$0")/.."

"${FLEXBAGEL_PYTHON:-python}" -m analysis.run_layer_update_difference \
  --base Qwen/Qwen2.5-VL-3B-Instruct \
  --expert endo=alrope/surg390k_qwen2_5-3b-vl-corrected \
  --expert rex=/flare/MatSciAI/xinxil/output/FlexBagel/rexvgradient_qwen2_5-3b-vl-8bits/final \
  --expert pathgen=alrope/pathgen_qwen2_5-3b-vl-retry \
  --top-percent "${TOP_PERCENT:-1}" \
  --output-dir "${ANALYSIS_OUTPUT_DIR:-/flare/MatSciAI/xinxil/output/analysis/per-layer-heterogeneous}"
