#!/usr/bin/env bash
# Per-layer FFN, attention, and embedding comparison for the chest X-ray set.
set -euo pipefail
cd "$(dirname "$0")/.."

"${FLEXBAGEL_PYTHON:-python}" -m analysis.run_layer_component_conflicts \
  --base Qwen/Qwen2.5-VL-3B-Instruct \
  --expert padchest=/flare/MatSciAI/xinxil/output/FlexBagel/rrg24_padchest_en_qwen2_5-3b-vl/final \
  --expert chexpert=/flare/MatSciAI/xinxil/output/FlexBagel/rrg24_chexpert_qwen2_5-3b-vl/final \
  --expert rex=/flare/MatSciAI/xinxil/output/FlexBagel/rexvgradient_qwen2_5-3b-vl-8bits/final \
  --top-percent "${TOP_PERCENT:-1}" \
  --output-dir "${ANALYSIS_OUTPUT_DIR:-/flare/MatSciAI/xinxil/output/analysis/per-layer-homogeneous}"
