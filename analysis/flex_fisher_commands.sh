#!/usr/bin/env bash
# Standalone Fisher group sensitivity for the local Flex Qwen2.5-VL MoE architecture.
# FLEX_MODEL_CHECKPOINT must point to saved weights/config.json, not modeling/ source.
set -euo pipefail
cd "$(dirname "$0")/.."

cohort="${1:-all}"
case "$cohort" in
  all|original|rrg24) ;;
  *) echo "Usage: FLEX_MODEL_CHECKPOINT=/path/to/final $0 [all|original|rrg24]" >&2; exit 2 ;;
esac
: "${FLEX_MODEL_CHECKPOINT:?Set FLEX_MODEL_CHECKPOINT to a flex_qwen2_5_vl_moe checkpoint directory}"
python_bin="${FLEXBAGEL_PYTHON:-python}"
output_dir="${ANALYSIS_OUTPUT_DIR:-/flare/MatSciAI/xinxil/output/analysis/flex-fisher}"
max_examples="${FISHER_MAX_EXAMPLES:-256}"
processor="${FLEX_PROCESSOR:-Qwen/Qwen2.5-VL-3B-Instruct}"
options=()
if [[ "${FISHER_ACTIVATION_CHECKPOINTING:-0}" == 1 ]]; then
  options+=(--activation-checkpointing)
fi

if [[ "$cohort" == all || "$cohort" == original ]]; then
  "$python_bin" -m analysis.run_flex_fisher \
    --model "$FLEX_MODEL_CHECKPOINT" \
    --processor "$processor" \
    --examples endo=/flare/MatSciAI/xinxil/data/surge396k/total_train_filtered.jsonl \
    --examples rex=/flare/MatSciAI/xinxil/data/rex/train_w_length.jsonl \
    --examples pathgen=/flare/MatSciAI/xinxil/data/pathgen/train_flatten_w_length_memory.jsonl \
    --output "$output_dir/original-fisher-mass.pt" \
    --resume-dir "$output_dir/original-fisher-mass.resume" \
    --max-examples "$max_examples" \
    "${options[@]}"
fi

if [[ "$cohort" == all || "$cohort" == rrg24 ]]; then
  "$python_bin" -m analysis.run_flex_fisher \
    --model "${FLEX_RRG24_MODEL_CHECKPOINT:-$FLEX_MODEL_CHECKPOINT}" \
    --processor "$processor" \
    --examples padchest=/flare/MatSciAI/xinxil/data/rrg24/padchest_en/train_w_length_memory.jsonl \
    --examples chexpert=/flare/MatSciAI/xinxil/data/rrg24/chexpert/train_w_length_memory.jsonl \
    --examples rex=/flare/MatSciAI/xinxil/data/rex/train_w_length.jsonl \
    --output "$output_dir/rrg24-fisher-mass.pt" \
    --resume-dir "$output_dir/rrg24-fisher-mass.resume" \
    --max-examples "$max_examples" \
    "${options[@]}"
fi
