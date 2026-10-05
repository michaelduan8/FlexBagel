#!/usr/bin/env bash
# Run from an Aurora allocation. This set compares the RRG24 PadChest and
# CheXpert checkpoints with the ReX checkpoint, independently of example_commands.sh.
# Re-run the Fisher command after a walltime stop to resume saved GPU shards.
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin="${FLEXBAGEL_PYTHON:-python}"
method="${1:-${ANALYSIS_METHOD:-all}}"
case "$method" in
  all|task-vectors|sign-conflicts|fisher) ;;
  *) echo "Usage: $0 [all|task-vectors|sign-conflicts|fisher]" >&2; exit 2 ;;
esac
output_dir="${ANALYSIS_OUTPUT_DIR:-/flare/MatSciAI/xinxil/output/analysis/rrg24/detailed}"
common=(
  --base Qwen/Qwen2.5-VL-3B-Instruct
  --expert padchest=/flare/MatSciAI/xinxil/output/FlexBagel/rrg24_padchest_en_qwen2_5-3b-vl/final
  --expert chexpert=/flare/MatSciAI/xinxil/output/FlexBagel/rrg24_chexpert_qwen2_5-3b-vl/final
  --expert rex=/flare/MatSciAI/xinxil/output/FlexBagel/rexvgradient_qwen2_5-3b-vl-8bits/final
)

if [[ "$method" == all || "$method" == task-vectors ]]; then
"$python_bin" -m analysis.run_expert_analysis \
  "${common[@]}" \
  --method task-vectors \
  --output "$output_dir/task-vectors.pt"
fi

if [[ "$method" == all || "$method" == sign-conflicts ]]; then
"$python_bin" -m analysis.run_expert_analysis \
  "${common[@]}" \
  --method sign-conflicts \
  --output "$output_dir/sign-conflicts.pt"
fi

if [[ "$method" == all || "$method" == fisher ]]; then
fisher_options=()
if [[ "${FISHER_ACTIVATION_CHECKPOINTING:-0}" == 1 ]]; then
  fisher_options+=(--activation-checkpointing)
fi
"$python_bin" -m analysis.run_expert_analysis \
  "${common[@]}" \
  --method fisher \
  "${fisher_options[@]}" \
  --examples padchest=/flare/MatSciAI/xinxil/data/rrg24/padchest_en/train_w_length_memory.jsonl \
  --examples chexpert=/flare/MatSciAI/xinxil/data/rrg24/chexpert/train_w_length_memory.jsonl \
  --examples rex=/flare/MatSciAI/xinxil/data/rex/train_w_length.jsonl \
  --output "$output_dir/fisher-sensitivity.pt" \
  --resume-dir "$output_dir/fisher-sensitivity-r1.resume" \
  --average-mode all \
  --max-examples 256
fi
