#!/usr/bin/env bash
# Run from an Aurora allocation. Each method writes its own result file.
# Re-run the Fisher command after a walltime stop to resume saved GPU shards.
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin="${FLEXBAGEL_PYTHON:-python}"
method="${1:-${ANALYSIS_METHOD:-all}}"
case "$method" in
  all|task-vectors|sign-conflicts|fisher) ;;
  *) echo "Usage: $0 [all|task-vectors|sign-conflicts|fisher]" >&2; exit 2 ;;
esac
output_dir="${ANALYSIS_OUTPUT_DIR:-/flare/MatSciAI/xinxil/output/analysis/detailed}"
common=(
  --base Qwen/Qwen2.5-VL-3B-Instruct
  --expert endo=alrope/surg390k_qwen2_5-3b-vl-corrected
  --expert rex=/flare/MatSciAI/xinxil/output/FlexBagel/rexvgradient_qwen2_5-3b-vl-8bits/final
  --expert pathgen=alrope/pathgen_qwen2_5-3b-vl-retry
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
fisher_suffix=""
if [[ -n "${FISHER_EXPERT:-}" ]]; then
  case "$FISHER_EXPERT" in
    endo|rex|pathgen) ;;
    *) echo "FISHER_EXPERT must be endo, rex, or pathgen" >&2; exit 2 ;;
  esac
  fisher_options+=(--fisher-expert "$FISHER_EXPERT")
  fisher_suffix="-$FISHER_EXPERT"
fi
if [[ "${FISHER_ACTIVATION_CHECKPOINTING:-0}" == 1 ]]; then
  fisher_options+=(--activation-checkpointing)
fi
"$python_bin" -m analysis.run_expert_analysis \
  "${common[@]}" \
  --method fisher \
  "${fisher_options[@]}" \
  --examples endo=/flare/MatSciAI/xinxil/data/surge396k/total_train_filtered.jsonl \
  --examples rex=/flare/MatSciAI/xinxil/data/rex/train_w_length.jsonl \
  --examples pathgen=/flare/MatSciAI/xinxil/data/pathgen/train_flatten_w_length_memory.jsonl \
  --output "$output_dir/fisher-sensitivity-all-components${fisher_suffix}.pt" \
  --resume-dir "$output_dir/fisher-sensitivity-all-components${fisher_suffix}.resume" \
  --average-mode all \
  --max-examples 256
fi
