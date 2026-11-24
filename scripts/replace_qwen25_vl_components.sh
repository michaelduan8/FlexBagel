#!/bin/bash
# Build one hybrid checkpoint per component: target weights with only that component taken from donor.
# Usage: bash scripts/replace_qwen25_vl_components.sh TARGET DONOR OUTPUT_ROOT [COMPONENT ...]
# Outputs land in OUTPUT_ROOT/hybrid_<component>. Defaults to all four components.
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 TARGET DONOR OUTPUT_ROOT [COMPONENT ...]" >&2
    exit 1
fi

target=$1
donor=$2
output_root=$3
shift 3
components=("$@")
if [ ${#components[@]} -eq 0 ]; then
    components=(vision_ffn vision_attention language_ffn language_attention)
fi

mkdir -p "$output_root"
for c in "${components[@]}"; do
    out="$output_root/hybrid_$c"
    if [ -e "$out" ]; then
        echo "Skipping $c (already exists: $out)"
        continue
    fi
    echo "Replacing $c -> $out"
    PYTHONPATH=. python modeling/upcycle/replace_qwen25_vl_components.py \
        --target "$target" \
        --donor "$donor" \
        --output "$out" \
        --components "$c"
done
