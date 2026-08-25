
#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <run_id> <model_id>" >&2
    exit 1
fi

run_id="$1"
model_id="$2"
num_gpus=2

if [[ "$run_id" != "flex_merged_pub_endo_qwen2_5-3b-vl-topk2-rp_topk3-10k-inf_topk1_norm" ]]; then
    PYTHONPATH=. python eval/vlm/eval/endochat/evaluate_endochat.py \
        --run_id "$run_id" \
        --model_id "$model_id" \
        --test_data_path /scratch1/duanm/data/surg_396k/C80/C80_test.json \
        --raw_data_dir /scratch1/duanm/data/surg_396k/C80/ \
        --split_str C80/ \
        --result_folder "results/endochat/$run_id/c80/" \
        --num_gpus "$num_gpus" \
        --num_fewshot 0 \
        --inference_backend transformers \
        --transformers_batch_size 32 \
        --num_experts_per_tok 1 \
        --norm_topk_prob
fi

PYTHONPATH=. python eval/vlm/eval/endochat/evaluate_endochat.py \
    --run_id "$run_id" \
    --model_id "$model_id" \
    --test_data_path /scratch1/duanm/data/surg_396k/CoPESD/CoPESD_test_extend_v3.json \
    --raw_data_dir /scratch1/duanm/data/surg_396k/CoPESD/ \
    --split_str CoPESD/ \
    --result_folder "results/endochat/$run_id/CoPESD/" \
    --num_gpus "$num_gpus" \
    --num_fewshot 0 \
    --inference_backend transformers \
    --transformers_batch_size 32 \
    --num_experts_per_tok 1 \
    --norm_topk_prob

PYTHONPATH=. python eval/vlm/eval/endochat/evaluate_endochat.py \
    --run_id "$run_id" \
    --model_id "$model_id" \
    --test_data_path /scratch1/duanm/data/surg_396k/EndoVis_part/endovis_test_resampled.json \
    --raw_data_dir /scratch1/duanm/data/surg_396k/EndoVis_part/ \
    --result_folder "results/endochat/$run_id/endovis/" \
    --num_gpus "$num_gpus" \
    --num_fewshot 0 \
    --inference_backend transformers \
    --transformers_batch_size 32 \
    --num_experts_per_tok 1 \
    --norm_topk_prob
