#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <model_id_suffix>"
    exit 1
fi

model_id_suffix="$1"

# PYTHONPATH=. python eval/vlm/eval/endochat/evaluate_endochat.py \
#     --run_id $model_id_suffix \
#     --model_id alrope/$model_id_suffix \
#     --test_data_path /scratch1/duanm/data/surg_396k/EndoVis_part/endovis_test_resampled.json \
#     --raw_data_dir /scratch1/duanm/data/surg_396k/EndoVis_part/ \
#     --result_folder results/endochat/$model_id_suffix/endovis/ \
#     --num_gpus 4 \
#     --num_fewshot 0 \
#     --inference_backend transformers \
#     --transformers_batch_size 32

# PYTHONPATH=. python eval/vlm/eval/endochat/evaluate_endochat.py \
#     --run_id $model_id_suffix \
#     --model_id alrope/$model_id_suffix \
#     --test_data_path /scratch1/duanm/data/surg_396k/CoPESD/CoPESD_test_extend_v3.json \
#     --raw_data_dir /scratch1/duanm/data/surg_396k/CoPESD/ \
#     --split_str CoPESD/ \
#     --result_folder results/endochat/$model_id_suffix/CoPESD/ \
#     --num_gpus 4 \
#     --num_fewshot 0 \
#     --inference_backend transformers \
#     --transformers_batch_size 32

# PYTHONPATH=. python eval/vlm/eval/endochat/evaluate_endochat.py \
#     --run_id $model_id_suffix \
#     --model_id alrope/$model_id_suffix \
#     --test_data_path /scratch1/duanm/data/surg_396k/C80/C80_test.json \
#     --raw_data_dir /scratch1/duanm/data/surg_396k/C80/ \
#     --split_str C80/ \
#     --result_folder results/endochat/$model_id_suffix/c80/ \
#     --num_gpus 4 \
#     --num_fewshot 0 \
#     --inference_backend transformers \
#     --transformers_batch_size 32

# for bench in CoPESD endovis c80; do
#     PYTHONPATH=. python eval/vlm/eval/endochat/endochat_eval.py \
#         --model_test_result_folder results/endochat/$model_id_suffix/$bench
# done

# ReXVQA Eval
PYTHONPATH=. python eval/vlm/eval/rexvqa/evaluate_rexvqa.py \
    --model_id alrope/$model_id_suffix \
    --test_data_path eval/vlm/eval/rexvqa/rexvqa_test.jsonl \
    --result_folder results/rexvqa/$model_id_suffix/ \
    --num_gpus 4 \
    --inference_backend transformers \
    --transformers_batch_size 32