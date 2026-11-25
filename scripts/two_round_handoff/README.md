# Two-round FlexOlmo: handoff to Delta AI (4x GH200)

Status as of 2026-11-24. The work so far ran on CARC (local evals) and Modal (training on 4x H200).

## 1. The main finding: precision

**`train/mm_tune.py` trains in pure bf16 by default.** It loads the model with `torch_dtype=torch.bfloat16`, and the
optimizer updates those bf16 weights directly. There is no float32 master copy. bf16 can only represent changes of
about 0.4% of a value, while an AdamW step at LR 2e-5 is about 2e-5 per weight. So **any weight larger than about
0.01 never changes**: its updates round to zero, every step.

Measured over all weights, against base Qwen2.5-VL-3B:

| Model | Weights changed |
|---|---|
| Our round-1 finetunes (RexGradient, PadChest, CheXpert halves) | 13.3–13.9% |
| alrope rexgradient-8bits / padchest / chexpert / pathgen-retry / surg390k-corrected | 13.5–16.3% |
| micdun surgpub_rarp50_video | 12.2% |
| alrope/rexvgradient_qwen2_5-3b-vl (recipe unknown: fp32 master weights or a higher LR) | 77.2% |
| Round-2 expert updates (bf16) | ~14% of expert weights, 0–1% of router rows |
| LayerNorm/RMSNorm weights, every bf16 model | ~0.2% (effectively frozen) |

With proper mixed precision (fp32 master weights, bf16 compute), nearly all weights move.

**Fix: `--trainable_params_fp32 True`** (new, opt-in, in `train/mm_tune.py`). It keeps *trainable* parameters in
float32 and frozen ones in bf16, runs forward/backward under bf16 autocast, and casts back to bf16 before the final save,
so the output format is unchanged. Test: `PYTHONPATH=. python -m train.test_trainable_fp32` (toy layer: pure bf16
changes 17.9% of weights, fp32 master changes 95.7%).

**Use it for every run on Delta.** All existing bf16-trained experts are undertrained in this sense. Comparisons among
them are internally consistent, but absolute numbers, and possibly some orderings, may change once training is done in
fp32.

Effect on round 2 (same data, LR, schedule and seed; RexVQA):
- bf16 round 2, saturated router: always worse than round 1 (0.745–0.753); PadChest diverged at LR 2e-5.
- fp32 round 2 + cosine routing: **0.7815** (top-3) vs round 1's 0.7659. PadChest is stable at 2e-5.

**Precision and routing were changed together, so their effects are not separated.** The fp32-only round-2 run was
stopped before training.

### Next run: fp32 dense control (planned)

RexGradient round-1 half, fp32 master weights, otherwise the same recipe as the bf16 finetune that scored **0.7597** on
RexVQA. This shows how much bf16 has been costing round 1. Then decide whether to redo round 1 in fp32 for all silos,
and add a one-round full-data fp32 baseline.

## 2. GH200 notes

- **aarch64 (ARM) CPUs:** build conda envs, flash-attn and vLLM for ARM. The Modal image's x86 flash-attn wheel won't work.
- **96 GB per GPU** (the H200s had 141 GB). Keep the effective batch at 192 (per-device × grad accumulation × 4 GPUs):
  - fp32 round 2 on the 3-expert BTX ran out of memory at batch 8 on H200 (127 GB used); batch 4 × 12 fit there.
    **Start at batch 2 × 24 on GH200.**
  - fp32 dense round 1 needs ~60 GB for weights, grads and Adam, plus activations. Start at batch 2 × 24 or 4 × 12.
- About 12% slower per step than bf16 on H200 (fp32 master weights).

## 3. Code changes (uncommitted on CARC; all opt-in, defaults unchanged)

| File | Change |
|---|---|
| `train/mm_tune.py` | `--trainable_params_fp32` (`upcast_trainable_params` / `cast_floating_params`) |
| `train/test_trainable_fp32.py` | test for the above |
| `modeling/flex_qwen2_5_vl_moe/modeling_flex_qwen2_5_vl_moe.py` | cosine routing: config `router_normalize`, `router_temperature` (text and vision MoE blocks) |
| `modeling/flex_qwen2_5_vl_moe/test_router_normalize.py` | test for the above (flag off = bit-identical to before) |
| `scripts/modal_mm_tuning/two_round/two_round_pipeline.py` | Modal-only pipeline: `--silos`, round1/round2 extra args, warmup, per-device batch |
| `scripts/two_round_handoff/` | this folder |

Run the existing tests too: `test_loss_normalization`, `test_gate_row_to_expert`, `test_gate_row_aggregation`,
`train.test_routing_overrides`, `train.test_multiturn_mask` (all pass).

## 4. Data

Silos: RexGradient-8bit, PadChest (`padchest_en`), CheXpert. Each silo is split 50/50 with seed 42; round 1 trains on
one half, round 2 on the other (disjoint).

```bash
python scripts/two_round_handoff/split_halves.py --source <silo train JSONL> --out-prefix <dir>/<silo>
```

This is verified byte-identical to the Modal halves for all three silos, given these source JSONLs (same rows, same order):
- RexGradient: `rexgradient_train_w_length_modal_8bit_flat.jsonl` (Modal volume `rexgradient_flat`; CARC copy
  `/scratch2/duanm/data/rexgradient/flat/`), 139,910 rows → 69,955 / 69,955
- PadChest: `traces/multimodal/rrg_padchest/train_w_length_memory_modal.jsonl`, 101,301 → 50,650 / 50,651
- CheXpert: `traces/multimodal/rrg_chexpert/train_w_length_memory_modal.jsonl`, 181,658 → 90,829 / 90,829

Split first, then rewrite image paths for Delta. Images live on Modal volumes `rexgradient_flat` (`images/<UID>.png`)
and `rrg24`. On CARC: RexGradient 8-bit in `/scratch2/duanm/data/rexgradient/deid_png/` (nested layout; map by file
name), and only small PadChest/CheXpert subsets in `/scratch2/duanm/data/rrg24/train_subset/`.

## 5. Recipe

All training uses `torchrun --nproc_per_node=4 train/mm_tune.py` with these common flags:
```
--num_train_epochs 1 --learning_rate 2e-5 --warmup_ratio 0.1 --gradient_checkpointing True --run_seed 42
--logging_steps 10 --save_n_epochs 0.25 --auto_resume True --skip_eval --delete_intermediate_checkpoints false
--trainable_params_fp32 True --per_device_train_batch_size <b> --gradient_accumulation_steps <48/b>
```

1. **Round 1, dense, one run per silo:** `--model Qwen/Qwen2.5-VL-3B-Instruct --datasets <silo>_round1.jsonl`
2. **Router seeds, one per silo:** 1000 rows of the silo's round-1 half through *its own* round-1 model:
   `python modeling/routing_embeddings/extract_hidden_states.py --model <silo round-1 model> --dataset <1000-row
   sample with local image paths> --query-amount 1000 --dtype bf16 --num-gpus 4 --output-path <silo>.pkl`
3. **BTX:** `python modeling/upcycle/mm_dense_to_expert_moe.py -m <rex> <pad> <chex> --embeddings-dict <rex.pkl>
   <pad.pkl> <chex.pkl> --merge-non-moe-weights --num-experts-per-tok 2 -t <btx_round1>`
   (expert order: RexGradient, PadChest, CheXpert)
4. **Cosine copy:** `python scripts/two_round_handoff/build_cosine_btx.py --src <btx_round1> --dst <btx_round1_cos100> --tau 100`
5. **Round 2, one run per silo, expert index e = 0/1/2:** `--model <btx_round1_cos100> --datasets <silo>_round2.jsonl
   --train_expert_idx e --num_experts_per_tok 3`
6. **Merge:** `python modeling/upcycle/merge_vl_experts.py -m <run0> <run1> <run2> -t <btx_round2>
   --experts-per-model "" 1 2 --shared-expert-index 0 --shared-from-model 0 --base-model-index 0
   --strict-shared-check --dtype bfloat16`. Before merging, check that each run changed only its own expert and router
   row (see `verify_run` in `scripts/modal_mm_tuning/rexsplit_round2/merge_round2.py`).
7. **Eval:**
   - RexVQA: `eval/vlm/eval/rexvqa/evaluate_rexvqa.py --inference_backend transformers --num_experts_per_tok 3`
     (cosine-trained models are best at top-3; also report `--num_experts_per_tok 1 --norm_topk_prob`).
   - Report perplexity: `eval/vlm/eval/perplexity/evaluate_perplexity.py --num_experts_per_tok 3 --no-norm_topk_prob`
     on `eval/vlm/eval/rrg24/perplexity_val/rrg24_{chexpert,padchest}_val.jsonl` and
     `eval/vlm/eval/rexvqa/rexgradient_test.jsonl --sample_size 1000 --sample_seed 42`.

## 6. Results so far (`rpc-halves-v1`)

| Model | RexVQA top-1 + norm | RexVQA top-3 | PPL PadChest / CheXpert / RexGradient |
|---|---|---|---|
| Dense RexGradient / PadChest / CheXpert (round-1 halves, bf16) | 0.7597 / 0.7504 / 0.7406 | | |
| Round-1 BTX (bf16 experts, hidden-state router) | 0.7677 | 0.7659 | 5.84 / 4.95 / 4.47 (cosine τ=100) |
| Round 2, bf16, saturated router (best variant) | 0.7450–0.7526 | | |
| Round 2, fp32 + cosine, RexGradient expert only | 0.7626 | 0.7755 | |
| Round 2, fp32 + cosine, PadChest expert only | | 0.7452 | 2.99 / 14.89 / 12.27 |
| **Round 2, fp32 + cosine, all three experts** | **0.7777** | **0.7815** | **4.18 / 3.68 / 2.59** |

- **A single retrained expert pulls the whole model toward its own silo** (see the PadChest-only row). Only the full set
  of three is balanced, so every silo must take part in round 2.
- **Why cosine routing:** hidden-state router rows have size 8–500, so scores scale with |h|², and ~80% of tokens put
  more than 0.99 weight on one expert. Top-3 was effectively top-1, and the router could not train. Cosine routing at
  τ=100 gives a top-expert weight of about 0.7.

## 7. Gotchas

- `evaluate_perplexity.py` overrides `CUDA_VISIBLE_DEVICES` with the worker rank, so two concurrent jobs on one node
  collide. It also defaults `--norm_topk_prob` **on**; pass `--no-norm_topk_prob` for top-3 evals.
- `modal volume get` once silently produced a partial download (0-byte files). Always md5-check downloads.
- The RexVQA eval generates exactly one token (the answer letter). A model that has stopped answering in
  multiple-choice format scores ~0; look at a few multi-token generations before trusting a near-zero score.
- `mm_tune` writes `num_experts_per_tok` into saved configs, so round-2 runs save top-3. `merge_vl_experts.py` refuses
  to merge models whose configs differ (for example a round-1 BTX saying top-2 vs round-2 runs saying top-3).
