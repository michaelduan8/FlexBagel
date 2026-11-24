# Many-to-one router rows: `gate_row_to_expert`

Optional field on `Flex_Qwen2_5_VLMoeTextConfig` and `Flex_Qwen2_5_VLMoeVisionConfig`. Default `None`: one
router (gate) row per expert, exactly as before — every code path below runs its original branch unchanged.

## What it does

`gate_row_to_expert = [e_0, e_1, ..., e_{R-1}]` gives each MoE layer's `mlp.gate` **R rows**, and row `r` routes
to expert `e_r`. Several rows may share an expert. The intended use is merging standard flex experts whose
routers (and non-FFN weights) were trained unfrozen: each source model keeps its own trained "public" row, and
all of them route to the single shared public expert, while each domain row routes to its own domain expert.

Example (3 source models, merged by `merge_vl_experts.py --many-to-one-public`):

| gate row | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| source | model 0 public | model 0 domain | model 1 public | model 1 domain | model 2 public | model 2 domain |
| expert | 0 (public) | 1 | 0 (public) | 2 | 0 (public) | 3 |

→ `gate_row_to_expert = [0, 1, 0, 2, 0, 3]`, `num_experts = 4`.

## Routing (MoE block forward)

1. `softmax` over **all R gate rows** (public probability is spread over its rows; no aggregation before softmax).
2. Each expert's score = its **best row's** probability (`max` over its rows).
3. `num_experts_per_tok` = top-k over **distinct experts** by that score; each selected expert runs **once**,
   weighted by its best row's probability. (So if two public rows are both among the highest rows, the public
   expert runs once and the next-best *other* expert fills the second slot.)
4. `norm_topk_prob` normalizes over the selected experts.

### Opt-in: summed row probabilities (`gate_row_aggregation = "sum"`)

`gate_row_aggregation` (text and vision configs) chooses how an expert's routing weight is formed from its rows:
`None` / `"max"` = its best row, as above (default, unchanged); `"sum"` = the **sum** of its rows' probabilities
(top-k over distinct experts by that sum, each selected expert weighted by it; `norm_topk_prob` as usual). With
`"sum"`, a block with rows `[r_0, r_1, r_2]` and `gate_row_to_expert = [0, 1, 1]` gives expert 1 the weight
`p_1 + p_2`, i.e. exactly what a 3-expert block with rows `[r_0, r_1, r_2]` gives its experts 1 and 2 together when
those two experts are identical. Use: a local 2-expert model in which one frozen expert (e.g. the weight average of
other collaborators' experts) stands in for several experts while keeping each of their router rows. The block,
the load-balancing loss (dispatch selection) and `monitor.py` (reported expert) all follow the setting; the
setting requires `gate_row_to_expert`.

## Training losses and monitoring

Where a probability distribution over experts is needed, an expert's probability is the **sum** of its rows'
probabilities (routing uses the max; a max per expert isn't a distribution):

- **Load-balancing aux loss** (`load_balancing_loss_func(..., gate_row_to_expert=...)`, text and vision): dispatch
  fraction from the same best-row top-k-over-experts selection as the block; router prob = summed row probs.
- **Depth-consistency / boundary losses and vision pooling**: `routed_only_router_logits` collapses gate rows to
  per-expert logits (logsumexp over each expert's rows), whose softmax is the summed row probability.
- **`train/mm_tune.py --train_expert_idx`**: the router-gradient mask keeps **all** rows that route to the trained
  expert (text/vision mapping chosen by parameter name). `--router_tuning_only` / `--unfreeze_non_ffn` need no
  change. (`train/finetune.py` and `train/mm_tune_mt.py` keep their own copies of the mask and are not updated.)
- **`monitor.py` (`MoeRoutingMonitor`)**: the reported expert is the block's top-1 (best row); entropy is over the
  per-expert summed distribution. Stats are per expert, not per row.

Router logits returned by the model (`output_router_logits`, `router_logits` / `vision_router_logits`) stay **per
row** (shape `[tokens, R]`), since they are the raw gate outputs.

## Constraints (validated in the config and the block)

- every expert must own at least one row; entries in `[0, num_experts)`;
- not combinable with `use_shared_expert` or `always_active_expert` (also rejected if `always_active_expert` is set
  on a block at runtime, e.g. by an eval override);
- `num_experts_per_tok <= num_experts` (top-k is over distinct experts).

## Building a model

```bash
PYTHONPATH=. python modeling/upcycle/merge_vl_experts.py \
    -m <standard flex model 1> <model 2> ... -t <output dir> \
    --many-to-one-public --merge-non-ffn-weights [--shared-expert-index 0] [--tokenizer <processor source>]
```

- the public (shared) expert FFN is copied once and must be **bit-identical** across sources (it is frozen in
  standard flex training); the merge fails otherwise;
- every source's public row and domain row(s) are kept, ordered per model as in the table above;
- `--merge-non-ffn-weights` averages non-FFN tensors over the source models, once each;
- the mapping is written to both `text_config` and `vision_config`.

## Tests

`PYTHONPATH=. python -m modeling.flex_qwen2_5_vl_moe.test_gate_row_to_expert` (or pytest): default path bit-exact to
the original routing, identity mapping equivalent, many-to-one semantics (public expert runs once, best-row
weight, normalization), losses, monitor, trainer mask, config validation, full-model forward and config round trip.
