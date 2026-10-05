# Expert checkpoint diagnostics

analysis.expert_analysis compares compatible base and expert checkpoints. Checkpoint keys and shapes must match. The Fisher estimator uses the model's unique floating point parameters, including frozen parameters in selected groups, and restores their gradient flags afterward. It excludes buffers and counts tied parameters once. State dictionaries may include buffers and tied aliases, so pass the canonical keys from model.named_parameters() to sensitivity analysis.

```python
from analysis.expert_analysis import (
    group_summary, task_vector_similarity, sign_conflicts,
    estimate_conditional_token_fisher, fisher_weighted_sensitivity,
)

parameter_names = [name for name, p in expert_model.named_parameters()
                   if p.is_floating_point()]
print(group_summary({name: base_state[name] for name in parameter_names}))
vectors = task_vector_similarity(base_state, expert_states, parameter_names=parameter_names)
conflicts = sign_conflicts(base_state, expert_states, top_percent=1.0,
                           parameter_names=parameter_names)
fishers = {
    name: estimate_conditional_token_fisher(model, examples_by_expert[name])
    for name, model in expert_models.items()
}
sensitivity = fisher_weighted_sensitivity(
    base_state, expert_states, fishers, parameter_names=parameter_names)
```

Use the same canonical parameter keys for all expert models. A Fisher example contains model-ready tensors including input_ids and labels of shape [1, sequence]. Labels equal input_ids at answer positions and are -100 elsewhere. Include image tensors and masks when needed. Put each model in evaluation mode. The estimator samples one independent target from the model distribution at each answer position under the reference prefix. It takes one gradient of the sum of sampled log probabilities divided by the square root of the number of answer positions. The expected squared gradient equals the mean of the individual token squared gradients; a particular sample has different noise. Examples are averaged equally. Fisher tensors use float32.

When coordinate-level Fisher is unnecessary, pass score_deltas as a mapping from metric names to canonical parameter delta maps. The estimator then returns Fisher mass and raw weighted sums by group without storing the full diagonal. Process each expert model and its deltas in sequence to limit memory.

Default groups have five overlapping views: component, architecture, expertization, component_architecture, and component_detail. Component groups are vision_tower, connector, and language_decoder. Architectural groups are architectural_ffn and architectural_non_ffn; dense mlp weights are FFN parameters. Expertization groups are expertized_ffn and shared_parameters. The connector is a distinct component. group_summary reports tensor and coordinate counts with example parameter names so group classification can be checked. Supply ParameterGroup(name, include_regex, exclude_regex) to customize groups.

task_vector_similarity returns update L2, update RMS, relative L2, and pairwise cosine. sign_conflicts selects the exact requested percentage of nonzero updates per group and expert, breaking ties in sorted parameter order.

fisher_weighted_sensitivity uses each expert's Fisher at that expert's weights. Its primary base_to_expert and expert_to_average fields are raw sums of F × delta². Secondary fields contain per-parameter means and Fisher-normalized RMS. Shares and percentages are calculated independently within each component, architectural, and expertization partition; overlapping partitions are never added together. Undefined zero-denominator ratios are None. Missing Fisher entries raise an error.

Equal-weight expert averaging is the default. Pass coefficients={expert_name: nonnegative_weight} for static averaging weights and base_coefficient for optional base participation. Weights are normalized together. Pass average_mask={parameter_name: bool_or_boolean_tensor} to average selected parameters or coordinates; omitted keys are preserved at each expert's own values. No averaged model is instantiated or evaluated.

## Command-line runs

[example_commands.sh](example_commands.sh) runs three independent methods and writes three files: `task-vectors.pt`, `sign-conflicts.pt`, and `fisher-sensitivity.pt`. The first two only compare checkpoint weights on CPU. Fisher uses the supplied JSONL files and all visible Aurora XPUs by default. Run one method with `bash analysis/example_commands.sh task-vectors`, `bash analysis/example_commands.sh sign-conflicts`, or `bash analysis/example_commands.sh fisher`. Re-running the Fisher command resumes its saved shards; the other two methods do not run again. For a next-eval resubmission, use [fisher_commands.sh](fisher_commands.sh) as the command file. The configured command uses the first 256 nonempty rows from each expert's JSONL.

~~~bash
python -m analysis.run_expert_analysis \
  --method fisher \
  --base /path/to/base \
  --expert a=/path/to/expert-a \
  --examples a=/flare/MatSciAI/xinxil/data/pathgen/train_flatten_w_length_memory.jsonl \
  --output /path/to/fisher-sensitivity.pt \
  --resume-dir /path/to/fisher-sensitivity.resume \
  --average-mode non-ffn \
  --max-examples 256
~~~

| Argument | Meaning |
| --- | --- |
| `--method task-vectors`, `sign-conflicts`, or `fisher` | Run one method and write only its result. Required. |
| `--base PATH` | Original base checkpoint. |
| `--expert NAME=PATH` | Named expert checkpoint. Repeat for every expert. |
| `--examples NAME=PATH` | PathGen-style JSONL file for a Fisher expert. Required for each expert only in Fisher mode. |
| `--output PATH` | Result `.pt` file for the selected method. Give each method a distinct path. |
| `--resume-dir DIR` | Fisher checkpoint directory. Defaults to `OUTPUT.resume`. Each expert and GPU shard is saved after every completed example. Re-run with the same arguments and directory to resume. |
| `--average-mode non-ffn` or `all` | Fisher only. Default `non-ffn` preserves FFNs and the connector; `all` averages every parameter, including the connector. |
| `--device auto`, `xpu:N`, `cuda:N`, or `cpu` | Fisher only. `auto` uses all visible Aurora XPUs or CUDA GPUs. |
| `--devices xpu:0,xpu:1` | Fisher only. Optional GPU subset. Resume with the same device list and order. |
| `--max-examples N` | Fisher only. First N nonempty rows per expert; 0 uses all rows. |
| `--processor PATH` | Fisher processor checkpoint; defaults to `--base`. |
| `--coefficient NAME=WEIGHT` | Fisher static averaging weight; repeat for every expert. Equal weights by default. |
| `--base-coefficient WEIGHT` | Fisher base participation; default 0. |
| `--top-percent PERCENT` | Sign-conflict selection percentage; default 1. |
| `--seed INTEGER` | Fisher target sampling seed; default 0. |
| `--trust-remote-code` | Allow custom Transformers checkpoint code. |

Resume files contain the raw group sums, number of completed examples, and the target-sampling RNG state. They are written atomically per GPU shard after a completed example. A manifest checks the checkpoint paths, JSONL file metadata, averaging settings, seed, and GPU layout before any saved scores are reused. Change `--resume-dir` when changing those inputs. An interrupted example is recomputed; completed examples are skipped, then shard scores are combined by example count. The final result is written atomically once all shards finish.

Each JSONL record must contain `images`, `conversation`, and a final assistant turn. The script follows `img_loc` on a user turn to place images before or after its text. It uses earlier turns as context and assigns Fisher targets only to the final assistant answer. Image paths may be absolute or relative to the JSONL file. Results can be read with `torch.load(path, weights_only=True)`.

For the RRG24 PadChest and CheXpert checkpoints plus ReX, use [rrg24_example_commands.sh](rrg24_example_commands.sh) with the same optional method argument. Its new outputs and Fisher resume shards default to `/flare/MatSciAI/xinxil/output/analysis/rrg24/detailed`. Use [rrg24_checkpoint_commands.sh](rrg24_checkpoint_commands.sh) for the two checkpoint-only methods or [rrg24_fisher_commands.sh](rrg24_fisher_commands.sh) to run/resume Fisher on its own.

The current `all` policy includes the connector. Old connector-preserving Fisher shards cannot be reused for this policy; choose a new output and resume directory. The example command uses `fisher-sensitivity-all-components.pt` and `fisher-sensitivity-all-components.resume` to preserve the earlier results.

## FFN breakdown by component

The `component_architecture` view contains `vision_ffn`, `vision_non_ffn`, `connector`, `language_ffn`, and `language_non_ffn`. Vision groups exclude the visual merger, which stays in the connector group. FFNs include dense MLP weights and explicitly named experts; router/gate modules remain non-FFN. Language non-FFN includes attention, embeddings, norms, and the output head.

All three analysis methods report these groups. Fisher percentage fields for this view use the sum of its five disjoint groups as their denominator. The two vision scores sum to `vision_tower`; the two language scores sum to `language_decoder`. Older Fisher files lack these intersection scores and cannot supply them post-hoc. The resume manifest now records group definitions and rejects older shards.

The `component_detail` view further divides each tower into mutually exclusive groups:

| Component | Detailed groups |
| --- | --- |
| Vision tower | `vision_ffn`, `vision_attention`, `vision_patch_embedding`, `vision_norm`, `vision_other` |
| Connector | `connector` (visual merger MLP and its normalization) |
| Language decoder | `language_ffn`, `language_attention`, `language_token_embedding`, `language_norm`, `language_output_head`, `language_other` |

`vision_other` and `language_other` capture unmatched parameters, including router/gate modules. A tied `lm_head` is counted once under the canonical token embedding name; an untied output head appears in `language_output_head`. The detailed groups sum to the three component totals, and each detailed group's percentage uses the total of this full view. The current base Qwen2.5-VL-3B checkpoint has no parameters in either residual group or the untied output-head group; future checkpoints may. Result files and resume shards created before this change do not contain the new scores. The example command files now default to isolated `detailed` output directories, and Fisher's resume manifest version is 4.

## Parallel comparative Fisher jobs

The comparative Fisher analysis can process one expert's examples per job with
`--fisher-expert NAME`. Each job still loads the base and all three expert
checkpoints so `expert_to_average` uses the same three-expert average. Only the
selected model receives Fisher backward passes. The output contains that
expert's `fisher_group_scores`; the checkpoint-only task-vector and sign-conflict
outputs remain separate.

The three ready-to-submit next-eval command scripts are
`fisher_endo_next_eval.sh`, `fisher_rex_next_eval.sh`, and
`fisher_pathgen_next_eval.sh`. They write distinct result files and resume
directories under `/flare/MatSciAI/xinxil/output/analysis/detailed`.
Run each through the shared next-eval PBS wrapper as a separate job. Re-submit
the same script after a walltime stop to resume its own saved shard.
The original `fisher_commands.sh` without `FISHER_EXPERT` still processes all
three experts sequentially and keeps its original output path.

## Capacity submissions

`capacity_analysis_job.sh` uses capacity's available `flexolmo` environment and enables `--activation-checkpointing`. This recomputes vision and language transformer blocks during backward while keeping the model in evaluation mode; it reduces stored activations without changing the Fisher definition. Runtime and memory use still depend on sequence lengths. Next-eval continues to use its newer `flexolmo-next-train` environment.

Both experiment command files accept `ANALYSIS_OUTPUT_DIR` for isolated results and resume shards; their defaults are `/flare/MatSciAI/xinxil/output/analysis/detailed` and `/flare/MatSciAI/xinxil/output/analysis/rrg24/detailed`. The original sets and RRG24 sets can therefore run in parallel without overwriting an earlier job's files.

## Standalone Fisher sensitivity for the local Flex MoE model

[run_flex_fisher.py](run_flex_fisher.py) loads a saved `flex_qwen2_5_vl_moe` checkpoint through the local model registration in `modeling/flex_qwen2_5_vl_moe`. It estimates conditional Fisher on each named JSONL dataset. No base checkpoint, task vector, or averaged checkpoint is involved. Results contain `fisher_mass` (the sum of estimated Fisher across a group), `mean_fisher` (per-coordinate Fisher), and `fisher_mass_percent` (the group's share within one partition). The detailed component groups and the broad views are both included. Percentages across overlapping views must not be added together. These scores describe where the model's answer likelihood is locally sensitive; they do not measure how far weights moved or establish which training dataset caused the sensitivity.

[flex_fisher_commands.sh](flex_fisher_commands.sh) provides the original Endo/ReX/PathGen and RRG24 PadChest/CheXpert/ReX dataset sets. Set `FLEX_MODEL_CHECKPOINT` to a **saved checkpoint directory** containing the custom model's weights and `config.json`; the `modeling/flex_qwen2_5_vl_moe` source directory is not a checkpoint. For a separate RRG24 model, set `FLEX_RRG24_MODEL_CHECKPOINT` too. For example:

~~~bash
FLEX_MODEL_CHECKPOINT=/flare/MatSciAI/xinxil/output/FlexBagel/btx_endochat_rexgradient_pathgen_router_tuned/final \
  bash analysis/flex_fisher_commands.sh original
~~~

Run `bash analysis/flex_fisher_commands.sh rrg24` or `all` with the corresponding checkpoint variables. The wrapper uses all visible Aurora GPUs automatically, 256 nonempty JSONL rows per dataset by default, and separate resumable output directories under `/flare/MatSciAI/xinxil/output/analysis/flex-fisher`. Set `FISHER_MAX_EXAMPLES`, `ANALYSIS_OUTPUT_DIR`, `FLEX_PROCESSOR`, or `FISHER_ACTIVATION_CHECKPOINTING=1` to adjust those settings. Resume by repeating the same command. Changing the model, examples, seed, group definitions, or GPU layout requires a new resume directory.

To run a different dataset directly:

~~~bash
python -m analysis.run_flex_fisher \
  --model /path/to/flex-moe-checkpoint/final \
  --examples task=/path/to/train.jsonl \
  --output /path/to/fisher-mass.pt \
  --max-examples 256
~~~

The CLI also accepts repeated `--examples NAME=JSONL`, `--processor PATH`, `--resume-dir DIR`, `--seed N`, `--device cpu|xpu:N|cuda:N`, `--devices xpu:0,xpu:1`, `--activation-checkpointing`, and `--trust-remote-code`. `--device auto` is the default and uses every visible GPU. Each GPU processes a disjoint shard of examples; completed examples are saved after each backward pass.

### Three specialist Flex checkpoints on next-eval

The command files [flex_fisher_endo_next_eval.sh](flex_fisher_endo_next_eval.sh), [flex_fisher_rex_next_eval.sh](flex_fisher_rex_next_eval.sh), and [flex_fisher_pathgen_next_eval.sh](flex_fisher_pathgen_next_eval.sh) measure Fisher mass on the matching EndoChat, ReX, and PathGen training JSONL files, respectively. They use the exact specialist checkpoints `alrope/endochat_qwen2_5-3b-vl-flex-topk2`, `/flare/MatSciAI/xinxil/output/FlexBagel/rexvgradient_qwen2_5-3b-vl-flex-topk2-8bits/final`, and `alrope/pathgen_qwen2_5-3b-vl-flex-topk2`. Each processes 256 nonempty examples, uses its model's processor, enables activation checkpointing, and writes a separate resumable result under `/flare/MatSciAI/xinxil/output/analysis/flex-fisher/three-models`. These files are intended for the Aurora next-eval PBS wrapper, which supplies `FLEXBAGEL_PYTHON`.

### Expert 0 versus expert 1 Fisher split

Add `--split-experts` to `run_flex_fisher.py` for Flex checkpoints whose vision and language FFNs are fully covered by experts 0 and 1. It adds `vision_expert_0`, `vision_expert_1`, `language_expert_0`, and `language_expert_1` to the existing broad and detailed groups. Each expert row reports `fisher_mass_percent` of the entire model, `within_expertized_percent` across all four expert groups, and `within_tower_ffn_percent` within its vision or language FFN. The runner rejects checkpoints where these four groups do not cover all FFN parameters. This measures Fisher sensitivity of the expert weights, not routing frequency or the fraction of tokens sent to an expert.

The next-eval command files `flex_fisher_endo_split_next_eval.sh`, `flex_fisher_rex_split_next_eval.sh`, and `flex_fisher_pathgen_split_next_eval.sh` write under `/flare/MatSciAI/xinxil/output/analysis/flex-fisher/three-models/expert-split`. They use new resume directories; the earlier aggregate expert results remain available.
