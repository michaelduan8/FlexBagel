"""Reusable checkpoint and conditional-Fisher analyses for Qwen2.5-VL experts.

All checkpoint arguments are state dictionaries with matching names and shapes.
Checkpoint distances are analyzed in float64 on CPU. Fisher accumulators use
float32. Component and architectural groups intentionally overlap.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch

TensorMap = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ParameterGroup:
    name: str
    include: str
    exclude: str | None = None

    def matches(self, key: str) -> bool:
        return bool(re.search(self.include, key)) and not (
            self.exclude and re.search(self.exclude, key)
        )


_VISION_PATTERN = r"^(?:model\.)?visual\.(?!merger\.)"
_LANGUAGE_PATTERN = r"^(?:model\.(?!visual\.)|language_model\.|lm_head\.)"
_FFN_PATTERN = r"\.(?:mlp|merger)\.(?!router\.|gate\.)"
_ATTENTION_PATTERN = r"\.(?:attn|self_attn|cross_attn)\."
_VISION_PATCH_PATTERN = r"\.patch_embed\."
_NORM_PATTERN = r"\.(?:[^.]*norm[^.]*|ln_q)\."
_TOKEN_EMBEDDING_PATTERN = r"\.embed_tokens\."
_OUTPUT_HEAD_PATTERN = r"^(?:model\.)?lm_head\."


def _scoped_pattern(scope: str, kind: str) -> str:
    return rf"^(?={scope})(?=.*{kind})"


def _alternative(*patterns: str) -> str:
    return "(?:" + "|".join(patterns) + ")"


_VISION_DETAIL_PATTERNS = (
    _FFN_PATTERN, _ATTENTION_PATTERN, _VISION_PATCH_PATTERN, _NORM_PATTERN,
)
_LANGUAGE_DETAIL_PATTERNS = (
    _FFN_PATTERN, _ATTENTION_PATTERN, _TOKEN_EMBEDDING_PATTERN,
    _NORM_PATTERN, _OUTPUT_HEAD_PATTERN,
)

DEFAULT_GROUPS = (
    ParameterGroup("vision_tower", _VISION_PATTERN),
    ParameterGroup("connector", r"^(?:model\.)?visual\.merger\."),
    ParameterGroup("language_decoder", _LANGUAGE_PATTERN),
    ParameterGroup("architectural_ffn", _FFN_PATTERN),
    ParameterGroup("architectural_non_ffn", r"^", _FFN_PATTERN),
    ParameterGroup("expertized_ffn", r"\.(?:mlp|merger)\.experts\.\d+\."),
    ParameterGroup("shared_parameters", r"^", r"\.(?:mlp|merger)\.experts\.\d+\."),
    ParameterGroup("vision_ffn", rf"^(?={_VISION_PATTERN})(?=.*{_FFN_PATTERN})"),
    ParameterGroup("vision_non_ffn", _VISION_PATTERN, _FFN_PATTERN),
    ParameterGroup("language_ffn", rf"^(?={_LANGUAGE_PATTERN})(?=.*{_FFN_PATTERN})"),
    ParameterGroup("language_non_ffn", _LANGUAGE_PATTERN, _FFN_PATTERN),
    ParameterGroup("vision_attention", _scoped_pattern(_VISION_PATTERN, _ATTENTION_PATTERN),
                   _FFN_PATTERN),
    ParameterGroup("vision_patch_embedding", _scoped_pattern(_VISION_PATTERN, _VISION_PATCH_PATTERN),
                   _alternative(_FFN_PATTERN, _ATTENTION_PATTERN)),
    ParameterGroup("vision_norm", _scoped_pattern(_VISION_PATTERN, _NORM_PATTERN),
                   _alternative(_FFN_PATTERN, _ATTENTION_PATTERN, _VISION_PATCH_PATTERN)),
    ParameterGroup("vision_other", _VISION_PATTERN, _alternative(*_VISION_DETAIL_PATTERNS)),
    ParameterGroup("language_attention", _scoped_pattern(_LANGUAGE_PATTERN, _ATTENTION_PATTERN),
                   _FFN_PATTERN),
    ParameterGroup("language_token_embedding", _scoped_pattern(_LANGUAGE_PATTERN, _TOKEN_EMBEDDING_PATTERN),
                   _alternative(_FFN_PATTERN, _ATTENTION_PATTERN)),
    ParameterGroup("language_norm", _scoped_pattern(_LANGUAGE_PATTERN, _NORM_PATTERN),
                   _alternative(_FFN_PATTERN, _ATTENTION_PATTERN, _TOKEN_EMBEDDING_PATTERN)),
    ParameterGroup("language_output_head", _scoped_pattern(_LANGUAGE_PATTERN, _OUTPUT_HEAD_PATTERN),
                   _alternative(_FFN_PATTERN, _ATTENTION_PATTERN, _TOKEN_EMBEDDING_PATTERN, _NORM_PATTERN)),
    ParameterGroup("language_other", _LANGUAGE_PATTERN, _alternative(*_LANGUAGE_DETAIL_PATTERNS)),
)


GROUP_PARTITIONS = {
    "component": ("vision_tower", "connector", "language_decoder"),
    "architecture": ("architectural_ffn", "architectural_non_ffn"),
    "expertization": ("expertized_ffn", "shared_parameters"),
    "component_architecture": (
        "vision_ffn", "vision_non_ffn", "connector",
        "language_ffn", "language_non_ffn",
    ),
    "component_detail": (
        "vision_ffn", "vision_attention", "vision_patch_embedding",
        "vision_norm", "vision_other", "connector", "language_ffn",
        "language_attention", "language_token_embedding", "language_norm",
        "language_output_head", "language_other",
    ),
}


def group_summary(parameters: TensorMap,
                  groups: Sequence[ParameterGroup] = DEFAULT_GROUPS,
                  example_limit: int = 3) -> dict:
    """Count parameter tensors and coordinates; show names for classification checks."""
    groups = _groups(groups)
    result = {g.name: {"tensor_count": 0, "parameter_count": 0, "examples": []}
              for g in groups}
    for name, tensor in sorted(parameters.items()):
        for group in groups:
            if group.matches(name):
                row = result[group.name]
                row["tensor_count"] += 1
                row["parameter_count"] += tensor.numel()
                if len(row["examples"]) < example_limit:
                    row["examples"].append(name)
    return result


def _groups(groups: Sequence[ParameterGroup]) -> tuple[ParameterGroup, ...]:
    groups = tuple(groups)
    names = [group.name for group in groups]
    if not names or len(names) != len(set(names)):
        raise ValueError("Groups must have distinct names and cannot be empty")
    return groups


def _keys(base: TensorMap, experts: Mapping[str, TensorMap],
          parameter_names: Iterable[str] | None = None) -> list[str]:
    if not experts:
        raise ValueError("At least one expert is required")
    keys = set(base)
    for name, expert in experts.items():
        if set(expert) != keys:
            raise ValueError(f"Checkpoint keys differ for expert {name!r}")
        for key in keys:
            if expert[key].shape != base[key].shape:
                raise ValueError(f"Shape differs for {name!r}, {key!r}")
            if expert[key].is_floating_point() != base[key].is_floating_point():
                raise ValueError(f"Tensor type differs for {name!r}, {key!r}")
    floating = {key for key in keys if base[key].is_floating_point()}
    if parameter_names is not None:
        selected = list(parameter_names)
        if len(selected) != len(set(selected)) or not set(selected) <= floating:
            raise ValueError("Parameter names must be unique floating checkpoint keys")
        floating = set(selected)
    return sorted(floating)


def _delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.detach().to(device="cpu", dtype=torch.float64) - b.detach().to(
        device="cpu", dtype=torch.float64
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def task_vector_similarity(
    base: TensorMap,
    experts: Mapping[str, TensorMap],
    groups: Sequence[ParameterGroup] = DEFAULT_GROUPS,
    *,
    parameter_names: Iterable[str] | None = None,
) -> dict:
    """Return L2 update magnitudes, relative magnitudes, and pairwise cosines.

    A cosine or relative magnitude with a zero denominator is returned as None.
    """
    groups = _groups(groups)
    keys = _keys(base, experts, parameter_names)
    names = list(experts)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    accum = {
        g.name: {
            "count": 0, "base_sq": 0.0,
            "update_sq": {name: 0.0 for name in names},
            "dot": {pair: 0.0 for pair in pairs},
        }
        for g in groups
    }
    for key in keys:
        matched = [g.name for g in groups if g.matches(key)]
        if not matched:
            continue
        base_value = base[key].detach().to(device="cpu", dtype=torch.float64)
        changes = {name: _delta(experts[name][key], base[key]) for name in names}
        base_sq = torch.sum(base_value.square()).item()
        squares = {name: torch.sum(value.square()).item() for name, value in changes.items()}
        dots = {pair: torch.sum(changes[pair[0]] * changes[pair[1]]).item() for pair in pairs}
        for group in matched:
            row = accum[group]
            row["count"] += base_value.numel()
            row["base_sq"] += base_sq
            for name in names:
                row["update_sq"][name] += squares[name]
            for pair in pairs:
                row["dot"][pair] += dots[pair]
    result = {}
    for group, row in accum.items():
        base_norm = math.sqrt(row["base_sq"])
        norms = {name: math.sqrt(value) for name, value in row["update_sq"].items()}
        result[group] = {
            "parameter_count": row["count"],
            "base_l2": base_norm,
            "experts": {
                name: {
                    "update_l2": norm,
                    "update_rms": math.sqrt(row["update_sq"][name] / row["count"]) if row["count"] else None,
                    "relative_l2": _safe_ratio(norm, base_norm),
                }
                for name, norm in norms.items()
            },
            "pairs": {
                pair: {
                    "dot": row["dot"][pair],
                    "cosine": _safe_ratio(row["dot"][pair], norms[pair[0]] * norms[pair[1]]),
                }
                for pair in pairs
            },
        }
    return result


def sign_conflicts(
    base: TensorMap,
    experts: Mapping[str, TensorMap],
    top_percent: float = 1.0,
    groups: Sequence[ParameterGroup] = DEFAULT_GROUPS,
    *,
    parameter_names: Iterable[str] | None = None,
) -> dict:
    """Select the exact top ceil(percent * nonzero / 100) per group and expert.

    Ties are broken by sorted parameter name, then flattened coordinate. The
    selection is global within each group, rather than repeated for each tensor.
    Only the top magnitudes and one parameter tensor are held in working memory.
    """
    if not 0 < top_percent <= 100:
        raise ValueError("top_percent must be in (0, 100]")
    groups = _groups(groups)
    keys = _keys(base, experts, parameter_names)
    names = list(experts)
    result = {}
    for group in groups:
        group_keys = [key for key in keys if group.matches(key)]
        parameter_count = sum(base[key].numel() for key in group_keys)
        nonzero_counts = {
            name: sum(int(torch.count_nonzero(_delta(experts[name][key], base[key])))
                      for key in group_keys)
            for name in names
        }
        selected_counts = {
            name: math.ceil(count * top_percent / 100)
            for name, count in nonzero_counts.items()
        }
        cutoffs = {}
        tied_remaining = {}
        for name in names:
            k = selected_counts[name]
            if not k:
                continue
            top = torch.empty(0, dtype=torch.float64)
            for key in group_keys:
                magnitudes = _delta(experts[name][key], base[key]).abs().reshape(-1)
                nonzero = magnitudes[magnitudes > 0]
                if nonzero.numel() > k:
                    nonzero = torch.topk(nonzero, k).values
                top = torch.cat((top, nonzero))
                if top.numel() > k:
                    top = torch.topk(top, k).values
            cutoff = top.min()
            cutoffs[name] = cutoff
            tied_remaining[name] = k - int((top > cutoff).sum())
        pairs = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs[(a, b)] = {"overlap_count": 0, "opposite_sign_count": 0}
        for key in group_keys:
            selected = {}
            for name in names:
                values = _delta(experts[name][key], base[key]).reshape(-1)
                signs = torch.zeros(values.numel(), dtype=torch.int8)
                if selected_counts[name]:
                    magnitudes = values.abs()
                    keep = magnitudes > cutoffs[name]
                    if tied_remaining[name]:
                        tied = torch.nonzero(magnitudes == cutoffs[name], as_tuple=True)[0]
                        chosen = tied[:tied_remaining[name]]
                        keep[chosen] = True
                        tied_remaining[name] -= chosen.numel()
                    signs[keep] = torch.sign(values[keep]).to(torch.int8)
                selected[name] = signs
            for pair, row in pairs.items():
                a, b = pair
                shared = (selected[a] != 0) & (selected[b] != 0)
                row["overlap_count"] += int(shared.sum())
                row["opposite_sign_count"] += int(((selected[a] != selected[b]) & shared).sum())
        for (a, b), row in pairs.items():
            overlap = row["overlap_count"]
            union_count = selected_counts[a] + selected_counts[b] - overlap
            row["overlap_fraction_a"] = _safe_ratio(overlap, selected_counts[a])
            row["overlap_fraction_b"] = _safe_ratio(overlap, selected_counts[b])
            row["jaccard"] = _safe_ratio(overlap, union_count)
            row["opposite_sign_fraction"] = _safe_ratio(row["opposite_sign_count"], overlap)
        result[group.name] = {
            "parameter_count": parameter_count,
            "nonzero_counts": nonzero_counts,
            "selected_counts": selected_counts,
            "pairs": pairs,
        }
    return result


def _unique_parameters(model: torch.nn.Module, groups: Sequence[ParameterGroup]) -> dict[str, torch.nn.Parameter]:
    """The same canonical names as named_parameters(), including frozen weights."""
    return {name: p for name, p in model.named_parameters(remove_duplicate=True)
            if p.is_floating_point() and any(g.matches(name) for g in groups)}


def estimate_conditional_token_fisher(
    model: torch.nn.Module,
    examples: Iterable[Mapping[str, torch.Tensor]],
    groups: Sequence[ParameterGroup] = DEFAULT_GROUPS,
    generator: torch.Generator | None = None,
    *,
    score_deltas: Mapping[str, TensorMap] | None = None,
    on_example_complete: Callable[[], None] | None = None,
    checkpoint_callback: Callable[[int, dict], None] | None = None,
    initial_scores: Mapping[str, Mapping] | None = None,
    initial_count: int = 0,
) -> dict:
    """Sample independent answer targets and take one score gradient per example.

    The sum of token log probabilities is divided by sqrt(answer length).
    Independent score functions have zero mean, so its squared gradient has
    the same expectation as the mean of per-token squared gradients. If
    score_deltas is supplied, accumulate weighted group sums directly rather
    than allocate coordinate-level Fisher tensors.
    """
    groups = _groups(groups)
    if initial_count < 0 or (initial_scores is None) != (initial_count == 0):
        raise ValueError("Resume scores and a positive example count must be supplied together")
    if initial_scores is not None and score_deltas is None:
        raise ValueError("Resuming requires direct group score accumulation")
    if model.training:
        raise ValueError("Set model.eval() before Fisher estimation")
    params = _unique_parameters(model, groups)
    if not params:
        raise ValueError("No floating parameters matched the groups")
    names = list(params)
    if score_deltas is not None:
        for metric, deltas in score_deltas.items():
            if set(deltas) != set(names):
                raise ValueError(f"Delta keys for {metric!r} must match selected parameters")
            for name in names:
                if deltas[name].shape != params[name].shape:
                    raise ValueError(f"Delta shape differs for {metric!r}, {name!r}")
        result = {g.name: {"parameter_count": 0, "fisher_mass": 0.0,
                           **{metric: 0.0 for metric in score_deltas}}
                  for g in groups}
        for name, p in params.items():
            for group in groups:
                if group.matches(name):
                    result[group.name]["parameter_count"] += p.numel()
        if initial_scores is not None:
            if set(initial_scores) != set(result):
                raise ValueError("Resume groups do not match the current model")
            for group, row in result.items():
                previous = initial_scores[group]
                if set(previous) != set(row) or previous["parameter_count"] != row["parameter_count"]:
                    raise ValueError(f"Resume parameter coverage differs for {group}")
                for metric in ("fisher_mass", *score_deltas):
                    row[metric] = float(previous[metric])
    else:
        result = {name: torch.zeros_like(p, device="cpu", dtype=torch.float32)
                  for name, p in params.items()}
    original_grad_flags = {id(p): p.requires_grad for p in params.values()}
    example_count = initial_count
    try:
        for p in params.values():
            p.requires_grad_(True)
        for example in examples:
            if "labels" not in example or "input_ids" not in example:
                raise ValueError("Each example requires input_ids and labels")
            labels = example["labels"]
            if labels.ndim != 2 or labels.shape[0] != 1 or labels.shape != example["input_ids"].shape:
                raise ValueError("input_ids and labels must have matching shape [1, sequence]")
            positions = torch.nonzero(labels[0] != -100, as_tuple=True)[0].tolist()
            if not positions or positions[0] == 0:
                raise ValueError("Each example needs answer labels after at least one prefix token")
            if not torch.equal(labels[0, positions], example["input_ids"][0, positions]):
                raise ValueError("Answer labels must match reference input_ids")
            model_inputs = {key: value for key, value in example.items() if key != "labels"}
            logits = model(**model_inputs).logits
            if logits.shape[0] != 1 or logits.shape[1] < labels.shape[1]:
                raise ValueError("Model logits do not align with input_ids")
            log_scores = []
            for position in positions:
                token_logits = logits[0, position - 1].float()
                probabilities = torch.softmax(token_logits.detach(), dim=-1)
                # A CPU generator works for XPU/CUDA as well; only vocabulary
                # probabilities are copied, not the model activations.
                target = torch.multinomial(probabilities.cpu(), 1, generator=generator).item()
                log_scores.append(torch.log_softmax(token_logits, dim=-1)[target])
            score = torch.stack(log_scores).sum() / math.sqrt(len(positions))
            gradients = (torch.autograd.grad(score, list(params.values()), allow_unused=True)
                         if score.requires_grad else (None,) * len(params))
            for name, gradient in zip(names, gradients):
                if gradient is None:
                    continue
                squared = gradient.detach().to(device="cpu", dtype=torch.float32).square_()
                if score_deltas is None:
                    result[name].add_(squared)
                else:
                    matched = [g.name for g in groups if g.matches(name)]
                    mass = squared.sum().item()
                    weighted = {metric: (squared * deltas[name].detach().to(
                        device="cpu", dtype=torch.float32).square()).sum().item()
                        for metric, deltas in score_deltas.items()}
                    for group in matched:
                        result[group]["fisher_mass"] += mass
                        for metric, value in weighted.items():
                            result[group][metric] += value
            example_count += 1
            if checkpoint_callback is not None:
                checkpoint_callback(example_count, result)
            if on_example_complete is not None:
                on_example_complete()
    finally:
        for p in params.values():
            p.requires_grad_(original_grad_flags[id(p)])
    if not example_count:
        raise ValueError("At least one example is required")
    if score_deltas is None:
        for value in result.values():
            value.div_(example_count)
    else:
        for row in result.values():
            for key in ("fisher_mass", *score_deltas):
                row[key] /= example_count
    return result


def _average_weights(names: list[str], coefficients: Mapping[str, float] | None,
                     base_coefficient: float) -> tuple[dict[str, float], float]:
    if coefficients is None:
        coefficients = {name: 1.0 for name in names}
    if set(coefficients) != set(names):
        raise ValueError("Average coefficients must name every expert exactly once")
    weights = {name: float(coefficients[name]) for name in names}
    base_weight = float(base_coefficient)
    if any(not math.isfinite(v) or v < 0 for v in (*weights.values(), base_weight)):
        raise ValueError("Average coefficients must be finite and nonnegative")
    total = sum(weights.values()) + base_weight
    if total <= 0:
        raise ValueError("Average coefficients must have positive total")
    return {name: value / total for name, value in weights.items()}, base_weight / total


def fisher_weighted_sensitivity(
    base: TensorMap,
    experts: Mapping[str, TensorMap],
    fishers: Mapping[str, TensorMap],
    groups: Sequence[ParameterGroup] = DEFAULT_GROUPS,
    *,
    parameter_names: Iterable[str] | None = None,
    coefficients: Mapping[str, float] | None = None,
    base_coefficient: float = 0.0,
    average_mask: Mapping[str, torch.Tensor | bool] | None = None,
) -> dict:
    """Return raw Fisher-weighted sums and secondary normalized statistics.

    parameter_names should be canonical named_parameters() keys when state
    dictionaries include buffers or tied aliases. Every selected key requires
    a Fisher entry; missing entries are an error.
    """
    groups = _groups(groups)
    keys = _keys(base, experts, parameter_names)
    names = list(experts)
    if set(fishers) != set(names):
        raise ValueError("Fisher names must match expert names")
    for name in names:
        if set(fishers[name]) != set(keys):
            raise ValueError(f"Fisher keys for {name!r} must match selected parameters")
    weights, base_weight = _average_weights(names, coefficients, base_coefficient)
    if average_mask is not None and not set(average_mask) <= set(keys):
        raise ValueError("Average mask contains unselected parameters")
    result = {g.name: {name: {"parameter_count": 0, "fisher_mass": 0.0,
                             "base_to_expert": 0.0, "expert_to_average": 0.0}
                       for name in names} for g in groups}
    counts = group_summary({key: base[key] for key in keys}, groups)
    for key in keys:
        matched = [g.name for g in groups if g.matches(key)]
        if not matched:
            continue
        mask = True if average_mask is None else average_mask.get(key, False)
        mask = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
        if mask.numel() != 1 and mask.shape != base[key].shape:
            raise ValueError(f"Average mask shape differs for {key!r}")
        average = base[key].detach().to(device="cpu", dtype=torch.float64) * base_weight
        for name in names:
            average = average + experts[name][key].detach().to(
                device="cpu", dtype=torch.float64) * weights[name]
        for name in names:
            f = fishers[name][key].detach().to(device="cpu", dtype=torch.float32)
            if f.shape != base[key].shape or not torch.isfinite(f).all() or (f < 0).any():
                raise ValueError(f"Invalid Fisher for {name!r}, {key!r}")
            update = _delta(experts[name][key], base[key]).float()
            to_average = torch.where(mask, _delta(experts[name][key], average).float(),
                                     torch.zeros_like(update))
            mass = f.sum().item()
            adaptation = (f * update.square()).sum().item()
            averaging = (f * to_average.square()).sum().item()
            for group in matched:
                row = result[group][name]
                row["parameter_count"] += f.numel()
                row["fisher_mass"] += mass
                row["base_to_expert"] += adaptation
                row["expert_to_average"] += averaging
    for group, rows in result.items():
        for row in rows.values():
            count, mass = row["parameter_count"], row["fisher_mass"]
            row["tensor_count"] = counts[group]["tensor_count"]
            row["examples"] = counts[group]["examples"]
            row["mean_fisher"] = _safe_ratio(mass, count)
            for field in ("base_to_expert", "expert_to_average"):
                row[field + "_per_parameter"] = _safe_ratio(row[field], count)
                weighted_mean = _safe_ratio(row[field], mass)
                row[field + "_fisher_rms"] = math.sqrt(weighted_mean) if weighted_mean is not None else None
    for partition, group_names in GROUP_PARTITIONS.items():
        present = [name for name in group_names if name in result]
        for expert in names:
            for field in ("base_to_expert", "expert_to_average"):
                total = sum(result[group][expert][field] for group in present)
                for group in present:
                    share = _safe_ratio(result[group][expert][field], total)
                    result[group][expert][field + "_share"] = share
                    result[group][expert][field + "_percent"] = (
                        100 * share if share is not None else None)
    return result
