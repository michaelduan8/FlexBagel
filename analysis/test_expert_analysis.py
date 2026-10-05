import math

import pytest
import torch

from analysis.expert_analysis import (
    DEFAULT_GROUPS,
    GROUP_PARTITIONS,
    group_summary,
    ParameterGroup,
    estimate_conditional_token_fisher,
    fisher_weighted_sensitivity,
    sign_conflicts,
    task_vector_similarity,
)


ALL = (ParameterGroup("all", r"^"),)


def test_default_groups_match_flexbagel_checkpoint_names():
    groups = {group.name: group for group in DEFAULT_GROUPS}
    assert groups["vision_tower"].matches("visual.blocks.0.attn.qkv.weight")
    assert groups["connector"].matches("visual.merger.experts.0.fc1.weight")
    assert groups["language_decoder"].matches("model.layers.0.self_attn.q_proj.weight")
    assert groups["language_decoder"].matches("model.norm.weight")
    assert not groups["language_decoder"].matches("visual.blocks.0.attn.qkv.weight")
    assert groups["expertized_ffn"].matches("model.layers.0.mlp.experts.0.up_proj.weight")
    assert not groups["shared_parameters"].matches("model.layers.0.mlp.experts.0.up_proj.weight")
    dense = "model.layers.0.mlp.up_proj.weight"
    assert groups["architectural_ffn"].matches(dense)
    assert not groups["architectural_non_ffn"].matches(dense)
    assert groups["shared_parameters"].matches(dense)
    assert not groups["connector"].matches(dense)
    assert groups["architectural_non_ffn"].matches("model.layers.0.self_attn.q_proj.weight")
    summary = group_summary({dense: torch.zeros(2), "visual.merger.fc1.weight": torch.zeros(3)})
    assert summary["architectural_ffn"]["parameter_count"] == 5
    assert dense in summary["architectural_ffn"]["examples"]
    assert summary["connector"]["tensor_count"] == 1


def test_task_vectors_and_sign_conflicts():
    base = {"w": torch.tensor([1., 2., 0., 0.])}
    experts = {
        "a": {"w": torch.tensor([3., 1., 0., 0.])},
        "b": {"w": torch.tensor([-1., 3., 0., 0.])},
    }
    vectors = task_vector_similarity(base, experts, ALL)["all"]
    assert vectors["experts"]["a"]["update_l2"] == pytest.approx(math.sqrt(5))
    assert vectors["experts"]["a"]["relative_l2"] == pytest.approx(1)
    assert vectors["pairs"][("a", "b")]["cosine"] == pytest.approx(-1)
    # At 50%, one of two nonzero coordinates is retained per expert.
    signs = sign_conflicts(base, experts, 50, ALL)["all"]
    assert signs["selected_counts"] == {"a": 1, "b": 1}
    assert signs["pairs"][("a", "b")]["overlap_count"] == 1
    assert signs["pairs"][("a", "b")]["opposite_sign_fraction"] == 1


def test_sign_conflicts_breaks_ties_across_tensors_in_key_order():
    base = {"b": torch.zeros(2), "a": torch.zeros(2)}
    experts = {
        "first": {"a": torch.tensor([1., 1.]), "b": torch.tensor([1., 1.])},
        "second": {"a": torch.tensor([-1., -1.]), "b": torch.tensor([0., -1.])},
    }
    row = sign_conflicts(base, experts, 50, ALL)["all"]
    assert row["selected_counts"] == {"first": 2, "second": 2}
    assert row["pairs"][("first", "second")]["overlap_count"] == 2
    assert row["pairs"][("first", "second")]["opposite_sign_count"] == 2


def test_fisher_weighted_distances_use_expert_local_fisher():
    base = {"w": torch.tensor([0., 0.])}
    experts = {"a": {"w": torch.tensor([2., 0.])}, "b": {"w": torch.tensor([0., 2.])}}
    fishers = {"a": {"w": torch.tensor([3., 1.])}, "b": {"w": torch.tensor([1., 4.])}}
    rows = fisher_weighted_sensitivity(base, experts, fishers, ALL)["all"]
    assert rows["a"]["base_to_expert"] == 12
    assert rows["a"]["expert_to_average"] == 4
    assert rows["b"]["base_to_expert"] == 16
    assert rows["b"]["expert_to_average"] == 5
    assert rows["a"]["base_to_expert_fisher_rms"] == pytest.approx(math.sqrt(3))


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))

    def forward(self, input_ids):
        logits = torch.stack([self.weight.expand_as(input_ids),
                              torch.zeros_like(input_ids, dtype=self.weight.dtype)], dim=-1)
        return type("Output", (), {"logits": logits})()


def _tokenwise_reference(model, example, targets):
    """The original per-token estimator, retained as a reference test."""
    logits = model(input_ids=example["input_ids"]).logits
    positions = torch.nonzero(example["labels"][0] != -100, as_tuple=True)[0].tolist()
    terms = []
    for position, target in zip(positions, targets):
        log_probability = torch.log_softmax(logits[0, position - 1], -1)[target]
        gradient, = torch.autograd.grad(log_probability, (model.weight,), retain_graph=True)
        terms.append(gradient.square())
    return torch.stack(terms).mean().item()


def test_conditional_fisher_one_backward_and_reference_expectation():
    model = TinyLM().eval()
    example = {"input_ids": torch.tensor([[0, 0, 1]]),
               "labels": torch.tensor([[-100, 0, 1]])}
    seed = 27
    completed = []
    fisher = estimate_conditional_token_fisher(
        model, [example], ALL, torch.Generator().manual_seed(seed),
        on_example_complete=lambda: completed.append(True),
    )
    assert completed == [True]
    assert fisher["weight"].dtype == torch.float32
    probability = torch.softmax(torch.tensor([0.4, 0.]), -1)
    replay = torch.Generator().manual_seed(seed)
    targets = [torch.multinomial(probability, 1, generator=replay).item() for _ in range(2)]
    score = sum(float(target == 0) - probability[0].item() for target in targets)
    assert fisher["weight"].item() == pytest.approx(score ** 2 / 2)
    # Enumerating independent targets shows that the cross terms cancel.
    fast_expectation = 0.0
    token_expectation = 0.0
    for a in range(2):
        for b in range(2):
            chance = probability[a].item() * probability[b].item()
            one = (float(a == 0) - probability[0].item())
            two = (float(b == 0) - probability[0].item())
            fast_expectation += chance * (one + two) ** 2 / 2
            token_expectation += chance * _tokenwise_reference(model, example, (a, b))
    assert fast_expectation == pytest.approx(token_expectation)
    model.weight.requires_grad_(False)
    scored = estimate_conditional_token_fisher(
        model, [example], ALL, torch.Generator().manual_seed(seed),
        score_deltas={"change": {"weight": torch.tensor(2.)}})
    assert scored["all"]["change"] == pytest.approx(4 * fisher["weight"].item())
    assert model.weight.requires_grad is False


def test_sensitivity_coverage_coefficients_masks_and_shares():
    base = {"model.layers.0.mlp.up_proj.weight": torch.zeros(2),
            "model.norm.weight": torch.zeros(1),
            "buffer": torch.ones(1)}
    experts = {
        "a": {"model.layers.0.mlp.up_proj.weight": torch.tensor([2., 0.]),
              "model.norm.weight": torch.tensor([1.]), "buffer": torch.ones(1)},
        "b": {"model.layers.0.mlp.up_proj.weight": torch.tensor([0., 2.]),
              "model.norm.weight": torch.tensor([3.]), "buffer": torch.ones(1)},
    }
    keys = ["model.layers.0.mlp.up_proj.weight", "model.norm.weight"]
    fishers = {name: {key: torch.ones_like(base[key]) for key in keys} for name in experts}
    with pytest.raises(ValueError, match="Fisher keys"):
        fisher_weighted_sensitivity(base, experts, fishers)
    rows = fisher_weighted_sensitivity(
        base, experts, fishers, parameter_names=keys,
        coefficients={"a": 3, "b": 1}, base_coefficient=0,
        average_mask={"model.layers.0.mlp.up_proj.weight": torch.tensor([True, False])})
    a = rows["architectural_ffn"]["a"]
    assert a["expert_to_average"] == pytest.approx(0.25)
    assert a["parameter_count"] == 2
    assert a["base_to_expert"] == 4
    assert rows["architectural_ffn"]["a"]["base_to_expert_share"] == pytest.approx(4 / 5)
    assert rows["architectural_non_ffn"]["a"]["base_to_expert_share"] == pytest.approx(1 / 5)
    assert rows["shared_parameters"]["a"]["base_to_expert_share"] == 1
    assert rows["expertized_ffn"]["a"]["base_to_expert_share"] == 0
    with pytest.raises(ValueError, match="Fisher keys"):
        fisher_weighted_sensitivity(base, experts, {"a": fishers["a"], "b": {}},
                                    parameter_names=keys)


def test_fisher_unique_tied_parameters_and_unused_frozen_component():
    model = TinyLM().eval()
    model.alias = model.weight
    model.unused = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
    model.weight.requires_grad_(False)
    example = {"input_ids": torch.tensor([[0, 1]]),
               "labels": torch.tensor([[-100, 1]])}
    fisher = estimate_conditional_token_fisher(
        model, [example], ALL, torch.Generator().manual_seed(1))
    assert set(fisher) == {"weight", "unused"}
    assert fisher["unused"].item() == 0
    assert not model.weight.requires_grad
    assert not model.unused.requires_grad
    only_unused = (ParameterGroup("unused", r"^unused$"),)
    result = estimate_conditional_token_fisher(model, [example], only_unused)
    assert result["unused"].item() == 0


def test_fisher_resume_matches_uninterrupted_sampling():
    model = TinyLM().eval()
    example = {"input_ids": torch.tensor([[0, 0, 1]]),
               "labels": torch.tensor([[-100, 0, 1]])}
    deltas = {"change": {"weight": torch.tensor(2.)}}
    full = estimate_conditional_token_fisher(
        model, [example] * 3, ALL, torch.Generator().manual_seed(19),
        score_deltas=deltas)

    generator = torch.Generator().manual_seed(19)
    checkpoints = []
    estimate_conditional_token_fisher(
        model, [example], ALL, generator, score_deltas=deltas,
        checkpoint_callback=lambda count, scores: checkpoints.append(
            (count, {group: row.copy() for group, row in scores.items()},
             generator.get_state().clone())))
    count, scores, generator_state = checkpoints[-1]
    generator.set_state(generator_state)
    resumed = estimate_conditional_token_fisher(
        model, [example] * 2, ALL, generator, score_deltas=deltas,
        initial_scores=scores, initial_count=count)
    assert resumed["all"]["fisher_mass"] == pytest.approx(
        full["all"]["fisher_mass"])
    assert resumed["all"]["change"] == pytest.approx(full["all"]["change"])


@pytest.mark.parametrize('name,expected', [
    ('model.visual.blocks.0.mlp.fc1.weight', 'vision_ffn'),
    ('visual.blocks.0.mlp.experts.2.fc1.weight', 'vision_ffn'),
    ('model.visual.blocks.0.attn.qkv.weight', 'vision_non_ffn'),
    ('visual.blocks.0.mlp.router.weight', 'vision_non_ffn'),
    ('model.visual.merger.mlp.0.weight', 'connector'),
    ('model.visual.merger.ln_q.weight', 'connector'),
    ('model.language_model.layers.0.mlp.gate_proj.weight', 'language_ffn'),
    ('model.layers.0.mlp.experts.1.up_proj.weight', 'language_ffn'),
    ('language_model.layers.0.self_attn.q_proj.weight', 'language_non_ffn'),
    ('model.language_model.layers.0.mlp.gate.weight', 'language_non_ffn'),
    ('lm_head.weight', 'language_non_ffn'),
])
def test_component_architecture_groups_are_disjoint(name, expected):
    from analysis.expert_analysis import GROUP_PARTITIONS

    groups = {group.name: group for group in DEFAULT_GROUPS}
    matches = [group for group in GROUP_PARTITIONS['component_architecture']
               if groups[group].matches(name)]
    assert matches == [expected]


def test_component_architecture_scores_reconstruct_component_totals():
    names = [
        'model.visual.blocks.0.mlp.fc1.weight',
        'model.visual.blocks.0.attn.qkv.weight',
        'model.visual.merger.mlp.0.weight',
        'model.language_model.layers.0.mlp.up_proj.weight',
        'model.language_model.layers.0.self_attn.q_proj.weight',
    ]
    base = {name: torch.zeros(index + 1) for index, name in enumerate(names)}
    experts = {expert: {name: torch.full_like(value, scale)
                        for name, value in base.items()}
               for expert, scale in [('a', 1.), ('b', 2.)]}
    fishers = {expert: {name: torch.ones_like(value) for name, value in base.items()}
               for expert in experts}
    rows = fisher_weighted_sensitivity(base, experts, fishers)
    for expert in experts:
        for metric in ('parameter_count', 'fisher_mass', 'base_to_expert', 'expert_to_average'):
            assert rows['vision_ffn'][expert][metric] + rows['vision_non_ffn'][expert][metric] == pytest.approx(
                rows['vision_tower'][expert][metric])
            assert rows['language_ffn'][expert][metric] + rows['language_non_ffn'][expert][metric] == pytest.approx(
                rows['language_decoder'][expert][metric])
        assert sum(rows[group][expert]['expert_to_average_percent'] for group in (
            'vision_ffn', 'vision_non_ffn', 'connector',
            'language_ffn', 'language_non_ffn')) == pytest.approx(100.)


@pytest.mark.parametrize("name,expected", [
    ("model.visual.blocks.0.mlp.fc1.weight", "vision_ffn"),
    ("visual.blocks.0.attn.qkv.weight", "vision_attention"),
    ("visual.patch_embed.proj.weight", "vision_patch_embedding"),
    ("model.visual.blocks.0.norm1.weight", "vision_norm"),
    ("visual.pos_embed", "vision_other"),
    ("visual.blocks.0.mlp.router.weight", "vision_other"),
    ("model.visual.merger.ln_q.weight", "connector"),
    ("model.layers.0.mlp.gate_proj.weight", "language_ffn"),
    ("model.layers.0.self_attn.q_proj.weight", "language_attention"),
    ("model.embed_tokens.weight", "language_token_embedding"),
    ("model.layers.0.input_layernorm.weight", "language_norm"),
    ("model.norm.weight", "language_norm"),
    ("lm_head.weight", "language_output_head"),
    ("model.layers.0.mlp.router.weight", "language_other"),
])
def test_component_detail_assigns_exactly_one_group(name, expected):
    groups = {group.name: group for group in DEFAULT_GROUPS}
    matches = [group for group in GROUP_PARTITIONS["component_detail"]
               if groups[group].matches(name)]
    assert matches == [expected]


def test_component_detail_reconstructs_component_scores():
    names = [
        "visual.blocks.0.mlp.fc1.weight",
        "visual.blocks.0.attn.qkv.weight",
        "visual.patch_embed.proj.weight",
        "visual.blocks.0.norm1.weight",
        "visual.pos_embed",
        "visual.merger.ln_q.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.mlp.router.weight",
    ]
    base = {name: torch.zeros(i + 1) for i, name in enumerate(names)}
    experts = {"a": {name: torch.ones_like(value) for name, value in base.items()}}
    fisher = {"a": {name: torch.ones_like(value) for name, value in base.items()}}
    summary = group_summary(base)
    detail = GROUP_PARTITIONS["component_detail"]
    assert sum(summary[group]["parameter_count"] for group in detail) == sum(
        value.numel() for value in base.values())
    rows = fisher_weighted_sensitivity(base, experts, fisher)
    for metric in ("parameter_count", "fisher_mass", "base_to_expert"):
        assert sum(rows[group]["a"][metric] for group in detail[:5]) == pytest.approx(
            rows["vision_tower"]["a"][metric])
        assert sum(rows[group]["a"][metric] for group in detail[6:]) == pytest.approx(
            rows["language_decoder"]["a"][metric])
    assert sum(rows[group]["a"]["base_to_expert_percent"] for group in detail) == pytest.approx(100)
