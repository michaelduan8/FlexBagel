import json

import pytest
import torch

from analysis import run_flex_fisher as cli
from analysis.test_run_expert_analysis import FakeProcessor


class TinyFlex(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"model_type": "flex_qwen2_5_vl_moe"})()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.language_model.layers[0].mlp = torch.nn.Module()
        self.model.language_model.layers[0].mlp.up_proj = torch.nn.Linear(1, 1, bias=False)

    def forward(self, input_ids, **kwargs):
        weight = self.model.language_model.layers[0].mlp.up_proj.weight.reshape(())
        logits = torch.stack((weight.expand_as(input_ids),
                              torch.zeros_like(input_ids, dtype=weight.dtype)), -1)
        return type("Output", (), {"logits": logits})()


def test_standalone_fisher_reports_mass_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_flex_model",
                        lambda path, trust_remote_code, device=None:
                        TinyFlex().eval().to(device or "cpu"))
    monkeypatch.setattr(cli, "load_processor", lambda *args: FakeProcessor())
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(json.dumps({
        "id": "0", "images": [],
        "conversation": [{"role": "user", "content": "Q", "img_loc": None},
                         {"role": "assistant", "content": "A"}],
    }) + "\n")
    output = tmp_path / "fisher.pt"
    command = ["--model", "flex", "--examples", f"task={dataset}",
               "--output", str(output), "--device", "cpu", "--max-examples", "1"]
    assert cli.main(command) == 0
    result = torch.load(output, weights_only=True)
    assert result["method"] == "standalone_fisher_mass"
    row = result["fisher_group_scores"]["task"]["language_ffn"]
    assert row["example_count"] == 1
    assert row["parameter_count"] == 1
    assert row["fisher_mass"] >= 0
    assert row["mean_fisher"] == pytest.approx(row["fisher_mass"])
    assert "base_to_expert" not in row and "expert_to_average" not in row
    assert result["fisher_group_scores"]["task"]["language_ffn"]["fisher_mass_percent"] == pytest.approx(100)
    monkeypatch.setattr(cli, "estimate_conditional_token_fisher",
                        lambda *args, **kwargs: pytest.fail("Completed shard recomputed"))
    assert cli.main(command) == 0
    assert torch.load(output, weights_only=True)["fisher_group_scores"] == result["fisher_group_scores"]


def test_flex_model_type_is_checked(monkeypatch):
    fake = TinyFlex()
    fake.config.model_type = "qwen2_5_vl"
    monkeypatch.setattr(cli, "load_model", lambda *args: fake)
    with pytest.raises(ValueError, match="expected flex_qwen2_5_vl_moe"):
        cli.load_flex_model("dense", False)


def test_two_expert_split_covers_ffns_and_reports_both_denominators():
    from analysis.expert_analysis import group_summary

    names = {
        "model.visual.blocks.0.mlp.experts.0.up_proj.weight": 1.0,
        "model.visual.blocks.0.mlp.experts.1.up_proj.weight": 3.0,
        "model.language_model.layers.0.mlp.experts.0.up_proj.weight": 2.0,
        "model.language_model.layers.0.mlp.experts.1.up_proj.weight": 4.0,
    }
    groups = cli.fisher_groups(True)
    summary = group_summary({name: torch.zeros(1) for name in names}, groups)
    cli.validate_expert_split(summary)
    scores = {"task": {
        group.name: {
            "parameter_count": summary[group.name]["parameter_count"],
            "fisher_mass": sum(value for name, value in names.items()
                               if group.matches(name)),
        }
        for group in groups
    }}
    cli.add_fisher_statistics(scores, summary, split_experts=True)
    rows = scores["task"]
    assert rows["vision_expert_0"]["fisher_mass_percent"] == pytest.approx(10)
    assert rows["vision_expert_0"]["within_expertized_percent"] == pytest.approx(10)
    assert rows["vision_expert_0"]["within_tower_ffn_percent"] == pytest.approx(25)
    assert rows["language_expert_1"]["fisher_mass_percent"] == pytest.approx(40)
    assert rows["language_expert_1"]["within_tower_ffn_percent"] == pytest.approx(100 * 4 / 6)
    assert sum(rows[group.name]["within_expertized_percent"]
               for group in cli.EXPERT_SPLIT_GROUPS) == pytest.approx(100)

    names["model.visual.blocks.0.mlp.experts.2.up_proj.weight"] = 5
    extra_summary = group_summary({name: torch.zeros(1) for name in names}, groups)
    with pytest.raises(ValueError, match="vision FFN parameters"):
        cli.validate_expert_split(extra_summary)
