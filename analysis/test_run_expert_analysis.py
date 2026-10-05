import json

import pytest
import torch
from PIL import Image

from analysis.run_expert_analysis import (
    average_parameter_names,
    encode_record,
    iter_jsonl,
    resolve_device,
)


class FakeProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        rendered = ""
        for message in messages:
            rendered += f"<{message['role']}>"
            for block in message["content"]:
                rendered += "<image>" if block["type"] == "image" else block["text"]
            rendered += "</>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, *, text, return_tensors, images=None):
        assert return_tensors == "pt"
        assert len(text) == 1
        assert images is None or all(image.mode == "RGB" for image in images)
        return {"input_ids": torch.tensor([[ord(c) for c in text[0]]]),
                "attention_mask": torch.ones((1, len(text[0])), dtype=torch.long)}


def test_all_averages_connector_and_non_ffn_preserves_it():
    names = [
        "model.visual.merger.fc1.weight",
        "model.visual.blocks.0.mlp.fc1.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate.weight",
    ]
    assert average_parameter_names(names, "all") == set(names)
    assert average_parameter_names(names, "non-ffn") == {
        "model.layers.0.self_attn.q_proj.weight"}


def test_jsonl_conversation_images_and_final_answer(tmp_path):
    image = tmp_path / "slide.png"
    Image.new("RGB", (2, 2), "red").save(image)
    record = {
        "id": "example",
        "images": [image.name],
        "conversation": [
            {"role": "user", "content": "Question", "img_loc": "after"},
            {"role": "assistant", "content": "Prior"},
            {"role": "user", "content": "Follow-up", "img_loc": None},
            {"role": "assistant", "content": "Final"},
        ],
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    rows = list(iter_jsonl(str(path), 1))
    assert len(rows) == 1
    encoded = encode_record(rows[0], FakeProcessor(), str(path))
    ids = encoded["input_ids"][0].tolist()
    text = "".join(map(chr, ids))
    assert "Question<image>" in text
    assert "Prior" in text
    label_text = "".join(chr(ids[i]) for i in
                         torch.nonzero(encoded["labels"][0] != -100, as_tuple=True)[0])
    assert label_text == "Final</>"
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="prefix tokens"):
        broken = FakeProcessor()
        broken.apply_chat_template = lambda *args, **kwargs: "unrelated"
        encode_record(record, broken, str(path))


def test_data_shards_cover_selected_examples_once(tmp_path):
    from analysis.run_expert_analysis import processed_examples

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(
        json.dumps({"id": str(i), "images": [],
                    "conversation": [
                        {"role": "user", "content": f"Q{i}", "img_loc": None},
                        {"role": "assistant", "content": str(i)},
                    ]}) + "\n"
        for i in range(6)
    ))
    from analysis.run_expert_analysis import count_jsonl_rows

    assert count_jsonl_rows(str(path), 5) == 5
    assert count_jsonl_rows(str(path), 0) == 6
    processor = FakeProcessor()
    counts = []
    answers = []
    for rank in range(2):
        counter = [0]
        shard = list(processed_examples(
            str(path), processor, torch.device("cpu"), 5,
            rank=rank, world_size=2, counter=counter))
        counts.append(counter[0])
        answers.extend("".join(
            chr(int(token)) for token in example["labels"][0]
            if int(token) != -100
        ) for example in shard)
    assert counts == [3, 2]
    assert sorted(answers) == [f"{i}</>" for i in range(5)]
    resumed = list(processed_examples(
        str(path), processor, torch.device("cpu"), 5,
        rank=0, world_size=2, skip_examples=2))
    assert len(resumed) == 1
    assert "".join(chr(int(token)) for token in resumed[0]["labels"][0]
                   if int(token) != -100) == "4</>"


def test_shard_scores_weighted_by_example_count():
    from analysis.run_expert_analysis import combine_shards

    def contribution(count, score):
        return {"a": {"count": count, "groups": {
            "all": {"parameter_count": 2, "fisher_mass": score,
                    "base_to_expert": score * 2,
                    "expert_to_average": score * 3}}}}
    rows = combine_shards([contribution(3, 2), contribution(1, 10)], ["a"])
    row = rows["a"]["all"]
    assert row["example_count"] == 4
    assert row["fisher_mass"] == 4
    assert row["base_to_expert"] == 8
    assert row["expert_to_average"] == 12
    with pytest.raises(ValueError, match="No examples"):
        combine_shards([{}], ["a"])


def test_multi_device_selection_requires_distinct_gpus(monkeypatch):
    from analysis import run_expert_analysis as cli

    monkeypatch.setattr(
        cli, "resolve_device",
        lambda name, *, set_current=True: torch.device(name),
    )
    assert [str(device) for device in cli.resolve_devices(
        "auto", "xpu:0,xpu:1")] == ["xpu:0", "xpu:1"]
    with pytest.raises(ValueError, match="distinct"):
        cli.resolve_devices("auto", "xpu:0,xpu:0")
    with pytest.raises(ValueError, match="only XPU or CUDA"):
        cli.resolve_devices("auto", "cpu:0,cpu:1")


def test_fisher_worker_uses_disjoint_jsonl_rows(tmp_path, monkeypatch):
    from argparse import Namespace
    from analysis import run_expert_analysis as cli

    class TinyCheckpoint(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].mlp = torch.nn.Module()
            self.model.layers[0].mlp.up_proj = torch.nn.Linear(1, 1, bias=False)
            self.model.layers[0].mlp.up_proj.weight.data.fill_(weight)

        def forward(self, input_ids, **kwargs):
            weight = self.model.layers[0].mlp.up_proj.weight.reshape(())
            logits = torch.stack(
                (weight.expand_as(input_ids),
                 torch.zeros_like(input_ids, dtype=weight.dtype)), dim=-1)
            return type("Output", (), {"logits": logits})()

    monkeypatch.setattr(cli, "load_model",
                        lambda path, trust_remote_code, device=None:
                        TinyCheckpoint({"base": 0.4, "expert": 0.5}[path]).eval().to(device or "cpu"))
    monkeypatch.setattr(cli, "load_processor", lambda *args: FakeProcessor())
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(
        json.dumps({"id": str(i), "images": [],
                    "conversation": [
                        {"role": "user", "content": "Q", "img_loc": None},
                        {"role": "assistant", "content": "A"},
                    ]}) + "\n" for i in range(3)
    ))
    args = Namespace(base="base", trust_remote_code=False,
                     average_mode="non-ffn", processor=None, seed=3,
                     max_examples=3)
    class ProgressEvents:
        def __init__(self):
            self.events = []

        def put(self, name):
            self.events.append(name)

    progress = ProgressEvents()
    shards = [
        cli.fisher_shard(rank, ["cpu", "cpu"], args,
                         {"a": "expert"}, {"a": str(path)},
                         {"a": 1.0}, 0.0, progress)
        for rank in range(2)
    ]
    assert progress.events == ["a"] * 3
    assert [shard["a"]["count"] for shard in shards] == [2, 1]
    combined = cli.combine_shards(shards, ["a"])
    assert combined["a"]["architectural_ffn"]["example_count"] == 3


def test_auto_selects_all_visible_gpus(monkeypatch):
    from analysis import run_expert_analysis as cli

    monkeypatch.setattr(cli.torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.xpu, "device_count", lambda: 3)
    assert [str(device) for device in cli.resolve_devices("auto", None)] == [
        "xpu:0", "xpu:1", "xpu:2"]

    monkeypatch.setattr(cli.torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 2)
    assert [str(device) for device in cli.resolve_devices("auto", None)] == [
        "cuda:0", "cuda:1"]

    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No GPU found"):
        cli.resolve_devices("auto", None)


def test_checkpoint_methods_write_independent_results(tmp_path, monkeypatch):
    from analysis import run_expert_analysis as cli

    base = {"model.language_model.layers.0.mlp.up_proj.weight": torch.tensor([1.])}
    experts = {"a": {key: value + 1 for key, value in base.items()}}
    monkeypatch.setattr(cli, "load_states", lambda *args: (
        base, experts, list(base)))
    monkeypatch.setattr(cli, "resolve_devices", lambda *args: (
        pytest.fail("Checkpoint methods must not resolve GPUs")))
    common = ["--base", "base", "--expert", "a=expert"]
    vector_path = tmp_path / "vectors.pt"
    sign_path = tmp_path / "signs.pt"
    assert cli.main([*common, "--method", "task-vectors",
                     "--output", str(vector_path)]) == 0
    assert cli.main([*common, "--method", "sign-conflicts",
                     "--output", str(sign_path)]) == 0
    vectors = torch.load(vector_path, weights_only=True)
    signs = torch.load(sign_path, weights_only=True)
    assert "task_vectors" in vectors and "sign_conflicts" not in vectors
    assert "sign_conflicts" in signs and "task_vectors" not in signs


def test_shard_checkpoint_restores_count_and_sampler(tmp_path):
    from analysis.run_expert_analysis import (
        load_shard_checkpoint, save_shard_checkpoint, shard_example_count,
    )

    assert [shard_example_count(5, rank, 2) for rank in range(2)] == [3, 2]
    path = tmp_path / "shard.pt"
    generator = torch.Generator().manual_seed(7)
    torch.rand(1, generator=generator)
    expected_next = torch.rand(1, generator=generator)
    generator.manual_seed(7)
    torch.rand(1, generator=generator)
    save_shard_checkpoint(path, 1, {"all": {"parameter_count": 1,
                                           "fisher_mass": 2.0}}, generator)
    restored = torch.Generator().manual_seed(99)
    count, scores = load_shard_checkpoint(path, 3, restored)
    assert count == 1 and scores["all"]["fisher_mass"] == 2.0
    assert torch.equal(torch.rand(1, generator=restored), expected_next)
    with pytest.raises(ValueError, match="Invalid saved example count"):
        load_shard_checkpoint(path, 0, restored)


def test_fisher_main_reuses_completed_checkpoint(tmp_path, monkeypatch):
    from analysis import run_expert_analysis as cli

    class TinyCheckpoint(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.language_model.layers[0].mlp = torch.nn.Module()
            self.model.language_model.layers[0].mlp.up_proj = torch.nn.Linear(
                1, 1, bias=False)
            self.model.language_model.layers[0].mlp.up_proj.weight.data.fill_(weight)

        def forward(self, input_ids, **kwargs):
            weight = self.model.language_model.layers[0].mlp.up_proj.weight.reshape(())
            logits = torch.stack((weight.expand_as(input_ids),
                                  torch.zeros_like(input_ids, dtype=weight.dtype)), -1)
            return type("Output", (), {"logits": logits})()

    monkeypatch.setattr(cli, "load_model", lambda path, trust_remote_code,
                        device=None: TinyCheckpoint({"base": 0.4,
                                                      "expert": 0.5}[path]).eval().to(device or "cpu"))
    monkeypatch.setattr(cli, "load_processor", lambda *args: FakeProcessor())
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "0", "images": [], "conversation": [
        {"role": "user", "content": "Q", "img_loc": None},
        {"role": "assistant", "content": "A"}]}) + "\n")
    output = tmp_path / "fisher.pt"
    command = ["--method", "fisher", "--base", "base", "--expert", "a=expert",
               "--examples", f"a={rows}", "--output", str(output),
               "--device", "cpu", "--max-examples", "1"]
    assert cli.main(command) == 0
    first = torch.load(output, weights_only=True)
    checkpoint = tmp_path / "fisher.pt.resume" / "a.rank0.pt"
    assert torch.load(checkpoint, weights_only=True)["count"] == 1
    monkeypatch.setattr(cli, "estimate_conditional_token_fisher",
                        lambda *args, **kwargs: pytest.fail("Completed shard recomputed"))
    assert cli.main(command) == 0
    second = torch.load(output, weights_only=True)
    assert second["fisher_group_scores"] == first["fisher_group_scores"]



def test_fisher_selected_expert_uses_all_weights_but_only_one_dataset(tmp_path, monkeypatch):
    from analysis import run_expert_analysis as cli

    class TinyCheckpoint(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].mlp = torch.nn.Module()
            self.model.layers[0].mlp.up_proj = torch.nn.Linear(1, 1, bias=False)
            self.model.layers[0].mlp.up_proj.weight.data.fill_(weight)

        def forward(self, input_ids, **kwargs):
            weight = self.model.layers[0].mlp.up_proj.weight.reshape(())
            logits = torch.stack((weight.expand_as(input_ids),
                                  torch.zeros_like(input_ids, dtype=weight.dtype)), -1)
            return type("Output", (), {"logits": logits})()

    weights = {"base": 0.4, "a": 0.5, "b": 0.6, "c": 0.7}
    loaded = []
    def load(path, trust_remote_code, device=None):
        loaded.append(path)
        return TinyCheckpoint(weights[path]).eval().to(device or "cpu")

    monkeypatch.setattr(cli, "load_model", load)
    monkeypatch.setattr(cli, "load_processor", lambda *args: FakeProcessor())
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "0", "images": [], "conversation": [
        {"role": "user", "content": "Q", "img_loc": None},
        {"role": "assistant", "content": "A"}]}) + "\n")
    output = tmp_path / "fisher-b.pt"
    command = ["--method", "fisher", "--base", "base",
               "--expert", "a=a", "--expert", "b=b", "--expert", "c=c",
               "--examples", f"b={rows}", "--fisher-expert", "b",
               "--output", str(output), "--device", "cpu", "--max-examples", "1"]
    assert cli.main(command) == 0
    result = torch.load(output, weights_only=True)
    assert set(result["fisher_group_scores"]) == {"b"}
    assert result["fisher_group_scores"]["b"]["architectural_ffn"]["example_count"] == 1
    assert set(loaded) == set(weights)
    assert loaded.count("b") == 2
    assert loaded.count("a") == loaded.count("c") == 1
    with pytest.raises(ValueError, match="matching --examples"):
        cli.main(["--method", "fisher", "--base", "base", "--expert", "b=b",
                  "--fisher-expert", "b", "--output", str(output), "--device", "cpu"])


def test_eval_activation_checkpointing_preserves_fisher():
    import copy
    from analysis.run_expert_analysis import enable_activation_checkpointing
    from analysis.expert_analysis import ParameterGroup, estimate_conditional_token_fisher

    class EvalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([
                torch.nn.Linear(2, 2, bias=False),
                torch.nn.Linear(2, 2, bias=False),
            ])

        def forward(self, input_ids):
            hidden = torch.nn.functional.one_hot(input_ids, 2).float()
            for layer in self.model.layers:
                hidden = torch.tanh(layer(hidden))
            return type('Output', (), {'logits': hidden})()

    torch.manual_seed(11)
    plain = EvalLM().eval()
    recomputed = copy.deepcopy(plain)
    enable_activation_checkpointing(recomputed)
    assert not recomputed.training
    assert all(not module.training for module in recomputed.modules())
    example = {'input_ids': torch.tensor([[0, 1, 0]]),
               'labels': torch.tensor([[-100, 1, 0]])}
    groups = (ParameterGroup('all', r'^'),)
    expected = estimate_conditional_token_fisher(
        plain, [example], groups, torch.Generator().manual_seed(9))
    actual = estimate_conditional_token_fisher(
        recomputed, [example], groups, torch.Generator().manual_seed(9))
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
