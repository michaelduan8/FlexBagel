"""Tests for train/multiturn_mask.py (mm_tune --supervise_all_assistant_turns).

Run with pytest, or without it:  cd FlexBagel && PYTHONPATH=. python -m train.test_multiturn_mask

Uses the real Qwen2.5-VL processor and tiny generated images/frames, so the chat template, image/video
placeholder expansion and both collators (TRL's image collator, VideoAwareVLMCollator) are exercised end to end.
"""
import functools
import json
import pathlib
import tempfile

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

from train.mm_tune import preprocess_dataset
from train.multiturn_mask import AllAssistantTurnsCollator, assistant_span_mask
from train.video_collator import VideoAwareVLMCollator

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
ANSWERS = ["The report: no nodules.", "Yes.", "left lung upper lobe (lung)", "6x4 mm"]


KINDS = ["image", "video"]


@functools.lru_cache(maxsize=1)
def get_processor():
    return AutoProcessor.from_pretrained(MODEL)


def make_rows(kind, n_turns):
    tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="mt_mask_"))
    paths = []
    for i in range(4):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (56, 56), (40 * i, 80, 120)).save(p)
        paths.append(str(p))
    conversation = []
    for t in range(n_turns):
        conversation += [
            {"role": "user", "content": f"Question {t}?", "img_loc": "before" if t == 0 else None},
            {"role": "assistant", "content": ANSWERS[t], "img_loc": None},
        ]
    row = {"id": f"r{kind}{n_turns}", "conversation": conversation}
    row.update({"videos": [paths], "video_fps": [1.0]} if kind == "video" else {"images": paths[:2]})
    f = tmp_path / f"{kind}{n_turns}.jsonl"
    f.write_text(json.dumps(row) + "\n")
    ds = load_dataset("json", data_files=str(f), split="train")
    return preprocess_dataset(ds, with_videos=(kind == "video"))


def make_collator(kind, processor):
    cls = VideoAwareVLMCollator if kind == "video" else DataCollatorForVisionLanguageModeling
    kwargs = {"default_video_fps": 1.0} if kind == "video" else {}
    return cls(processor=processor, completion_only_loss=True, dataset_text_field="text", **kwargs)


def supervised_spans(labels):
    spans, cur = [], []
    for tok in labels.tolist():
        if tok != -100:
            cur.append(tok)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans


def test_every_assistant_turn_supervised_exactly():
    for kind in KINDS:
        check_every_turn(kind)


def check_every_turn(kind):
    processor = get_processor()
    ds = make_rows(kind, 4)
    inner = make_collator(kind, processor)
    batch = AllAssistantTurnsCollator(inner, processor.tokenizer)([ds[0]])
    spans = [processor.tokenizer.decode(s) for s in supervised_spans(batch["labels"][0])]
    assert spans == [a + "<|im_end|>\n" for a in ANSWERS]
    # no vision placeholder, role header or user text is ever a label
    tok = processor.tokenizer
    vision_ids = {tok.convert_tokens_to_ids(t) for t in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")}
    sup = set(batch["labels"][0][batch["labels"][0] != -100].tolist())
    assert not sup & vision_ids
    assert tok.convert_tokens_to_ids("<|im_start|>") not in sup
    # the vision branch still got its inputs
    key = "pixel_values_videos" if kind == "video" else "pixel_values"
    assert key in batch and batch[key].numel() > 0


def test_last_turn_matches_default_collator():
    """The default (flag off) path supervises only the last turn: the wrapper's last span must equal it."""
    for kind in KINDS:
        check_last_turn(kind)


def check_last_turn(kind):
    processor = get_processor()
    ds = make_rows(kind, 4)
    inner = make_collator(kind, processor)
    plain = inner([ds[0]])
    assert [processor.tokenizer.decode(s) for s in supervised_spans(plain["labels"][0])] == [ANSWERS[-1] + "<|im_end|>\n"]
    wrapped = AllAssistantTurnsCollator(inner, processor.tokenizer)([ds[0]])
    assert torch.equal(plain["input_ids"], wrapped["input_ids"])
    last = plain["labels"][0] != -100
    assert torch.equal(wrapped["labels"][0][last], plain["labels"][0][last])


def test_single_turn_labels_identical():
    processor = get_processor()
    for kind in KINDS:
        ds = make_rows(kind, 1)
        inner = make_collator(kind, processor)
        plain = inner([ds[0]])
        wrapped = AllAssistantTurnsCollator(inner, processor.tokenizer)([ds[0]])
        assert torch.equal(plain["input_ids"], wrapped["input_ids"])
        assert torch.equal(plain["labels"], wrapped["labels"])


def test_padding_masked_in_batch():
    """Rows of different lengths: padding is never supervised and each row keeps its own turns."""
    processor = get_processor()
    long_ds, short_ds = make_rows("image", 4), make_rows("image", 2)
    batch = AllAssistantTurnsCollator(make_collator("image", processor), processor.tokenizer)([long_ds[0], short_ds[0]])
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    spans = [processor.tokenizer.decode(s) for s in supervised_spans(batch["labels"][1])]
    assert spans == [a + "<|im_end|>\n" for a in ANSWERS[:2]]


def test_span_mask_truncated_turn():
    """A turn cut off by max_length (no end token) is supervised up to the cut."""
    ids = torch.tensor([[1, 9, 2, 5, 6, 3, 2, 1, 9, 2, 7, 8]])
    m = assistant_span_mask(ids, header_ids=[1, 9, 2], end_id=3, newline_id=2)
    assert m[0].tolist() == [False] * 3 + [True] * 4 + [False] * 3 + [True] * 2


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
