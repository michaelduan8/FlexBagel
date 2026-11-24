"""
Supervise every assistant turn of a multi-turn conversation (mm_tune --supervise_all_assistant_turns).

mm_tune's preprocessing turns a conversation into prompt = all turns but the last (earlier assistant answers
included) and completion = the last assistant turn, and the collators supervise only the completion. This
wrapper runs the inner collator unchanged and then re-derives the labels from input_ids: a token is supervised
iff it lies inside an assistant turn's content, i.e. after a `<|im_start|>assistant\\n` header up to and
including the turn's `<|im_end|>` and the newline that follows it. For the last turn this is exactly the
completion the default collators supervise ("answer<|im_end|>\\n"); earlier assistant turns get the same
treatment. Everything else stays -100: system/user turns, image and video tokens, the role headers and padding.

Rows with a single assistant turn get identical labels with or without the wrapper.
"""
from __future__ import annotations

import torch


def assistant_span_mask(input_ids: torch.Tensor, header_ids: list[int], end_id: int, newline_id: int | None) -> torch.Tensor:
    """Bool mask [B, L]: True for tokens inside assistant turns (content + end token + following newline)."""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    h = len(header_ids)
    for b, row in enumerate(input_ids.tolist()):
        i, n = 0, len(row)
        while i <= n - h:
            if row[i:i + h] != header_ids:
                i += 1
                continue
            j = i + h
            while j < n and row[j] != end_id:
                j += 1
            stop = j + 1  # include <|im_end|>
            if newline_id is not None and stop < n and row[stop] == newline_id:
                stop += 1  # and the newline the chat template puts after it
            mask[b, i + h:min(stop, n)] = True
            i = stop
    return mask


class AllAssistantTurnsCollator:
    """Wraps a VLM collator so that labels cover every assistant turn, not only the last one."""

    def __init__(self, inner, tokenizer, header: str = "<|im_start|>assistant\n", end: str = "<|im_end|>"):
        self.inner = inner
        self.header_ids = tokenizer.encode(header, add_special_tokens=False)
        end_ids = tokenizer.encode(end, add_special_tokens=False)
        assert len(end_ids) == 1, f"end marker {end!r} must be a single token, got {end_ids}"
        self.end_id = end_ids[0]
        newline = tokenizer.encode("\n", add_special_tokens=False)
        self.newline_id = newline[0] if len(newline) == 1 else None

    def __call__(self, examples):
        batch = self.inner(examples)
        mask = assistant_span_mask(batch["input_ids"], self.header_ids, self.end_id, self.newline_id)
        mask &= batch["attention_mask"].bool()
        labels = batch["input_ids"].clone()
        labels[~mask] = -100
        batch["labels"] = labels
        return batch
