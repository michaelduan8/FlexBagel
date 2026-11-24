"""
Video-aware VLM collator for prompt-completion SFT data.

TRL's DataCollatorForVisionLanguageModeling only forwards `images` to the processor. This subclass also
handles rows with a `videos` column (one list of frame paths per video) and a matching `video_fps` column:
frames are loaded as RGB PIL images and passed to the processor's video branch together with the per-video
fps, so Qwen2.5-VL gets `pixel_values_videos`, `video_grid_thw` and `second_per_grid_ts`.

Batches without any video are delegated to the parent collator unchanged, so image-only runs behave exactly
as with TRL's default collator. The video path mirrors TRL 0.22.2's `_collate_prompt_completion`.
"""

from PIL import Image as PILImage
import torch
from trl.data_utils import apply_chat_template, is_conversational
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
from trl.trainer.utils import flush_left


def load_frames(frame_paths):
    frames = []
    for path in frame_paths:
        with PILImage.open(path) as img:
            frames.append(img.convert("RGB"))
    return frames


class VideoAwareVLMCollator(DataCollatorForVisionLanguageModeling):
    def __init__(self, *args, default_video_fps: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_video_fps = default_video_fps

    def _collate_prompt_completion(self, examples):
        if not any(example.get("videos") for example in examples):
            return super()._collate_prompt_completion(examples)

        if self.pad_to_multiple_of is not None:
            raise NotImplementedError("pad_to_multiple_of is not supported for prompt-completion VLM data.")

        # Flat lists over the batch, in prompt order (the processor consumes placeholders sequentially)
        images = [img for example in examples for img in (example.get("images") or [])]
        videos, fps = [], []
        for example in examples:
            example_videos = example.get("videos") or []
            example_fps = example.get("video_fps") or [self.default_video_fps] * len(example_videos)
            assert len(example_fps) == len(example_videos), "video_fps must have one value per video"
            videos.extend(load_frames(frames) for frames in example_videos)
            fps.extend(self.default_video_fps if f is None else float(f) for f in example_fps)

        # mm_tune rows already carry typed content with image/video placeholders, so no
        # prepare_multimodal_messages step is needed (it would only prepend image placeholders).
        if is_conversational(examples[0]):
            examples = [apply_chat_template(example, self.processor) for example in examples]

        prompts = [example["prompt"] for example in examples]
        completions = [example["completion"] for example in examples]

        processed_prompts = self.processor(
            images=images or None,
            videos=videos,
            fps=fps,
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )
        processed_completions = self.processor(
            text=completions,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )

        prompt_ids, completion_ids = processed_prompts["input_ids"], processed_completions["input_ids"]
        prompt_mask, completion_mask = processed_prompts["attention_mask"], processed_completions["attention_mask"]
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)

        attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)

        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            completion_mask = completion_mask[:, : self.max_length]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if self.completion_only_loss:
            labels[completion_mask == 0] = -100

        output = processed_prompts  # holds pixel_values(_videos), *_grid_thw, second_per_grid_ts
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["labels"] = labels
        return output
