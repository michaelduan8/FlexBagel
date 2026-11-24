"""Tests for the Trainer loss normalization of Flex_Qwen2_5_VLMoeForConditionalGeneration.

The forward takes **kwargs, so the HF Trainer passes num_items_in_batch (label tokens of the whole accumulated,
cross-device batch) and does not divide the loss by gradient_accumulation_steps itself. The loss must therefore be
sum(token losses) / num_items_in_batch, and the router aux losses must be weighted by the micro-batch's token share.

Run with pytest, or without it:  PYTHONPATH=. python -m modeling.flex_qwen2_5_vl_moe.test_loss_normalization
"""
import tempfile

import torch
import torch.nn.functional as F

from modeling.flex_qwen2_5_vl_moe import Flex_Qwen2_5_VLMoeConfig, Flex_Qwen2_5_VLMoeForConditionalGeneration

HIDDEN, VOCAB = 16, 64


def tiny_model(aux_coef=0.0, seed=0):
    torch.manual_seed(seed)
    cfg = Flex_Qwen2_5_VLMoeConfig(
        text_config=dict(hidden_size=HIDDEN, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, vocab_size=VOCAB, num_experts=3, num_experts_per_tok=2,
                         output_router_logits=aux_coef > 0, router_aux_loss_coef=aux_coef,
                         rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=HIDDEN,
                           num_experts=3, num_experts_per_tok=2),
    )
    return Flex_Qwen2_5_VLMoeForConditionalGeneration(cfg).eval()


def micro_batches():
    """Two micro-batches with different numbers of supervised tokens (prompt tokens masked with -100)."""
    g = torch.Generator().manual_seed(1)
    out = []
    for length, masked in ((10, 2), (7, 5)):
        ids = torch.randint(0, VOCAB, (1, length), generator=g)
        labels = ids.clone()
        labels[:, :masked] = -100
        out.append((ids, labels))
    return out


def count(labels):
    return int((labels[..., 1:] != -100).sum())


def token_ce_sum(model, ids, labels):
    logits = model(input_ids=ids).logits.float()
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")


def test_without_num_items_loss_is_plain_mean():
    model = tiny_model()
    ids, labels = micro_batches()[0]
    with torch.no_grad():
        loss = model(input_ids=ids, labels=labels).loss
        ref = token_ce_sum(model, ids, labels) / count(labels)
    assert torch.allclose(loss, ref, atol=1e-5), (loss, ref)


def test_accumulated_micro_batches_sum_to_global_token_mean():
    model = tiny_model()
    batches = micro_batches()
    total = sum(count(lab) for _, lab in batches)
    with torch.no_grad():
        summed = sum(model(input_ids=i, labels=l, num_items_in_batch=torch.tensor(total)).loss for i, l in batches)
        ref = sum(token_ce_sum(model, i, l) for i, l in batches) / total
    assert torch.allclose(summed, ref, atol=1e-5), (summed, ref)
    # the old behaviour (per-micro-batch means, not divided by the accumulation steps) is ~2x this here
    with torch.no_grad():
        old = sum(model(input_ids=i, labels=l).loss for i, l in batches)
    assert old > 1.5 * ref


def test_aux_loss_is_token_weighted_mean_over_accumulated_batch():
    coef = 0.5
    model = tiny_model(aux_coef=coef)
    batches = micro_batches()
    total = sum(count(lab) for _, lab in batches)
    with torch.no_grad():
        summed = sum(model(input_ids=i, labels=l, num_items_in_batch=total).loss for i, l in batches)
        ce = sum(token_ce_sum(model, i, l) for i, l in batches) / total
        aux = [model(input_ids=i, labels=l).aux_loss for i, l in batches]
        aux_mean = sum(a * count(l) for a, (_, l) in zip(aux, batches)) / total
    assert torch.allclose(summed, ce + coef * aux_mean, atol=1e-5), (summed, ce + coef * aux_mean)


def test_hf_trainer_logs_true_mean_loss_with_gradient_accumulation():
    """End to end: Trainer with gradient_accumulation_steps=2 logs the token-mean loss of the accumulated batch."""
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, TrainerCallback

    model = tiny_model()
    batches = micro_batches()
    total = sum(count(lab) for _, lab in batches)
    with torch.no_grad():
        ref = float(sum(token_ce_sum(model, i, l) for i, l in batches) / total)

    ds = Dataset.from_list([{"input_ids": i[0].tolist(), "labels": l[0].tolist()} for i, l in batches])
    collate = lambda feats: {k: torch.tensor([f[k] for f in feats]) for k in ("input_ids", "labels")}
    logged = []

    class Grab(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                logged.append(logs["loss"])

    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(output_dir=tmp, per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                 max_steps=1, logging_steps=1, learning_rate=0.0, report_to=[], use_cpu=True,
                                 save_strategy="no", dataloader_drop_last=False, seed=0)
        trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate, callbacks=[Grab()])
        trainer.train_dataloader = None
        trainer.train()
    assert trainer.model_accepts_loss_kwargs
    assert logged and abs(logged[0] - ref) < 1e-3 * max(1.0, ref), (logged, ref)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
