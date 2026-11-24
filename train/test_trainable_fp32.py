"""Tests for mm_tune --trainable_params_fp32 (float32 master weights for trainable params only).

Run with pytest, or without it:  cd FlexBagel && PYTHONPATH=. python -m train.test_trainable_fp32
"""
import torch

from train.mm_tune import cast_floating_params, upcast_trainable_params

STEPS, LR = 50, 2e-5


def make_model():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64))
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.02 + 0.02 * torch.sign(torch.randn_like(p)))  # |w| ~ 0.02, like the experts
    model.to(torch.bfloat16)
    for p in model[0].parameters():  # layer 0 frozen, layer 1 trainable
        p.requires_grad_(False)
    return model


def train(model):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    torch.manual_seed(1)
    for _ in range(STEPS):
        x = torch.randn(32, 64, dtype=torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(x).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()


def changed_fraction(before, after):
    return (before.float() != after.float()).float().mean().item()


def test_upcast_only_trainable():
    model = make_model()
    assert upcast_trainable_params(model) == 2  # layer-1 weight + bias
    assert all(p.dtype == torch.bfloat16 for p in model[0].parameters())
    assert all(p.dtype == torch.float32 for p in model[1].parameters())
    assert cast_floating_params(model, torch.bfloat16) == 2
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


def test_bf16_updates_round_away_but_fp32_master_keeps_them():
    pure = make_model()
    start = pure[1].weight.detach().clone()
    frozen_start = pure[0].weight.detach().clone()
    train(pure)
    frac_pure = changed_fraction(start, pure[1].weight)

    master = make_model()
    upcast_trainable_params(master)
    train(master)
    cast_floating_params(master, torch.bfloat16)  # what mm_tune does before saving
    frac_master = changed_fraction(start, master[1].weight)

    print(f"pure bf16: {frac_pure:.1%} of trainable weights changed; fp32 master (saved as bf16): {frac_master:.1%}")
    assert frac_pure < 0.5, frac_pure  # most bf16 updates of ~LR on |w|~0.02 round to zero
    assert frac_master > 0.9, frac_master  # accumulated fp32 updates survive the final bf16 cast
    assert torch.equal(master[0].weight, frozen_start)  # frozen weights untouched


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
