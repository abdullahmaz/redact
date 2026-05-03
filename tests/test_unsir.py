"""End-to-end smoke test for UNSIR.

Trains a tiny model on a tiny synthetic dataset, runs UNSIR, and asserts
that forget-class accuracy collapses while retain accuracy is preserved.
This catches obvious regressions in the noise / impair / repair phases
without needing a full CIFAR-10 training run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

from redact.unsir import UnsirConfig, run_unsir


def make_toy_data(n_per_class: int = 400, n_classes: int = 4, dim: int = 8, seed: int = 0):
    """Each class is a Gaussian blob centred at one corner of a hypercube."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, dim) * 4.0
    xs, ys = [], []
    for c in range(n_classes):
        x = centers[c] + 0.5 * torch.randn(n_per_class, dim, generator=g)
        xs.append(x)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(xs)
    Y = torch.cat(ys)
    perm = torch.randperm(X.size(0), generator=g)
    return X[perm], Y[perm]


class ToyMLP(nn.Module):
    def __init__(self, dim: int = 8, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # Accept either (B, dim) or (B, C, H, W) for compatibility with UNSIR's
        # noise tensors. We just flatten.
        return self.net(x.view(x.size(0), -1))


def train_baseline(model, X, Y, epochs: int = 30, lr: float = 0.05):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    dl = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)
    for _ in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()


def acc_per_class(model, X, Y, n_classes: int) -> list[float]:
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1)
    out = []
    for c in range(n_classes):
        m = Y == c
        if m.sum() == 0:
            out.append(0.0)
        else:
            out.append((pred[m] == c).float().mean().item())
    return out


def test_unsir_forgets_target_class():
    torch.manual_seed(42)
    n_classes, dim = 4, 8
    X, Y = make_toy_data(n_per_class=400, n_classes=n_classes, dim=dim)
    Xtest, Ytest = make_toy_data(n_per_class=100, n_classes=n_classes, dim=dim, seed=1)

    model = ToyMLP(dim=dim, n_classes=n_classes)
    train_baseline(model, X, Y, epochs=25)
    base = acc_per_class(model, Xtest, Ytest, n_classes)
    print("baseline per-class acc:", [f"{a:.2f}" for a in base])
    assert all(a > 0.85 for a in base), f"baseline too weak: {base}"

    forget = [1]
    retain_mask = ~torch.isin(Y, torch.tensor(forget))
    Xr, Yr = X[retain_mask], Y[retain_mask]

    class _IntLabel(torch.utils.data.Dataset):
        def __init__(self, X, Y): self.X, self.Y = X, Y
        def __len__(self): return self.X.size(0)
        def __getitem__(self, i): return self.X[i], int(self.Y[i].item())

    retain_sub = Subset(_IntLabel(Xr, Yr), list(range(min(200, Xr.size(0)))))

    cfg = UnsirConfig(
        forget_classes=forget,
        num_classes=n_classes,
        noise_steps=80, noise_lr=0.1, noise_lambda=0.1, noise_batch_size=128,
        impair_epochs=2, impair_lr=0.05,
        repair_epochs=1, repair_lr=0.01,
        input_shape=(dim,),
        device="cpu",
    )
    timings = run_unsir(model, retain_sub, cfg)
    after = acc_per_class(model, Xtest, Ytest, n_classes)
    print("after UNSIR per-class acc:", [f"{a:.2f}" for a in after])
    print("timings:", timings)

    forget_acc = after[forget[0]]
    retain_accs = [a for c, a in enumerate(after) if c not in forget]
    assert forget_acc < 0.20, f"forget class still at {forget_acc:.2f}"
    assert min(retain_accs) > 0.70, f"retain accs collapsed: {retain_accs}"


if __name__ == "__main__":
    test_unsir_forgets_target_class()
    print("OK — UNSIR smoke test passed.")
