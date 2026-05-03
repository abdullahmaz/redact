"""Train the baseline ResNet-18 on CIFAR-10.

This is a minimal, GPU-friendly training loop. The paper targets ~78%
test accuracy at 40 epochs. For the demo we accept anywhere ~70-78% so
the unlearning effect is clearly visible without lengthy training.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import cifar10, loader
from .evaluate import per_class_accuracy
from .model import build_resnet18_cifar, device


def train(epochs: int = 12, lr: float = 0.05, batch_size: int = 128,
          out_path: str = "checkpoints/baseline.pt") -> None:
    dev = device()
    print(f"Device: {dev}")

    train_set = cifar10(train=True)
    test_set = cifar10(train=False, augment=False)
    train_dl = loader(train_set, batch_size=batch_size, shuffle=True)
    test_dl = loader(test_set, batch_size=256, shuffle=False)

    model = build_resnet18_cifar(num_classes=10).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        n = 0
        correct = 0
        for x, y in train_dl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
            correct += (out.argmax(1) == y).sum().item()
        sched.step()
        train_loss = running / n
        train_acc = correct / n

        accs = per_class_accuracy(model, test_dl, num_classes=10, device=dev)
        test_acc = sum(accs) / len(accs)
        dt = time.perf_counter() - t0
        print(f"epoch {ep+1:02d}/{epochs}  loss={train_loss:.3f}  "
              f"train_acc={train_acc*100:.2f}%  test_acc={test_acc*100:.2f}%  "
              f"{dt:.1f}s")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"state_dict": model.state_dict(), "test_acc": test_acc,
                        "epochs_trained": ep + 1}, out_path)
            print(f"  -> saved baseline (test_acc={test_acc*100:.2f}%) -> {out_path}")

    print(f"Done. Best test accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--out", type=str, default="checkpoints/baseline.pt")
    args = p.parse_args()
    train(epochs=args.epochs, lr=args.lr, out_path=args.out)
