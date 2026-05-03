"""End-to-end test on CIFAR-10 baseline checkpoint.

Loads the trained baseline, runs UNSIR forgetting class 0 (airplane),
checks that ADf collapses and ADr is preserved.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from redact.data import cifar10, retain_subset, loader
from redact.evaluate import per_class_accuracy, adf_adr
from redact.model import build_resnet18_cifar, device
from redact.unsir import UnsirConfig, run_unsir


def main():
    dev = device()
    print(f"Device: {dev}")
    ckpt = torch.load("checkpoints/baseline.pt", map_location=dev)
    model = build_resnet18_cifar(10).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded baseline (test_acc={ckpt['test_acc']*100:.2f}%, "
          f"epochs={ckpt['epochs_trained']})")

    test_set = cifar10(train=False, augment=False)
    train_set = cifar10(train=True, augment=False)
    test_dl = loader(test_set, batch_size=256, shuffle=False)

    base_accs = per_class_accuracy(model, test_dl, 10, dev)
    print("baseline per-class:", [f"{a*100:.1f}" for a in base_accs])

    forget = [0]  # airplane
    retain_sub = retain_subset(train_set, forget, per_class=500)
    cfg = UnsirConfig(forget_classes=forget, num_classes=10,
                      noise_steps=40, noise_batch_size=256,
                      impair_lr=0.02, repair_lr=0.01)

    t0 = time.perf_counter()
    timings = run_unsir(model, retain_sub, cfg)
    after_accs = per_class_accuracy(model, test_dl, 10, dev)
    adf, adr = adf_adr(after_accs, forget)
    print(f"after UNSIR per-class: {[f'{a*100:.1f}' for a in after_accs]}")
    print(f"ADf={adf*100:.2f}%  ADr={adr*100:.2f}%  total UNSIR={timings['total']:.2f}s")
    print(f"timings: noise={timings['noise']:.2f}s impair={timings['impair']:.2f}s repair={timings['repair']:.2f}s")

    assert adf < 0.10, f"forget accuracy still {adf*100:.1f}%"
    assert adr > 0.65, f"retain mean {adr*100:.1f}% too low"
    print("OK")


if __name__ == "__main__":
    main()
