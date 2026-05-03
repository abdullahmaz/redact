"""Evaluation helpers: per-class accuracy, ADf / ADr, sample predictions."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def per_class_accuracy(model: nn.Module, loader: DataLoader, num_classes: int, device: torch.device) -> list[float]:
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        for c in range(num_classes):
            mask = y == c
            total[c] += mask.sum().item()
            correct[c] += (pred[mask] == c).sum().item()
    accs = []
    for c in range(num_classes):
        accs.append(float(correct[c]) / float(total[c]) if total[c] > 0 else 0.0)
    return accs


def adf_adr(per_class: list[float], forget_classes: Iterable[int]) -> tuple[float, float]:
    """Mean accuracy on forget classes (ADf) and on retain classes (ADr)."""
    forget = set(int(c) for c in forget_classes)
    f_accs = [a for c, a in enumerate(per_class) if c in forget]
    r_accs = [a for c, a in enumerate(per_class) if c not in forget]
    adf = sum(f_accs) / len(f_accs) if f_accs else 0.0
    adr = sum(r_accs) / len(r_accs) if r_accs else 0.0
    return adf, adr


@torch.no_grad()
def sample_predictions(model: nn.Module, dataset, indices: list[int], device: torch.device) -> list[dict]:
    """Return prediction info for a list of dataset indices."""
    model.eval()
    out = []
    for i in indices:
        x, y = dataset[i]
        logits = model(x.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        out.append({
            "index": int(i),
            "true": int(y),
            "pred": pred,
            "confidence": float(probs[pred]),
            "true_class_prob": float(probs[int(y)]),
        })
    return out
