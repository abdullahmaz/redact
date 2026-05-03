"""UNSIR: Unlearning by Selective Impair and Repair.

Implements the three phases of:
  Tarun et al., "Fast Yet Effective Machine Unlearning",
  arXiv:2111.08947v5.

Phase 1: error-maximizing noise generation.
Phase 2: impair step — one epoch on retain_sub ∪ noise at high LR.
Phase 3: repair step — one epoch on retain_sub at low LR.

Crucially, the forget-set samples are *never* shown to the optimizer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset


@dataclass
class UnsirConfig:
    forget_classes: list[int] = field(default_factory=list)
    num_classes: int = 10
    noise_steps: int = 40
    noise_lr: float = 0.1
    noise_lambda: float = 0.1
    noise_batch_size: int = 256
    impair_epochs: int = 1
    impair_lr: float = 0.02
    repair_epochs: int = 1
    repair_lr: float = 0.01
    input_shape: tuple = (3, 32, 32)
    device: str = "cuda"


def _device(cfg: UnsirConfig) -> torch.device:
    return torch.device(cfg.device if torch.cuda.is_available() else "cpu")


# -------------------- Phase 1: noise --------------------

def generate_class_noise(
    model: nn.Module,
    target_class: int,
    cfg: UnsirConfig,
    on_step: Callable[[int, float], None] | None = None,
) -> torch.Tensor:
    """Optimize a noise tensor that the *frozen* model classifies confidently
    as `target_class`. The objective minimises  L(f(N), target) + λ ||N||_2.

    Returns a tensor of shape (batch, C, H, W) on the configured device.
    """
    device = _device(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    shape = (cfg.noise_batch_size, *cfg.input_shape)
    noise = torch.randn(shape, device=device, requires_grad=True)
    optim = torch.optim.Adam([noise], lr=cfg.noise_lr)
    target = torch.full((cfg.noise_batch_size,), target_class, dtype=torch.long, device=device)

    for step in range(cfg.noise_steps):
        optim.zero_grad(set_to_none=True)
        logits = model(noise)
        ce = F.cross_entropy(logits, target)
        reg = cfg.noise_lambda * noise.pow(2).mean()
        # Minimising ce drives logits *toward* target_class — i.e. we are
        # synthesising an input the model thinks is strongly that class.
        # When trained on as if it were a non-target sample, this becomes
        # the "anti-pattern" used to push the decision boundary.
        loss = ce + reg
        loss.backward()
        optim.step()
        if on_step is not None:
            on_step(step, float(loss.item()))

    # Re-enable grads for subsequent fine-tuning.
    for p in model.parameters():
        p.requires_grad_(True)
    return noise.detach()


# -------------------- Phase 2 & 3: impair / repair --------------------

class _NoiseDataset(Dataset):
    """Dataset wrapping a noise tensor with anti-labels for impair training.

    Per the paper, the noise is mixed into the retain stream and labelled
    with a *retain* class so the gradient pushes weights away from the
    forget-class decision region while remaining stable on retain classes.
    For multi-class forgetting, each forget class has its own noise tensor;
    we round-robin assign retain class labels.
    """

    def __init__(self, noise: torch.Tensor, labels: torch.Tensor):
        assert noise.size(0) == labels.size(0)
        self.noise = noise.cpu()
        self.labels = labels.cpu()

    def __len__(self) -> int:
        return self.noise.size(0)

    def __getitem__(self, idx: int):
        return self.noise[idx], int(self.labels[idx])


def _build_noise_dataset(
    noises: dict[int, torch.Tensor],
    retain_classes: list[int],
) -> _NoiseDataset:
    """Concatenate per-class noise into one dataset with retain-class labels.

    Each chunk of noise is given a label drawn from retain_classes
    (round-robin). This is the "anti-sample with wrong-class label" trick
    that drives the impair step.
    """
    pieces_x: list[torch.Tensor] = []
    pieces_y: list[torch.Tensor] = []
    for i, (_, n) in enumerate(noises.items()):
        retain_label = retain_classes[i % len(retain_classes)]
        labels = torch.full((n.size(0),), retain_label, dtype=torch.long)
        pieces_x.append(n)
        pieces_y.append(labels)
    return _NoiseDataset(torch.cat(pieces_x, dim=0), torch.cat(pieces_y, dim=0))


def impair(
    model: nn.Module,
    retain_sub: Dataset,
    noises: dict[int, torch.Tensor],
    cfg: UnsirConfig,
    on_batch: Callable[[int, int, float], None] | None = None,
) -> None:
    """Fine-tune one (or more) epochs on retain_sub ∪ noise at high LR."""
    device = _device(cfg)
    forget_set = set(cfg.forget_classes)
    retain_classes = [c for c in range(cfg.num_classes) if c not in forget_set]
    noise_ds = _build_noise_dataset(noises, retain_classes)
    combined = ConcatDataset([retain_sub, noise_ds])
    dl = DataLoader(combined, batch_size=128, shuffle=True, num_workers=0,
                    pin_memory=torch.cuda.is_available())
    optim = torch.optim.SGD(model.parameters(), lr=cfg.impair_lr, momentum=0.9)
    model.train()
    for epoch in range(cfg.impair_epochs):
        for b, (x, y) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optim.step()
            if on_batch is not None:
                on_batch(epoch, b, float(loss.item()))


def repair(
    model: nn.Module,
    retain_sub: Dataset,
    cfg: UnsirConfig,
    on_batch: Callable[[int, int, float], None] | None = None,
) -> None:
    """Fine-tune one (or more) epochs on retain_sub only at low LR."""
    device = _device(cfg)
    dl = DataLoader(retain_sub, batch_size=128, shuffle=True, num_workers=0,
                    pin_memory=torch.cuda.is_available())
    optim = torch.optim.SGD(model.parameters(), lr=cfg.repair_lr, momentum=0.9)
    model.train()
    for epoch in range(cfg.repair_epochs):
        for b, (x, y) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optim.step()
            if on_batch is not None:
                on_batch(epoch, b, float(loss.item()))


# -------------------- Orchestration --------------------

def run_unsir(
    model: nn.Module,
    retain_sub: Dataset,
    cfg: UnsirConfig,
    progress: Callable[[str, dict], None] | None = None,
) -> dict:
    """Run the full UNSIR pipeline. Returns timing info per phase."""
    timings: dict[str, float] = {}

    # Phase 1
    t0 = time.perf_counter()
    noises: dict[int, torch.Tensor] = {}
    for c in cfg.forget_classes:
        if progress:
            progress("noise_class_start", {"class": c})

        def _step(s, l, _c=c):
            if progress:
                progress("noise_step", {"class": _c, "step": s, "loss": l})

        noises[c] = generate_class_noise(model, c, cfg, on_step=_step)
        if progress:
            progress("noise_class_done", {"class": c})
    timings["noise"] = time.perf_counter() - t0

    # Phase 2
    t0 = time.perf_counter()
    if progress:
        progress("impair_start", {})

    def _ib(e, b, l):
        if progress:
            progress("impair_batch", {"epoch": e, "batch": b, "loss": l})

    impair(model, retain_sub, noises, cfg, on_batch=_ib)
    if progress:
        progress("impair_done", {})
    timings["impair"] = time.perf_counter() - t0

    # Phase 3
    t0 = time.perf_counter()
    if progress:
        progress("repair_start", {})

    def _rb(e, b, l):
        if progress:
            progress("repair_batch", {"epoch": e, "batch": b, "loss": l})

    repair(model, retain_sub, cfg, on_batch=_rb)
    if progress:
        progress("repair_done", {})
    timings["repair"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    return timings
