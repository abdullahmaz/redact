"""Process-wide model + dataset state shared by the Flask app.

We keep a single model in memory and rebuild it from the saved checkpoint
whenever the user resets, so the UI feels live.
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from .data import cifar10, retain_subset, loader, CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
from .evaluate import per_class_accuracy
from .model import build_resnet18_cifar, device


@dataclass
class AppState:
    checkpoint_path: str
    test_set: object = None
    test_loader: object = None
    train_set: object = None  # used for sampling Dr_sub during unlearning
    raw_test_set: object = None  # un-augmented, for visual sample previews
    model: torch.nn.Module = None
    device: torch.device = None
    baseline_per_class: list[float] = field(default_factory=list)
    current_per_class: list[float] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reload_model(self) -> None:
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model = build_resnet18_cifar(num_classes=10).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()


def init_state(checkpoint_path: str = "checkpoints/baseline.pt") -> AppState:
    s = AppState(checkpoint_path=checkpoint_path)
    s.device = device()
    print(f"[redact] device: {s.device}")
    print(f"[redact] loading test set...")
    s.test_set = cifar10(train=False, augment=False)
    s.raw_test_set = s.test_set  # transforms already non-augmented
    s.test_loader = loader(s.test_set, batch_size=256, shuffle=False)
    print(f"[redact] loading train set...")
    s.train_set = cifar10(train=True, augment=False)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {checkpoint_path}. "
            "Run `python -m redact.train` first."
        )
    s.reload_model()
    print(f"[redact] computing baseline per-class accuracy...")
    s.baseline_per_class = per_class_accuracy(s.model, s.test_loader, 10, s.device)
    s.current_per_class = list(s.baseline_per_class)
    return s


# -------------------- denormalisation for visual previews --------------------

_MEAN = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
_STD = torch.tensor(CIFAR10_STD).view(3, 1, 1)


def tensor_to_png_bytes(x: torch.Tensor) -> bytes:
    """Inverse-normalise a CIFAR-10 tensor and return PNG bytes."""
    x = x.detach().cpu()
    x = x * _STD + _MEAN
    x = x.clamp(0, 1) * 255
    x = x.byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(x)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
