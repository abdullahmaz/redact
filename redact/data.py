"""CIFAR-10 dataset utilities and retain/forget splits for UNSIR.

We support two backends, picked automatically:

1. The standard `torchvision.datasets.CIFAR10` if its tar.gz is reachable
   (the canonical UofT host is occasionally 503 for us).
2. A fallback that reads the HuggingFace parquet mirror at
   `data/hf/{train,test}.parquet` — pre-decoded into RAM at startup.

Both backends expose the same fields used downstream: `targets`, `__len__`,
indexing returning (tensor, label).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset


# torchvision CIFAR-10 official label order
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _train_transform() -> T.Compose:
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _eval_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# -------------------- HF parquet backend --------------------

class _Cifar10HF(Dataset):
    """CIFAR-10 read from HuggingFace parquet, pre-decoded to a uint8 array."""

    def __init__(self, parquet_path: str | Path, transform=None):
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_path))
        rows = table.to_pydict()
        n = len(rows["label"])
        # decode every PNG up-front; CIFAR-10 fits in RAM.
        arr = np.empty((n, 32, 32, 3), dtype=np.uint8)
        for i, img_struct in enumerate(rows["img"]):
            png_bytes = img_struct["bytes"]
            with Image.open(io.BytesIO(png_bytes)) as im:
                arr[i] = np.array(im.convert("RGB"))
        self.data = arr  # (N, H, W, C) uint8
        self.targets = list(int(x) for x in rows["label"])
        self.classes = CIFAR10_CLASSES
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.data[idx])
        label = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# -------------------- Dispatcher --------------------

def cifar10(data_root: str | Path = "data", train: bool = True, augment: bool | None = None):
    """Return a CIFAR-10 dataset.

    augment defaults to True for train, False for test. Falls back to the
    HF parquet copy at `data/hf/{train,test}.parquet` if the torchvision
    download host is unreachable.
    """
    if augment is None:
        augment = train
    transform = _train_transform() if augment else _eval_transform()

    hf_path = Path(data_root) / "hf" / ("train.parquet" if train else "test.parquet")
    if hf_path.exists():
        return _Cifar10HF(hf_path, transform=transform)

    return torchvision.datasets.CIFAR10(
        root=str(data_root), train=train, download=True, transform=transform,
    )


def loader(dataset: Dataset, batch_size: int = 128, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


def indices_by_class(dataset, target_classes: Iterable[int]) -> list[int]:
    targets = _get_targets(dataset)
    target_set = set(int(c) for c in target_classes)
    return [i for i, t in enumerate(targets) if int(t) in target_set]


def indices_excluding_class(dataset, exclude_classes: Iterable[int]) -> list[int]:
    targets = _get_targets(dataset)
    excl = set(int(c) for c in exclude_classes)
    return [i for i, t in enumerate(targets) if int(t) not in excl]


def _get_targets(dataset):
    if hasattr(dataset, "targets"):
        return dataset.targets
    return [dataset[i][1] for i in range(len(dataset))]


def retain_subset(dataset, forget_classes: Iterable[int], per_class: int = 1000, seed: int = 0) -> Subset:
    """A small balanced subset of the retain set used by impair/repair (Dr_sub).

    `per_class` samples per non-forget class (paper uses small Dr_sub).
    """
    g = torch.Generator().manual_seed(seed)
    forget = set(int(c) for c in forget_classes)
    targets = _get_targets(dataset)
    by_class: dict[int, list[int]] = {}
    for i, t in enumerate(targets):
        t = int(t)
        if t in forget:
            continue
        by_class.setdefault(t, []).append(i)
    chosen: list[int] = []
    for c, idxs in by_class.items():
        idxs_t = torch.tensor(idxs)
        perm = torch.randperm(len(idxs_t), generator=g)[:per_class]
        chosen.extend(idxs_t[perm].tolist())
    return Subset(dataset, chosen)
