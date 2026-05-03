"""ResNet-18 adapted for 32x32 CIFAR-10 inputs.

The torchvision ImageNet ResNet-18 starts with a 7x7 stride-2 conv plus
a 3x3 max-pool, which destroys spatial resolution on 32x32 images. The
standard CIFAR adaptation is to swap the stem for a 3x3 stride-1 conv
and drop the max-pool, which is what we do here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    model = tvm.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
