"""Video augmentation pipeline for raw mouth-crop videos.

LRS3 (and similar VSR datasets) ship mouth crops as ~96x96 greyscale-equivalent RGB.

train:
    /255.0 → RandomCrop(crop_size) → Grayscale → AdaptiveTimeMask → Normalize(mean,std)
val/test:
    /255.0 → CenterCrop(crop_size) → Grayscale → Normalize(mean,std)

Input is [T, C, H, W] uint8 or float (from load_video). Output is [T, 1, crop_size, crop_size] float.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torchvision


class _FunctionalModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class AdaptiveTimeMask(nn.Module):
    """Mask spans of length `window` at `stride` intervals to zero."""

    def __init__(self, window: int = 10, stride: int = 25):
        super().__init__()
        self.window = int(window)
        self.stride = int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            t = int(t.item()); t_end = int(t_end.item())
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end_abs = t_start + t_end
            t_end_abs = min(t_end_abs, length)
            cloned[t_start:t_end_abs] = 0
        return cloned


class VideoTransform(nn.Module):
    """[T, C, H, W] uint8 or float → [T, 1, crop, crop] float, normalized.

    Knobs:
        train_crop ∈ {"random", "center"} — crop mode for the train subset.
            "random" (default): adds slight shift augmentation
            "center": CenterCrop only (matches val/test distribution)
        enable_time_mask: whether AdaptiveTimeMask is applied on the train subset.
    """

    def __init__(
        self,
        subset: str = "train",
        crop_size: int = 88,
        normalize_mean: float = 0.421,
        normalize_std: float = 0.165,
        time_mask_window: int = 10,
        time_mask_stride: int = 25,
        train_crop: str = "random",
        enable_time_mask: bool = True,
    ):
        super().__init__()
        self.subset = subset
        if train_crop not in ("random", "center"):
            raise ValueError(f"train_crop must be 'random' or 'center', got {train_crop!r}")
        layers: list[nn.Module] = [
            _FunctionalModule(lambda x: x.float() / 255.0 if x.dtype == torch.uint8 else x.float())
        ]
        if subset == "train" and train_crop == "random":
            layers.append(torchvision.transforms.RandomCrop(crop_size))
        else:
            layers.append(torchvision.transforms.CenterCrop(crop_size))
        layers.append(torchvision.transforms.Grayscale())
        if subset == "train" and enable_time_mask:
            layers.append(AdaptiveTimeMask(time_mask_window, time_mask_stride))
        layers.append(torchvision.transforms.Normalize(normalize_mean, normalize_std))
        self.pipeline = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pipeline(x)
