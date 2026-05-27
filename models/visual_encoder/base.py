"""VisualEncoder ABC for dllm-vsr.

Train/inference flow:
    raw video [B,C,T,H,W]  ─encode──▶  [B,T',D]  ─downsampler+fc(adapter)──▶  [B,T'',dllm_hidden]

DLLMVSRModel holds `self.v_encoder: VisualEncoder` with no per-type branching.
To add a new encoder: (a) subclass base.VisualEncoder in a new file (b) register in __init__.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class VisualEncoder(ABC, nn.Module):
    # class attrs (subclasses override)
    feat_dim: int = 0           # output channels D
    output_fps: float = 25.0    # output token frame rate (Hz)

    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """encoder forward.

        Args:
            x: [B, C, T, H, W] raw video
            padding_mask: [B, T] (True=pad). If None, all valid.
        Returns:
            feat: [B, T', D]  (D = self.feat_dim)
            out_padding_mask: [B, T'] (True=pad) or None
        """
        raise NotImplementedError

    def input_dtype(self) -> torch.dtype:
        """Expected input dtype for the encoder. Defaults to the encoder's parameter dtype."""
        params = list(self.parameters())
        if not params:
            return torch.float32
        return params[0].dtype
