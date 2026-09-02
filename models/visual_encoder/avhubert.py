"""AV-HuBERT visual encoder — loads fairseq checkpoint, frozen forward."""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
from torch import Tensor

from .base import VisualEncoder


# Vendored AV-HuBERT repo location (third_party/avhubert). Override via AVHUBERT_REPO env var.
DEFAULT_AVHUBERT_REPO = os.environ.get(
    "AVHUBERT_REPO",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "avhubert",
    )),
)


class AVHubertEncoder(VisualEncoder):
    def __init__(
        self,
        ckpt_path: str,
        feat_dim: int = 1024,
        output_fps: float = 25.0,
        repo_path: Optional[str] = None,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.output_fps = float(output_fps)
        self.ckpt_path = ckpt_path
        self.repo_path = repo_path or DEFAULT_AVHUBERT_REPO
        self._encoder: Optional[torch.nn.Module] = None
        self.load()

    def load(self) -> None:
        # Vendored avhubert lives in third_party/; ensure its parent dir is importable.
        parent = os.path.dirname(self.repo_path)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from fairseq import checkpoint_utils  # noqa: WPS433 (lazy import — fairseq is heavy)
        from avhubert.hubert_asr import HubertEncoderWrapper

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([self.ckpt_path])
        self._encoder = HubertEncoderWrapper(models[0])
        self._encoder.eval()
        self._encoder.requires_grad_(False)

    def encode(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """AV-HuBERT video frontend expects [B, C=1, T, H, W] (channel-before-time);
        raw collate gives [B, T, C, H, W] so we permute here.
        """
        if self._encoder is None:
            raise RuntimeError("AVHubertEncoder.load() not called yet")
        if x.dim() == 5:
            # [B, T, C, H, W] -> [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        elif x.dim() == 4:
            # [B, T, H, W] (assume greyscale) -> [B, 1, T, H, W]
            x = x.unsqueeze(1).contiguous()
        # else: assume already 5D channel-first

        with torch.no_grad():
            out = self._encoder(
                source={"audio": None, "video": x},
                padding_mask=padding_mask,
            )
            feat = out["encoder_out"].transpose(0, 1)  # [T,B,D] -> [B,T,D]
            pad = out.get("padding_mask", padding_mask)
        return feat, pad

    def train(self, mode: bool = True):
        super().train(mode)
        # always keep encoder in eval (frozen)
        if self._encoder is not None:
            self._encoder.eval()
        return self
