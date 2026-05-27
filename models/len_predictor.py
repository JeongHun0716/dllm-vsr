# coding=utf-8
"""Lightweight length predictor.

Architecture:
  video [B, T, 1, 88, 88] (raw)
    ↓ USR2 (frozen)                    → [B, T', 1280]
    ↓ Linear proj  1280 → hidden       → [B, T', hidden]
    ↓ + sinusoidal positional embed
    ↓ prepend learnable <LEN> query    → [B, T'+1, hidden]
    ↓ Transformer encoder × n_layers
    ↓ take position 0 (LEN query out)  → [B, hidden]
    ↓ MLP head  hidden → len_max       → logits [B, len_max]
    loss = CE(logits, K-1)             # K = transcript+EOS length, 1..len_max
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.visual_encoder import build_visual_encoder


def _sinusoid_pos_embed(seq_len: int, dim: int, device, dtype) -> torch.Tensor:
    """[seq_len, dim] sinusoidal positional embedding."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class LenPredictor(nn.Module):
    def __init__(
        self,
        *,
        visual_encoder_cfg: dict,
        enc_dim: int,
        hidden: int = 384,
        n_layers: int = 2,
        n_heads: int = 6,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
        len_max: int = 150,
        max_video_frames: int = 600,    # pos embed buffer size
    ):
        super().__init__()
        self.len_max = int(len_max)
        self.hidden = int(hidden)

        # 1) Visual encoder (frozen)
        self.v_encoder = build_visual_encoder(visual_encoder_cfg)
        for p in self.v_encoder.parameters():
            p.requires_grad = False

        # 2) projection
        self.proj = nn.Linear(int(enc_dim), hidden)

        # 3) learnable LEN query token
        self.len_query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.len_query, std=0.02)

        # 4) Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 5) LEN classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.len_max),
        )

        # precompute pos embedding as buffer (sinusoidal, not learnable)
        self.register_buffer(
            "_pos_embed",
            _sinusoid_pos_embed(max_video_frames + 1, hidden, device="cpu", dtype=torch.float32),
            persistent=False,
        )

    def video_input_dtype(self):
        return self.proj.weight.dtype

    def forward(
        self,
        video_feats: torch.Tensor,                       # [B, T, 1, H, W] raw
        video_pad: Optional[torch.Tensor] = None,        # [B, T] True=pad
        K_target: Optional[torch.Tensor] = None,         # [B], 1..len_max, optional (for loss)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1) frozen visual encode
        with torch.no_grad():
            v_enc, v_pad = self.v_encoder.encode(video_feats, video_pad)
            # v_enc: [B, T', enc_dim], v_pad: [B, T'] True=pad

        # 2) project to hidden
        x = self.proj(v_enc.to(self.proj.weight.dtype))      # [B, T', hidden]
        B, Tp, _ = x.shape

        # 3) prepend LEN query
        len_q = self.len_query.expand(B, 1, self.hidden)     # [B, 1, hidden]
        x = torch.cat([len_q, x], dim=1)                     # [B, 1+T', hidden]

        # 4) add pos embed
        pos = self._pos_embed[: 1 + Tp].to(dtype=x.dtype, device=x.device)   # [1+T', hidden]
        x = x + pos.unsqueeze(0)

        # 5) build attention key_padding_mask. LEN query is always valid (False).
        if v_pad is not None:
            len_pad_col = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            kpm = torch.cat([len_pad_col, v_pad.bool()], dim=1)     # [B, 1+T']
        else:
            kpm = None

        # 6) Transformer encode
        y = self.encoder(x, src_key_padding_mask=kpm)        # [B, 1+T', hidden]

        # 7) LEN query output
        len_repr = y[:, 0]                                   # [B, hidden]
        logits = self.head(len_repr)                          # [B, len_max]

        loss = None
        if K_target is not None:
            target = (K_target.long() - 1).clamp(0, self.len_max - 1)
            loss = nn.functional.cross_entropy(logits, target)

        return logits, loss

    def print_trainable_params(self):
        tot = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LenPredictor] trainable: {train:,} / {tot:,} ({train/tot*100:.2f}%)")
