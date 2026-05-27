"""USR / USR 2.0 visual encoder (ahaliassos/usr, ahaliassos/usr2).

ResNet + Transformer hybrid (resnet_transformer_{base,baseplus,large,huge}).
Adds upstream repo (default `<dllm parent>/usr2/`) to sys.path for import.

Ckpt state_dict keys are deeply prefixed (`model._orig_mod.model.backbone.encoder.<X>`);
prefix must be stripped before loading into USRModel.backbone.encoder.

VSR-only: pass xs_v=video, xs_a=None to encoder forward. Output [B, T', adim].
adim per backbone (matches resnet_transformer_{name}.yaml):
    base       → 768
    baseplus   → 1024
    large      → 1024   (USR paper Large — 24 enc layers)
    huge       → 1280   (USR2 paper Huge — 32 enc layers)
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
from torch import Tensor

from .base import VisualEncoder


# upstream slim copy location — defaults to dllm-vsr/third_party/usr2. Override via USR2_REPO env var.
DEFAULT_USR2_REPO = os.environ.get(
    "USR2_REPO",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "usr2",
    )),
)

_BACKBONE_FEAT_DIM = {
    "base": 768,
    "baseplus": 1024,
    "large": 1024,
    "huge": 1280,
}

_CKPT_PREFIX_CANDIDATES = (
    "model._orig_mod.model.",   # torch.compile + Lightning artifact (USR2 v2 ckpt)
    "model._orig_mod.",
    "_orig_mod.model.",         # USR (original paper) Large ckpt
    "_orig_mod.",
    "model.",
)


def _purge_espnet_from_sys_modules() -> None:
    """USR2 and Auto-AVSR both vendor the same `espnet` top-level namespace, so
    importing one after the other collides in sys.modules cache. To use both, we
    clear espnet/models from sys.modules right before loading the encoder."""
    to_purge = [m for m in list(sys.modules) if m == "espnet" or m.startswith("espnet.") or m == "models" or m.startswith("models.")]
    for m in to_purge:
        del sys.modules[m]


class USR2Encoder(VisualEncoder):
    def __init__(
        self,
        ckpt_path: str,
        backbone: str = "huge",
        feat_dim: Optional[int] = None,
        output_fps: float = 25.0,
        repo_path: Optional[str] = None,
    ):
        super().__init__()
        if backbone not in _BACKBONE_FEAT_DIM:
            raise ValueError(f"USR2 backbone must be one of {list(_BACKBONE_FEAT_DIM)}, got {backbone}")
        self.backbone = backbone
        self.feat_dim = int(feat_dim) if feat_dim is not None else _BACKBONE_FEAT_DIM[backbone]
        self.output_fps = float(output_fps)
        self.ckpt_path = ckpt_path
        self.repo_path = repo_path or DEFAULT_USR2_REPO
        self._encoder: Optional[torch.nn.Module] = None
        self.load()

    def load(self) -> None:
        from omegaconf import OmegaConf

        # cleanup if another encoder already claimed espnet namespace (so we use USR2's vendored code)
        _purge_espnet_from_sys_modules()
        # ensure USR2 repo sits at the front of sys.path
        if self.repo_path in sys.path:
            sys.path.remove(self.repo_path)
        sys.path.insert(0, self.repo_path)
        from models.usr import USRModel  # noqa: E402

        backbone_yaml = os.path.join(
            self.repo_path, "conf", "model", "backbone",
            f"resnet_transformer_{self.backbone}.yaml",
        )
        if not os.path.isfile(backbone_yaml):
            raise FileNotFoundError(f"USR2 backbone config missing: {backbone_yaml}")
        backbone_cfg = OmegaConf.load(backbone_yaml)
        cfg = OmegaConf.create({"model": {"backbone": backbone_cfg}})

        usr_model = USRModel(cfg)
        sd_raw = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
            sd_raw = sd_raw["state_dict"]
        elif isinstance(sd_raw, dict) and "model_state_dict" in sd_raw:
            sd_raw = sd_raw["model_state_dict"]

        # Strip the longest matching prefix
        prefix_to_strip = ""
        for p in _CKPT_PREFIX_CANDIDATES:
            if any(k.startswith(p) for k in sd_raw.keys()):
                prefix_to_strip = p
                break
        if prefix_to_strip:
            sd_clean = {
                (k[len(prefix_to_strip):] if k.startswith(prefix_to_strip) else k): v
                for k, v in sd_raw.items()
            }
        else:
            sd_clean = dict(sd_raw)

        missing, unexpected = usr_model.load_state_dict(sd_clean, strict=False)
        # Permissive: USR ckpt may contain ASR-side decoder/CTC keys we don't use
        if missing:
            n_critical_missing = sum(1 for k in missing if k.startswith("backbone.encoder."))
            if n_critical_missing > 0:
                raise RuntimeError(
                    f"USR2 ckpt missing {n_critical_missing} critical encoder keys; first 5: {missing[:5]}"
                )
        usr_model.eval()
        usr_model.requires_grad_(False)
        # we only call backbone.encoder
        self._encoder = usr_model.backbone.encoder

    def encode(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """encode raw video.

        Args:
            x: [B, T, H, W] grayscale (USR convention) or [B, T, 1, H, W] (with channel).
               If channel present, squeeze.
            padding_mask: [B, T] True=pad. Encoder expects mask=[B, 1, T] (non-pad mask).
        Returns:
            feat: [B, T', feat_dim]
            out_padding_mask: [B, T'] True=pad, or None
        """
        if self._encoder is None:
            raise RuntimeError("USR2Encoder.load() not called yet")
        if x.dim() == 5 and x.size(2) == 1:
            x = x.squeeze(2)  # [B, T, H, W]
        elif x.dim() != 4:
            raise ValueError(f"USR2 expects [B,T,H,W] or [B,T,1,H,W], got {tuple(x.shape)}")

        # USR encoder forward expects masks: [B, 1, T] non-pad mask (True=valid)
        masks = None
        if padding_mask is not None:
            non_pad = (~padding_mask).unsqueeze(1)  # [B, 1, T]
            masks = non_pad

        with torch.no_grad():
            feat = self._encoder(xs_v=x, xs_a=None, masks=masks, return_all=False)
            # USR encoder may return (xs, masks) or (xs,)
            if isinstance(feat, tuple):
                feat = feat[0]

        # output padding mask: no time-axis stride change (frontend3D stride=(1,2,2)) → same T
        return feat, padding_mask

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder is not None:
            self._encoder.eval()
        return self
