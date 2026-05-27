"""Visual encoder registry for dllm-vsr v2.

`model.visual_encoder.type` in the config must be one of the keys registered here.
To add a new encoder: (1) write models/visual_encoder/<name>.py (2) register in ENCODERS dict.
"""
from __future__ import annotations

from typing import Type

from .base import VisualEncoder
from .avhubert import AVHubertEncoder
from .usr2 import USR2Encoder


ENCODERS: dict[str, Type[VisualEncoder]] = {
    "avhubert": AVHubertEncoder,
    "usr2": USR2Encoder,
}


def build_visual_encoder(cfg) -> VisualEncoder:
    """Accepts either an OmegaConf node or a plain dict.

    Example cfg (avhubert):
        {"type": "avhubert", "ckpt_path": "/path/to/large_vox_iter5.pt", "enc_dim": 1024}
    Example cfg (usr2):
        {"type": "usr2", "ckpt_path": "/path/to/usr2_huge.pt", "backbone": "huge"}
    """
    # unified dict-like access (OmegaConf supports attr/index, dict supports index only)
    def _get(k, default=None):
        if isinstance(cfg, dict):
            return cfg.get(k, default)
        return cfg.get(k, default) if hasattr(cfg, "get") else getattr(cfg, k, default)

    def _has(k):
        if isinstance(cfg, dict):
            return k in cfg
        try:
            return k in cfg
        except TypeError:
            return hasattr(cfg, k)

    enc_type = str(_get("type"))
    if enc_type not in ENCODERS:
        raise ValueError(
            f"Unknown visual encoder type: {enc_type}. Available: {list(ENCODERS.keys())}"
        )
    cls = ENCODERS[enc_type]

    # common kwargs — only override enc_dim if user specified it (defaults differ per encoder)
    kwargs: dict = {}
    if _has("enc_dim"):
        kwargs["feat_dim"] = int(_get("enc_dim"))
    if _has("output_fps"):
        kwargs["output_fps"] = float(_get("output_fps"))

    # per-encoder extra kwargs
    if enc_type == "avhubert":
        kwargs["ckpt_path"] = str(_get("ckpt_path"))
    elif enc_type == "usr2":
        kwargs["ckpt_path"] = str(_get("ckpt_path"))
        if _has("backbone"):
            kwargs["backbone"] = str(_get("backbone"))

    return cls(**kwargs)


__all__ = [
    "VisualEncoder",
    "AVHubertEncoder",
    "USR2Encoder",
    "ENCODERS",
    "build_visual_encoder",
]
