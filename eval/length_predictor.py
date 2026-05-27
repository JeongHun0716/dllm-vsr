# coding=utf-8
"""Run length predictor on an LRS3 split (single GPU) and dump per-sample
top-1 prediction + ±5 probability range for downstream decoding use.

Output: <ckpt_dir>/len_pred_<split>.jsonl
Each line:
{
  "utt_id": "...",
  "K_target": 47,
  "pred_K": 46,
  "K_range": [41, 42, ..., 51],
  "probs":   [0.001, 0.005, ..., 0.42, ..., 0.002],
  "argmax_prob": 0.42,
  "K_target_prob": 0.38,
  "entropy": 1.23
}

Also writes summary metrics (Acc, Acc±1/±3/±5, MAE) for the split.

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval/length_predictor.py \
      config=configs/lrs3/len_pred_usr2.yaml \
      experiment.eval_checkpoint=ckpt/usr2/len_pred

Optional overrides:
  eval.split=val                          # split name; resolves to <manifest_root>/<split>.tsv|.wrd
  eval.manifest_tsv=/path/to/val.tsv      # direct manifest path
  eval.label_wrd=/path/to/val.wrd         # direct label path
"""
from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# sys.path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_DLLM_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _DLLM_ROOT)

from omegaconf import OmegaConf

from training.utils import get_config
from training.data import VSRDataset
from models.len_predictor import LenPredictor


logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: str):
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = [p for p in out.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.name.split("-")[1]))
    return ckpts[-1]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float32

    # ===== model =====
    visual_enc_cfg = OmegaConf.to_container(config.model.visual_encoder, resolve=True)
    model = LenPredictor(
        visual_encoder_cfg=visual_enc_cfg,
        enc_dim=int(config.model.enc_dim),
        hidden=int(config.model.hidden),
        n_layers=int(config.model.n_layers),
        n_heads=int(config.model.n_heads),
        ffn_dim=int(config.model.ffn_dim),
        dropout=0.0,                              # eval
        len_max=int(config.model.len_max),
        max_video_frames=int(config.dataset.vsr.max_video_frames) + 8,
    )
    len_max = int(config.model.len_max)

    # checkpoint
    eval_ckpt = config.experiment.get("eval_checkpoint", None)
    if eval_ckpt:
        ckpt_dir = Path(str(eval_ckpt))
    else:
        ckpt_dir = find_latest_checkpoint(config.experiment.output_dir)
    if ckpt_dir is None or not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_dir}")
    ckpt_path = ckpt_dir / "trainable_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"trainable_model.pt not found in {ckpt_dir}")
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"Loaded {ckpt_path}  Missing(ok): {len(missing)}  Unexpected: {len(unexpected)}")

    model = model.to(device=device, dtype=dtype)
    model.eval()

    # ===== dataset (full test 1321) =====
    raw_cfg = config.dataset.get("raw_video", {})
    vsr_cfg = config.dataset.vsr
    manifest_root = str(vsr_cfg.manifest_root)
    common = dict(
        dllm_tokenizer_path=config.model.dllm.tokenizer_path,
        modalities=list(vsr_cfg.modalities),
        video_root=vsr_cfg.get("video_root", None),
        crop_size=int(raw_cfg.get("crop_size", 88)),
        normalize_mean=float(raw_cfg.get("normalize_mean", 0.421)),
        normalize_std=float(raw_cfg.get("normalize_std", 0.165)),
        time_mask_window=int(raw_cfg.get("time_mask_window", 10)),
        time_mask_stride=int(raw_cfg.get("time_mask_stride", 25)),
        train_crop="center",
        enable_time_mask=False,
    )
    # Allow override via eval.split (e.g., "val" / "test") and CLI overrides for tsv/wrd.
    split_name = str(config.eval.get("split", "test")) if config.get("eval", None) is not None else "test"
    tsv_override = config.eval.get("manifest_tsv", None) if config.get("eval", None) is not None else None
    wrd_override = config.eval.get("label_wrd", None) if config.get("eval", None) is not None else None
    test_ds = VSRDataset(
        manifest_path=str(tsv_override) if tsv_override else os.path.join(manifest_root, f"{split_name}.tsv"),
        label_paths=[str(wrd_override) if wrd_override else os.path.join(manifest_root, f"{split_name}.wrd")],
        subset="test",
        max_video_frames=int(vsr_cfg.max_video_frames),
        **common,
    )
    logger.info(f"{split_name} set: {len(test_ds)} samples")

    # keep it simple with B=1 (only a few minutes, no need to batch)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=test_ds.collate_fn,
        num_workers=int(config.dataset.dataloader.get("num_workers", 4)),
        pin_memory=True,
    )

    # ===== output =====
    out_dir = ckpt_dir
    out_jsonl = out_dir / f"len_pred_{split_name}.jsonl"
    out_summary = out_dir / f"len_pred_{split_name}_summary.json"

    total = 0
    correct = 0
    within1 = 0
    within3 = 0
    within5 = 0
    sum_abs = 0.0

    range_radius = 5    # ±5

    with open(out_jsonl, "w") as f_jsonl, torch.no_grad():
        for batch in test_loader:
            video_feats = batch.get("video_features", None)
            video_pad = batch.get("video_padding_mask", None)
            if video_feats is None:
                continue
            video_feats = video_feats.to(device=device, dtype=dtype)
            if video_pad is not None:
                video_pad = video_pad.to(device)

            K_list = batch.get("K", None)
            utt_id_list = batch.get("utt_id", ["?"])
            K_target_int = int(K_list[0])

            logits, _ = model(video_feats=video_feats, video_pad=video_pad, K_target=None)
            probs = F.softmax(logits.float(), dim=-1).squeeze(0)   # [len_max]
            pred_K = int(probs.argmax().item()) + 1                # 1..len_max

            # ±5 range around pred_K
            lo = max(1, pred_K - range_radius)
            hi = min(len_max, pred_K + range_radius)
            K_range = list(range(lo, hi + 1))
            range_probs = probs[lo - 1: hi].tolist()    # idx (K-1)

            # entropy
            entropy = float(-(probs * (probs.clamp_min(1e-12)).log()).sum().item())

            rec = {
                "utt_id": utt_id_list[0] if isinstance(utt_id_list, list) else utt_id_list,
                "K_target": K_target_int,
                "pred_K": pred_K,
                "K_range": K_range,
                "probs": [round(p, 6) for p in range_probs],
                "argmax_prob": round(float(probs[pred_K - 1].item()), 6),
                "K_target_prob": round(float(probs[K_target_int - 1].item()), 6),
                "entropy": round(entropy, 4),
            }
            f_jsonl.write(json.dumps(rec) + "\n")

            # summary metrics
            diff = abs(pred_K - K_target_int)
            total += 1
            if diff == 0: correct += 1
            if diff <= 1: within1 += 1
            if diff <= 3: within3 += 1
            if diff <= 5: within5 += 1
            sum_abs += diff

            if total % 100 == 0:
                logger.info(f"[{total}/{len(test_ds)}] running Acc(=K)={correct/total*100:.2f}% "
                            f"Acc(±5)={within5/total*100:.2f}%")

    summary = {
        "ckpt": str(ckpt_dir),
        "total": total,
        "acc_eq":   correct / total,
        "acc_pm1":  within1 / total,
        "acc_pm3":  within3 / total,
        "acc_pm5":  within5 / total,
        "mae":      sum_abs / total,
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("===== FINAL =====")
    logger.info(json.dumps(summary, indent=2))
    logger.info(f"jsonl:   {out_jsonl}")
    logger.info(f"summary: {out_summary}")


if __name__ == "__main__":
    main()
