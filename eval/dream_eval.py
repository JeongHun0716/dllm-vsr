# coding=utf-8
"""Unified Dream eval — Fast-dLLM generate (subsumes 1-step argmax / blockwise / fastdllm).

Single decoding path: `generate_with_dual_cache_embeds_dream`.
The three previous "modes" become parameter choices:

  - 1-step argmax:   alg=fixed_k  block_size=answer_len  fixed_k=answer_len  steps=1
  - block-wise:      alg=threshold  dual_cache=false  (functionally = clean modeling)
  - fast-dLLM (fast): alg=threshold  dual_cache=true   (KV cache reuse)

Common knobs:
  block_size, steps, dual_cache, alg ∈ {threshold, fixed_k, dynamic},
  threshold, fixed_k, factor, answer_len, tail_token (pad|eos|int),
  eos_tail_mode (fill|attn_zero|truncate), max_samples.

Usage:
  CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_configs/1_gpu.yaml \\
    eval/dream_eval.py config=configs/lrs3/usr2_v2.yaml \\
    experiment.eval_checkpoint=ckpt/usr2/dream_stage2 \\
    eval.alg=threshold eval.block_size=8 eval.threshold=0.9 eval.dual_cache=false

Note: Fast-dLLM patched Dream backbone is forced via `DLLM_VSR_USE_FAST_DREAM=1`
(set at the top of this file, before any model import).
"""
from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

# sys.path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_DLLM_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _DLLM_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.pop("DLLM_VSR_USE_KV_CACHE_MODELING", None)
os.environ["DLLM_VSR_USE_FAST_DREAM"] = "1"  # always use patched Dream (dual_cache flag controls KV cache use)

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from safetensors.torch import load_file as safe_load_file

import editdistance

from training.utils import get_config
from training.data import VSRDataset
from models.model import build_model_and_tokenizer
from evaluation.fast_dream_generate import generate_with_dual_cache_embeds_dream

logger = get_logger(__name__, log_level="INFO")


# ============================================================================
# Helpers
# ============================================================================

def find_latest_checkpoint(output_dir: str):
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = [p for p in out.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.name.split("-")[1]))
    return ckpts[-1]


def load_checkpoint_weights(model, ckpt_dir: Path):
    trainable_path = ckpt_dir / "trainable_model.safetensors"
    sd = safe_load_file(str(trainable_path))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"Loaded: {trainable_path}  Missing(ok): {len(missing)}  Unexpected: {len(unexpected)}")


def build_test_loader(config, tokenizer_path):
    raw_cfg = config.dataset.get("raw_video", {})
    vsr_cfg = config.dataset.vsr
    manifest_root = str(vsr_cfg.manifest_root)
    test_dataset = VSRDataset(
        manifest_path=os.path.join(manifest_root, "test.tsv"),
        label_paths=[os.path.join(manifest_root, "test.wrd")],
        subset="test",
        max_video_frames=int(vsr_cfg.max_video_frames),
        dllm_tokenizer_path=tokenizer_path,
        modalities=list(vsr_cfg.modalities),
        video_root=vsr_cfg.get("video_root", None),
        crop_size=int(raw_cfg.get("crop_size", 88)),
        normalize_mean=float(raw_cfg.get("normalize_mean", 0.421)),
        normalize_std=float(raw_cfg.get("normalize_std", 0.165)),
        time_mask_window=int(raw_cfg.get("time_mask_window", 10)),
        time_mask_stride=int(raw_cfg.get("time_mask_stride", 25)),
        enable_time_mask=False,
    )
    return DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=config.dataset.dataloader.get("num_workers", 0),
        pin_memory=True,
    )


# ============================================================================
# Eval loop
# ============================================================================

@torch.no_grad()
def run_eval(*, model, dataloader, tokenizer, accelerator, mask_id,
             block_size: int, steps: int, dual_cache: bool,
             alg: str, threshold: float, fixed_k: int, factor: float,
             answer_len: int, tail_token_id: int, eos_tail_mode: str,
             max_samples: int):
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    v_in_dtype = unwrapped.video_input_dtype()

    n_err, n_total, total_nfe = 0, 0, 0
    samples = []

    for bidx, batch in enumerate(dataloader, start=1):
        if max_samples > 0 and bidx > max_samples:
            break
        video_feats = batch.get("video_features", None)
        if video_feats is None:
            continue
        if video_feats.is_floating_point():
            video_feats = video_feats.to(device=accelerator.device, dtype=v_in_dtype)
        video_pad = batch.get("video_padding_mask", None)
        if video_pad is not None:
            video_pad = video_pad.to(accelerator.device)
        instructions = [batch["input_ids"][0].to(accelerator.device)]
        post_user = batch.get("post_user_ids", None)
        if post_user is not None:
            post_user = post_user.to(accelerator.device)

        label_ids = batch["labels"][0]
        label_text = tokenizer.decode(label_ids.tolist(), skip_special_tokens=True).strip()
        # answer length: oracle (label-based) if model.use_oracle_length else fixed canvas
        if unwrapped.config.use_oracle_length:
            gen_length = int(label_ids.numel())
        else:
            gen_length = int(answer_len)

        v_enc, len_v, _ = unwrapped.encode_video(video_feats=video_feats, video_pad=None, apply_fc=True)
        pred_text, _, nfe = generate_with_dual_cache_embeds_dream(
            model=unwrapped.dllm, tokenizer=tokenizer, instructions=instructions,
            v_feats=v_enc[:, :len_v[0]], post_user_prefix=post_user, mask_id=mask_id,
            gen_length=gen_length, steps=steps, block_length=block_size,
            alg=alg, threshold=threshold, fixed_k=fixed_k, factor=factor,
            dual_cache=dual_cache, tail_token_id=tail_token_id,
            eos_tail_mode=eos_tail_mode,
        )
        total_nfe += nfe
        pred_text = pred_text.replace(tokenizer.eos_token, "").strip() if tokenizer.eos_token else pred_text.strip()

        n_err += editdistance.eval(pred_text.split(), label_text.split())
        n_total += len(label_text.split())
        samples.append({"idx": bidx, "gt": label_text, "pred": pred_text, "nfe": nfe})

        if bidx % 25 == 0 or bidx == len(dataloader):
            wer = 100.0 * n_err / max(1, n_total)
            avg_nfe = total_nfe / max(1, bidx)
            param = (f"th={threshold}" if alg == "threshold" else
                     f"k={fixed_k}" if alg == "fixed_k" else f"f={factor}")
            logger.info(f"[alg={alg} b={block_size} {param} dual={dual_cache}] "
                        f"[{bidx}/{len(dataloader)}] WER={wer:.2f}% avg_nfe={avg_nfe:.1f}")

    return (
        {"wer": 100.0 * n_err / max(1, n_total), "n_err": n_err, "n_total": n_total,
         "total_nfe": total_nfe, "avg_nfe": total_nfe / max(1, len(samples)),
         "n_samples": len(samples)},
        samples,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    logging.basicConfig(level=logging.INFO)
    config = get_config()

    accelerator = Accelerator(mixed_precision=config.training.mixed_precision)
    if config.training.seed is not None:
        set_seed(config.training.seed)

    model, tokenizer, mask_id, dllm_cfg_yaml = build_model_and_tokenizer(config)

    eval_ckpt = config.experiment.get("eval_checkpoint", None)
    if eval_ckpt is not None and str(eval_ckpt).strip() != "":
        ckpt_dir = Path(str(eval_ckpt))
    else:
        ckpt_dir = find_latest_checkpoint(config.experiment.output_dir)
    if ckpt_dir is None or not Path(ckpt_dir).exists():
        raise FileNotFoundError(f"checkpoint not found at {ckpt_dir}")
    logger.info(f"[dream_eval] Using checkpoint: {ckpt_dir}")
    load_checkpoint_weights(model, Path(ckpt_dir))

    test_loader = build_test_loader(config, dllm_cfg_yaml.tokenizer_path)
    model = accelerator.prepare(model)

    # ----- eval hparams -----
    block_size = int(config.eval.get("block_size", 8))
    steps = int(config.eval.get("steps", 128))
    dual_cache = bool(config.eval.get("dual_cache", False))
    alg = str(config.eval.get("alg", "threshold"))
    threshold = float(config.eval.get("threshold", 0.9))
    fixed_k = int(config.eval.get("fixed_k", 1))
    factor = float(config.eval.get("factor", 1.0))
    answer_len = int(config.eval.get("answer_len", int(config.model.get("max_seq_len", 108))))
    eos_tail_mode = str(config.eval.get("eos_tail_mode", "fill"))
    max_samples = int(config.eval.get("max_samples", -1))

    unwrapped_meta = accelerator.unwrap_model(model)
    tail_token_cfg = config.eval.get("tail_token", None)
    if tail_token_cfg is not None:
        tt = str(tail_token_cfg).lower()
        tail_token_id = (int(unwrapped_meta.pad_id) if tt == "pad"
                         else int(tokenizer.eos_token_id) if tt == "eos"
                         else int(tail_token_cfg))
    else:
        tail_token_id = int(tokenizer.eos_token_id)

    logger.info(
        f"params: alg={alg}, block_size={block_size}, steps={steps}, dual_cache={dual_cache}, "
        f"threshold={threshold}, fixed_k={fixed_k}, factor={factor}, "
        f"answer_len={answer_len}, tail_token_id={tail_token_id}, "
        f"eos_tail_mode={eos_tail_mode} "
        f"(use_oracle_length={unwrapped_meta.config.use_oracle_length})"
    )

    metrics, samples = run_eval(
        model=model, dataloader=test_loader, tokenizer=tokenizer,
        accelerator=accelerator, mask_id=mask_id,
        block_size=block_size, steps=steps, dual_cache=dual_cache,
        alg=alg, threshold=threshold, fixed_k=fixed_k, factor=factor,
        answer_len=answer_len, tail_token_id=tail_token_id,
        eos_tail_mode=eos_tail_mode,
        max_samples=max_samples,
    )

    # ----- output dir naming -----
    cache_tag = "dc" if dual_cache else "nc"
    mode_tag = (f"th{str(threshold).replace('.', 'p')}" if alg == "threshold"
                else f"k{fixed_k}" if alg == "fixed_k"
                else f"f{str(factor).replace('.', 'p')}")
    slot_tag = f"l{answer_len}"
    eos_tag = "" if eos_tail_mode == "fill" else f"_eos{eos_tail_mode}"
    if tail_token_id == int(unwrapped_meta.pad_id):
        tail_tag = "tailpad"
    elif tail_token_id == int(tokenizer.eos_token_id):
        tail_tag = "taileos"
    else:
        tail_tag = f"tail{tail_token_id}"
    try:
        ckpt_step = int(str(ckpt_dir).rstrip("/").split("checkpoint-")[-1].split("/")[0])
        suffix = f"_ckpt{ckpt_step}"
    except (ValueError, IndexError):
        suffix = ""

    out_dir = Path(config.experiment.output_dir) / (
        f"eval_dream_{alg}_b{block_size}_s{steps}_{mode_tag}_{cache_tag}_{tail_tag}_{slot_tag}{eos_tag}{suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        **metrics,
        "alg": alg, "block_size": block_size, "steps": steps, "dual_cache": dual_cache,
        "threshold": threshold, "fixed_k": fixed_k, "factor": factor,
        "answer_len": answer_len, "tail_token_id": tail_token_id,
        "eos_tail_mode": eos_tail_mode,
        "ckpt": str(ckpt_dir),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(rec, f, indent=2)
    with open(out_dir / "samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    logger.info(f"[FINAL alg={alg} b={block_size} {mode_tag} dual={dual_cache}] "
                f"WER={metrics['wer']:.2f}%  avg_nfe={metrics['avg_nfe']:.1f}  → {out_dir/'result.json'}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
