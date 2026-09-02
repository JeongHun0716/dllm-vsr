# coding=utf-8
"""Shared evaluation core for the Dream (dLLM) VSR decoder.

Entry point using this module:
  - eval_main.py  : one decoding setting, sharded across GPUs, fast.

Sharding strategy: the test set is split into disjoint strided shards (one per
process, no duplicated samples), each process decodes its shard at batch_size=1,
and per-shard error counts are summed across processes (accelerator.reduce) for
an exact global WER — one `accelerate launch` command, one number.
"""
import os

# Patched Dream backbone (cache_mode controls KV cache use) — set before model import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ["DLLM_VSR_USE_FAST_DREAM"] = "1"

import time
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from accelerate.logging import get_logger
from accelerate.utils import gather_object
from safetensors.torch import load_file as safe_load_file

import editdistance

from training.data import VSRDataset
from models.model import build_model_and_tokenizer
from evaluation.fast_dream_generate import GENERATORS

logger = get_logger(__name__, log_level="INFO")


# ============================================================
# Checkpoint utils
# ============================================================
def find_latest_checkpoint(output_dir: str):
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = [p for p in out.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.name.split("-")[1]))
    return ckpts[-1]


def resolve_checkpoint(config):
    eval_ckpt = config.experiment.get("eval_checkpoint", None)
    if eval_ckpt is not None and str(eval_ckpt).strip() != "":
        ckpt_dir = Path(str(eval_ckpt))
    else:
        ckpt_dir = find_latest_checkpoint(config.experiment.output_dir)
    if ckpt_dir is None or not Path(ckpt_dir).exists():
        raise FileNotFoundError(
            "Checkpoint not found. Set experiment.eval_checkpoint or ensure "
            f"{config.experiment.output_dir}/checkpoint-* exists."
        )
    return Path(ckpt_dir)


def load_checkpoint_weights(model, ckpt_dir: Path):
    trainable_path = ckpt_dir / "trainable_model.safetensors"
    sd = safe_load_file(str(trainable_path))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"Loaded: {trainable_path}  Missing(ok): {len(missing)}  Unexpected: {len(unexpected)}")


# ============================================================
# Build model / sharded data
# ============================================================
def build_model_for_eval(config, accelerator):
    """Build the model + tokenizer, load the checkpoint, move to this process's
    device. Eval has no backward pass, so we skip accelerator.prepare()/DDP — each
    process holds a full replica and decodes its own data shard.
    Returns (model, tokenizer, mask_id, dllm_cfg_yaml, ckpt_dir).
    """
    model, tokenizer, mask_id, dllm_cfg_yaml = build_model_and_tokenizer(config)
    ckpt_dir = resolve_checkpoint(config)
    logger.info(f"[EVAL] Using checkpoint: {ckpt_dir}")
    load_checkpoint_weights(model, ckpt_dir)
    model.to(accelerator.device)
    return model, tokenizer, mask_id, dllm_cfg_yaml, ckpt_dir


def build_sharded_loader(config, accelerator, tokenizer_path, num_workers: int = 0, split: str = "test"):
    """Disjoint strided shard of the eval set for this process (no duplicates)."""
    raw_cfg = config.dataset.get("raw_video", {})
    vsr_cfg = config.dataset.vsr
    manifest_root = str(vsr_cfg.manifest_root)
    dataset = VSRDataset(
        manifest_path=os.path.join(manifest_root, f"{split}.tsv"),
        label_paths=[os.path.join(manifest_root, f"{split}.wrd")],
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

    rank = accelerator.process_index
    world = accelerator.num_processes
    indices = list(range(rank, len(dataset), world))   # strided, disjoint; union = full set
    shard = Subset(dataset, indices)

    loader = DataLoader(
        shard, batch_size=1, shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, len(dataset), len(indices)


def resolve_tail_token_id(config, model, tokenizer):
    """tail_token (pad|eos|int) → token id. Defaults to PAD, which matches the
    stage-2 training distribution (positions beyond EOS are PAD).
    """
    tail_token_cfg = config.eval.get("tail_token", None)
    if tail_token_cfg is None:
        return int(model.pad_id)
    tt = str(tail_token_cfg).lower()
    if tt == "pad":
        return int(model.pad_id)
    if tt == "eos":
        return int(tokenizer.eos_token_id)
    return int(tail_token_cfg)


# ============================================================
# Run one decoding setting
# ============================================================
@torch.no_grad()
def run_one_setting(
    *,
    model,
    loader,
    tokenizer,
    accelerator,
    mask_id: int,
    cache_mode: str,
    alg: str,
    threshold,
    fixed_k: int,
    factor: float,
    block_size: int,
    steps: int,
    answer_len: int,
    tail_token_id: int,
    use_oracle_length: bool,
    max_samples: int = -1,
    log_every: int = 25,
    return_samples: bool = False,
    tag: str = "",
):
    """Decode this process's shard, then sum error counts across processes for an
    exact global WER. Returns (metrics, samples) — samples is [] unless
    return_samples=True (then it is the gathered list from all shards).
    """
    if cache_mode not in GENERATORS:
        raise ValueError(f"cache_mode must be one of {list(GENERATORS)}, got {cache_mode!r}")
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    v_in_dtype = unwrapped.video_input_dtype()
    gen_fn = GENERATORS[cache_mode]

    n_err, n_total, n_done, total_nfe = 0, 0, 0, 0
    local_samples = []
    t0 = time.perf_counter()

    for bidx, batch in enumerate(loader, start=1):
        if max_samples > 0 and bidx > max_samples:
            break
        video_feats = batch.get("video_features", None)
        if video_feats is None:
            continue
        if video_feats.is_floating_point():
            video_feats = video_feats.to(device=accelerator.device, dtype=v_in_dtype)

        instructions = [batch["input_ids"][0].to(accelerator.device)]
        post_user = batch.get("post_user_ids", None)
        if post_user is not None:
            post_user = post_user.to(accelerator.device)

        label_ids = batch["labels"][0]
        label_text = tokenizer.decode(label_ids.tolist(), skip_special_tokens=True).strip()
        cur_gen_length = int(label_ids.numel()) if use_oracle_length else int(answer_len)

        v_enc, len_v, _ = unwrapped.encode_video(video_feats=video_feats, video_pad=None, apply_fc=True)
        _, pred_ids, nfe = gen_fn(
            model=unwrapped.dllm, tokenizer=tokenizer, instructions=instructions,
            v_feats=v_enc[:, :len_v[0]], post_user_prefix=post_user, mask_id=mask_id,
            gen_length=cur_gen_length, steps=steps, block_length=block_size,
            alg=alg, threshold=threshold, fixed_k=fixed_k, factor=factor,
            tail_token_id=tail_token_id,
        )
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

        n_err += editdistance.eval(pred_text.split(), label_text.split())
        n_total += len(label_text.split())
        total_nfe += int(nfe)
        n_done += 1
        if return_samples:
            local_samples.append({"gt": label_text, "pred": pred_text, "nfe": int(nfe)})

        if accelerator.is_main_process and (n_done % log_every == 0):
            wer_mid = 100.0 * n_err / max(1, n_total)
            logger.info(f"[{tag}] rank0 {n_done} done | local WER={wer_mid:.2f}% ({n_err}/{n_total})")

    # Sum per-shard counts across all processes for an exact global WER.
    counts = torch.tensor([n_err, n_total, n_done, total_nfe], dtype=torch.long, device=accelerator.device)
    counts = accelerator.reduce(counts, reduction="sum")
    g_err, g_total, g_done, g_nfe = [int(v) for v in counts.tolist()]

    elapsed = time.perf_counter() - t0
    metrics = {
        "wer": 100.0 * g_err / max(1, g_total),
        "n_err": g_err, "n_total": g_total, "n_samples": g_done,
        "avg_nfe": g_nfe / max(1, g_done),
        "time_sec": elapsed, "sec_per_sample": elapsed / max(1, g_done),
    }
    samples = gather_object(local_samples) if return_samples else []
    return metrics, samples
