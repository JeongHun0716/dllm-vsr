# coding=utf-8
"""Standalone trainer for LenPredictor (length-only sanity check).

Pipeline:
  raw video → USR2 (frozen) → Linear → +pos → prepend <LEN> → Transformer × 2 → head → CE(K)

Uses VSRDataset; the collate adds K (transcript+EOS length) to the batch dict.
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# sys.path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_DLLM_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _DLLM_ROOT)

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from training.utils import get_config, AverageMeter
from models.lr_schedulers import get_scheduler
from training.data import VSRDataset
from training.bucketbatchsampler import MaxFramesGlobalSortBatchShuffleSampler
from models.len_predictor import LenPredictor


logger = get_logger(__name__, log_level="INFO")


def _build_dataset(cfg, manifest_path, label_paths, subset, max_frames):
    """Build VSRDataset; the collate adds K (= labels.numel()) for the length head."""
    raw_cfg = cfg.dataset.get("raw_video", {})
    vsr_cfg = cfg.dataset.vsr
    return VSRDataset(
        manifest_path=manifest_path,
        label_paths=label_paths,
        subset=subset,
        max_video_frames=int(max_frames),
        dllm_tokenizer_path=cfg.model.dllm.tokenizer_path,
        modalities=list(vsr_cfg.modalities),
        video_root=vsr_cfg.get("video_root", None),
        crop_size=int(raw_cfg.get("crop_size", 88)),
        normalize_mean=float(raw_cfg.get("normalize_mean", 0.421)),
        normalize_std=float(raw_cfg.get("normalize_std", 0.165)),
        time_mask_window=int(raw_cfg.get("time_mask_window", 10)),
        time_mask_stride=int(raw_cfg.get("time_mask_stride", 25)),
        train_crop=str(raw_cfg.get("train_crop", "random")),
        enable_time_mask=bool(raw_cfg.get("enable_time_mask", True)),
    )


def main():
    logging.basicConfig(level=logging.INFO)
    config = get_config()

    accelerator = Accelerator(mixed_precision=config.training.mixed_precision)
    if config.training.seed is not None:
        set_seed(config.training.seed)

    # With the DeepSpeed plugin, dynamic batching makes the dataloader batch_size None,
    # which makes accelerator.prepare raise ValueError. Pin a dummy 1; batch_sampler decides the real batch.
    if hasattr(accelerator, "deepspeed_plugin") and accelerator.deepspeed_plugin is not None:
        accelerator.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1

    # ===== output dir / log =====
    output_dir = Path(config.experiment.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, output_dir / "config.yaml")

    # ===== model =====
    logger.info("Building LenPredictor")
    visual_enc_cfg = OmegaConf.to_container(config.model.visual_encoder, resolve=True)
    model = LenPredictor(
        visual_encoder_cfg=visual_enc_cfg,
        enc_dim=int(config.model.enc_dim),
        hidden=int(config.model.hidden),
        n_layers=int(config.model.n_layers),
        n_heads=int(config.model.n_heads),
        ffn_dim=int(config.model.ffn_dim),
        dropout=float(config.model.get("dropout", 0.1)),
        len_max=int(config.model.len_max),
        max_video_frames=int(config.dataset.vsr.max_video_frames) + 8,
    )
    model.print_trainable_params()

    # ===== optional: load weights from existing ckpt (finetune_from) =====
    finetune_from = config.experiment.get("finetune_from", None)
    if finetune_from is not None and str(finetune_from).strip() != "":
        ckpt_path = Path(str(finetune_from)) / "trainable_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"finetune_from ckpt not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        # `missing` should only contain frozen v_encoder.* keys (those are excluded at save time).
        non_v_missing = [k for k in missing if not k.startswith("v_encoder.")]
        if accelerator.is_main_process:
            logger.info(f"[finetune_from] loaded {ckpt_path}  unexpected={len(unexpected)}  non-v_encoder missing={len(non_v_missing)}")

    # ===== optimizer / scheduler =====
    opt_cfg = config.optimizer.params
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt_cfg.learning_rate,
        betas=(opt_cfg.beta1, opt_cfg.beta2),
        weight_decay=opt_cfg.weight_decay,
        eps=opt_cfg.epsilon,
    )

    max_train_steps = config.training.max_train_steps * accelerator.num_processes
    warmup_steps = config.lr_scheduler.params.warmup_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    # ===== dataset =====
    manifest_root = str(config.dataset.vsr.manifest_root)
    train_manifest = os.path.join(manifest_root, "train.tsv")
    train_labels = [os.path.join(manifest_root, "train.wrd")]
    test_manifest = os.path.join(manifest_root, "test.tsv")
    test_labels = [os.path.join(manifest_root, "test.wrd")]

    train_ds = _build_dataset(config, train_manifest, train_labels, "train",
                              config.dataset.vsr.max_video_frames)
    test_ds = _build_dataset(config, test_manifest, test_labels, "test",
                             config.dataset.vsr.max_video_frames)

    train_sampler = MaxFramesGlobalSortBatchShuffleSampler(
        lengths=train_ds.lengths,
        max_tokens=int(config.training.max_tokens),
        max_batch_size=config.training.get("max_batch_size", None),
        shuffle_batches=True,
        drop_last=False,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42,
        cost_pair=bool(config.training.get("cost_pair", True)),
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        collate_fn=train_ds.collate_fn,
        num_workers=config.dataset.dataloader.num_workers,
        pin_memory=True,
    )

    test_sampler = MaxFramesGlobalSortBatchShuffleSampler(
        lengths=test_ds.lengths,
        max_tokens=int(config.training.max_tokens),
        max_batch_size=config.training.get("max_batch_size", None),
        shuffle_batches=False,
        drop_last=False,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42,
        cost_pair=bool(config.training.get("cost_pair", True)),
    )
    test_loader = DataLoader(
        test_ds, batch_sampler=test_sampler,
        collate_fn=test_ds.collate_fn,
        num_workers=config.dataset.dataloader.num_workers,
        pin_memory=True,
    )

    accelerator.even_batches = False
    # The DeepSpeed plugin requires a fixed dataloader batch_size, but our batch_sampler
    # uses dynamic batching so batch_size is None and prepare would fail. Skip the dataloader
    # in prepare and pass only model/optimizer/scheduler (same approach as training/train.py).
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # ===== training =====
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Dynamic batch: max_tokens per device = {config.training.max_tokens}")

    global_step = 0
    epoch = 0
    save_every = int(config.experiment.save_every)
    eval_every = int(config.experiment.eval_every)
    log_every = int(config.experiment.log_every)
    v_in_dtype = accelerator.unwrap_model(model).video_input_dtype()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()     # K accuracy (top-1)
    acc1_meter = AverageMeter()    # K within ±1
    acc3_meter = AverageMeter()    # K within ±3
    acc5_meter = AverageMeter()    # K within ±5

    end = time.time()

    while global_step < config.training.max_train_steps:
        model.train()
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        for batch in train_loader:
            video_feats = batch.get("video_features", None)
            video_pad = batch.get("video_padding_mask", None)
            if video_feats is None:
                continue
            if video_feats.is_floating_point():
                video_feats = video_feats.to(device=accelerator.device, dtype=v_in_dtype)
            if video_pad is not None:
                video_pad = video_pad.to(accelerator.device)

            # K target — batch dict "K" is a list of ints (collate keeps it as a list).
            K_list = batch.get("K", None)
            if K_list is None:
                # fallback: labels (in len_only mode that's a 1-token [len_id]) cannot yield K, skip
                continue
            K_target = torch.tensor(
                [int(k) for k in K_list], device=accelerator.device, dtype=torch.long
            )

            logits, loss = model(video_feats=video_feats, video_pad=video_pad, K_target=K_target)

            # metrics
            with torch.no_grad():
                pred_K = logits.argmax(dim=-1) + 1                     # 1..len_max
                diff = (pred_K - K_target).abs()
                acc = (diff == 0).float().mean().item()
                acc1 = (diff <= 1).float().mean().item()
                acc3 = (diff <= 3).float().mean().item()
                acc5 = (diff <= 5).float().mean().item()
            n_b = int(video_feats.size(0))
            loss_meter.update(float(loss.detach().item()), n=n_b)
            acc_meter.update(acc, n=n_b)
            acc1_meter.update(acc1, n=n_b)
            acc3_meter.update(acc3, n=n_b)
            acc5_meter.update(acc5, n=n_b)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), float(config.training.max_grad_norm))
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % log_every == 0 and accelerator.is_main_process:
                logger.info(
                    f"Step: {global_step}  Loss: {loss_meter.avg:.4f}  "
                    f"Acc(=K): {acc_meter.avg*100:.2f}%  "
                    f"Acc(±1): {acc1_meter.avg*100:.2f}%  "
                    f"Acc(±3): {acc3_meter.avg*100:.2f}%  "
                    f"Acc(±5): {acc5_meter.avg*100:.2f}%  "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.6g}  "
                    f"BSZ(local): {int(video_feats.size(0))}  "
                    f"Time(step): {(time.time()-end)/log_every:.2f}s"
                )
                loss_meter.reset(); acc_meter.reset()
                acc1_meter.reset(); acc3_meter.reset(); acc5_meter.reset()
                end = time.time()

            if global_step % save_every == 0 and accelerator.is_main_process:
                ckpt = output_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                state = accelerator.unwrap_model(model).state_dict()
                # Save only trainable params, excluding the frozen USR2 encoder.
                state = {k: v for k, v in state.items() if not k.startswith("v_encoder.")}
                torch.save(state, ckpt / "trainable_model.pt")
                with open(ckpt / "metadata.json", "w") as f:
                    json.dump({"global_step": global_step, "epoch": epoch}, f)
                logger.info(f"[save] {ckpt}")

            if global_step % eval_every == 0:
                _evaluate(accelerator, model, test_loader, config, global_step, v_in_dtype)
                model.train()

            if global_step >= config.training.max_train_steps:
                break

        epoch += 1

    # final eval
    _evaluate(accelerator, model, test_loader, config, global_step, v_in_dtype)


@torch.no_grad()
def _evaluate(accelerator, model, test_loader, config, step, v_in_dtype):
    """Evaluate on FULL test set by gathering across all DDP ranks."""
    model.eval()
    # local counters (per rank)
    local_total = 0
    local_correct = 0
    local_within1 = 0
    local_within3 = 0
    local_within5 = 0
    local_sum_abs = 0.0
    local_loss_sum = 0.0
    for batch in test_loader:
        video_feats = batch.get("video_features", None)
        video_pad = batch.get("video_padding_mask", None)
        if video_feats is None:
            continue
        if video_feats.is_floating_point():
            video_feats = video_feats.to(device=accelerator.device, dtype=v_in_dtype)
        if video_pad is not None:
            video_pad = video_pad.to(accelerator.device)
        K_list = batch.get("K", None)
        if K_list is None:
            continue
        K_target = torch.tensor(
            [int(k) for k in K_list], device=accelerator.device, dtype=torch.long
        )
        logits, loss = model(video_feats=video_feats, video_pad=video_pad, K_target=K_target)
        pred_K = logits.argmax(dim=-1) + 1
        diff = (pred_K - K_target).abs()
        local_total += K_target.numel()
        local_correct += int((diff == 0).sum().item())
        local_within1 += int((diff <= 1).sum().item())
        local_within3 += int((diff <= 3).sum().item())
        local_within5 += int((diff <= 5).sum().item())
        local_sum_abs += float(diff.sum().item())
        local_loss_sum += float(loss.item()) * K_target.numel()

    # gather across ranks: sum scalar counts
    stats = torch.tensor(
        [local_total, local_correct, local_within1, local_within3, local_within5, local_sum_abs, local_loss_sum],
        device=accelerator.device, dtype=torch.float64,
    )
    if accelerator.num_processes > 1:
        stats = accelerator.reduce(stats, reduction="sum")
    total = int(stats[0].item())
    correct = int(stats[1].item())
    within1 = int(stats[2].item())
    within3 = int(stats[3].item())
    within5 = int(stats[4].item())
    sum_abs = float(stats[5].item())
    loss_sum = float(stats[6].item())

    if accelerator.is_main_process and total > 0:
        logger.info(
            f"[EVAL @ {step}]  total={total}  loss={loss_sum/total:.4f}  "
            f"Acc(=K): {correct/total*100:.2f}%  "
            f"Acc(±1): {within1/total*100:.2f}%  "
            f"Acc(±3): {within3/total*100:.2f}%  "
            f"Acc(±5): {within5/total*100:.2f}%  "
            f"MAE: {sum_abs/total:.2f}"
        )


if __name__ == "__main__":
    main()
