# coding=utf-8
# Copyright 2025 MMaDA Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
# Add dllm-vsr/ to sys.path so `models.*`, `training.*`, `avhubert.*` are importable.
_HERE = os.path.dirname(os.path.abspath(__file__))                  # dllm-vsr/training
_DLLM_ROOT = os.path.dirname(_HERE)                                  # dllm-vsr/
sys.path.insert(0, _DLLM_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import time
from pathlib import Path

from omegaconf import OmegaConf
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.utils import get_config, flatten_omega_conf, AverageMeter
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from models.model import build_model_and_tokenizer
from training.data import VSRDataset
from training.bucketbatchsampler import MaxFramesGlobalSortBatchShuffleSampler

from torch.utils.data import DataLoader

from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers.modeling_utils import load_sharded_checkpoint


logger = get_logger(__name__, log_level="INFO")


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    accelerator.even_batches = False

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        # Dynamic sampler decides batch size; pin the DeepSpeed nominal field to 1.
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
        # accelerate YAML hard-codes deepspeed_config.gradient_accumulation_steps=1, so
        # Accelerator(gradient_accumulation_steps=N) is not propagated to the DS engine — sync explicitly.
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = (
            int(config.training.gradient_accumulation_steps)
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    model, tokenizer, mask_id, dllm_cfg_yaml = build_model_and_tokenizer(config)

    # ===== Freeze all, then enable only LoRA + adapters =====
    for p in model.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    for m in [model.video_downsampler, model.fc_v]:
        for p in m.parameters():
            p.requires_grad = True

    model.print_trainable_params()

    ##################################
    #   Optimizer and LR scheduler   #
    ##################################
    optimizer_config = config.optimizer.params

    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # LR scheduler step count: summed across all processes.
    max_train_steps = config.training.max_train_steps * accelerator.num_processes
    warmup_steps = config.lr_scheduler.params.warmup_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    ##################################
    #         DATALOADER             #
    ##################################
    logger.info("Creating dataloaders and lr_scheduler")
    dataset_config = config.dataset.dataloader

    raw_cfg = config.dataset.get("raw_video", {})
    vsr_cfg = config.dataset.vsr

    # Simplified schema: dataset.vsr.{manifest_root, max_video_frames, modalities}
    # -> manifest_root/{train,test}.tsv, labels are .wrd files alongside the .tsv.
    import os as _os
    manifest_root = str(vsr_cfg.manifest_root)
    common_max_frames = int(vsr_cfg.max_video_frames)
    common_modalities = list(vsr_cfg.modalities)
    train_manifest_path = _os.path.join(manifest_root, "train.tsv")
    train_label_paths = [_os.path.join(manifest_root, "train.wrd")]
    test_manifest_path = _os.path.join(manifest_root, "test.tsv")
    test_label_paths = [_os.path.join(manifest_root, "test.wrd")]

    common_ds_kwargs = dict(
        dllm_tokenizer_path=dllm_cfg_yaml.tokenizer_path,
        modalities=common_modalities,
        video_root=vsr_cfg.get("video_root", None),
        crop_size=int(raw_cfg.get("crop_size", 88)),
        normalize_mean=float(raw_cfg.get("normalize_mean", 0.421)),
        normalize_std=float(raw_cfg.get("normalize_std", 0.165)),
        time_mask_window=int(raw_cfg.get("time_mask_window", 10)),
        time_mask_stride=int(raw_cfg.get("time_mask_stride", 25)),
        train_crop=str(raw_cfg.get("train_crop", "random")),
        enable_time_mask=bool(raw_cfg.get("enable_time_mask", True)),
    )

    def _build_dataset(manifest_path, label_paths, subset, max_frames):
        return VSRDataset(
            manifest_path=manifest_path,
            label_paths=label_paths,
            subset=subset,
            max_video_frames=int(max_frames),
            **common_ds_kwargs,
        )

    def build_train_loader(max_frames: int):
        ds = _build_dataset(
            manifest_path=train_manifest_path,
            label_paths=train_label_paths,
            subset="train",
            max_frames=max_frames,
        )
        bs = MaxFramesGlobalSortBatchShuffleSampler(
            lengths=ds.lengths,
            max_tokens=int(config.training.max_tokens),
            max_batch_size=config.training.get("max_batch_size", None),
            shuffle_batches=True,
            drop_last=False,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
            cost_pair=bool(config.training.get("cost_pair", True)),
        )
        loader = DataLoader(
            ds,
            batch_sampler=bs,
            collate_fn=ds.collate_fn,
            num_workers=dataset_config.num_workers,
            pin_memory=True,
        )
        return loader, ds

    # Curriculum maps epoch_idx -> max_video_frames; when disabled, uses dataset.vsr.max_video_frames.
    cur_cfg = vsr_cfg.get("curriculum", None)
    curriculum_enabled = bool(cur_cfg is not None and cur_cfg.get("enabled", False))
    if curriculum_enabled:
        curriculum_schedule = [int(x) for x in cur_cfg.schedule]
        logger.info(f"[curriculum] enabled, schedule={curriculum_schedule}")
    else:
        curriculum_schedule = None
    default_train_max = common_max_frames

    def get_max_frames_for_epoch(epoch_idx: int) -> int:
        if curriculum_schedule is None:
            return default_train_max
        return curriculum_schedule[min(epoch_idx, len(curriculum_schedule) - 1)]

    # Train dataloader is built after the RESUME metadata is parsed (we need first_epoch first).
    train_dataloader_avsr = None
    dataset = None
    current_max_frames = None

    test_dataset = _build_dataset(
        manifest_path=test_manifest_path,
        label_paths=test_label_paths,
        subset="test",
        max_frames=common_max_frames,
    )

    test_batch_sampler = MaxFramesGlobalSortBatchShuffleSampler(
        lengths=test_dataset.lengths,
        max_tokens=int(config.training.max_tokens),
        max_batch_size=config.training.get("max_batch_size", None),
        shuffle_batches=False,
        drop_last=False,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42,
        cost_pair=bool(config.training.get("cost_pair", True)),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        collate_fn=test_dataset.collate_fn,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
    )

    #################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    def find_latest_checkpoint(output_dir: str):
        out = Path(output_dir)
        if not out.exists():
            return None
        ckpts = [p for p in out.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
        if not ckpts:
            return None
        ckpts.sort(key=lambda p: int(p.name.split("-")[1]))
        return ckpts[-1]

    # (A0) finetune_from: only honored when resume_from_checkpoint=False.
    # Loads trainable weights (LoRA + adapters) from another run as the starting point.
    # global_step/epoch start at 0 (LR scheduler / optimizer state are fresh).
    finetune_from = config.experiment.get("finetune_from", None)
    finetune_ckpt_dir = None
    if not config.experiment.resume_from_checkpoint and finetune_from:
        finetune_ckpt_dir = Path(str(finetune_from))
        if not finetune_ckpt_dir.exists():
            raise FileNotFoundError(f"finetune_from path not found: {finetune_ckpt_dir}")
        logger.info(f"[finetune_from] will load weights from: {finetune_ckpt_dir}")

    # (A) Read ckpt metadata (step, epoch_idx) first so we can pick first_epoch.
    ckpt_dir = None
    if config.experiment.resume_from_checkpoint:
        ckpt_dir = find_latest_checkpoint(config.experiment.output_dir)
        logger.info(f"Latest checkpoint dir: {ckpt_dir}")
        if ckpt_dir is not None:
            global_step = int(ckpt_dir.name.split("-")[1])
            meta_path = ckpt_dir / "metadata.json"
            epoch_from_meta = None
            if meta_path.exists():
                try:
                    with meta_path.open("r") as f:
                        meta = json.load(f)
                    ev = meta.get("epoch_idx", None)
                    if ev is not None:
                        epoch_from_meta = int(ev)
                except Exception:
                    epoch_from_meta = None
            first_epoch = epoch_from_meta if epoch_from_meta is not None else 0
            logger.info(
                f"[RESUME meta] global_step={global_step} epoch_idx={first_epoch} "
                f"(epoch_from_meta={'found' if epoch_from_meta is not None else 'missing'})"
            )
        else:
            logger.info("resume_from_checkpoint=True but no checkpoint-* dirs found.")
    else:
        logger.info("Not resuming from checkpoint")

    # (B) Build dataloader using max_frames for first_epoch.
    init_max = get_max_frames_for_epoch(first_epoch)
    train_dataloader_avsr, dataset = build_train_loader(init_max)
    current_max_frames = init_max
    print(
        "process index : ",
        accelerator.process_index, ", ", accelerator.num_processes,
        "Length: ", len(dataset),
    )
    if accelerator.is_main_process:
        logger.info(
            f"[loader/init] first_epoch={first_epoch} max_frames={init_max} "
            f"n_samples={len(dataset)} curriculum={'on' if curriculum_enabled else 'off'}"
        )

    # (C) Load model state dict if a ckpt exists. When finetune_from is set and
    # resume_from_checkpoint=False, that ckpt is used as initial weights.
    if ckpt_dir is None and finetune_ckpt_dir is not None:
        ckpt_dir = finetune_ckpt_dir
        logger.info(f"[finetune_from] using {ckpt_dir} as initial weights (step=0)")
    if ckpt_dir is not None:
        trainable_path = ckpt_dir / "trainable_model.safetensors"
        if trainable_path.exists():
            sd = safe_load_file(str(trainable_path))
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f"Loaded trainable checkpoint: {trainable_path}")
            logger.info(f"Missing keys (ok): {len(missing)} | Unexpected keys: {len(unexpected)}")
        else:
            unwrapped_dir = ckpt_dir / "unwrapped_model"
            if not unwrapped_dir.exists():
                raise FileNotFoundError(
                    f"No supported checkpoint format found in {ckpt_dir}."
                )
            pt_bin = unwrapped_dir / "pytorch_model.bin"
            if pt_bin.exists():
                state_dict = torch.load(str(pt_bin), map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict, strict=True)
                del state_dict
                logger.info(f"Loaded full checkpoint: {pt_bin}")
            elif (unwrapped_dir / "pytorch_model.bin.index.json").exists() \
                    or (unwrapped_dir / "model.safetensors.index.json").exists():
                load_sharded_checkpoint(model, str(unwrapped_dir))
                logger.info(f"Loaded sharded checkpoint from: {unwrapped_dir}")
            else:
                raise FileNotFoundError(
                    f"Unsupported checkpoint contents in {unwrapped_dir}."
                )

    ##################################
    #       Prepare accelerator      #
    ##################################
    logger.info("Preparing model, and optimizer")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    if config.experiment.resume_from_checkpoint and ckpt_dir is not None:
        # Fast-forward the LR scheduler by replaying what the train loop does once per step.
        # AcceleratedScheduler internally steps the underlying scheduler num_processes times,
        # so calling step() global_step times restores the LR to the resume-point value.
        # Optimizer state (Adam 1st/2nd moments) is NOT restored — empirically still stable.
        if global_step > 0:
            for _ in range(global_step):
                lr_scheduler.step()
            logger.info(
                f"[RESUME] LR scheduler fast-forwarded to step {global_step} "
                f"(LR={lr_scheduler.get_last_lr()[0]:.6g})"
            )
        logger.info("[RESUME] Loaded model weights + LR position. Optimizer state restarts.")

    ##################################
    #             Training           #
    ##################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Dynamic batch: max_tokens per device = {config.training.max_tokens}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    running_correct = 0
    running_masked = 0
    running_answer = 0
    running_loss_sum = 0.0
    running_bsz_sum = 0.0
    last_batch_cost_local = 0.0  # per-rank batch cost of last step; gathered every log_every

    epoch = first_epoch
    while global_step < config.training.max_train_steps:
        # Curriculum: pick max_frames at epoch start; rebuild dataloader on change.
        desired_max = get_max_frames_for_epoch(epoch)
        if desired_max != current_max_frames:
            train_dataloader_avsr, dataset = build_train_loader(desired_max)
            current_max_frames = desired_max
            if accelerator.is_main_process:
                logger.info(
                    f"[curriculum] epoch={epoch} max_frames={desired_max} n_samples={len(dataset)}"
                )

        model.train()
        if hasattr(train_dataloader_avsr, "batch_sampler") and hasattr(train_dataloader_avsr.batch_sampler, "set_epoch"):
            train_dataloader_avsr.batch_sampler.set_epoch(epoch)
        v_in_dtype = accelerator.unwrap_model(model).video_input_dtype()
        for batch_idx, batch in enumerate(train_dataloader_avsr):
            video_feats = batch.get("video_features", None)            # [B, C, T, H, W] or None
            video_pad   = batch.get("video_padding_mask", None)        # [B, T] True=pad or None

            if video_feats is not None and video_feats.is_floating_point():
                video_feats = video_feats.to(device=accelerator.device, dtype=v_in_dtype)
            if video_pad is not None:
                video_pad = video_pad.to(accelerator.device)

            # batch cost (pad_max_sq) — proxy for VRAM use, computed on video_features.
            if video_feats is not None:
                _B, _T = video_feats.size(0), video_feats.size(1)
                last_batch_cost_local = float(_B * _T * _T)
            else:
                last_batch_cost_local = 0.0

            # 1) Encode video
            v_enc, len_v = None, None
            if video_feats is not None:
                v_enc, len_v, _ = model.encode_video(
                    video_feats=video_feats,
                    video_pad=video_pad,
                    apply_fc=True,
                )

            # 2) Build instruction/labels list
            instructions = [x.to(accelerator.device) for x in batch["input_ids"]]
            labels_list = batch.get("labels", None)
            if labels_list is not None:
                labels_list = [x.to(accelerator.device) for x in labels_list]
            post_user = batch.get("post_user_ids", None)
            if post_user is not None:
                post_user = post_user.to(accelerator.device)

            # 3) Make dllm tensors
            dllm_inputs, attention_mask, prompt_mask, dllm_labels = model.prepare_dllm_inputs_labels(
                tokenizer=tokenizer,
                instructions=instructions,
                v_feats=v_enc,
                len_v=len_v,
                labels=labels_list,
                post_user_prefix=post_user,
            )

            # 4) Corrupt only answer region (masking)
            noisy_inputs, labels_m, p_mask, answer_lengths, masked_indices = model.corrupt_answer_region(
                dllm_inputs=dllm_inputs,
                prompt_mask=prompt_mask,
                labels=dllm_labels,
                mask_id=mask_id,
                eps=1e-3,
                eos_token_id=tokenizer.eos_token_id,
            )

            with accelerator.accumulate(model):
                logits, loss = model(
                    inputs_embeds=noisy_inputs,
                    labels=labels_m,
                    masked_indices=masked_indices,
                    p_mask=p_mask,
                    attention_mask=attention_mask,
                    prompt_mask=prompt_mask,
                )

                if "input_ids" in batch:
                    local_bsz = len(batch["input_ids"])
                elif video_feats is not None:
                    local_bsz = int(video_feats.size(0))
                else:
                    raise RuntimeError("Cannot infer local batch size")

                running_loss_sum += float(loss.detach().item()) * float(local_bsz)
                running_bsz_sum += float(local_bsz)

                # ---- per-step running accuracy (no comm) ----
                with torch.no_grad():
                    pred_ids = logits.argmax(dim=-1)
                    eos_id = tokenizer.eos_token_id
                    if eos_id is None:
                        eos_id = config.model.eos_token_id
                    pad_id = tokenizer.pad_token_id

                    label_pos = (labels_m != -100)
                    answer_pos = (~prompt_mask.bool()) & (attention_mask == 1) & label_pos
                    mask_pos = masked_indices.bool() & label_pos

                    B, L = labels_m.shape
                    pos = torch.arange(L, device=labels_m.device).unsqueeze(0).expand(B, L)

                    eos_mask = answer_pos & (labels_m == pad_id)
                    INF = L
                    eos_idx = torch.where(eos_mask, pos, torch.full_like(pos, INF)).min(dim=1).values
                    eos_idx = torch.clamp(eos_idx, max=L-1)
                    upto_eos = pos <= eos_idx.unsqueeze(1)

                    valid_pos = answer_pos & mask_pos & upto_eos
                    answer_upto = answer_pos & upto_eos

                    masked_in_answer = valid_pos.sum()
                    answer_total = answer_upto.sum()
                    correct = ((pred_ids == labels_m) & valid_pos).sum()

                    running_correct += int(correct.item())
                    running_masked += int(masked_in_answer.item())
                    running_answer += int(answer_total.item())

                accelerator.backward(loss)
                do_update = accelerator.sync_gradients
                if do_update:
                    # NaN/Inf guard: check grad norm and skip the step if it blew up.
                    # DeepSpeed returns a float, plain DDP may return a Tensor.
                    grad_norm_val = accelerator.clip_grad_norm_(
                        model.parameters(),
                        config.training.max_grad_norm if config.training.max_grad_norm is not None else 1e9,
                    )
                    gn = float(grad_norm_val.item()) if torch.is_tensor(grad_norm_val) else float(grad_norm_val)
                    skip_step = math.isnan(gn) or math.isinf(gn)
                    if skip_step:
                        if accelerator.is_main_process:
                            logger.warning(f"[NaN guard] step={global_step+1}: grad_norm={gn:.4g}, skipping update")
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if (
                        global_step % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                    ):
                        log_grad_norm(model, accelerator, global_step)

            if do_update:
                batch_time_m.update(time.time() - end)
                end = time.time()

                do_log = (global_step % config.experiment.log_every == 0)

                if do_log:
                    epoch_float = epoch + (batch_idx + 1) / max(1, len(train_dataloader_avsr))

                    correct_t = torch.tensor([running_correct], device=accelerator.device, dtype=torch.long)
                    masked_t = torch.tensor([running_masked], device=accelerator.device, dtype=torch.long)
                    answer_t = torch.tensor([running_answer], device=accelerator.device, dtype=torch.long)
                    loss_sum_t = torch.tensor([running_loss_sum], device=accelerator.device, dtype=torch.float32)
                    bsz_sum_t = torch.tensor([running_bsz_sum], device=accelerator.device, dtype=torch.float32)

                    correct_all = accelerator.gather(correct_t).sum()
                    masked_all = accelerator.gather(masked_t).sum()
                    answer_all = accelerator.gather(answer_t).sum()
                    loss_sum_all = accelerator.gather(loss_sum_t).sum()
                    bsz_sum_all = accelerator.gather(bsz_sum_t).sum()

                    masked_acc_running = (correct_all.float() / masked_all.clamp(min=1).float()).item()
                    mask_ratio_running = (masked_all.float() / answer_all.clamp(min=1).float()).item()
                    avg_loss = (loss_sum_all / bsz_sum_all.clamp(min=1)).item()

                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * local_bsz / batch_time_m.val
                    )
                    bsz_t = torch.tensor([local_bsz], device=accelerator.device, dtype=torch.long)
                    bsz_all = accelerator.gather(bsz_t).flatten()  # [num_processes]
                    global_bsz = int(bsz_all.sum().item())
                    bsz_min = int(bsz_all.min().item())
                    bsz_max = int(bsz_all.max().item())
                    grad_accum = int(config.training.gradient_accumulation_steps)
                    effective_bsz = global_bsz * grad_accum  # total utts per loss update
                    samples_per_second_global = grad_accum * global_bsz / batch_time_m.val

                    # Batch cost distribution — diagnoses load balance across DDP ranks.
                    cost_t = torch.tensor([last_batch_cost_local], device=accelerator.device, dtype=torch.float32)
                    all_costs = accelerator.gather(cost_t).flatten()
                    cost_min = float(all_costs.min().item())
                    cost_max = float(all_costs.max().item())
                    cost_mean = float(all_costs.mean().item())
                    cost_spread = cost_max - cost_min
                    cost_imbalance = (cost_spread / cost_max) if cost_max > 0 else 0.0

                    # VRAM peak over the log window; reset at the end of this window.
                    vram_peak_local_gb = torch.cuda.max_memory_allocated(accelerator.device) / 1e9
                    vram_peak_t = torch.tensor([vram_peak_local_gb], device=accelerator.device, dtype=torch.float32)
                    vram_peaks = accelerator.gather(vram_peak_t).flatten()
                    vram_peak_max = float(vram_peaks.max().item())
                    torch.cuda.reset_peak_memory_stats(accelerator.device)

                    logs = {
                        "epoch": epoch_float,
                        "step_loss": avg_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_global,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                        "mask_ratio_running": mask_ratio_running,
                        "masked_acc_running": masked_acc_running,
                        "batch_cost/mean": cost_mean,
                        "batch_cost/max": cost_max,
                        "batch_cost/min": cost_min,
                        "batch_cost/spread": cost_spread,
                        "batch_cost/imbalance": cost_imbalance,  # 0~1, 0=balanced
                        "bsz/local": local_bsz,            # utts on this rank
                        "bsz/local_min": bsz_min,           # min across ranks
                        "bsz/local_max": bsz_max,           # max across ranks
                        "bsz/global": global_bsz,           # sum across ranks (= utts per step)
                        "bsz/effective": effective_bsz,     # x grad_accum (one loss update)
                        "vram/peak_gb_max": vram_peak_max,  # max VRAM peak across ranks during window
                    }
                    accelerator.log(logs, step=global_step)

                    # (EARLY_STOP rule disabled — long runs proceed indefinitely)

                    if accelerator.is_main_process:
                        logger.info(
                            f"Epoch: {epoch_float:.3f} "
                            f"Step: {global_step} "
                            f"Loss: {avg_loss:0.4f} "
                            f"MaskRatio(r): {mask_ratio_running*100:0.2f}% "
                            f"MaskedAcc(r): {masked_acc_running*100:0.2f}% "
                            f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_m.val:0.4f} "
                            f"LR: {lr_scheduler.get_last_lr()[0]:0.6f} "
                            f"BSZ(local/global/eff): {local_bsz}/{global_bsz}/{effective_bsz} "
                            f"BCost(mean/max/imbal): {cost_mean/1e3:0.1f}k/{cost_max/1e3:0.1f}k/{cost_imbalance*100:0.1f}% "
                            f"VRAM(peak): {vram_peak_max:0.1f}GB"
                        )

                    if accelerator.is_main_process:
                        b = 0
                        with torch.no_grad():
                            pred_ids = logits.argmax(dim=-1)
                            eos_id = tokenizer.eos_token_id
                            if eos_id is None:
                                eos_id = config.model.eos_token_id
                            MAX_PRINT = 200

                            ans_idx = torch.nonzero(
                                (~prompt_mask[b].bool()) & (attention_mask[b] == 1),
                                as_tuple=False
                            ).squeeze(1)

                            if ans_idx.numel() == 0:
                                logger.info(f"[SAMPLE@{global_step}] (no answer tokens)")
                            else:
                                gt_full = labels_m[b, ans_idx]
                                pred_full = pred_ids[b, ans_idx]
                                mask_full = masked_indices[b, ans_idx].bool()

                                eos_where = torch.nonzero(gt_full == eos_id, as_tuple=False)
                                if eos_where.numel() > 0:
                                    k = int(eos_where[0].item()) + 1
                                else:
                                    k = min(gt_full.size(0), MAX_PRINT)

                                gt_ids = gt_full[:k]
                                pred_ans_ids = pred_full[:k]
                                local_mask = mask_full[:k]

                                inp_ids = gt_ids.clone()
                                mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else mask_id
                                inp_ids[local_mask] = mask_token_id

                                logger.info(f"[SAMPLE@{global_step}] Input : {tokenizer.decode(inp_ids.tolist())}")
                                logger.info(f"[SAMPLE@{global_step}] Label : {tokenizer.decode(gt_ids.tolist())}")
                                logger.info(f"[SAMPLE@{global_step}] Pred  : {tokenizer.decode(pred_ans_ids.tolist())}")

                    batch_time_m.reset()
                    data_time_m.reset()
                    running_correct = 0
                    running_masked = 0
                    running_answer = 0
                    running_loss_sum = 0.0
                    running_bsz_sum = 0.0

                if global_step % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step, epoch_idx=epoch)
                    test_logs = evaluate_masked_acc(
                        model=model,
                        test_dataloader=test_dataloader,
                        tokenizer=tokenizer,
                        config=config,
                        mask_id=mask_id,
                        accelerator=accelerator,
                    )
                    accelerator.log(test_logs, step=global_step)

                    if accelerator.is_main_process:
                        logger.info(
                            f"[TEST@{global_step}] masked_acc={test_logs['test/masked_acc']*100:.2f}% "
                            f"mask_ratio={test_logs['test/mask_ratio']*100:.2f}%"
                        )
                    model.train()

            if global_step >= config.training.max_train_steps:
                break
        epoch += 1

    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, global_step, epoch_idx=epoch)
    accelerator.end_training()


def save_checkpoint(model, config, accelerator, global_step, epoch_idx=None):
    """Save only trainable params (LoRA + adapters) as a safetensors file."""
    output_dir = config.experiment.output_dir
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped = accelerator.unwrap_model(model)
    trainable_sd = {}

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        import deepspeed
        for name, p in unwrapped.named_parameters():
            if not p.requires_grad:
                continue
            with deepspeed.zero.GatheredParameters([p], modifier_rank=0):
                if accelerator.is_main_process:
                    trainable_sd[name] = p.detach().cpu().clone()
    else:
        sd = unwrapped.state_dict()
        for name, p in unwrapped.named_parameters():
            if p.requires_grad and name in sd:
                trainable_sd[name] = sd[name].detach().cpu()

    if accelerator.is_main_process:
        safe_save_file(trainable_sd, str(save_path / "trainable_model.safetensors"))
        with (save_path / "metadata.json").open("w") as f:
            meta = {
                "global_step": int(global_step),
                "num_trainable_tensors": len(trainable_sd),
            }
            if epoch_idx is not None:
                meta["epoch_idx"] = int(epoch_idx)
            json.dump(meta, f, indent=2)
    accelerator.wait_for_everyone()


@torch.no_grad()
def evaluate_masked_acc(
    model,
    test_dataloader,
    tokenizer,
    config,
    mask_id,
    accelerator,
):
    model.eval()

    total_correct = 0
    total_masked = 0
    total_answer = 0

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = config.model.eos_token_id

    v_in_dtype = accelerator.unwrap_model(model).video_input_dtype()

    for batch in test_dataloader:
        video_feats = batch.get("video_features", None)
        video_pad = batch.get("video_padding_mask", None)

        if video_feats is not None and video_feats.is_floating_point():
            video_feats = video_feats.to(device=accelerator.device, dtype=v_in_dtype)
        if video_pad is not None:
            video_pad = video_pad.to(accelerator.device)

        v_enc, len_v = None, None
        if video_feats is not None:
            v_enc, len_v, _ = model.encode_video(video_feats=video_feats, video_pad=video_pad, apply_fc=True)

        instructions = [x.to(accelerator.device) for x in batch["input_ids"]]
        labels_list = batch.get("labels", None)
        if labels_list is not None:
            labels_list = [x.to(accelerator.device) for x in labels_list]
        post_user = batch.get("post_user_ids", None)
        if post_user is not None:
            post_user = post_user.to(accelerator.device)

        dllm_inputs, attention_mask, prompt_mask, dllm_labels = model.prepare_dllm_inputs_labels(
            tokenizer=tokenizer,
            instructions=instructions,
            v_feats=v_enc,
            len_v=len_v,
            labels=labels_list,
            post_user_prefix=post_user,
        )

        noisy_inputs, labels_m, p_mask, answer_lengths, masked_indices = model.corrupt_answer_region(
            dllm_inputs=dllm_inputs,
            prompt_mask=prompt_mask,
            labels=dllm_labels,
            mask_id=mask_id,
            eps=1e-3,
            force_mask_prob=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )

        logits, loss = model(
            inputs_embeds=noisy_inputs,
            labels=labels_m,
            masked_indices=masked_indices,
            p_mask=p_mask,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
        )

        pred_ids = logits.argmax(dim=-1)
        pad_id = tokenizer.pad_token_id

        label_pos = (labels_m != -100)
        answer_pos = (~prompt_mask.bool()) & (attention_mask == 1) & label_pos
        mask_pos = masked_indices.bool() & label_pos

        B, L = labels_m.shape
        pos = torch.arange(L, device=labels_m.device).unsqueeze(0).expand(B, L)

        eos_mask = answer_pos & (labels_m == pad_id)
        INF = L
        eos_idx = torch.where(eos_mask, pos, torch.full_like(pos, INF)).min(dim=1).values
        eos_idx = torch.clamp(eos_idx, max=L-1)
        upto_eos = pos <= eos_idx.unsqueeze(1)

        valid_pos = answer_pos & mask_pos & upto_eos
        answer_upto = answer_pos & upto_eos

        correct = ((pred_ids == labels_m) & valid_pos).sum()
        masked = valid_pos.sum()
        answer = answer_upto.sum()

        total_correct += int(correct.item())
        total_masked += int(masked.item())
        total_answer += int(answer.item())

    correct_t = torch.tensor([total_correct], device=accelerator.device, dtype=torch.long)
    masked_t = torch.tensor([total_masked], device=accelerator.device, dtype=torch.long)
    answer_t = torch.tensor([total_answer], device=accelerator.device, dtype=torch.long)

    correct_all = accelerator.gather(correct_t).sum()
    masked_all = accelerator.gather(masked_t).sum()
    answer_all = accelerator.gather(answer_t).sum()

    masked_acc = (correct_all.float() / masked_all.clamp(min=1).float()).item()
    mask_ratio = (masked_all.float() / answer_all.clamp(min=1).float()).item()

    return {"test/masked_acc": masked_acc, "test/mask_ratio": mask_ratio}


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
