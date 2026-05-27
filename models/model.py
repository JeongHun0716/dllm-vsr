from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, PretrainedConfig
from peft import LoraConfig, get_peft_model

import os as _os
from omegaconf import OmegaConf

from .visual_encoder import VisualEncoder, build_visual_encoder


# Padding token ID for Dream (Qwen2-base) — reuse unused image-pad slot.
PAD_ID = 151655  # <|image_pad|>


class Projector(nn.Module):
    """FC adapter (LN at input, GELU between linears):
    - num_layers=1: LN(input_dim) -> Linear(input_dim, output_dim)
    - num_layers=2: LN(input_dim) -> Linear(input_dim, hidden_dim) -> GELU -> Linear(hidden_dim, output_dim)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers not in (1, 2):
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")
        self.num_layers = int(num_layers)
        if num_layers == 1:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DLLMVSRConfig(PretrainedConfig):
    model_type = "dllm_vsr"

    def __init__(
        self,
        # LLM backbone: Dream-org/Dream-v0-Instruct-7B (HF AutoModel + trust_remote_code).
        dllm_pretrained_path: str = "Dream-org/Dream-v0-Instruct-7B",
        dllm_hidden: int = 3584,

        # Visual encoder (registry-driven). e.g., {"type": "avhubert", "ckpt_path": "...", "enc_dim": 1024}
        # or {"type": "usr2", "ckpt_path": "...", "backbone": "huge"}
        visual_encoder_cfg: Optional[dict] = None,

        # Adapter: Conv1d+FC mapping encoder output to LLM hidden.
        adapter_kernel: int = 2,
        adapter_stride: int = 2,
        adapter_num_fc_layers: int = 2,   # 1: single Linear, 2: 2-Linear with hidden (default)
        # If None, auto-computed as floor((enc_dim + dllm_hidden) / 2); otherwise used as given.
        adapter_hidden_dim: Optional[int] = None,

        # LoRA
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules=("q_proj", "v_proj", "k_proj", "o_proj"),

        # If True, use transcript+EOS as-is (no PAD padding up to max_seq_len).
        # Assumes ground-truth transcript length is known at train/inference (oracle length).
        use_oracle_length: bool = False,

        # Max answer-region length (PAD canvas size when use_oracle_length=False).
        # When use_oracle_length=True, transcript+EOS exceeding this length is truncated.
        max_seq_len: int = 100,

        # Whether to force-mask the first EOS position in corrupt_answer_region.
        # Default True (legacy — stronger variable-length termination signal). Set False
        # for natural random masking only (Dream MDM standard training convention).
        force_first_eos_mask: bool = True,

        # When use_oracle_length=True, include batch-alignment tail PAD positions as loss
        # targets (attention_mask=1, label=pad_id). Unlike fixed-length pad-loss to
        # max_seq_len, pad ratio is bounded by (batch max - sample len), avoiding PAD
        # dominance. Ignored if use_oracle_length=False.
        train_batch_pad_loss: bool = False,

        # Activation checkpointing on the LLM backbone (saves VRAM).
        gradient_checkpointing: bool = False,

        # misc
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dllm_pretrained_path = dllm_pretrained_path
        self.dllm_hidden = dllm_hidden

        if visual_encoder_cfg is None:
            raise ValueError("visual_encoder_cfg is required (e.g., {'type': 'avhubert', 'ckpt_path': '...'})")
        self.visual_encoder_cfg: dict = dict(visual_encoder_cfg)

        self.adapter_kernel = adapter_kernel
        self.adapter_stride = adapter_stride
        self.adapter_num_fc_layers = int(adapter_num_fc_layers)
        self.adapter_hidden_dim = int(adapter_hidden_dim) if adapter_hidden_dim is not None else None

        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # peft accepts str ("all-linear") too; keep str as-is, coerce list/tuple to list.
        if isinstance(lora_target_modules, str):
            self.lora_target_modules = lora_target_modules
        else:
            self.lora_target_modules = list(lora_target_modules)

        self.use_oracle_length = bool(use_oracle_length)
        self.max_seq_len = int(max_seq_len)
        self.force_first_eos_mask = bool(force_first_eos_mask)
        self.train_batch_pad_loss = bool(train_batch_pad_loss)

        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.torch_dtype = torch_dtype

    @property
    def enc_dim(self) -> int:
        """Visual encoder output dim (Conv1d adapter input)."""
        return int(self.visual_encoder_cfg.get("enc_dim", 1024))


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


class DLLMVSRModel(nn.Module):
    config_class = DLLMVSRConfig

    def __init__(self, config: DLLMVSRConfig):
        super().__init__()
        self.config = config
        self.dllm: Optional[nn.Module] = None     # Dream backbone
        self.v_encoder: Optional[VisualEncoder] = None
        self.video_downsampler: Optional[nn.Conv1d] = None
        self.fc_v: Optional[Projector] = None
        self.max_seq_len = int(getattr(config, "max_seq_len", 100))
        self.pad_id = PAD_ID

    def build_components(self):
        cfg = self.config

        # ===== visual encoder (built from registry, frozen) =====
        self.v_encoder = build_visual_encoder(cfg.visual_encoder_cfg)
        self.v_encoder.eval()
        self.v_encoder.requires_grad_(False)
        # opt-in: cast frozen encoder to bf16/fp16 for VRAM/speed.
        # Applied only if visual_encoder.dtype is set; otherwise ckpt dtype is kept (usually fp32).
        enc_dtype_str = cfg.visual_encoder_cfg.get("dtype") if cfg.visual_encoder_cfg else None
        if enc_dtype_str:
            self.v_encoder.to(dtype=_dtype_from_str(enc_dtype_str))

        # ===== adapters (trainable) =====
        enc_dim = self.v_encoder.feat_dim
        self.video_downsampler = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=enc_dim,
            kernel_size=cfg.adapter_kernel,
            stride=cfg.adapter_stride,
            padding=0,
        )

        if cfg.adapter_hidden_dim is not None:
            hidden_dim = int(cfg.adapter_hidden_dim)
        else:
            hidden_dim = math.floor((enc_dim + cfg.dllm_hidden) / 2)
        self.fc_v = Projector(
            input_dim=enc_dim,
            hidden_dim=hidden_dim,
            output_dim=cfg.dllm_hidden,
            num_layers=cfg.adapter_num_fc_layers,
        )

    def move_components_to(self, device):
        for m in [self.v_encoder, self.video_downsampler, self.fc_v]:
            if m is not None:
                m.to(device)

    def video_input_dtype(self):
        """Expected dtype of video input — delegated to the encoder."""
        if self.v_encoder is not None:
            return self.v_encoder.input_dtype()
        return next(self.fc_v.parameters()).dtype

    def print_trainable_params(self):
        total, trainable = 0, 0
        for p in self.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

    def apply_lora_if_needed(self):
        cfg = self.config
        if not cfg.use_lora:
            return
        if self.dllm is None:
            raise RuntimeError("self.dllm is None. Load backbone before applying LoRA.")

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
        )
        self.dllm = get_peft_model(self.dllm, lora_cfg).base_model.model

    def train(self, mode: bool = True):
        super().train(mode)
        # v_encoder is frozen → always eval, regardless of encoder type.
        if self.v_encoder is not None:
            self.v_encoder.eval()
        return self

    @classmethod
    def from_pretrained(cls, config: DLLMVSRConfig):
        m = cls(config)
        dtype = _dtype_from_str(config.torch_dtype)

        # DLLM_VSR_USE_FAST_DREAM=1 → load patched DreamModel (supports dual_cache +
        # replace_position for Fast-dLLM style block-wise KV-cache decoding).
        # Default: HF AutoModel (trust_remote_code) loads the standard Dream backbone.
        if _os.environ.get("DLLM_VSR_USE_FAST_DREAM") == "1":
            from evaluation.fast_dream_model import DreamModel
            m.dllm = DreamModel.from_pretrained(
                config.dllm_pretrained_path,
                torch_dtype=dtype,
            )
        else:
            from transformers import AutoModel
            m.dllm = AutoModel.from_pretrained(
                config.dllm_pretrained_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        if config.gradient_checkpointing and hasattr(m.dllm, "gradient_checkpointing_enable"):
            try:
                m.dllm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                m.dllm.gradient_checkpointing_enable()

        m.build_components()
        m.apply_lora_if_needed()
        return m

    def _get_wte(self):
        """Find the token embedding (nn.Embedding) in the Dream (Qwen2-base) backbone."""
        for name, mod in self.dllm.named_modules():
            if isinstance(mod, nn.Embedding) and (
                name.endswith("embed_tokens") or name.endswith("wte")
            ):
                return mod
        # fallback: first nn.Embedding
        for name, mod in self.dllm.named_modules():
            if isinstance(mod, nn.Embedding):
                return mod
        raise RuntimeError("Could not find token embedding in backbone")

    # =========================================================
    # Encode video.
    # Encoder takes raw video [B,C,T,H,W].
    # All encoders are frozen → output passes through downsampler (Conv1d) → fc_v (Projector).
    # =========================================================
    def encode_video(
        self,
        video_feats: Tensor,
        video_pad: Optional[Tensor] = None,  # [B, T] True=pad
        apply_fc: bool = True,
    ) -> Tuple[Tensor, List[int], Optional[Tensor]]:
        if self.video_downsampler is None or self.fc_v is None or self.v_encoder is None:
            raise RuntimeError("Components not built; call build_components()")

        v_enc, pad = self.v_encoder.encode(video_feats, video_pad)

        # Match adapter weight dtype: encoder may be cast to bf16/fp16. Mixed-precision
        # autocast handles this during training, but unwrapped eval forwards run outside
        # autocast, so cast explicitly to avoid bf16-input / fp32-bias mismatch.
        adapter_dtype = self.video_downsampler.weight.dtype
        if v_enc.dtype != adapter_dtype:
            v_enc = v_enc.to(adapter_dtype)

        # Conv1d downsample (trainable). [B,T,D] -> [B,T',D]
        v_enc = self.video_downsampler(v_enc.transpose(1, 2)).transpose(1, 2)

        # Length computation assumes stride=2 (exact conv output formula is a future TODO).
        v_pad_ds = None
        if pad is not None:
            v_pad_ds = pad[:, 1::2]
            v_pad_ds = v_pad_ds[:, : v_enc.size(1)]
            v_valid = (~v_pad_ds).long()
            len_v = v_valid.sum(dim=1).tolist()
        else:
            len_v = [v_enc.size(1)] * v_enc.size(0)

        if apply_fc:
            v_enc = self.fc_v(v_enc)  # [B, T', dllm_hidden]

        return v_enc, len_v, v_pad_ds

    # ---------- prepare dllm inputs/labels ----------
    def prepare_dllm_inputs_labels(
        self,
        tokenizer,
        instructions: List[Tensor],
        v_feats: Tensor,            # [B, T, D]
        len_v: List[int],
        labels: Optional[List[Tensor]] = None,
        post_user_prefix: Optional[Tensor] = None,  # eot + assistant header (inserted after visual feat)
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Per-sample sequence layout:
            [instruction ids (user header+content)] +
            [visual feature embeddings] +
            [post_user ids (eot + assistant header)] +
            [label ids (transcript + EOS, training only)]

        Visual feat sits inside the user turn (after instruction, before eot) — LLaVA-style.
        post_user_prefix=None falls back to legacy behavior (visual after assistant header).

        Returns:
            dllm_inputs:    [B, T_max, D] padded input embeddings
            attention_mask: [B, T_max] (1=valid, 0=pad)
            prompt_mask:    [B, T_max] (1=prompt or pad, 0=answer) or None
            dllm_labels:    [B, T_max] with -100 for non-label positions, or None
        """
        wte = self._get_wte()
        # Compute post_user embedding once and share across all samples in the batch.
        post_user_emb = None
        if post_user_prefix is not None:
            pu_ids = post_user_prefix.to(dtype=torch.long, device=v_feats.device if v_feats is not None else "cpu")
            post_user_emb = wte(pu_ids)  # [L_post, D]

        # Right-padding after answer-EOS is always PAD; the model learns PAD = end-padding.
        label_pad_id = self.pad_id

        B = len(instructions)
        dllm_input_list: List[Tensor] = []
        dllm_labels_list: List[Tensor] = []
        lengths: List[int] = []

        for i, inst in enumerate(instructions):
            inst_ids = inst.to(dtype=torch.long)
            inst_emb = wte(inst_ids)  # [L_inst, D]

            v = v_feats[i]  # [T, D]
            v_len = min(len_v[i], v.size(0))
            feat_emb = v[:v_len, :]  # [v_len, D]

            if labels is not None:
                lbl_ids = labels[i].to(dtype=torch.long, device=inst.device)
                if self.config.use_oracle_length:
                    # Oracle: use transcript+EOS as-is (no truncate/pad).
                    pass
                else:
                    if lbl_ids.numel() > self.max_seq_len:
                        lbl_ids = lbl_ids[: self.max_seq_len]
                    elif lbl_ids.numel() < self.max_seq_len:
                        pad_len = self.max_seq_len - lbl_ids.numel()
                        lbl_ids = torch.cat(
                            [lbl_ids, torch.full((pad_len,), label_pad_id, dtype=torch.long, device=inst.device)],
                            dim=0,
                        )

                lbl_emb = wte(lbl_ids)  # [L_lbl, D]  (variable in oracle, max_seq_len otherwise)
                # LLaVA-style: instruction -> visual feat -> eot+assistant_hdr -> label.
                # If post_user_emb is None, fall back to legacy (visual after assistant header).
                if post_user_emb is not None:
                    combined = torch.cat([inst_emb, feat_emb, post_user_emb, lbl_emb], dim=0)
                else:
                    combined = torch.cat([inst_emb, feat_emb, lbl_emb], dim=0)

                # PAD positions remain as loss targets so the model learns to predict PAD.
                # In oracle mode there is no PAD; transcript+EOS is the entire target.
                lbl_for_loss = lbl_ids

                mask = torch.full((combined.size(0),), -100, dtype=torch.long, device=inst.device)
                offset = inst_emb.size(0) + feat_emb.size(0)
                if post_user_emb is not None:
                    offset += post_user_emb.size(0)
                mask[offset: offset + lbl_ids.numel()] = lbl_for_loss

                dllm_labels_list.append(mask)
            else:
                if post_user_emb is not None:
                    combined = torch.cat([inst_emb, feat_emb, post_user_emb], dim=0)
                else:
                    combined = torch.cat([inst_emb, feat_emb], dim=0)

            dllm_input_list.append(combined)
            lengths.append(combined.size(0))

        max_len = max(lengths)
        D = dllm_input_list[0].size(1)
        device = dllm_input_list[0].device

        # Right-padding alignment: content always starts at position 0, PAD fills the tail
        # (attention=0, label=-100). Keeps absolute positions consistent between eval B=1
        # and train B>1 (matters for RoPE-sensitive encoders).
        pad_emb = wte(torch.tensor([self.pad_id], device=device, dtype=torch.long)).squeeze(0)

        dllm_inputs = pad_emb.view(1, 1, D).expand(B, max_len, D).clone()
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for b, seq in enumerate(dllm_input_list):
            L = seq.size(0)
            dllm_inputs[b, :L] = seq
            attention_mask[b, :L] = 1

        if labels is not None:
            dllm_labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
            for b, m in enumerate(dllm_labels_list):
                L = m.size(0)
                dllm_labels[b, :L] = m
            # train_batch_pad_loss: include batch-alignment tail PAD positions as loss
            # targets (active only with use_oracle_length=True). Setting attention_mask=1
            # lets the model attend and learn "this slot is post-answer padding". PAD ratio
            # is bounded by (batch max - sample len), avoiding the late-training PAD
            # overfitting seen with fixed-length pad-loss.
            if self.config.train_batch_pad_loss and self.config.use_oracle_length:
                for b, m in enumerate(dllm_labels_list):
                    L = m.size(0)
                    if L < max_len:
                        dllm_labels[b, L:max_len] = self.pad_id
                        attention_mask[b, L:max_len] = 1
            prompt_mask = ((attention_mask == 0) | (dllm_labels == -100)).to(torch.long)
        else:
            dllm_labels = None
            prompt_mask = None

        return dllm_inputs, attention_mask, prompt_mask, dllm_labels

    # ---------- mask answer region ----------
    def corrupt_answer_region(
        self,
        dllm_inputs: Tensor,
        prompt_mask: Tensor,
        labels: Tensor,
        mask_id: int,
        eps: float = 1e-3,
        force_mask_prob: Optional[float] = None,
        eos_token_id: Optional[int] = None,   # if given, always mask the first EOS in answer
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        wte = self._get_wte()
        B, L, D = dllm_inputs.shape
        device = dllm_inputs.device

        if force_mask_prob is None:
            t = torch.rand(B, device=device)
            p_mask = (1 - eps) * t + eps
        else:
            p = max(0.0, min(1.0, float(force_mask_prob)))
            p_mask = torch.full((B,), p, device=device)

        p_mask = p_mask[:, None].repeat(1, L)
        masked_indices = (torch.rand((B, L), device=device) < p_mask)
        masked_indices = masked_indices & (~prompt_mask.bool())  # answer-region only

        # Legacy: always mask the first EOS in the answer to strengthen variable-length-end learning.
        # Disable via config.force_first_eos_mask=False (Dream MDM standard random masking only).
        answer_pos_bool = (~prompt_mask.bool())
        batch_idx = torch.arange(B, device=device)
        if eos_token_id is not None and getattr(self.config, "force_first_eos_mask", True):
            label_pos = (labels != -100)  # -100 marks prompt positions
            is_eos = (labels == int(eos_token_id)) & answer_pos_bool & label_pos
            B_, Lc = is_eos.shape
            pos = torch.arange(Lc, device=device).unsqueeze(0).expand(B_, Lc)
            big = Lc + 1
            first_eos_idx = torch.where(is_eos, pos, torch.full_like(pos, big)).min(dim=1).values  # [B]
            has_eos = first_eos_idx < big
            # vectorized: masked_indices[b, first_eos_idx[b]] = True where has_eos[b]
            safe_idx = first_eos_idx.clamp(max=Lc - 1)
            eos_set = torch.zeros_like(masked_indices)
            eos_set[batch_idx, safe_idx] = True
            masked_indices = masked_indices | (eos_set & has_eos.unsqueeze(1))

        # Force at least one mask per answer region — vectorized.
        # If a sample has zero masks in the answer, pick one random answer position.
        # Original called per-sample randint (B RNG draws); here we score [B,L] in one go
        # and argmax. Distribution-equivalent but RNG sequence differs (no training impact).
        any_masked_in_ans = (masked_indices & answer_pos_bool).any(dim=1)
        has_any_ans = answer_pos_bool.any(dim=1)
        need_one = has_any_ans & ~any_masked_in_ans
        rand_scores = torch.where(
            answer_pos_bool,
            torch.rand((B, L), device=device),
            torch.full((B, L), -1.0, device=device),
        )
        chosen_pos = rand_scores.argmax(dim=1)  # [B]
        one_set = torch.zeros_like(masked_indices)
        one_set[batch_idx, chosen_pos] = True
        masked_indices = masked_indices | (one_set & need_one.unsqueeze(1))

        mask_emb = wte(torch.tensor([mask_id], device=device, dtype=torch.long)).squeeze(0)
        noisy_inputs = dllm_inputs.clone()
        noisy_inputs[masked_indices] = mask_emb

        valid = (labels != -100)
        answer_len = valid.sum(dim=-1, keepdim=True)
        answer_lengths = answer_len.repeat(1, L)

        return noisy_inputs, labels, p_mask, answer_lengths, masked_indices

    # ---------- forward ----------
    def forward(
        self,
        inputs_embeds,          # [B, L, D]  (noisy_inputs)
        labels,                 # [B, L]     (labels_m)
        masked_indices,         # [B, L] bool
        p_mask=None,            # [B, L] or None
        attention_mask=None,    # [B, L]
        prompt_mask=None,       # [B, L] long (1=prompt/pad, 0=answer). Needed by overlap_block branch.
    ):
        # Dream's modeling_dream.py forwards attention_mask directly to SDPA's attn_mask.
        # SDPA requires a shape compatible with [B, H, L_q, L_k], so convert our 2D [B, L]
        # to 4D bool [B, 1, 1, L_k] (broadcastable; True=attend, False=mask out).
        am = attention_mask
        if am is not None and am.dim() == 2:
            am = am.to(torch.bool)[:, None, None, :]
        out = self.dllm(inputs_embeds=inputs_embeds, attention_mask=am)
        logits = out.logits  # [B, L, V]

        if not masked_indices.any():
            return logits, torch.tensor(0.0, device=logits.device)

        loss_tok = F.cross_entropy(
            logits[masked_indices],
            labels[masked_indices],
            ignore_index=-100,
            reduction="none",
        )

        if p_mask is not None:
            pm = p_mask.to(logits.device)[masked_indices].clamp_min(1e-6)
            loss_tok = loss_tok / pm

        loss = loss_tok.mean()
        return logits, loss

    # ---------- generate ----------
    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        instructions,        # List[Tensor], len=1 (bs=1 only)
        v_feats,             # Tensor [B, T, D] or None
        mask_id: int,
        answer_len: int = 100,
        num_steps: int = 10,
        tail_after_eos: int = 0,
        post_user_prefix: Optional[Tensor] = None,  # eot+assistant header (inserted after visual feat; same as training)
        block_size: Optional[int] = None,  # if set, override num_steps = ceil(answer_len / block_size)
        threshold: Optional[float] = None, # static confidence threshold; only commit positions with conf > threshold (top-1 fallback to avoid stall)
    ):
        """
        Returns:
            pred_text: str
            pred_ids:  List[int]
        """
        self.eval()
        device = next(self.parameters()).device
        wte = self._get_wte()

        # 1) encode video
        v_enc = None
        if v_feats is not None:
            v_enc, len_v, _ = self.encode_video(video_feats=v_feats, video_pad=None, apply_fc=True)

        # 2) inst_emb + feat_emb
        assert len(instructions) == 1, "this function only supports bs=1"
        inst = instructions[0].to(device=device, dtype=torch.long)
        inst_emb = wte(inst)  # [L_inst, D]

        if v_enc is not None:
            v = v_enc[0]
            v_len = min(len_v[0], v.size(0))
            feat_emb = v[:v_len, :]
        else:
            feat_emb = torch.empty((0, inst_emb.size(-1)), device=device, dtype=inst_emb.dtype)
        feat_emb = feat_emb.to(dtype=inst_emb.dtype)

        # post_user prefix (eot + assistant header) — inserted between visual and answer if given
        if post_user_prefix is not None:
            pu_ids = post_user_prefix.to(device=device, dtype=torch.long)
            post_user_emb = wte(pu_ids).to(dtype=inst_emb.dtype)
        else:
            post_user_emb = None

        # Answer region starts fully masked.
        mask_emb = wte(torch.tensor([mask_id], device=device, dtype=torch.long)).squeeze(0)
        ans_emb = mask_emb.view(1, -1).repeat(answer_len, 1)
        if post_user_emb is not None:
            combined = torch.cat([inst_emb, feat_emb, post_user_emb, ans_emb], dim=0)
        else:
            combined = torch.cat([inst_emb, feat_emb, ans_emb], dim=0)
        L = combined.size(0)

        inputs_embeds = combined.unsqueeze(0).contiguous()
        attention_mask = torch.ones((1, L), device=device, dtype=torch.long)

        prompt_mask = torch.ones((1, L), device=device, dtype=torch.long)
        answer_start = inst_emb.size(0) + feat_emb.size(0)
        if post_user_emb is not None:
            answer_start += post_user_emb.size(0)
        answer_end = answer_start + answer_len
        prompt_mask[:, answer_start:answer_end] = 0

        masked_indices = torch.zeros((1, L), device=device, dtype=torch.bool)
        masked_indices[:, answer_start:answer_end] = True

        pred_ids = torch.full((answer_len,), mask_id, device=device, dtype=torch.long)

        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = getattr(self.dllm.config, "eos_token_id", None)

        # Tail = PAD, matching the training convention where post-EOS slots are PAD.
        # Attend (1) because PAD is a trained target.
        tail_token_id = self.pad_id
        tail_attn_value = 1
        tail_emb = wte(torch.tensor([tail_token_id], device=device, dtype=torch.long)).squeeze(0)

        # If block_size is given, it caps per-step commits (= k_per_step) and num_steps
        # is overridden to ceil(answer_len / block_size).
        if block_size is not None and block_size > 0:
            k_per_step = int(block_size)
            num_steps = (answer_len + k_per_step - 1) // k_per_step
        else:
            k_per_step = max(1, answer_len // max(1, num_steps))

        # 4) iterative unmask
        for step in range(num_steps):
            cur_mask_pos = masked_indices[0, answer_start:answer_end]
            remaining = int(cur_mask_pos.sum().item())
            if remaining == 0:
                break

            k = remaining if step == num_steps - 1 else min(k_per_step, remaining)

            am = attention_mask
            if am is not None and am.dim() == 2:
                am = am.to(torch.bool)[:, None, None, :]
            out = self.dllm(inputs_embeds=inputs_embeds, attention_mask=am)
            logits = out.logits

            ans_logits = logits[0, answer_start:answer_end, :]
            masked_idx = torch.nonzero(cur_mask_pos, as_tuple=False).squeeze(1)

            sub = ans_logits[masked_idx]
            probs = F.softmax(sub, dim=-1)
            conf, tok = probs.max(dim=-1)

            # Threshold mode: commit only positions with conf > threshold (top-1 fallback
            # to avoid stall). On the last step, fall back to plain top-k (force progress).
            if threshold is not None and step < num_steps - 1:
                pass_mask = conf > float(threshold)
                if pass_mask.any():
                    cand_conf = torch.where(pass_mask, conf, torch.full_like(conf, -1.0))
                    eff_k = int(min(k, int(pass_mask.sum().item())))
                    topk_conf, topk_local = torch.topk(cand_conf, k=eff_k, largest=True)
                else:
                    # All below threshold → commit top-1 to avoid stall.
                    topk_conf, topk_local = torch.topk(conf, k=1, largest=True)
            else:
                topk_conf, topk_local = torch.topk(conf, k=k, largest=True)
            chosen_ans_positions = masked_idx[topk_local]

            pred_ids[chosen_ans_positions] = tok[topk_local]

            chosen_global = chosen_ans_positions + answer_start
            new_emb = wte(tok[topk_local])
            inputs_embeds[0, chosen_global, :] = new_emb
            masked_indices[0, chosen_global] = False

            # If EOS appeared, fill the rest with tail tokens.
            if eos_id is not None:
                eos_positions = (pred_ids == eos_id).nonzero(as_tuple=False).squeeze(-1)
                if eos_positions.numel() > 0:
                    eos_pos = int(eos_positions.min().item())
                    if eos_pos + 1 < answer_len:
                        tail = torch.arange(eos_pos + 1, answer_len, device=device)
                        pred_ids[tail] = tail_token_id

                        tail_global = tail + answer_start
                        inputs_embeds[0, tail_global, :] = tail_emb
                        masked_indices[0, tail_global] = False
                        attention_mask[0, tail_global] = tail_attn_value

        # Decode (truncate at EOS).
        out_ids = pred_ids.detach().cpu().tolist()
        if eos_id is not None and eos_id in out_ids:
            eos_pos = out_ids.index(eos_id)
            cut = eos_pos + 1 + max(0, int(tail_after_eos))
            out_ids = out_ids[:cut]

        pred_text = tokenizer.decode(out_ids)
        return pred_text, out_ids


# ============================================================================
# Builder helpers (OmegaConf config → model / tokenizer).
# Shared by training/train.py and all eval/* scripts.
# ============================================================================

def build_model(config) -> "DLLMVSRModel":
    """OmegaConf config → DLLMVSRModel (no tokenizer)."""
    dllm_cfg = config.model.dllm
    enc_cfg = OmegaConf.to_container(config.model.visual_encoder, resolve=True)
    adapter_cfg = config.model.adapter
    dllm_cfg = DLLMVSRConfig(
        dllm_pretrained_path=str(dllm_cfg.pretrained_model_path),
        dllm_hidden=int(dllm_cfg.hidden_size),
        visual_encoder_cfg=enc_cfg,
        adapter_kernel=int(adapter_cfg.kernel),
        adapter_stride=int(adapter_cfg.stride),
        adapter_num_fc_layers=int(adapter_cfg.get("num_fc_layers", 2)),
        adapter_hidden_dim=(int(adapter_cfg.hidden_dim) if adapter_cfg.get("hidden_dim", None) is not None else None),
        use_lora=bool(config.model.lora.enable),
        lora_r=int(config.model.lora.r),
        lora_alpha=int(config.model.lora.alpha),
        lora_dropout=float(config.model.lora.dropout),
        lora_target_modules=(
            config.model.lora.target_modules
            if isinstance(config.model.lora.target_modules, str)
            else tuple(config.model.lora.target_modules)
        ),
        use_oracle_length=bool(config.model.get("use_oracle_length", False)),
        max_seq_len=int(config.model.get("max_seq_len", 100)),
        force_first_eos_mask=bool(config.model.get("force_first_eos_mask", True)),
        train_batch_pad_loss=bool(config.model.get("train_batch_pad_loss", False)),
        gradient_checkpointing=bool(config.model.get("gradient_checkpointing", False)),
        torch_dtype=str(config.model.get("torch_dtype", "bfloat16")),
    )
    return DLLMVSRModel.from_pretrained(dllm_cfg)


def build_model_and_tokenizer(config):
    """OmegaConf config → (model, tokenizer, mask_id, dllm_cfg_yaml)."""
    dllm_cfg_yaml = config.model.dllm
    tokenizer = AutoTokenizer.from_pretrained(
        dllm_cfg_yaml.tokenizer_path, padding_side="left", trust_remote_code=True
    )
    model = build_model(config)
    mask_id = getattr(model.dllm.config, "mask_token_id", None) or tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("mask_token_id not found")
    return model, tokenizer, mask_id, dllm_cfg_yaml
