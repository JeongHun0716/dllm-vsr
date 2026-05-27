# Copyright 2025 NVIDIA / dllm-vsr
# Adapted from Fast-dLLM Dream (NVlabs) for VSR pipeline (inputs_embeds based).
"""Block-wise decoding with dual KV-cache (Fast-dLLM style) for the Dream backbone.

Inputs: encoded visual feature + instruction token IDs + post_user prefix (chat eot+assistant hdr).
Outputs: predicted answer text + token IDs + nfe (number of forward evaluations).

Decoding flow:
    1) Build inputs_embeds = [prepad? + inst + visual_feat + post_user + MASK x gen_length].
    2) Process left-to-right semi-AR by block (block_length at a time).
    3) Within each block:
         - First step: full forward, commit the single highest-conf position (logits use Dream NTP shift).
         - Later steps: re-forward only the block slice (dual_cache + replace_position),
           committing by confidence threshold or quota.
    4) On EOS, fill subsequent positions with tail_token_id (matches training convention).
       - tail_token_id unset: eos_id.
       - For pad-loss-trained models: pass pad_id (matches training distribution).

dual_cache=True reuses prefix KV cache during block refinement -> faster.
dual_cache=False re-forwards from the prefix every step (slower but exact).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F


def _shift_logits_for_ntp(logits: torch.Tensor) -> torch.Tensor:
    """No-op: our Dream-VSR is trained MDM-style (position i logit = position i prediction),
    so the original Fast-dLLM NTP shift does not match our training convention.
    """
    return logits


def _select_threshold(conf: torch.Tensor, threshold: float) -> torch.Tensor:
    """Mode A: select all positions with conf > threshold; fall back to top-1 if none.
    Args: conf [N]  → returns local indices [k].
    """
    pass_mask = conf > threshold
    if pass_mask.any():
        return torch.nonzero(pass_mask, as_tuple=False).squeeze(-1)
    return torch.argmax(conf).unsqueeze(0)


def _select_fixed_k(conf: torch.Tensor, k: int) -> torch.Tensor:
    """Mode B: commit exactly k per step (or fewer if remaining < k).
    Args: conf [N], k int → returns local indices [k'].
    """
    n_valid = (conf > -float("inf")).sum().item()
    k = max(1, min(int(k), int(n_valid)))
    _, top_local = torch.topk(conf, k=k)
    return top_local


def _select_dynamic_factor(conf: torch.Tensor, factor: float) -> torch.Tensor:
    """Mode C: rank-dependent threshold. threshs[i] = 1 - factor/(i+2); top-1 always commits.
    After sorting, commit until the first i where conf[i] < threshs[i].

    Equivalent to Fast-dLLM `get_transfer_index_dynamic`.
    """
    valid = conf > -float("inf")
    n = int(valid.sum().item())
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=conf.device)

    sorted_conf, sorted_idx = torch.sort(conf, descending=True)   # [N]
    # threshs[i] = 1 - factor/(i+2);  threshs[0] = -1 (always commit)
    ranks = torch.arange(n, device=conf.device, dtype=torch.float64)
    threshs = 1.0 - float(factor) / (ranks + 2.0)
    threshs[0] = -1.0
    sc = sorted_conf[:n].to(torch.float64)
    below = sc < threshs
    if below.any():
        top_i = int(torch.nonzero(below, as_tuple=False).min().item())
    else:
        top_i = n
    top_i = max(1, top_i)
    return sorted_idx[:top_i]


@torch.no_grad()
def generate_with_dual_cache_embeds_dream(
    *,
    model,
    tokenizer,
    instructions: List[torch.Tensor],         # list of token id tensors (user prefix per sample)
    v_feats: Optional[torch.Tensor],          # [B, T_v, D] visual feature embeddings (already through encoder+adapter)
    post_user_prefix: Optional[torch.Tensor], # token ids for eot + assistant header
    mask_id: int,
    gen_length: int,
    steps: int = 32,
    block_length: int = 32,
    alg: str = "threshold",                   # "threshold" | "fixed_k" | "dynamic"
    threshold: Optional[float] = 0.9,         # for alg="threshold"
    fixed_k: int = 1,                          # for alg="fixed_k": tokens to commit per step
    factor: float = 1.0,                       # for alg="dynamic": looser → more commits
    temperature: float = 0.0,
    dual_cache: bool = True,
    eos_fill: bool = True,
    tail_token_id: Optional[int] = None,       # None -> use eos_id; otherwise fill with this token
    eos_tail_mode: str = "fill",               # "fill" | "attn_zero" | "truncate"
):
    """Run block-wise dual-cache generation on Dream backbone with visual-feature embeddings.

    Returns:
        pred_text (str), pred_ids (List[int]), nfe (int)
    """
    assert len(instructions) == 1, "bs=1 only"
    device = next(model.parameters()).device
    wte = model.get_input_embeddings()
    dllm_dtype = next(model.parameters()).dtype

    inst = instructions[0].to(device=device, dtype=torch.long)
    inst_len = int(inst.numel())

    if v_feats is None:
        feat_emb = torch.zeros((1, 0, wte.weight.size(1)), device=device, dtype=dllm_dtype)
        T = 0
    else:
        feat_emb = v_feats.to(device=device, dtype=dllm_dtype)
        if feat_emb.dim() == 2:
            feat_emb = feat_emb.unsqueeze(0)
        T = int(feat_emb.size(1))

    inst_emb = wte(inst).unsqueeze(0)  # [1, inst_len, D]
    mask_emb = wte(torch.tensor([mask_id], device=device, dtype=torch.long))  # [1, D]
    ans_emb = mask_emb.unsqueeze(1).repeat(1, gen_length, 1)                   # [1, gen_length, D]

    # Post-user prefix (eot + assistant header) — placed after visual feat, before the answer.
    if post_user_prefix is not None:
        pu_ids = post_user_prefix.to(device=device, dtype=torch.long)
        pu_emb = wte(pu_ids).unsqueeze(0) if pu_ids.dim() == 1 else wte(pu_ids)
    else:
        pu_emb = torch.zeros((1, 0, wte.weight.size(1)), device=device, dtype=dllm_dtype)
    pu_len = int(pu_emb.size(1))

    pre_pad_len = 0
    prepad_emb = torch.zeros((1, 0, wte.weight.size(1)), device=device, dtype=dllm_dtype)

    inputs_embeds = torch.cat(
        [prepad_emb, inst_emb, feat_emb, pu_emb, ans_emb], dim=1
    ).contiguous().to(dllm_dtype)
    B, L, D = inputs_embeds.shape

    # 4D bool attention mask is needed when pre-pad exists or in attn_zero mode.
    # Always maintain am1d regardless of eos_tail_mode so EOS occurrence can update it dynamically.
    am1d = torch.ones((B, L), device=device, dtype=torch.bool)
    if pre_pad_len > 0:
        am1d[:, :pre_pad_len] = False

    def _build_attn_mask():
        """Return 4D attention mask, or None when no masking is needed."""
        if am1d.all():
            return None
        return torch.logical_and(
            am1d.unsqueeze(1).unsqueeze(-2),  # [B,1,1,L]
            am1d.unsqueeze(1).unsqueeze(-1),  # [B,1,L,1]
        )  # → [B,1,L,L] bool

    attention_mask = _build_attn_mask()

    answer_start = pre_pad_len + inst_len + T + pu_len
    answer_end = answer_start + gen_length

    # Track committed token IDs; mask_id default = uncommitted.
    # Non-answer regions hold dummy values (ignored at decode).
    x = torch.full((B, L), mask_id, dtype=torch.long, device=device)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    # tail token: caller may override to match the training distribution (e.g. pad_id for pad-loss models).
    # Default falls back to EOS fill (Fast-dLLM convention).
    effective_tail_id = int(tail_token_id) if tail_token_id is not None else eos_id
    tail_emb = (
        wte(torch.tensor([effective_tail_id], device=device, dtype=torch.long)).to(dllm_dtype)
        if effective_tail_id is not None else None
    )

    # ---- helpers ----
    def _commit(positions: torch.Tensor, tok_ids: torch.Tensor):
        """positions: 1D long [k], tok_ids: 1D long [k] — write x[:,positions] and inputs_embeds."""
        if positions.numel() == 0:
            return
        new_emb = wte(tok_ids).to(dllm_dtype)  # [k, D]
        x[0, positions] = tok_ids
        inputs_embeds[0, positions, :] = new_emb

    # Cursor (input physical length) for truncate mode after EOS occurrence; 0 means disabled.
    trunc_end = [0]   # mutable container (nonlocal-friendly)

    def _apply_eos_fill():
        """Handle positions after EOS within the answer; dispatch on eos_tail_mode.
        - "fill"      (default): tail = EOS token + attention=1
        - "attn_zero": tail = EOS token (for skip check) + attention=0
        - "truncate" : tail = EOS token (for skip check); subsequent forwards slice the input
        """
        nonlocal attention_mask
        if not eos_fill or eos_id is None or tail_emb is None:
            return
        ans = x[0, answer_start:answer_end]
        eos_positions = (ans == eos_id).nonzero(as_tuple=False).squeeze(-1)
        if eos_positions.numel() == 0:
            return
        first_eos = int(eos_positions.min().item())
        if first_eos + 1 >= gen_length:
            return
        tail = torch.arange(answer_start + first_eos + 1, answer_end, device=device)
        # Common: write a non-mask_id so subsequent skip checks treat it as committed.
        x[0, tail] = effective_tail_id
        inputs_embeds[0, tail, :] = tail_emb
        if eos_tail_mode == "attn_zero":
            am1d[0, tail] = False
            attention_mask = _build_attn_mask()
        elif eos_tail_mode == "truncate":
            if trunc_end[0] == 0:   # set once at first EOS
                trunc_end[0] = int(answer_start + first_eos + 1)

    # ---- block schedule ----
    # Round down; any remainder is handled as a final partial block.
    full_blocks = gen_length // block_length
    last_partial = gen_length - full_blocks * block_length     # 0 means no partial block
    num_blocks = full_blocks + (1 if last_partial > 0 else 0)
    base = max(1, steps // num_blocks)
    rem = steps - base * num_blocks
    steps_per_block_list = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    steps_per_block_list = [max(1, s) for s in steps_per_block_list]
    # Per-block actual lengths: only the last may be partial; the rest are block_length.
    block_lens = [block_length] * full_blocks
    if last_partial > 0:
        block_lens.append(last_partial)

    nfe = 0

    cursor = answer_start
    for nb in range(num_blocks):
        blk_len = block_lens[nb]
        s = cursor
        e = cursor + blk_len
        cursor = e
        if blk_len <= 0:
            continue
        if (x[:, s:e] == mask_id).sum().item() == 0:
            continue

        steps_per_block = steps_per_block_list[nb]

        # === Block step 0: full forward, get prefix KV cache, commit first token ===
        # In truncate mode, if a prior block committed EOS, the mask=0 skip above bypasses this branch,
        # so step 0 forward always runs with truncate inactive -> the full input is used.
        out_full = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = out_full.past_key_values
        logits = _shift_logits_for_ntp(out_full.logits)
        nfe += 1

        # Restrict to current block masked positions; pick top-k by confidence.
        block_mask = (x[0, s:e] == mask_id)
        block_logits = logits[0, s:e, :]                          # [blk_len, V]
        probs = F.softmax(block_logits.to(torch.float64), dim=-1)
        conf, tok = probs.max(dim=-1)                              # [blk_len]
        conf = torch.where(block_mask, conf, torch.full_like(conf, -float("inf")))

        # First step: commit only the single highest-conf position as a safe starting point.
        top1_local = int(torch.argmax(conf).item())
        _commit(torch.tensor([s + top1_local], device=device, dtype=torch.long),
                tok[top1_local:top1_local+1].to(torch.long))
        _apply_eos_fill()

        # === Block refinement loop ===
        if dual_cache:
            replace_position = torch.zeros((B, L), device=device, dtype=torch.bool)
            replace_position[:, s:e] = True

        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum().item() == 0:
                break

            if eos_tail_mode == "truncate" and trunc_end[0] > 0:
                # After EOS commit: slice input up to trunc_end and run full forward (no cache).
                tend = int(trunc_end[0])
                out_t = model(
                    inputs_embeds=inputs_embeds[:, :tend, :],
                    attention_mask=None,
                    use_cache=False,
                )
                logits_t = _shift_logits_for_ntp(out_t.logits)   # [B, tend, V]
                # Slice block logits [s:e]. If e > tend, the tail is already non-mask,
                # so block_mask turns it into -inf -> safe to zero-pad here.
                if e <= tend:
                    logits_blk = logits_t[:, s:e, :]
                else:
                    head = logits_t[:, s:tend, :]
                    n_pad = e - tend
                    pad = torch.zeros(B, n_pad, head.shape[-1], device=device, dtype=head.dtype)
                    logits_blk = torch.cat([head, pad], dim=1)
            elif dual_cache:
                blk_embeds = inputs_embeds[:, s:e, :]
                # When forwarding only the block slice, mask is None (all positions valid); pre_pad lives in the prefix.
                out_blk = model(
                    inputs_embeds=blk_embeds,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                    dual_cache=True,
                    replace_position=replace_position,
                )
                logits_blk = _shift_logits_for_ntp(out_blk.logits)   # [B, blk_len, V]
            else:
                # No-cache fallback: full forward each iteration (slow but correct).
                out_blk = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
                logits_blk = _shift_logits_for_ntp(out_blk.logits)[:, s:e, :]
            nfe += 1

            # Compute confidence over still-masked positions in block.
            block_mask = (x[0, s:e] == mask_id)                    # [blk_len]
            probs = F.softmax(logits_blk[0].to(torch.float64), dim=-1)
            conf, tok = probs.max(dim=-1)                           # [blk_len]
            conf = torch.where(block_mask, conf, torch.full_like(conf, -float("inf")))

            n_remaining = int(block_mask.sum().item())

            if alg == "threshold":
                chosen_local = _select_threshold(conf, float(threshold))
            elif alg == "fixed_k":
                chosen_local = _select_fixed_k(conf, int(fixed_k))
            elif alg == "dynamic":
                chosen_local = _select_dynamic_factor(conf, float(factor))
            else:
                raise ValueError(f"Unknown alg: {alg}")

            positions = (s + chosen_local).to(torch.long)
            tok_ids = tok[chosen_local].to(torch.long)
            _commit(positions, tok_ids)
            _apply_eos_fill()

        # Final safety net: force-commit any still-masked positions so no block is left incomplete.
        block_mask = (x[0, s:e] == mask_id)
        if block_mask.any():
            if eos_tail_mode == "truncate" and trunc_end[0] > 0:
                tend = int(trunc_end[0])
                out_full2 = model(inputs_embeds=inputs_embeds[:, :tend, :], attention_mask=None, use_cache=False)
                logits_full = _shift_logits_for_ntp(out_full2.logits)
                if e <= tend:
                    block_logits = logits_full[0, s:e, :]
                else:
                    head = logits_full[0, s:tend, :]
                    n_pad = e - tend
                    pad = torch.zeros(n_pad, head.shape[-1], device=device, dtype=head.dtype)
                    block_logits = torch.cat([head, pad], dim=0)
            else:
                out_full2 = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
                logits_full = _shift_logits_for_ntp(out_full2.logits)
                block_logits = logits_full[0, s:e, :]
            probs = F.softmax(block_logits.to(torch.float64), dim=-1)
            conf, tok = probs.max(dim=-1)
            still_masked = torch.nonzero(block_mask, as_tuple=False).squeeze(-1)
            positions = (s + still_masked).to(torch.long)
            tok_ids = tok[still_masked].to(torch.long)
            _commit(positions, tok_ids)
            _apply_eos_fill()
            nfe += 1

    # Decode answer
    out_ids = x[0, answer_start:answer_end].detach().cpu().tolist()
    if eos_id is not None and eos_id in out_ids:
        eos_pos = out_ids.index(eos_id)
        out_ids = out_ids[: eos_pos + 1]
    pred_text = tokenizer.decode(out_ids)
    return pred_text, out_ids, nfe
