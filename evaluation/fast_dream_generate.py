# Copyright 2025 NVIDIA / dllm-vsr
# Adapted from Fast-dLLM Dream (NVlabs) for VSR pipeline (inputs_embeds based).
"""Block-wise masked-diffusion decoders for the Dream backbone (embeds input).

Three cache strategies, each its OWN function (mirrors upstream
`generate` / `generate_with_prefix_cache` / `generate_with_dual_cache`):

  - generate_embeds_dream                  : no cache. Re-forward the full canvas
                                             every step. EXACT reference, slowest.
  - generate_with_prefix_cache_embeds_dream: cache KV strictly before the block;
                                             re-forward block(+suffix) each step.
  - generate_with_dual_cache_embeds_dream  : cache prefix AND suffix; re-forward
                                             only the block slice via
                                             replace_position. Fastest.

Canvas layout matches training (LLaVA-style):
    [ inst | visual_feat | post_user (<eot><assistant-header>) | answer(MASK) ]

Common knobs: block_length, steps, alg ∈ {threshold, fixed_k, dynamic},
threshold, fixed_k, factor, tail_token_id.

NOTES
-----
* Dream-VSR is bidirectional + trained MDM-style (position i logit predicts
  position i), so there is no NTP shift (`_shift_logits_for_ntp` is a no-op).
* attention_mask is left None on every model call: eval is batch_size=1 with no
  padding, so full bidirectional attention is correct (and a full-length mask
  would slice to empty against the shorter cached-block query).
* The Dream backbone caches KV as 3-D tensors [B, seq, n_kv*head_dim] (pre-view,
  pre-rotary), so prefix truncation slices the seq dim: k[:, :s]. The dual path
  must pass `dual_cache=True` explicitly (unlike LLaDA, which infers it).
* tail_token_id: token used to fill positions after a committed EOS. The eval
  entry points pass PAD (matching stage-2 training); falls back to EOS if None.
* Because the model is bidirectional, the cached prefix/suffix KV is computed
  while the answer is still masked, so cached modes can score *slightly*
  differently from `generate_embeds_dream`. When EOS-fill rewrites tail
  embeddings outside the current block the cached KV goes stale; instead of a
  separate re-warm forward (which would waste an nfe), the next step is promoted
  to a FULL forward that both refreshes the cache and supplies that step's logits
  (the `cache_stale` flag).

Fast-dLLM patched Dream backbone is forced via `DLLM_VSR_USE_FAST_DREAM=1`
(set by the eval entrypoint before any model import).
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn.functional as F


def _shift_logits_for_ntp(logits: torch.Tensor) -> torch.Tensor:
    """No-op: Dream-VSR is trained MDM-style (position i logit = position i
    prediction), so the original Fast-dLLM NTP shift does not apply.
    """
    return logits


# ============================================================================
# Selection algorithms (per-block, confidence based)
# ============================================================================
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


# ============================================================================
# Shared canvas / helpers
# ============================================================================
def _build_canvas(*, model, tokenizer, instructions, v_feats, post_user_prefix,
                  mask_id, gen_length, tail_token_id):
    """Build [inst | feats | post_user | answer(MASK)] embeds + id canvas."""
    device = next(model.parameters()).device
    wte = model.get_input_embeddings()
    dllm_dtype = next(model.parameters()).dtype

    assert len(instructions) == 1, "bs=1 only"
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

    inst_emb = wte(inst).unsqueeze(0)                                          # [1, inst_len, D]
    mask_emb = wte(torch.tensor([mask_id], device=device, dtype=torch.long))   # [1, D]
    ans_emb = mask_emb.unsqueeze(1).repeat(1, gen_length, 1)                   # [1, gen_length, D]

    if post_user_prefix is not None:
        pu_ids = post_user_prefix.to(device=device, dtype=torch.long).view(-1)
        pu_emb = wte(pu_ids).unsqueeze(0)
    else:
        pu_ids = torch.zeros((0,), device=device, dtype=torch.long)
        pu_emb = torch.zeros((1, 0, wte.weight.size(1)), device=device, dtype=dllm_dtype)
    pu_len = int(pu_emb.size(1))

    inputs_embeds = torch.cat(
        [inst_emb, feat_emb, pu_emb, ans_emb], dim=1
    ).contiguous().to(dllm_dtype)
    B, L, D = inputs_embeds.shape

    answer_start = inst_len + T + pu_len
    answer_end = answer_start + gen_length

    # Track committed token IDs; mask_id default = uncommitted.
    # Non-answer regions hold dummy values (only the answer span is ever read).
    x = torch.full((B, L), mask_id, dtype=torch.long, device=device)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    effective_tail_id = int(tail_token_id) if tail_token_id is not None else eos_id
    tail_emb = (
        wte(torch.tensor([effective_tail_id], device=device, dtype=torch.long)).to(dllm_dtype)
        if effective_tail_id is not None else None
    )

    return SimpleNamespace(
        device=device, wte=wte, dllm_dtype=dllm_dtype,
        inputs_embeds=inputs_embeds, x=x, B=B, L=L, D=D,
        answer_start=answer_start, answer_end=answer_end, gen_length=gen_length,
        mask_id=mask_id, eos_id=eos_id,
        effective_tail_id=effective_tail_id, tail_emb=tail_emb,
    )


def _block_schedule(gen_length, block_length, steps):
    """Round up to allow a partial last block; distribute steps across blocks."""
    num_blocks = math.ceil(gen_length / block_length)
    base = max(1, steps // num_blocks)
    rem = steps - base * num_blocks
    spb = [max(1, base + (1 if i < rem else 0)) for i in range(num_blocks)]
    return num_blocks, spb


def _commit(c, positions: torch.Tensor, tok_ids: torch.Tensor):
    """Write x[:,positions] and inputs_embeds for committed tokens."""
    if positions.numel() == 0:
        return
    new_emb = c.wte(tok_ids).to(c.dllm_dtype)   # [k, D]
    c.x[0, positions] = tok_ids
    c.inputs_embeds[0, positions, :] = new_emb


def _select(c, logits_blk, s, e, *, alg, threshold, fixed_k, factor):
    """Pick block positions to commit from logits_blk [B, blk_len, V].
    Returns (positions [k] global, tok_ids [k]).
    """
    block_mask = (c.x[0, s:e] == c.mask_id)                       # [blk_len]
    probs = F.softmax(logits_blk[0].to(torch.float64), dim=-1)
    conf, tok = probs.max(dim=-1)                                 # [blk_len]
    conf = torch.where(block_mask, conf, torch.full_like(conf, -float("inf")))

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
    return positions, tok_ids


def _apply_eos_fill(c) -> bool:
    """If EOS appears in the answer span, overwrite the tail with tail_token_id
    (ids + embeds) so it is not re-predicted. Returns True if the canvas changed
    (so cached modes can re-warm the KV cache to stay consistent).
    """
    if c.eos_id is None or c.tail_emb is None:
        return False
    ans = c.x[0, c.answer_start:c.answer_end]
    eos_positions = (ans == c.eos_id).nonzero(as_tuple=False).squeeze(-1)
    if eos_positions.numel() == 0:
        return False
    first_eos = int(eos_positions.min().item())
    if first_eos + 1 >= c.gen_length:
        return False
    tail = torch.arange(c.answer_start + first_eos + 1, c.answer_end, device=c.device)
    changed = bool((c.x[0, tail] != c.effective_tail_id).any().item())
    c.x[0, tail] = c.effective_tail_id
    c.inputs_embeds[0, tail, :] = c.tail_emb
    return changed


def _step0_commit_top1(model, c, s, e, *, use_cache):
    """Block step 0: full forward over the whole canvas, commit the single
    highest-confidence masked position (a safe starting point), then EOS-fill.
    Returns (past_key_values_or_None, eos_changed).
    """
    out_full = model(inputs_embeds=c.inputs_embeds, attention_mask=None, use_cache=use_cache)
    logits = _shift_logits_for_ntp(out_full.logits)
    block_mask = (c.x[0, s:e] == c.mask_id)
    block_logits = logits[0, s:e, :]
    probs = F.softmax(block_logits.to(torch.float64), dim=-1)
    conf, tok = probs.max(dim=-1)
    conf = torch.where(block_mask, conf, torch.full_like(conf, -float("inf")))
    top1_local = int(torch.argmax(conf).item())
    _commit(c, torch.tensor([s + top1_local], device=c.device, dtype=torch.long),
            tok[top1_local:top1_local + 1].to(torch.long))
    changed = _apply_eos_fill(c)
    return out_full.past_key_values, changed


def _safety_net(model, c, s, e) -> int:
    """Force-commit any still-masked block positions via a final full forward so
    no block is left incomplete. Returns the number of forwards performed (0/1).
    """
    block_mask = (c.x[0, s:e] == c.mask_id)
    if not block_mask.any():
        return 0
    out_full2 = model(inputs_embeds=c.inputs_embeds, attention_mask=None, use_cache=False)
    logits_full = _shift_logits_for_ntp(out_full2.logits)
    block_logits = logits_full[0, s:e, :]
    probs = F.softmax(block_logits.to(torch.float64), dim=-1)
    conf, tok = probs.max(dim=-1)
    still_masked = torch.nonzero(block_mask, as_tuple=False).squeeze(-1)
    positions = (s + still_masked).to(torch.long)
    tok_ids = tok[still_masked].to(torch.long)
    _commit(c, positions, tok_ids)
    _apply_eos_fill(c)
    return 1


def _block_iter(c, block_length):
    """Yield (s, e, block_index) for each block of the answer span."""
    num_blocks = math.ceil(c.gen_length / block_length)
    for nb in range(num_blocks):
        s = c.answer_start + nb * block_length
        e = min(c.answer_start + (nb + 1) * block_length, c.answer_end)
        yield s, e, nb


def _finish(c, tokenizer):
    out_ids = c.x[0, c.answer_start:c.answer_end].detach().cpu().tolist()
    if c.eos_id is not None and c.eos_id in out_ids:
        eos_pos = out_ids.index(c.eos_id)
        out_ids = out_ids[: eos_pos + 1]
    pred_text = tokenizer.decode(out_ids)
    return pred_text, out_ids


# ============================================================================
# 1) No cache (exact reference)
# ============================================================================
@torch.no_grad()
def generate_embeds_dream(
    *, model, tokenizer, instructions: List[torch.Tensor],
    v_feats: Optional[torch.Tensor], post_user_prefix: Optional[torch.Tensor],
    mask_id: int, gen_length: int, steps: int = 32, block_length: int = 32,
    alg: str = "threshold", threshold: Optional[float] = 0.9, fixed_k: int = 1,
    factor: float = 1.0, tail_token_id: Optional[int] = None,
):
    """Re-forward the full canvas every refinement step. Slowest, exact reference."""
    c = _build_canvas(model=model, tokenizer=tokenizer, instructions=instructions,
                      v_feats=v_feats, post_user_prefix=post_user_prefix,
                      mask_id=mask_id, gen_length=gen_length, tail_token_id=tail_token_id)
    _, spb = _block_schedule(gen_length, block_length, steps)
    nfe = 0

    for s, e, nb in _block_iter(c, block_length):
        if e - s <= 0:
            continue
        if (c.x[:, s:e] == c.mask_id).sum().item() == 0:
            continue

        _step0_commit_top1(model, c, s, e, use_cache=False)
        nfe += 1

        for i in range(1, spb[nb]):
            if (c.x[:, s:e] == c.mask_id).sum().item() == 0:
                break
            out = model(inputs_embeds=c.inputs_embeds, attention_mask=None, use_cache=False)
            logits_blk = _shift_logits_for_ntp(out.logits)[:, s:e, :]
            nfe += 1
            positions, tok_ids = _select(c, logits_blk, s, e,
                                         alg=alg, threshold=threshold, fixed_k=fixed_k, factor=factor)
            _commit(c, positions, tok_ids)
            _apply_eos_fill(c)

        nfe += _safety_net(model, c, s, e)

    pred_text, out_ids = _finish(c, tokenizer)
    return pred_text, out_ids, nfe


# ============================================================================
# 2) Prefix cache (cache KV before the block; re-forward block+suffix)
# ============================================================================
@torch.no_grad()
def generate_with_prefix_cache_embeds_dream(
    *, model, tokenizer, instructions: List[torch.Tensor],
    v_feats: Optional[torch.Tensor], post_user_prefix: Optional[torch.Tensor],
    mask_id: int, gen_length: int, steps: int = 32, block_length: int = 32,
    alg: str = "threshold", threshold: Optional[float] = 0.9, fixed_k: int = 1,
    factor: float = 1.0, tail_token_id: Optional[int] = None,
):
    """Cache KV strictly before the current block; re-forward block(+suffix)
    against it each step. The suffix is always fresh, so only the prefix is
    approximated.
    """
    c = _build_canvas(model=model, tokenizer=tokenizer, instructions=instructions,
                      v_feats=v_feats, post_user_prefix=post_user_prefix,
                      mask_id=mask_id, gen_length=gen_length, tail_token_id=tail_token_id)
    _, spb = _block_schedule(gen_length, block_length, steps)
    nfe = 0

    for s, e, nb in _block_iter(c, block_length):
        if e - s <= 0:
            continue
        if (c.x[:, s:e] == c.mask_id).sum().item() == 0:
            continue

        # step 0: full forward warms the cache; keep only KV strictly before block.
        past_full, cache_stale = _step0_commit_top1(model, c, s, e, use_cache=True)
        past = [(k[:, :s], v[:, :s]) for (k, v) in past_full]
        nfe += 1

        for i in range(1, spb[nb]):
            if (c.x[:, s:e] == c.mask_id).sum().item() == 0:
                break
            if cache_stale:
                # EOS-fill changed the canvas -> cached [:s] prefix KV is stale. Refresh
                # with a FULL forward and REUSE its logits this step, so the refresh is
                # not a wasted forward (saves the +1 nfe a separate re-warm would cost).
                out = model(inputs_embeds=c.inputs_embeds, attention_mask=None, use_cache=True)
                past = [(k[:, :s], v[:, :s]) for (k, v) in out.past_key_values]
                logits_blk = _shift_logits_for_ntp(out.logits)[:, s:e, :]
                cache_stale = False
            else:
                # Re-forward block(+suffix) against the cached prefix KV (standard concat).
                out = model(inputs_embeds=c.inputs_embeds[:, s:, :], attention_mask=None,
                            past_key_values=past, use_cache=False)
                logits_blk = _shift_logits_for_ntp(out.logits)[:, : e - s, :]
            nfe += 1
            positions, tok_ids = _select(c, logits_blk, s, e,
                                         alg=alg, threshold=threshold, fixed_k=fixed_k, factor=factor)
            _commit(c, positions, tok_ids)
            if _apply_eos_fill(c):
                cache_stale = True

        nfe += _safety_net(model, c, s, e)

    pred_text, out_ids = _finish(c, tokenizer)
    return pred_text, out_ids, nfe


# ============================================================================
# 3) Dual cache (cache prefix+suffix; re-forward only the block slice)
# ============================================================================
@torch.no_grad()
def generate_with_dual_cache_embeds_dream(
    *, model, tokenizer, instructions: List[torch.Tensor],
    v_feats: Optional[torch.Tensor], post_user_prefix: Optional[torch.Tensor],
    mask_id: int, gen_length: int, steps: int = 32, block_length: int = 32,
    alg: str = "threshold", threshold: Optional[float] = 0.9, fixed_k: int = 1,
    factor: float = 1.0, tail_token_id: Optional[int] = None,
):
    """Cache prefix AND suffix; re-forward only the block slice via
    replace_position. Fastest. (Dream requires the explicit dual_cache=True flag.)
    """
    c = _build_canvas(model=model, tokenizer=tokenizer, instructions=instructions,
                      v_feats=v_feats, post_user_prefix=post_user_prefix,
                      mask_id=mask_id, gen_length=gen_length, tail_token_id=tail_token_id)
    _, spb = _block_schedule(gen_length, block_length, steps)
    nfe = 0

    for s, e, nb in _block_iter(c, block_length):
        if e - s <= 0:
            continue
        if (c.x[:, s:e] == c.mask_id).sum().item() == 0:
            continue

        # step 0: full forward warms the full prefix+suffix cache.
        past, cache_stale = _step0_commit_top1(model, c, s, e, use_cache=True)
        nfe += 1

        replace_position = torch.zeros((c.B, c.L), device=c.device, dtype=torch.bool)
        replace_position[:, s:e] = True

        for i in range(1, spb[nb]):
            if (c.x[:, s:e] == c.mask_id).sum().item() == 0:
                break
            if cache_stale:
                # EOS-fill changed positions outside the block -> the cached suffix KV is
                # stale. Refresh with a FULL forward and REUSE its logits this step, so the
                # refresh is not a wasted forward (saves the +1 nfe a separate re-warm cost).
                out_blk = model(inputs_embeds=c.inputs_embeds, attention_mask=None, use_cache=True)
                past = out_blk.past_key_values
                logits_blk = _shift_logits_for_ntp(out_blk.logits)[:, s:e, :]
                cache_stale = False
            else:
                # Re-forward only the block slice; block KV replaces into the full cache.
                out_blk = model(inputs_embeds=c.inputs_embeds[:, s:e, :], attention_mask=None,
                                past_key_values=past, use_cache=True,
                                dual_cache=True, replace_position=replace_position)
                logits_blk = _shift_logits_for_ntp(out_blk.logits)   # [B, blk_len, V]
            nfe += 1
            positions, tok_ids = _select(c, logits_blk, s, e,
                                         alg=alg, threshold=threshold, fixed_k=fixed_k, factor=factor)
            _commit(c, positions, tok_ids)
            if _apply_eos_fill(c):
                cache_stale = True

        nfe += _safety_net(model, c, s, e)

    pred_text, out_ids = _finish(c, tokenizer)
    return pred_text, out_ids, nfe


# Dispatch table for callers that select the cache strategy by string.
GENERATORS = {
    "none": generate_embeds_dream,
    "prefix": generate_with_prefix_cache_embeds_dream,
    "dual": generate_with_dual_cache_embeds_dream,
}
