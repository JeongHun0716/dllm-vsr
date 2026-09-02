"""Multi-K canvas decoder: per-candidate canvas + pinned EOS at position K-1 (paper main).

Decodes the length-candidate pool (K_pred ± 5 by default) in one batch per
utterance. For each candidate K in the pool:
  - Positions [0..K-2]  = <mask>  (transcript positions to decode)
  - Position  [K-1]     = EOS    (pinned; the canvas ends here by default)
  - Iteratively commit masked positions with confidence-based selection
    (per-step commit cap `block_size` + confidence threshold, on a fixed canvas
    with a pinned EOS suffix).
  - lm_score = mean log(commit_conf) over decoded positions.
The pool is decoded in a single batched forward pass per step.

Output: shard_*_raw.jsonl with per-K candidate rows. The offline rerank stage
(scripts/eval/rerank/) selects the final transcript within a search radius R
(default 3) and can report the full R sweep.
"""
from __future__ import annotations
import os, sys, json, math, time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLLM_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _DLLM_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DLLM_VSR_USE_FAST_DREAM"] = "1"

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import load_file as safe_load_file
from transformers import AutoTokenizer
import editdistance

from training.utils import get_config
from training.data import VSRDataset
from models.model import build_model


def load_len_predictions(jsonl_path: str) -> dict:
    table = {}
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line.strip())
            table[d["utt_id"]] = d
    return table


@torch.no_grad()
def decode_canvas_with_pinned_eos_batched(
    *, model, wte, inst_emb, feat_emb, post_user_emb, mask_id, eos_id, pad_id,
    K_list, canvas_len, block_size, threshold, cache_mode: str = "none",
):
    """Batched: each row b has its EOS pinned at position K_list[b].

    cache_mode:
      - "none"  : re-forward the whole [prompt | canvas] every step (exact).
      - "prefix": the prompt (inst+feat+post_user) is fixed across all steps, so
                  warm its KV once and re-forward only the canvas slice each step.
                  EOS is pinned (no EOS-fill events), so no re-warm is needed. The
                  cached prompt KV is computed while the canvas is masked, so (Dream
                  being bidirectional) "prefix" is an approximation of "none".

    Returns:
        pred_ids   [B, canvas_len]  (final canvas tokens)
        commit_conf [B, canvas_len]  (commit-time conf; 0 at pinned positions)
        nfe         List[int]         (model forward steps per row, conservative)
    """
    device = next(model.parameters()).device
    dllm_dtype = next(model.parameters()).dtype
    B = len(K_list)

    L_prompt = inst_emb.size(0) + feat_emb.size(1) + post_user_emb.size(0)
    D = inst_emb.size(-1)

    mask_emb_v = wte(torch.tensor([mask_id], device=device, dtype=torch.long)).squeeze(0).to(dllm_dtype)
    eos_emb_v = wte(torch.tensor([eos_id], device=device, dtype=torch.long)).squeeze(0).to(dllm_dtype)
    pad_emb_t_v = wte(torch.tensor([pad_id], device=device, dtype=torch.long)).squeeze(0).to(dllm_dtype)

    # Prompt shared per row
    prompt_emb = torch.cat([inst_emb, feat_emb[0], post_user_emb], dim=0).to(dllm_dtype)

    L_total = L_prompt + canvas_len
    inputs_embeds = torch.empty(B, L_total, D, device=device, dtype=dllm_dtype)
    attention_mask = torch.ones(B, L_total, device=device, dtype=torch.long)
    pred_ids = torch.full((B, canvas_len), mask_id, device=device, dtype=torch.long)
    commit_conf = torch.zeros(B, canvas_len, device=device)
    # per-row mask: True where still masked (decoding target)
    masked = torch.zeros(B, canvas_len, dtype=torch.bool, device=device)

    for b in range(B):
        Kb = int(K_list[b])
        # K from the length predictor counts transcript tokens + EOS
        # → decode K-1 transcript positions, pin EOS at position K-1.
        Ntrans = max(0, Kb - 1)
        inputs_embeds[b, :L_prompt] = prompt_emb
        if Ntrans > 0:
            inputs_embeds[b, L_prompt : L_prompt + Ntrans] = mask_emb_v
            masked[b, :Ntrans] = True
        if Ntrans < canvas_len:
            inputs_embeds[b, L_prompt + Ntrans] = eos_emb_v
            pred_ids[b, Ntrans] = eos_id
        if Ntrans + 1 < canvas_len:
            inputs_embeds[b, L_prompt + Ntrans + 1 : L_total] = pad_emb_t_v
            pred_ids[b, Ntrans + 1 :] = pad_id

    nfe = [0] * B
    done = torch.zeros(B, dtype=torch.bool, device=device)
    prompt_kv = None   # cache_mode="prefix": fixed prompt KV, warmed once
    # Each step commits up to `block_size` highest-confidence masked positions
    # per row (threshold fallback to top-1), so max(K_list) steps always suffice.
    for step in range(max(K_list)):
        if bool(done.all()):
            break
        # Dream: attention_mask passed as [B, 1, 1, L_total] bool (key-side).
        am = attention_mask.to(torch.bool)[:, None, None, :]
        if cache_mode == "prefix":
            if prompt_kv is None:
                # warm: full forward; cache the (fixed) prompt KV; reuse its logits this step.
                out = model(inputs_embeds=inputs_embeds, attention_mask=am, use_cache=True)
                prompt_kv = [(k[:, :L_prompt], v[:, :L_prompt]) for (k, v) in out.past_key_values]
                ans_logits = out.logits[:, L_prompt:, :]            # [B, canvas_len, V]
            else:
                # cached: forward only the canvas slice against the fixed prompt KV (concat).
                out = model(inputs_embeds=inputs_embeds[:, L_prompt:, :], attention_mask=am,
                            past_key_values=prompt_kv, use_cache=False)
                ans_logits = out.logits                              # [B, canvas_len, V]
        else:
            out = model(inputs_embeds=inputs_embeds, attention_mask=am)
            ans_logits = out.logits[:, L_prompt:, :]                 # [B, canvas_len, V]

        for b in range(B):
            if bool(done[b]):
                continue
            nfe[b] += 1
            row_masked = masked[b]
            if not bool(row_masked.any()):
                done[b] = True
                continue
            masked_idx = torch.nonzero(row_masked, as_tuple=False).squeeze(1)
            sub = ans_logits[b, masked_idx, :].float()
            probs = F.softmax(sub, dim=-1)
            conf, tok_pred = probs.max(dim=-1)

            # threshold commit: pass if conf > th; else top-1 fallback
            # (always commit at least 1 position to make progress)
            pass_mask = conf > float(threshold)
            n_block = min(int(block_size), int(masked_idx.numel()))
            if bool(pass_mask.any()):
                eff_k = min(n_block, int(pass_mask.sum().item()))
                cand_conf = torch.where(pass_mask, conf, torch.full_like(conf, -1.0))
                topk_conf, topk_local = torch.topk(cand_conf, k=eff_k, largest=True)
            else:
                topk_conf, topk_local = torch.topk(conf, k=1, largest=True)
            chosen = masked_idx[topk_local]              # global positions in canvas
            committed_tokens = tok_pred[topk_local]
            pred_ids[b, chosen] = committed_tokens
            commit_conf[b, chosen] = topk_conf

            new_emb = wte(committed_tokens).to(dllm_dtype)
            inputs_embeds[b, L_prompt + chosen, :] = new_emb
            masked[b, chosen] = False
            if not bool(masked[b].any()):
                done[b] = True
    return pred_ids, commit_conf, nfe


def main():
    config = get_config()
    accelerator = Accelerator()
    eval_cfg = config.get("eval", OmegaConf.create({}))
    n_samples = int(eval_cfg.get("n_samples", -1))
    len_jsonl = str(eval_cfg.get("len_jsonl"))
    # λ values for the online length-prior-only progress display; the final
    # (λ, β) rerank runs offline over the raw jsonl (scripts/eval/rerank/).
    lambdas = eval_cfg.get("lambdas", [0.9])
    # Sharding: under a multi-process `accelerate launch`, derive the shard from
    # the process rank so a single command fans out across all GPUs. For a plain
    # single-process run, fall back to explicit eval.shard_idx / eval.num_shards.
    if accelerator.num_processes > 1:
        shard_idx = accelerator.process_index
        num_shards = accelerator.num_processes
    else:
        shard_idx = int(eval_cfg.get("shard_idx", 0))
        num_shards = int(eval_cfg.get("num_shards", 1))
    dump_raw = bool(eval_cfg.get("dump_raw", True))
    canvas_len = int(eval_cfg.get("canvas_len", 32))
    block_size = int(eval_cfg.get("block_size", 32))
    threshold = float(eval_cfg.get("threshold", 0.9))
    cache_mode = str(eval_cfg.get("cache_mode", "none"))   # none | prefix (prompt-KV reuse)
    # The full dumped window (±5) is decoded by default; the rerank stage
    # applies the search radius R. Set eval.range_radius>=0 to narrow the
    # decoded pool (optional).
    range_radius = int(eval_cfg.get("range_radius", -1))
    # canvas sizing: True (default) sizes the batch canvas by its max K;
    # False pads every candidate to eval.canvas_len.
    batch_max_canvas = bool(eval_cfg.get("batch_max_canvas", True))

    if isinstance(lambdas, str):
        lambdas = [float(x) for x in lambdas.split(",")]
    else:
        lambdas = [float(x) for x in lambdas]

    ckpt_dir = Path(str(eval_cfg.ckpt_path))
    out_dir = Path(eval_cfg.get("out_dir", str(ckpt_dir / "canvas_pinnedeos_shards")))
    out_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"checkpoint: {ckpt_dir}")
        print(f"canvas={canvas_len} block_size={block_size} threshold={threshold} "
              f"cache_mode={cache_mode} R={range_radius}")
        print(f"len_jsonl: {len_jsonl}")
        print(f"out_dir:   {out_dir}")
        print(f"num_shards: {num_shards}")
    print(f"shard:     {shard_idx}/{num_shards}  (device={accelerator.device})")

    device = accelerator.device
    model = build_model(config).to(device)
    sd = safe_load_file(str(ckpt_dir / "trainable_model.safetensors"))
    model.load_state_dict(sd, strict=False)
    model.eval()

    tok_path = str(config.model.dllm.tokenizer_path)
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, padding_side="left")
    mask_id = int(getattr(model.dllm.config, "mask_token_id", None) or tok.mask_token_id)
    eos_id = int(tok.eos_token_id)
    pad_id = int(getattr(model, "pad_id", eos_id))

    manifest_root = str(config.dataset.vsr.manifest_root)
    raw_cfg = config.dataset.get("raw_video", {})
    split_name = str(eval_cfg.get("split", "test"))
    ds = VSRDataset(
        manifest_path=os.path.join(manifest_root, f"{split_name}.tsv"),
        label_paths=[os.path.join(manifest_root, f"{split_name}.wrd")],
        subset="test",
        dllm_tokenizer_path=tok_path,
        max_video_frames=int(config.dataset.vsr.max_video_frames),
        modalities=list(config.dataset.vsr.modalities),
        video_root=config.dataset.vsr.get("video_root", None),
        crop_size=int(raw_cfg.get("crop_size", 88)),
        normalize_mean=float(raw_cfg.get("normalize_mean", 0.421)),
        normalize_std=float(raw_cfg.get("normalize_std", 0.165)),
        time_mask_window=int(raw_cfg.get("time_mask_window", 10)),
        time_mask_stride=int(raw_cfg.get("time_mask_stride", 25)),
        enable_time_mask=False,
    )

    len_table = load_len_predictions(len_jsonl)
    print(f"loaded len predictions: {len(len_table)} entries")

    wte = model._get_wte()
    v_in_dtype = model.video_input_dtype()

    stats = {lam: {"n_match": 0, "wer_err": 0, "wer_total": 0} for lam in lambdas}
    n_oracle_err = 0; n_oracle_gt = 0
    n_processed = 0

    N_run = len(ds) if n_samples <= 0 else min(n_samples, len(ds))
    my_indices = list(range(shard_idx, N_run, num_shards))
    print(f"[shard {shard_idx}] will process {len(my_indices)}/{N_run}")

    n_skipped_video = 0
    n_skipped_len = 0
    raw_f = None
    if dump_raw:
        raw_path = out_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}_raw.jsonl"
        raw_f = open(raw_path, "w")
        print(f"[dump_raw] {raw_path}")

    EPS = 1e-12
    for idx in my_indices:
        item = ds[idx]
        if item["video_features"] is None:
            n_skipped_video += 1
            continue
        utt_id = item["utt_id"]
        if utt_id not in len_table:
            n_skipped_len += 1
            continue
        rec = len_table[utt_id]
        v_feat = item["video_features"].unsqueeze(0).to(device, dtype=v_in_dtype)
        inst = item["input_ids"].to(device)
        post_user = item["post_user_ids"].to(device)
        labels = item["labels"]
        oracle = int(labels.numel())
        assert oracle == rec["K_target"], f"mismatch utt={utt_id}"
        gt = ds.labels[idx].strip()

        K_range = rec["K_range"]
        probs_len = rec["probs"]
        pred_K = int(rec["pred_K"])
        # filter: search window |K - pred_K| <= R, K >= 1 (need at least 1 token
        # before EOS) and K < canvas_len
        candidates = [(k, p) for k, p in zip(K_range, probs_len)
                      if 1 <= k < canvas_len
                      and (range_radius < 0 or abs(k - pred_K) <= range_radius)]

        # per-utterance wall-clock (video encode + all-candidate decode) for RTF
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_utt = time.perf_counter()

        with torch.no_grad():
            v_enc, len_v, _ = model.encode_video(video_feats=v_feat, video_pad=None, apply_fc=True)
            v = v_enc[0]
            v_len = min(int(len_v[0]), v.size(0))
            feat_emb = v[:v_len, :].to(dtype=wte.weight.dtype).unsqueeze(0)
            inst_emb = wte(inst).to(dtype=wte.weight.dtype)
            post_user_emb = wte(post_user).to(dtype=wte.weight.dtype)

            K_list = [c[0] for c in candidates]
            # K already includes EOS. canvas must hold transcript (K-1) + EOS (1) = K positions.
            eff_canvas = max(K_list) if batch_max_canvas else canvas_len
            pred_ids_b, commit_conf_b, nfe_b = decode_canvas_with_pinned_eos_batched(
                model=model.dllm, wte=wte, inst_emb=inst_emb, feat_emb=feat_emb,
                post_user_emb=post_user_emb, mask_id=mask_id, eos_id=eos_id, pad_id=pad_id,
                K_list=K_list, canvas_len=eff_canvas, block_size=block_size, threshold=threshold,
                cache_mode=cache_mode,
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_sec = time.perf_counter() - t_utt

        rows = []
        for b, (K, p_len) in enumerate(candidates):
            Ntrans = max(0, K - 1)
            ans_tokens = pred_ids_b[b, :Ntrans].tolist()
            commit_confs_raw = commit_conf_b[b, :Ntrans].clamp_min(EPS).cpu().tolist()
            commit_logconfs = [math.log(max(c, EPS)) for c in commit_confs_raw]
            lm_score_mean = sum(commit_logconfs) / len(commit_logconfs) if Ntrans > 0 else float("-inf")
            pred_text = tok.decode(ans_tokens, skip_special_tokens=False)
            pred_clean = pred_text.replace(tok.eos_token, "").strip()
            pr_words = pred_clean.split()
            gt_words = gt.split()
            ed = editdistance.eval(pr_words, gt_words)
            rows.append({
                "K": K, "lm_score": lm_score_mean, "p_len": p_len,
                "log_p_len": math.log(max(p_len, EPS)),
                "pred": pred_clean, "ed": ed, "nfe": nfe_b[b],
                # consumed by the offline rerank scripts
                "commit_confs": commit_confs_raw,    # [Ntrans] per-position commit-time conf
                "tokens": ans_tokens,                # token id sequence
            })

        n_gt_words = len(gt.split())
        # Include utterance in eval even when oracle K is not among candidates
        # (real deployment also has utts outside the lenpred ±R range; in that
        # case the rerank winner's prediction is used).
        # ed_oracle is only recorded when oracle K is in candidates.
        if any(r["K"] == oracle for r in rows):
            ed_oracle = next(r["ed"] for r in rows if r["K"] == oracle)
            n_oracle_err += ed_oracle
            n_oracle_gt += n_gt_words
        else:
            ed_oracle = -1   # oracle K not in candidates (sentinel)
            # do NOT accumulate n_oracle_err / n_oracle_gt (oracle WER metric)
            # but rerank lambda WER is still accumulated

        if raw_f is not None:
            raw_f.write(json.dumps({
                "utt_id": utt_id,
                "oracle_K": oracle,
                "K_target": rec["K_target"],
                "pred_K_lenpred": rec["pred_K"],
                "argmax_prob": rec.get("argmax_prob", None),
                "n_gt_words": n_gt_words,
                "ed_oracle": ed_oracle,
                "elapsed_sec": elapsed_sec,
                "rows": [
                    {"K": r["K"], "lm_score": r["lm_score"], "log_p_len": r["log_p_len"],
                     "p_len": r["p_len"], "ed": r["ed"], "pred": r["pred"], "nfe": r["nfe"],
                     "commit_confs": r["commit_confs"], "tokens": r["tokens"]}
                    for r in rows
                ],
            }) + "\n")
            raw_f.flush()

        for lam in lambdas:
            best = max(rows, key=lambda r: r["lm_score"] + lam * r["log_p_len"])
            stats[lam]["n_match"] += int(best["K"] == oracle)
            stats[lam]["wer_err"] += best["ed"]
            stats[lam]["wer_total"] += n_gt_words

        n_processed += 1
        log_every = int(eval_cfg.get("log_every", 25))
        is_last = (idx == my_indices[-1])
        if n_processed % log_every == 0 or is_last:
            wer_oracle = 100 * n_oracle_err / max(1, n_oracle_gt)
            line = f"[shard {shard_idx} {n_processed}/{len(my_indices)}] oracle={wer_oracle:.2f}%"
            for lam in lambdas:
                s = stats[lam]
                w = 100 * s["wer_err"] / max(1, s["wer_total"])
                line += f" | λ{lam}: w={w:.2f}%"
            print(line, flush=True)

    print()
    print(f"=== shard {shard_idx} done ({n_processed}) ===")
    if n_skipped_video or n_skipped_len:
        # surface silent drops: a missing utterance skews WER comparability
        print(f"[warn] skipped {n_skipped_video} utt(s) with unreadable video, "
              f"{n_skipped_len} utt(s) missing from len_jsonl")
    wer_oracle = 100 * n_oracle_err / max(1, n_oracle_gt)
    print(f"oracle WER: {wer_oracle:.2f}%")
    for lam in lambdas:
        s = stats[lam]
        w = 100 * s["wer_err"] / max(1, s["wer_total"])
        print(f"  λ={lam:.2f}: WER={w:.2f}%")

    result = {
        "ckpt": str(ckpt_dir),
        "len_jsonl": len_jsonl,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "n_processed": n_processed,
        "n_oracle_err": int(n_oracle_err),
        "n_oracle_gt": int(n_oracle_gt),
        "decoder": "canvas-pinned-eos",
        "canvas_len": canvas_len, "block_size": block_size, "threshold": threshold,
        "range_radius": range_radius,
        "lambdas": [float(l) for l in lambdas],
        "per_lambda": {
            str(lam): {
                "n_match": int(stats[lam]["n_match"]),
                "wer_err": int(stats[lam]["wer_err"]),
                "wer_total": int(stats[lam]["wer_total"]),
            } for lam in lambdas
        },
    }
    out_path = out_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[write] {out_path}")

    if raw_f is not None:
        raw_f.close()

    # Barrier: don't let the launcher return until every shard has written its files,
    # so the downstream rerank/merge step sees a complete set of shards.
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
