# `eval/`

Inference entry points. Typically invoked through the runners in `scripts/eval/`; see the [top-level README](../README.md) for usage.

| File | Role |
|---|---|
| `length_predictor.py`        | Length-predictor inference → per-utterance `len_pred_<split>.jsonl` (top-1 + the K_pred ± 5 candidate pool with length probabilities) |
| `eval_core.py`               | Shared sharded eval core: model/loader build + `run_one_setting` (decode shard → `accelerator.reduce` → exact global WER) |
| `eval_main.py`               | Single-setting Dream forward, sharded across GPUs (`accelerate launch --num_processes N eval/eval_main.py`). `eval.cache_mode ∈ {none,prefix,dual}`, `eval.alg ∈ {threshold,fixed_k,dynamic}` |
| `dream_canvas_pinned_eos.py` | Multi-K canvas + pinned-EOS decoder (paper main; produces `shard_*_raw.jsonl` for rerank) |
