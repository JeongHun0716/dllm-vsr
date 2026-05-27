# `eval/`

Inference entry points. Typically invoked through the runners in `scripts/eval/`; see the [top-level README](../README.md) for usage.

| File | Role |
|---|---|
| `length_predictor.py`        | Length-predictor inference → per-utterance `len_pred_<split>.jsonl` (top-1 + ±5 probability range) |
| `dream_eval.py`              | Single-K Dream forward (block-wise iterative commit; supports `dual_cache` for Fast-dLLM) |
| `dream_canvas_pinned_eos.py` | Multi-K canvas + pinned-EOS probe (paper main; produces `shard_*_raw.jsonl` for rerank) |
