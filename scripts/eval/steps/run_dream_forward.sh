#!/usr/bin/env bash
# Single-K Dream forward eval (Row 1/2: greedy/threshold, no rerank).
# Usage:
#   CONFIG=configs/lrs3/usr2_v2.yaml CKPT=ckpt/usr2/dream_stage2 \
#   GPU=0 bash scripts/eval/steps/run_dream_forward.sh
#
# Optional overrides (default in parens):
#   ALG=threshold (threshold|fixed_k|...), THRESHOLD=0.9, BLOCK_SIZE=32, STEPS=128,
#   DUAL_CACHE=false (true = Fast-dLLM KV cache reuse),
#   ANSWER_LEN=32 (fixed canvas length when USE_ORACLE_LENGTH=false),
#   USE_ORACLE_LENGTH=false (true = use ground-truth transcript length; test info leakage)
GPU=${GPU:-0}
CONFIG=${CONFIG:?usage: CONFIG=configs/lrs3/usr2_v2.yaml ...}
CKPT=${CKPT:?usage: CKPT=ckpt/usr2/dream_stage2}
ALG=${ALG:-threshold}
THRESHOLD=${THRESHOLD:-0.9}
BLOCK_SIZE=${BLOCK_SIZE:-32}
STEPS=${STEPS:-128}
DUAL_CACHE=${DUAL_CACHE:-false}
ANSWER_LEN=${ANSWER_LEN:-32}
USE_ORACLE_LENGTH=${USE_ORACLE_LENGTH:-false}
CUDA_VISIBLE_DEVICES=${GPU} python eval/dream_eval.py \
    config=${CONFIG} \
    experiment.eval_checkpoint=${CKPT} \
    model.use_oracle_length=${USE_ORACLE_LENGTH} \
    eval.alg=${ALG} \
    eval.threshold=${THRESHOLD} \
    eval.block_size=${BLOCK_SIZE} \
    eval.steps=${STEPS} \
    eval.dual_cache=${DUAL_CACHE} \
    eval.answer_len=${ANSWER_LEN}
