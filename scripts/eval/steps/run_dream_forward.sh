#!/usr/bin/env bash
# Single-setting Dream forward eval (implicit-length / oracle-length decoding, no reranking).
# Sharded across NUM_GPUS via accelerate -> one exact WER (no manual shard merge).
# Usage:
#   CONFIG=configs/lrs3/usr2_v2.yaml CKPT=ckpt/usr2/dream_stage2 \
#   NUM_GPUS=8 bash scripts/eval/steps/run_dream_forward.sh
#
# Optional overrides (default in parens):
#   ALG=threshold (threshold|fixed_k|dynamic), THRESHOLD=0.9, BLOCK_SIZE=32, STEPS=128,
#   CACHE_MODE=none (none|prefix|dual; none = exact reference, dual = Fast-dLLM KV reuse),
#   ANSWER_LEN=32 (fixed canvas length when USE_ORACLE_LENGTH=false),
#   USE_ORACLE_LENGTH=false (true = use ground-truth transcript length; test info leakage),
#   NUM_GPUS=1 (data-parallel shard count)
CONFIG=${CONFIG:?usage: CONFIG=configs/lrs3/usr2_v2.yaml ...}
CKPT=${CKPT:?usage: CKPT=ckpt/usr2/dream_stage2}
ALG=${ALG:-threshold}
THRESHOLD=${THRESHOLD:-0.9}
BLOCK_SIZE=${BLOCK_SIZE:-32}
STEPS=${STEPS:-128}
CACHE_MODE=${CACHE_MODE:-none}
ANSWER_LEN=${ANSWER_LEN:-32}
USE_ORACLE_LENGTH=${USE_ORACLE_LENGTH:-false}
TAIL_TOKEN=${TAIL_TOKEN:-pad}   # pad | eos — token used to fill positions after a committed EOS
NUM_GPUS=${NUM_GPUS:-1}

MULTI_GPU=""
if [[ "${NUM_GPUS}" -gt 1 ]]; then MULTI_GPU="--multi_gpu"; fi

accelerate launch --num_processes "${NUM_GPUS}" ${MULTI_GPU} \
    eval/eval_main.py \
    config=${CONFIG} \
    experiment.eval_checkpoint=${CKPT} \
    model.use_oracle_length=${USE_ORACLE_LENGTH} \
    eval.alg=${ALG} \
    eval.threshold=${THRESHOLD} \
    eval.block_size=${BLOCK_SIZE} \
    eval.steps=${STEPS} \
    eval.cache_mode=${CACHE_MODE} \
    eval.answer_len=${ANSWER_LEN} \
    eval.tail_token=${TAIL_TOKEN}
