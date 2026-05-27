#!/usr/bin/env bash
# Multi-K canvas + pinned-EOS rerank (Row 4 main result).
# Usage:
#   CONFIG=configs/lrs3/usr2_v2.yaml CKPT=ckpt/usr2/dream_stage2 \
#   LEN_JSONL=ckpt/usr2/len_pred/len_pred_test.jsonl \
#   GPU=0 bash scripts/eval/steps/run_canvas_probe.sh
#
# Optional:
#   CANVAS_LEN=108, BLOCK_SIZE=32, THRESHOLD=0.9,
#   LAMBDAS="0,0.5,1.0,2.0,5.0" (rerank λ grid),
#   OUT_DIR=<ckpt>/canvas108_pinnedeos_shards
GPU=${GPU:-0}
CONFIG=${CONFIG:?usage: CONFIG=configs/lrs3/usr2_v2.yaml ...}
CKPT=${CKPT:?usage: CKPT=ckpt/usr2/dream_stage2}
LEN_JSONL=${LEN_JSONL:?usage: LEN_JSONL=ckpt/usr2/len_pred/len_pred_test.jsonl}
CANVAS_LEN=${CANVAS_LEN:-108}
BLOCK_SIZE=${BLOCK_SIZE:-32}
THRESHOLD=${THRESHOLD:-0.9}
LAMBDAS=${LAMBDAS:-"0,0.5,1.0,2.0,5.0"}
CUDA_VISIBLE_DEVICES=${GPU} python eval/dream_canvas_pinned_eos.py \
    config=${CONFIG} \
    eval.ckpt_path=${CKPT} \
    eval.len_jsonl=${LEN_JSONL} \
    eval.canvas_len=${CANVAS_LEN} \
    eval.block_size=${BLOCK_SIZE} \
    eval.threshold=${THRESHOLD} \
    eval.lambdas=${LAMBDAS} \
    ${OUT_DIR:+eval.out_dir=${OUT_DIR}}
