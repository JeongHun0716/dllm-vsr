#!/usr/bin/env bash
# Dump per-utterance length predictions (.jsonl) for downstream rerank.
# Usage:
#   CONFIG=configs/lrs3/len_pred_usr2.yaml CKPT=ckpt/usr2/len_pred \
#   GPU=0 bash scripts/eval/steps/run_length_predictor.sh
GPU=${GPU:-0}
CONFIG=${CONFIG:?usage: CONFIG=configs/lrs3/len_pred_usr2.yaml ...}
CKPT=${CKPT:?usage: CKPT=ckpt/usr2/len_pred}
CUDA_VISIBLE_DEVICES=${GPU} python eval/length_predictor.py \
    config=${CONFIG} \
    experiment.eval_checkpoint=${CKPT}
