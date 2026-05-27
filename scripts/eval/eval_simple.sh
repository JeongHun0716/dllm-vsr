#!/usr/bin/env bash
#
# Decoding:
#   - fixed canvas: answer_len=32 (use_oracle_length=false)
#   - block-wise iterative commit: block_size=32, threshold=0.9
#   - no dual-cache (standard MDM forward)
#
# Expected (paper):
#   - USR2     = 20.5%
#   - AvHubert = 23.1%
#
# Prereqs (one-time):
#   conda activate dllm-vsr
#   bash scripts/setup_paths.sh /abs/path/to/lrs3_videos
#
# Usage (default USR2 on GPU 0):
#   bash scripts/eval/eval_simple.sh
#
# To run AvHubert instead:
#   BACKBONE=avhubert bash scripts/eval/eval_simple.sh
#
# ETA: ~9 minutes (single GPU)
set -e

BACKBONE=${BACKBONE:-usr2}     # usr2 | avhubert
GPU=${GPU:-0}

echo "===== ${BACKBONE} | single-K Dream forward (fixed canvas=32, no lenpred) ====="
CONFIG=configs/lrs3/${BACKBONE}_v2.yaml \
CKPT=ckpt/${BACKBONE}/dream_stage2 \
GPU=${GPU} \
USE_ORACLE_LENGTH=false \
ANSWER_LEN=32 \
BLOCK_SIZE=32 \
THRESHOLD=0.9 \
DUAL_CACHE=false \
bash scripts/eval/steps/run_dream_forward.sh

echo
echo "===== Done ====="
echo "FINAL WER line is logged above."
