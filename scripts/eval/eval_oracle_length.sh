#!/usr/bin/env bash
#
# Decoding:
#   - oracle length: gen_length = ground-truth transcript+EOS length (test info leakage)
#   - block-wise iterative commit: block_size=32, threshold=0.9
#   - no dual-cache (standard MDM forward)
#
# This is an upper-bound reference (ground-truth length is never available at deployment).
# Use eval_simple.sh for the fixed-canvas baseline (paper Row 1),
# or eval_main.sh for the paper main result with the length predictor.
#
# Prereqs (one-time):
#   conda activate dllm-vsr
#   bash scripts/setup_paths.sh /abs/path/to/lrs3_videos
#
# Usage (default USR2 on GPU 0):
#   bash scripts/eval/eval_oracle_length.sh
#
# To run AvHubert instead:
#   BACKBONE=avhubert bash scripts/eval/eval_oracle_length.sh
#
# ETA: ~6 minutes (single GPU)
set -e

BACKBONE=${BACKBONE:-usr2}     # usr2 | avhubert
GPU=${GPU:-0}

echo "===== ${BACKBONE} | single-K Dream forward (oracle length) ====="
CONFIG=configs/lrs3/${BACKBONE}_v2.yaml \
CKPT=ckpt/${BACKBONE}/dream_stage2 \
GPU=${GPU} \
USE_ORACLE_LENGTH=true \
BLOCK_SIZE=32 \
THRESHOLD=0.9 \
DUAL_CACHE=false \
bash scripts/eval/steps/run_dream_forward.sh

echo
echo "===== Done ====="
echo "FINAL WER line is logged above."
