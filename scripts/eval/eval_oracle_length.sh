#!/usr/bin/env bash
#
# Decoding:
#   - oracle-length decoding: canvas length = ground-truth transcript+EOS length (upper bound)
#   - full-parallel confidence-based commitment (block_size=32 = canvas, threshold=0.9)
#   - no KV cache (exact bidirectional forward)
#
# This is an upper-bound reference (ground-truth length is never available at deployment).
# Use eval_implicit_length.sh for implicit-length decoding,
# or eval_length_guided.sh for length-guided candidate decoding (paper main).
#
# Expected (paper):
#   - USR2     = 17.7%
#   - AvHubert = 20.2%
#
# Prereqs (one-time):
#   conda activate dllm-vsr
#   bash scripts/setup_paths.sh /abs/path/to/lrs3_mouth_rois_root
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

echo "===== ${BACKBONE} | oracle-length decoding ====="
CUDA_VISIBLE_DEVICES=${GPU} \
CONFIG=configs/lrs3/${BACKBONE}_v2.yaml \
CKPT=ckpt/${BACKBONE}/dream_stage2 \
USE_ORACLE_LENGTH=true \
BLOCK_SIZE=32 \
THRESHOLD=0.9 \
CACHE_MODE=none \
bash scripts/eval/steps/run_dream_forward.sh

echo
echo "===== Done ====="
echo "FINAL WER line is logged above."
