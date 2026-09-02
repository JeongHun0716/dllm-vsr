#!/usr/bin/env bash
#
# Decoding:
#   - implicit-length decoding: fixed canvas T=32 (answer_len=32), length inferred via EOS
#   - full-parallel confidence-based commitment (block_size=32 = canvas, threshold=0.9)
#   - no KV cache (exact bidirectional forward)
#
# Expected with defaults (positions after EOS filled with PAD, matching the
# stage-2 training distribution):
#   - USR2     = 20.4%
#   - AvHubert = 23.0%
# The paper's implicit-length numbers (20.5% / 23.1%) were measured with EOS
# tail filling; add eval.tail_token=eos to reproduce them exactly.
#
# Prereqs (one-time):
#   conda activate dllm-vsr
#   bash scripts/setup_paths.sh /abs/path/to/lrs3_mouth_rois_root
#
# Usage (default USR2 on GPU 0):
#   bash scripts/eval/eval_implicit_length.sh
#
# To run AvHubert instead:
#   BACKBONE=avhubert bash scripts/eval/eval_implicit_length.sh
#
# ETA: ~9 minutes (single GPU)
set -e

BACKBONE=${BACKBONE:-usr2}     # usr2 | avhubert
GPU=${GPU:-0}

echo "===== ${BACKBONE} | implicit-length decoding (fixed canvas T=32) ====="
CUDA_VISIBLE_DEVICES=${GPU} \
CONFIG=configs/lrs3/${BACKBONE}_v2.yaml \
CKPT=ckpt/${BACKBONE}/dream_stage2 \
USE_ORACLE_LENGTH=false \
ANSWER_LEN=32 \
BLOCK_SIZE=32 \
THRESHOLD=0.9 \
CACHE_MODE=none \
NUM_GPUS=1 \
bash scripts/eval/steps/run_dream_forward.sh

echo
echo "===== Done ====="
echo "FINAL WER line is logged above."
