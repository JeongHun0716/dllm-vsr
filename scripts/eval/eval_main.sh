#!/usr/bin/env bash
#
# 8-GPU pipeline for ONE backbone (default USR2):
#   Step 1: length predictor → len_pred_test.jsonl       (single GPU)
#   Step 2: multi-K canvas + pinned-EOS probe on test    (8-shard parallel on GPU 0-7)
#   Step 3: apply paper-best (λ, β) rerank to test       (offline, CPU)
#
# Paper-best (λ, β) for LRS3:
#   - USR2:     (0.9, 0.6) → expected 19.5%
#   - AvHubert: (0.9, 0.7) → expected 21.9%
#
# Prereqs (one-time):
#   conda activate dllm-vsr
#   bash scripts/setup_paths.sh /abs/path/to/lrs3_videos
#
# Usage (default USR2):
#   bash scripts/eval/eval_main.sh
#
# To run AvHubert instead:
#   BACKBONE=avhubert bash scripts/eval/eval_main.sh
#
# For the full data-leakage-free pipeline (val-tune then apply test),
# see scripts/eval/rerank/run_tuning_pipeline.sh.
#
# ETA on 8x RTX 3090: ~15-25 minutes
set -e

BACKBONE=${BACKBONE:-usr2}     # usr2 | avhubert
PYBIN=${PYBIN:-python}
CANVAS_LEN=${CANVAS_LEN:-32}
BLOCK_SIZE=${BLOCK_SIZE:-32}
THRESHOLD=${THRESHOLD:-0.9}

# Paper-best (val-tuned) rerank weights for LRS3
if [[ "${BACKBONE}" == "usr2" ]]; then
    LAM=0.9
    BETA=0.6
elif [[ "${BACKBONE}" == "avhubert" ]]; then
    LAM=0.9
    BETA=0.7
else
    echo "error: BACKBONE must be 'usr2' or 'avhubert', got '${BACKBONE}'" >&2
    exit 1
fi

LENPRED_CKPT=ckpt/${BACKBONE}/len_pred
DREAM_CKPT=ckpt/${BACKBONE}/dream_stage2
LEN_CFG=configs/lrs3/len_pred_${BACKBONE}.yaml
DREAM_CFG=configs/lrs3/${BACKBONE}_v2.yaml
TEST_OUT=${DREAM_CKPT}/canvas${CANVAS_LEN}_b${BLOCK_SIZE}_test

mkdir -p eval_logs

# ===================================================================
# Step 1: lenpred jsonl on test  (single GPU)
# ===================================================================
echo "[$(date +%H:%M:%S)] Step 1: lenpred test  (${BACKBONE})"
if [[ -f "${LENPRED_CKPT}/len_pred_test.jsonl" ]]; then
    echo "  [skip] ${LENPRED_CKPT}/len_pred_test.jsonl exists"
else
    CUDA_VISIBLE_DEVICES=0 ${PYBIN} eval/length_predictor.py \
        config=${LEN_CFG} \
        experiment.eval_checkpoint=${LENPRED_CKPT} \
        eval.split=test
fi
echo "[$(date +%H:%M:%S)] Step 1 done"

# ===================================================================
# Step 2: multi-K canvas probe on test  (8-shard parallel on GPU 0-7)
# ===================================================================
echo
echo "[$(date +%H:%M:%S)] Step 2: multi-K canvas probe on test  (${BACKBONE}, 8 shards on GPU 0-7)"
mkdir -p "${TEST_OUT}"
if compgen -G "${TEST_OUT}/shard_*_raw.jsonl" > /dev/null; then
    echo "  [skip] ${TEST_OUT} already has shards"
else
    PIDS=()
    for SH in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=${SH} ${PYBIN} eval/dream_canvas_pinned_eos.py \
            config=${DREAM_CFG} \
            eval.ckpt_path=${DREAM_CKPT} \
            eval.len_jsonl=${LENPRED_CKPT}/len_pred_test.jsonl \
            eval.canvas_len=${CANVAS_LEN} \
            eval.block_size=${BLOCK_SIZE} \
            eval.threshold=${THRESHOLD} \
            eval.batch_max_canvas=true \
            eval.shard_idx=${SH} \
            eval.num_shards=8 \
            eval.split=test \
            eval.out_dir=${TEST_OUT} \
            > ${TEST_OUT}/shard_${SH}.log 2>&1 &
        PIDS+=($!)
    done
    echo "  launched 8 shards (PIDs: ${PIDS[*]})"
    wait "${PIDS[@]}"
fi
echo "[$(date +%H:%M:%S)] Step 2 done"

# ===================================================================
# Step 3: apply paper-best (λ, β) rerank
# ===================================================================
echo
echo "[$(date +%H:%M:%S)] Step 3: apply paper-best rerank  (λ=${LAM}, β=${BETA})"
${PYBIN} scripts/eval/rerank/apply_fixed_weights.py \
    --test-dir ${TEST_OUT} \
    --lam ${LAM} --beta ${BETA} --tag ${BACKBONE}

echo
echo "===== Done ====="
echo "Paper Table 1 Row 4 result is above (test WER)."
