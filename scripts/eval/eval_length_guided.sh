#!/usr/bin/env bash
#
# 8-GPU pipeline for ONE backbone (default USR2) on ONE dataset (default LRS3):
#   Step 1: length predictor → len_pred_<split>.jsonl (K_pred ± 5 candidate pool; single GPU)
#   Step 2: length-guided candidate decoding              (8-shard parallel on GPU 0-7)
#   Step 3: reranking within search radius R             (offline, CPU)
#
# Expected WER with paper defaults (rerank search radius R=3, val-tuned (λ, β)):
#              LRS3 test    WildVSR
#   USR2:        19.4%       39.9%     (λ, β) = (0.9, 0.6)
#   AvHubert:    21.9%       46.1%     (λ, β) = (0.9, 0.7)
#
# Prereqs (one-time):
#   conda activate dllm-vsr
#   bash scripts/setup_paths.sh /abs/path/to/lrs3_mouth_rois_root [/abs/path/to/WildVSR]
#
# Usage (default USR2 on LRS3):
#   bash scripts/eval/eval_length_guided.sh
#
# Variants:
#   BACKBONE=avhubert bash scripts/eval/eval_length_guided.sh
#   DATASET=wildvsr   bash scripts/eval/eval_length_guided.sh
#   R=5               bash scripts/eval/eval_length_guided.sh   # search radius applied at the rerank
#                                                      # stage (default 3); decoding always
#                                                      # evaluates the ±5 candidate pool
#
# For the full data-leakage-free pipeline (val-tune then apply test),
# see scripts/eval/rerank/run_tuning_pipeline.sh.
#
# ETA on 8x RTX 3090: ~15-25 minutes (LRS3)
set -e

BACKBONE=${BACKBONE:-usr2}     # usr2 | avhubert
DATASET=${DATASET:-lrs3}       # lrs3 | wildvsr
R=${R:-3}                      # rerank-stage search radius (decoding covers the full ±5 pool)
PYBIN=${PYBIN:-python}
BLOCK_SIZE=${BLOCK_SIZE:-32}
THRESHOLD=${THRESHOLD:-0.9}

# Dataset: manifest root, split name, canvas length (WildVSR clips are longer)
if [[ "${DATASET}" == "lrs3" ]]; then
    MANIFEST_ROOT=manifest/lrs3
    SPLIT=test
    CANVAS_LEN=${CANVAS_LEN:-32}
elif [[ "${DATASET}" == "wildvsr" ]]; then
    MANIFEST_ROOT=manifest/wildvsr
    SPLIT=wildvsr
    CANVAS_LEN=${CANVAS_LEN:-96}
else
    echo "error: DATASET must be 'lrs3' or 'wildvsr', got '${DATASET}'" >&2
    exit 1
fi

# Paper-best (LRS3-val-tuned) rerank weights
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
TEST_OUT=${DREAM_CKPT}/canvas${CANVAS_LEN}_b${BLOCK_SIZE}_${SPLIT}

# ===================================================================
# Step 1: length predictor on ${SPLIT}  (single GPU)
# ===================================================================
echo "[$(date +%H:%M:%S)] Step 1: length predictor on ${SPLIT}  (${BACKBONE})"
if [[ -f "${LENPRED_CKPT}/len_pred_${SPLIT}.jsonl" ]]; then
    echo "  [skip] ${LENPRED_CKPT}/len_pred_${SPLIT}.jsonl exists (shipped with the checkpoints; delete it to regenerate)"
else
    CUDA_VISIBLE_DEVICES=0 ${PYBIN} eval/length_predictor.py \
        config=${LEN_CFG} \
        experiment.eval_checkpoint=${LENPRED_CKPT} \
        dataset.vsr.manifest_root=${MANIFEST_ROOT} \
        eval.split=${SPLIT}
fi
echo "[$(date +%H:%M:%S)] Step 1 done"

# ===================================================================
# Step 2: length-guided candidate decoding on ${SPLIT}  (8-shard parallel on GPU 0-7)
# ===================================================================
echo
echo "[$(date +%H:%M:%S)] Step 2: length-guided candidate decoding on ${SPLIT}  (${BACKBONE}, accelerate 8-way data-parallel on GPU 0-7)"
mkdir -p "${TEST_OUT}"
if compgen -G "${TEST_OUT}/shard_*_raw.jsonl" > /dev/null; then
    echo "  [skip] ${TEST_OUT} already has shards"
else
    # One command, 8 processes: accelerate assigns each process a GPU and a rank;
    # the script derives shard_idx/num_shards from the rank (no manual loop needed).
    accelerate launch --config_file accelerate_configs/8_gpu.yaml \
        eval/dream_canvas_pinned_eos.py \
        config=${DREAM_CFG} \
        eval.ckpt_path=${DREAM_CKPT} \
        eval.len_jsonl=${LENPRED_CKPT}/len_pred_${SPLIT}.jsonl \
        eval.canvas_len=${CANVAS_LEN} \
        eval.block_size=${BLOCK_SIZE} \
        eval.threshold=${THRESHOLD} \
        eval.batch_max_canvas=true \
        dataset.vsr.manifest_root=${MANIFEST_ROOT} \
        eval.split=${SPLIT} \
        eval.out_dir=${TEST_OUT} \
        2>&1 | tee "${TEST_OUT}/accelerate_run.log"
fi
echo "[$(date +%H:%M:%S)] Step 2 done"

# ===================================================================
# Step 3: apply paper-best (λ, β) rerank
# ===================================================================
echo
echo "[$(date +%H:%M:%S)] Step 3: reranking within search radius R=${R}  (λ=${LAM}, β=${BETA})"
${PYBIN} scripts/eval/rerank/apply_fixed_weights.py \
    --test-dir ${TEST_OUT} \
    --lam ${LAM} --beta ${BETA} --R ${R} --tag ${BACKBONE}_${DATASET}

echo
echo "===== Done ====="
echo "Length-guided candidate decoding result is above (${DATASET} WER)."
