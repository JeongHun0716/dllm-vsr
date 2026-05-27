#!/usr/bin/env bash
# Val tuning pipeline:
#   1. Run lenpred inference on LRS3 val (998 samples) for both backbones.
#   2. Run multi-K dream_canvas_pinned_eos probe on val with val lenpred jsonl.
#   3. (offline) Tune (λ, β) on val raw jsonl, apply to test raw jsonl.
set -e

USR2_LENPRED_CKPT=ckpt/usr2/len_pred
AV_LENPRED_CKPT=ckpt/avhubert/len_pred
USR2_CFG=configs/lrs3/usr2_v2.yaml
USR2_CKPT=ckpt/usr2/dream_stage2
AV_CFG=configs/lrs3/avhubert_v2.yaml
AV_CKPT=ckpt/avhubert/dream_stage2

# ===== Step 1: Lenpred inference on val =====
echo "[$(date +%H:%M:%S)] Step 1: lenpred inference on val (USR2 + AvHub in parallel)"
CUDA_VISIBLE_DEVICES=0 python eval/length_predictor.py \
    config=configs/lrs3/len_pred_usr2.yaml \
    experiment.eval_checkpoint=${USR2_LENPRED_CKPT} \
    eval.split=val \
    > /tmp/val_lenpred_usr2.log 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=1 python eval/length_predictor.py \
    config=configs/lrs3/len_pred_avhubert.yaml \
    experiment.eval_checkpoint=${AV_LENPRED_CKPT} \
    eval.split=val \
    > /tmp/val_lenpred_avhub.log 2>&1 &
PID2=$!
wait $PID1 $PID2
echo "[$(date +%H:%M:%S)] Lenpred inference done"

USR2_VAL_JSONL=${USR2_LENPRED_CKPT}/len_pred_val.jsonl
AV_VAL_JSONL=${AV_LENPRED_CKPT}/len_pred_val.jsonl

if [ ! -f "${USR2_VAL_JSONL}" ] || [ ! -f "${AV_VAL_JSONL}" ]; then
    echo "ERROR: val lenpred jsonl missing"; exit 1
fi

# ===== Step 2: Multi-K probes on val =====
echo "[$(date +%H:%M:%S)] Step 2: multi-K probes on val"
USR2_VAL_OUT=${USR2_CKPT}/canvas32_b32_val
AV_VAL_OUT=${AV_CKPT}/canvas32_b32_val
mkdir -p "${USR2_VAL_OUT}" "${AV_VAL_OUT}"

# USR2 on GPU 0-3, AvHub on GPU 4-7 (4 shards each)
for SH in 0 1 2 3; do
  PORT=$((30000 + SH))
  CUDA_VISIBLE_DEVICES=${SH} nohup accelerate launch \
      --config_file accelerate_configs/1_gpu.yaml --main_process_port=${PORT} \
      eval/dream_canvas_pinned_eos.py \
      config=${USR2_CFG} eval.ckpt_path=${USR2_CKPT} \
      eval.len_jsonl=${USR2_VAL_JSONL} eval.canvas_len=32 \
      eval.block_size=32 eval.threshold=0.9 eval.batch_max_canvas=true \
      eval.shard_idx=${SH} eval.num_shards=4 \
      eval.split=val \
      eval.out_dir=${USR2_VAL_OUT} \
      > ${USR2_VAL_OUT}/shard_${SH}.log 2>&1 &
done
for SH in 0 1 2 3; do
  GPU=$((SH + 4))
  PORT=$((30100 + SH))
  CUDA_VISIBLE_DEVICES=${GPU} nohup accelerate launch \
      --config_file accelerate_configs/1_gpu.yaml --main_process_port=${PORT} \
      eval/dream_canvas_pinned_eos.py \
      config=${AV_CFG} eval.ckpt_path=${AV_CKPT} \
      eval.len_jsonl=${AV_VAL_JSONL} eval.canvas_len=32 \
      eval.block_size=32 eval.threshold=0.9 eval.batch_max_canvas=true \
      eval.shard_idx=${SH} eval.num_shards=4 \
      eval.split=val \
      eval.out_dir=${AV_VAL_OUT} \
      > ${AV_VAL_OUT}/shard_${SH}.log 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] Multi-K val probes done. Now run offline rerank tuning."
echo "USR2 val probe: ${USR2_VAL_OUT}"
echo "AvHub val probe: ${AV_VAL_OUT}"
