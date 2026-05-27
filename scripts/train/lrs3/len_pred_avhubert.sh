#!/usr/bin/env bash
# Lightweight length predictor — AvHubert backbone, 4 GPU (DeepSpeed ZeRO-2).
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-8896}
CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --config_file accelerate_configs/1_node_4_gpus_deepspeed_zero2.yaml \
    --main_process_port=${PORT} \
    training/train_len_pred.py \
    config=configs/lrs3/len_pred_avhubert.yaml
