#!/usr/bin/env bash
# Lightweight length predictor — USR2 backbone, 4 GPU (DeepSpeed ZeRO-2).
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-8892}
CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --config_file accelerate_configs/1_node_4_gpus_deepspeed_zero2.yaml \
    --main_process_port=${PORT} \
    training/train_len_pred.py \
    config=configs/lrs3/len_pred_usr2.yaml
