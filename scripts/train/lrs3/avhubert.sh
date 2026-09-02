#!/usr/bin/env bash
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-8892}
CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port=${PORT} \
    training/train.py \
    config=configs/lrs3/avhubert.yaml
