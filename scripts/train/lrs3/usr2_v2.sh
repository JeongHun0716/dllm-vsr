#!/usr/bin/env bash
# USR2-Huge + Dream v2 training. v2 convention:
#   - use_oracle_length=true (variable-length transcript+EOS)
#   - force_first_eos_mask=false (random masking only)
#   - train_batch_pad_loss=true (batch tail PAD as loss target)
#   - finetune_from=USR2 stage1 ckpt-42000
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-8893}
CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port=${PORT} \
    training/train.py \
    config=configs/lrs3/usr2_v2.yaml
