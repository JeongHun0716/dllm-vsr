#!/usr/bin/env bash
# AvHubert-dream v2 — standard multimodal dLLM training convention.
#   - use_oracle_length=true (variable-length transcript+EOS)
#   - force_first_eos_mask=false (random masking only)
#   - train_batch_pad_loss=true (only batch-alignment tail PAD is a loss target)
#   - finetune_from=stage1 ckpt-16000
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-8892}
CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port=${PORT} \
    training/train.py \
    config=configs/lrs3/avhubert_v2.yaml
