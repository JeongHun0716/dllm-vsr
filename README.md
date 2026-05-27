# Diffusion Large Language Models for Visual Speech Recognition

This repository contains the PyTorch implementation of the following paper:
> **Diffusion Large Language Models for Visual Speech Recognition**<br>
> <br>
> Authors: Jeong Hun Yeo, Chae Won Kim, Hyeongseop Rha, Yong Man Ro<br>
> **Paper Link**: [http://arxiv.org/abs/XXXX.XXXXX](http://arxiv.org/abs/XXXX.XXXXX)


## Introduction
DLLM-VSR adapts a discrete Diffusion Large Language Model (DLLM, Dream-7B) to visual speech recognition. Instead of left-to-right autoregressive generation, it iteratively denoises a fixed-length masked canvas via confidence-based unmasking, then reranks length-guided candidates to produce the final transcript.


## Setup

```bash
conda create -n dllm-vsr python=3.9 -y
conda activate dllm-vsr
git clone https://github.com/JeongHun0716/dllm-vsr
cd dllm-vsr

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


## Preparation

LRS3 raw videos must be preprocessed into **mouth ROIs** (face detection → mean-face alignment → mouth region cropping). Follow [auto-avsr/preparation](https://github.com/mpc001/auto_avsr/tree/main/preparation).

Point the bundled manifests (`manifest/433h/*.tsv`) at your mouth-ROI root:
```bash
bash scripts/setup_paths.sh /abs/path/to/lrs3_mouth_rois_root
```


## Model Zoo

### External pretrained models (download from original repos)

| Component       | Model              | Source                                                                                | Local path |
|---|---|---|---|
| Visual encoder  | AV-HuBERT (Large)  | [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert)           | `ckpt/avhubert/large_vox_iter5.pt`  |
| Visual encoder  | USR 2.0 (Huge)     | [ahaliassos/usr2](https://github.com/ahaliassos/usr2)                                 | `ckpt/usr2/usr2_huge_pretrain.pth`  |
| DLLM backbone   | Dream-7B           | auto-downloaded by `transformers` (`Dream-org/Dream-v0-Instruct-7B`)                  | `~/.cache/huggingface/hub/`         |

### Trained checkpoints (this paper)

All entries are trained on **LRS3 (433h)** and hosted at [jh-y/dllm-vsr](https://huggingface.co/jh-y/dllm-vsr). Bulk download:

```bash
huggingface-cli download jh-y/dllm-vsr --local-dir ckpt
```

| Backbone           | Component                       | Local path                    |
|---|---|---|
| AV-HuBERT (Large)  | Dream stage-2 (LoRA + adapter)  | `ckpt/avhubert/dream_stage2/` |
| AV-HuBERT (Large)  | Length predictor                | `ckpt/avhubert/len_pred/`     |
| USR 2.0 (Huge)     | Dream stage-2 (LoRA + adapter)  | `ckpt/usr2/dream_stage2/`     |
| USR 2.0 (Huge)     | Length predictor                | `ckpt/usr2/len_pred/`         |


## Evaluation

Results on LRS3 test (WER %):

| Decoding | USR 2.0 | AV-HuBERT |
|---|:---:|:---:|
| Direct                                        | 20.5 | 23.1 |
| Length-guided candidate decoding (paper main) | **19.5** | **21.9** |
| Oracle-length (upper-bound reference)         | 17.7 | 20.2 |

Examples use USR 2.0; prepend `BACKBONE=avhubert` for AV-HuBERT.

```bash
# Direct decoding (~9 min, 1 GPU)
bash scripts/eval/eval_simple.sh

# Length-guided candidate decoding — paper main (~15-25 min, 8 GPUs)
bash scripts/eval/eval_main.sh

# Oracle-length (upper-bound reference)
bash scripts/eval/eval_oracle_length.sh
```

`eval_main.sh` runs three stages internally: length predictor → 8-shard multi-K canvas + pinned-EOS probe → rerank with paper-best `(λ, β)` (USR 2.0 `(0.9, 0.6)`, AV-HuBERT `(0.9, 0.7)`).


## Training

8-GPU DeepSpeed ZeRO-2 launchers (AV-HuBERT variants follow the same pattern):
```bash
bash scripts/train/lrs3/usr2.sh             # stage 1
bash scripts/train/lrs3/usr2_v2.sh          # stage 2 (warm-start from stage 1)
bash scripts/train/lrs3/len_pred_usr2.sh    # length predictor
```

Single-GPU example:
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file accelerate_configs/1_gpu.yaml \
    training/train.py config=configs/lrs3/usr2.yaml
```

OmegaConf CLI overrides are supported:
```bash
accelerate launch ... training/train.py \
    config=configs/lrs3/usr2.yaml \
    training.max_train_steps=100 model.lora.r=8
```

Stage-2 configs (`configs/lrs3/*_v2.yaml`) reference a stage-1 checkpoint via `experiment.finetune_from` — train stage 1 first and update the path, or remove the field.


## Citation

```bibtex
@article{yeo2026dllmvsr,
  title={Diffusion Large Language Models for Visual Speech Recognition},
  author={Yeo, Jeong Hun and Kim, Chae Won and Rha, Hyeongseop and Ro, Yong Man},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```


## Acknowledgement

This project builds on [Dream-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B), [AV-HuBERT](https://github.com/facebookresearch/av_hubert), [USR 2.0](https://github.com/ahaliassos/usr2), [MMaDA](https://github.com/gen-verse/mmada), and [Fast-dLLM](https://github.com/HKUNLP/Fast-dLLM). We thank the developers of these projects and the open-source community.
