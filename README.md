<div align="center">

# [EMNLP 2026] Diffusion Large Language Models for Visual Speech Recognition

**Jeong Hun Yeo, Chae Won Kim, Hyeongseop Rha, Yong Man Ro**<br>
Integrated Vision Language Lab, KAIST

[![arXiv](https://img.shields.io/badge/arXiv-2605.28456-b31b1b.svg)](https://arxiv.org/abs/2605.28456)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/jh-y/dllm-vsr)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<img width="1748" height="946" alt="image" src="https://github.com/user-attachments/assets/1cde2f06-55d5-4b6c-b05a-b1510d42a8aa" />

</div>

This repository contains the official PyTorch implementation of **DLLM-VSR**, the first Diffusion Large Language Model (DLLM)-based framework for Visual Speech Recognition.

## News
- **2026-09**: Code and checkpoints released. Paper accepted at **EMNLP 2026** ([arXiv](https://arxiv.org/abs/2605.28456)).


## Introduction
DLLM-VSR adapts a discrete Diffusion Large Language Model (DLLM, Dream-7B) to visual speech recognition. Instead of left-to-right autoregressive decoding, it performs flexible-order masked denoising on a fixed-length canvas: confidence-based unmasking commits high-confidence positions first and uses them as bidirectional context for visually ambiguous ones. A two-stage masked-denoising training strategy separates visual-to-text content alignment from length modeling, and length-guided candidate decoding decodes a pool of transcript-length hypotheses from the video and selects the final transcript by joint reranking.


## Setup

```bash
conda create -n dllm-vsr python=3.9 -y
conda activate dllm-vsr
git clone https://github.com/JeongHun0716/dllm-vsr
cd dllm-vsr

# PyTorch (CUDA 11.8; adjust the cu* tag if your driver differs)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Other Python dependencies
pip install -r requirements.txt

# fairseq (needed by the AV-HuBERT encoder; install as editable from upstream)
git clone https://github.com/facebookresearch/fairseq
cd fairseq && pip install --editable ./ && cd ..
```


## Preparation

LRS3 raw videos must be preprocessed into **mouth ROIs** (face detection → mean-face alignment → mouth region cropping). Follow [auto-avsr/preparation](https://github.com/mpc001/auto_avsr/tree/main/preparation).

For out-of-domain evaluation, download the [WildVSR test set](https://github.com/YasserdahouML/VSR_test_set) — its videos are already distributed as 96×96 mouth-ROI crops, so no extra preprocessing is needed.

Point the bundled manifests (`manifest/lrs3`, `manifest/wildvsr`) at your video roots (the 2nd argument is optional):
```bash
bash scripts/setup_paths.sh /abs/path/to/lrs3_mouth_rois_root /abs/path/to/WildVSR
```


## Model Zoo

### External pretrained models (download from original repos)

| Component       | Model              | Source                                                                                | Local path |
|---|---|---|---|
| Visual encoder  | AV-HuBERT (Large)  | [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert)           | `ckpt/avhubert/large_vox_iter5.pt`  |
| Visual encoder  | USR 2.0 (Huge, LRS2+LRS3+Vox2+AVS pretrain) | [ahaliassos/usr2](https://github.com/ahaliassos/usr2)        | `ckpt/usr2/usr2_huge_pretrain.pth`  |
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
| Implicit-length decoding                      | 20.5 | 23.1 |
| Length-guided candidate decoding (paper main) | **19.4** | **21.9** |
| Oracle-length decoding (upper bound)          | 17.7 | 20.2 |

Out-of-domain results on WildVSR (WER %, same LRS3-trained models and LRS3-val-tuned weights):

| Decoding | USR 2.0 | AV-HuBERT |
|---|:---:|:---:|
| Length-guided candidate decoding | **39.9** | **46.1** |

Examples use USR 2.0; prepend `BACKBONE=avhubert` for AV-HuBERT.

```bash
# Implicit-length decoding (~9 min, 1 GPU)
bash scripts/eval/eval_implicit_length.sh

# Length-guided candidate decoding — paper main (~15-25 min, 8 GPUs)
bash scripts/eval/eval_length_guided.sh

# Out-of-domain: WildVSR test set (requires manifest/wildvsr setup; see Preparation)
DATASET=wildvsr bash scripts/eval/eval_length_guided.sh

# Oracle-length decoding (upper bound)
bash scripts/eval/eval_oracle_length.sh
```

The length-predictor outputs used in the paper (`ckpt/<backbone>/len_pred/len_pred_{test,wildvsr}.jsonl`, the K_pred ± 5 candidate pools) are included in the checkpoint download, so Step 1 below is skipped automatically; delete them to regenerate.

`eval_length_guided.sh` runs three stages internally: length predictor (dumps the K_pred ± 5 candidate pool) → 8-shard candidate decoding on pinned-EOS canvases → offline reranking that selects within a search radius R (default 3) and reports the full R = 1–5 sweep, using val-tuned `(λ, β)` (USR 2.0 `(0.9, 0.6)`, AV-HuBERT `(0.9, 0.7)`). Set `R=<r>` to change the reported operating radius.


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
  journal={arXiv preprint arXiv:2605.28456},
  year={2026}
}
```


## Acknowledgement

This project builds on [Dream-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B), [AV-HuBERT](https://github.com/facebookresearch/av_hubert), [USR 2.0](https://github.com/ahaliassos/usr2), [MMaDA](https://github.com/gen-verse/mmada), and [Fast-dLLM](https://github.com/HKUNLP/Fast-dLLM). We thank the developers of these projects and the open-source community.
