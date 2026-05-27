# Diffusion Large Language Models for Visual Speech Recognition

This repository contains the PyTorch implementation of the following paper:
> **Diffusion Large Language Models for Visual Speech Recognition**<br>
> <br>
> Authors: Jeong Hun Yeo, Chae Won Kim, Hyeongseop Rha, Yong Man Ro<br>
> **Paper Link**: [http://arxiv.org/abs/XXXX.XXXXX](http://arxiv.org/abs/XXXX.XXXXX)


## Environment Setup
```bash
conda create -n dllm-vsr python=3.9 -y
conda activate dllm-vsr
git clone https://github.com/JeongHun0716/dllm-vsr
cd dllm-vsr
```
```bash
# PyTorch and related packages
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


## Preparation

### LRS3 video preprocessing
LRS3 raw videos must be lip-cropped (face detection + mouth ROI). Follow the preprocessing pipeline in [auto-avsr](https://github.com/mpc001/auto_avsr) (`preparation/` directory) to produce lip-cropped clips.

### Manifest setup
LRS3 manifests (train/val/test tsv + wrd, with `{LRS3_ROOT}` placeholder on line 0) are bundled under `manifest/433h/`. Point them at your local lip-cropped LRS3 video root with the one-time setup script:

```bash
bash scripts/setup_paths.sh /abs/path/to/lip_cropped_lrs3_root
```

This rewrites the placeholder in `manifest/433h/{train,val,test}.tsv` to the absolute path you provide.

Resulting structure:
```
dllm-vsr/
├── accelerate_configs/                 single/multi-GPU + DeepSpeed ZeRO-2/3 yaml
├── configs/lrs3/                       per-model training/eval configs
├── manifest/433h/                      LRS3 train/val/test manifests (tsv/wrd)
├── ckpt/                               paper checkpoints (downloaded — see below)
│   ├── usr2/
│   │   ├── usr2_huge_pretrain.pth      frozen encoder (original-repo download)
│   │   ├── dream_stage2/                paper main result (LRS3 19.5%)
│   │   └── len_pred/                    length predictor
│   └── avhubert/
│       ├── large_vox_iter5.pt          frozen encoder (original-repo download)
│       ├── dream_stage2/                paper main result (LRS3 21.9%)
│       └── len_pred/
├── scripts/
│   ├── setup_paths.sh                  one-time LRS3 video-root setup
│   ├── train/lrs3/                     training launchers
│   └── eval/                           reproducing scripts (simple / oracle / main)
├── models/                             DLLMVSRModel + visual encoders + length predictor
├── training/                           train.py (Dream LoRA) + train_len_pred.py
├── eval/                               inference entry points
├── evaluation/                         Fast-dLLM patched Dream backbone
└── third_party/                        vendored AV-HuBERT and USR 2.0 repos
```


## Pretrained Models
1. Download the `AV-HuBERT Large` model from this [link](https://github.com/facebookresearch/av_hubert).
2. Download the `USR 2.0 Huge` model from this [link](https://github.com/ahaliassos/usr2).
3. Download `dllm-vsr` LoRA / adapter / length-predictor checkpoints from the Hugging Face Hub (link below).
4. The Dream-7B base weights (`Dream-org/Dream-v0-Instruct-7B`) are downloaded automatically by `transformers` on first run (~14 GB, cached under `~/.cache/huggingface/hub/`).

After downloading, place files as follows:
- `large_vox_iter5.pt` → `ckpt/avhubert/large_vox_iter5.pt`
- `usr2_huge_pretrain.pth` → `ckpt/usr2/usr2_huge_pretrain.pth`
- `dllm-vsr` checkpoints → `ckpt/{usr2,avhubert}/{dream_stage2,len_pred}/...`

### `dllm-vsr` checkpoints (HF Hub)

All four trained checkpoints are hosted in a single Hugging Face Hub model repo: [jh-y/dllm-vsr](https://huggingface.co/jh-y/dllm-vsr).

```bash
huggingface-cli download jh-y/dllm-vsr --local-dir ckpt
```

| Backbone | Stage 2 (Dream LoRA + adapter) | Length predictor |
|---|---|---|
| USR 2.0 (Huge) | `ckpt/usr2/dream_stage2/`   | `ckpt/usr2/len_pred/`   |
| AV-HuBERT (Large) | `ckpt/avhubert/dream_stage2/` | `ckpt/avhubert/len_pred/` |


## Evaluation

### Results on LRS3 test (WER, %)

All entries are trained on **LRS3 (433h)** only.

| Decoding | USR 2.0 | AV-HuBERT |
|---|:---:|:---:|
| Direct                                        | 20.5 | 23.1 |
| Length-guided candidate decoding (paper main) | **19.5** | **21.9** |
| Oracle-length (upper-bound reference)         | 17.7 | 20.2 |

Examples below use USR 2.0. To run AV-HuBERT instead, prepend `BACKBONE=avhubert` to any command.

### Direct decoding
```bash
bash scripts/eval/eval_simple.sh
# Expected: USR 2.0 = 20.5%   (~9 minutes on 1 GPU)
```

### Length-guided candidate decoding (paper main)
```bash
bash scripts/eval/eval_main.sh
# Expected: USR 2.0 = 19.5%   (~15-25 minutes on 8 GPUs)
```

`eval_main.sh` internally runs:
1. Length predictor → `ckpt/usr2/len_pred/len_pred_test.jsonl`
2. 8-shard multi-K canvas + pinned-EOS probe on test
3. `scripts/eval/rerank/apply_fixed_weights.py` with paper-best `(λ, β) = (0.9, 0.6)` for USR 2.0

### Oracle-length decoding (upper-bound reference)
```bash
bash scripts/eval/eval_oracle_length.sh
# Uses the ground-truth transcript length as the answer canvas (test-info leakage).
```


## Training
Pre-set 8-GPU DeepSpeed ZeRO-2 launchers (the AV-HuBERT variants share the same pattern):
```bash
bash scripts/train/lrs3/usr2.sh             # stage 1 (oracle length)
bash scripts/train/lrs3/usr2_v2.sh          # stage 2 (PAD-in-loss + warm-start from stage 1)
bash scripts/train/lrs3/len_pred_usr2.sh    # length predictor (USR 2.0 features)
```

Single-GPU example (any config under `configs/lrs3/`):
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file accelerate_configs/1_gpu.yaml \
    training/train.py config=configs/lrs3/usr2.yaml
```

Configs in `configs/lrs3/` cover both stages. OmegaConf CLI overrides are supported:
```bash
accelerate launch ... training/train.py \
    config=configs/lrs3/usr2.yaml \
    training.max_train_steps=100 \
    model.lora.r=8
```

The stage-2 configs (`configs/lrs3/*_v2.yaml`) reference a stage-1 checkpoint via `experiment.finetune_from`. Either train stage 1 first and update the path, or remove the field if not warm-starting.


## Citation
If you find this work useful in your research, please cite the paper:
```bibtex
@article{yeo2026dllmvsr,
  title={Diffusion Large Language Models for Visual Speech Recognition},
  author={Yeo, Jeong Hun and Kim, Chae Won and Rha, Hyeongseop and Ro, Yong Man},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```


## Acknowledgement
This project builds on the [Dream-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B), [AV-HuBERT](https://github.com/facebookresearch/av_hubert), [USR 2.0](https://github.com/ahaliassos/usr2), [MMaDA](https://github.com/gen-verse/mmada), and [Fast-dLLM](https://github.com/HKUNLP/Fast-dLLM) code. We would like to thank the developers of these projects for their contributions and the open-source community for making this work possible.
