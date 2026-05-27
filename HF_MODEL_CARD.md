---
license: mit
library_name: pytorch
tags:
- visual-speech-recognition
- lip-reading
- discrete-diffusion
- lrs3
datasets:
- lrs3
---

# DLLM-VSR: Diffusion Large Language Models for Visual Speech Recognition

Paper checkpoints for **DLLM-VSR** — adapting the Dream-7B discrete-diffusion LLM to Visual Speech Recognition (VSR) on LRS3.

- Paper: [arxiv.org/abs/XXXX.XXXXX](http://arxiv.org/abs/XXXX.XXXXX)
- Code: [github.com/JeongHun0716/dllm-vsr](https://github.com/JeongHun0716/dllm-vsr)
- Authors: Jeong Hun Yeo, Chae Won Kim, Hyeongseop Rha, Yong Man Ro

## Contents

| Path | Description | Size |
|---|---|---|
| `usr2/dream_stage2/`     | USR 2.0 + Dream-7B stage 2 (LoRA + adapter) | 117 MB |
| `usr2/len_pred/`         | Length predictor for USR 2.0 features        | 8.2 MB |
| `avhubert/dream_stage2/` | AV-HuBERT + Dream-7B stage 2                 | 102 MB |
| `avhubert/len_pred/`     | Length predictor for AV-HuBERT features      | 8.0 MB |

Each `dream_stage2/` holds `trainable_model.safetensors` (LoRA adapters + visual-feature projector). Each `len_pred/` holds `trainable_model.pt` (small Transformer over visual features).

**Note**: Visual encoder weights (USR 2.0 Huge, AV-HuBERT Large) are **not** redistributed here. Download them from the original repos:
- AV-HuBERT: https://github.com/facebookresearch/av_hubert
- USR 2.0:   https://github.com/ahaliassos/usr2

## Results on LRS3 test (WER, %)

All entries are trained on **LRS3 (433h)** only.

| Decoding | USR 2.0 | AV-HuBERT |
|---|:---:|:---:|
| Direct                                        | 20.5 | 23.1 |
| Length-guided candidate decoding (paper main) | **19.5** | **21.9** |
| Oracle-length (upper-bound reference)         | 17.7 | 20.2 |

## Usage

```bash
huggingface-cli download jh-y/dllm-vsr --local-dir ckpt
```

Then follow the [code repo's README](https://github.com/JeongHun0716/dllm-vsr) for environment setup, preprocessing (auto-avsr pipeline), and inference scripts.

## Citation

```bibtex
@article{yeo2026dllmvsr,
  title={Diffusion Large Language Models for Visual Speech Recognition},
  author={Yeo, Jeong Hun and Kim, Chae Won and Rha, Hyeongseop and Ro, Yong Man},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
