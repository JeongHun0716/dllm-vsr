# coding=utf-8
"""VSR dataset — raw video backend only.

Loads video_rel (mp4) from the LRS3 manifest via torchvision.io.read_video then augments.
collate output: video_features [B, T, 1, H, W] (greyscale, normalized) + padding_mask.

Manifest format (tab-separated, 6 columns):
    line 0: <root_dir>
    line N: <prefix>\t<video_rel>\t<audio_rel>\t<n_video_frames>\t<n_audio_samples>\t<duration_sec>
"""
from __future__ import annotations

import logging
import os
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from training.transforms import VideoTransform


logger = logging.getLogger(__name__)


def _load_manifest(
    manifest_path: str,
    label_paths: List[str],
    max_video_frames: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]], List[int], List[str]]:
    """
    Returns:
      root, entries=[{video_rel, utt_id}], frames, labels
    utt_id = os.path.splitext(video_rel)[0]
    """
    with open(label_paths[0]) as f:
        all_labels: List[str] = [line.rstrip("\n") for line in f]

    entries: List[Dict[str, Any]] = []
    frames: List[int] = []
    labels: List[str] = []

    with open(manifest_path) as f:
        root = f.readline().strip()
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            if i >= len(all_labels):
                break

            video_rel = parts[1] if len(parts) > 1 else None
            if not video_rel:
                continue
            try:
                video_frames = int(parts[3])
            except ValueError:
                continue
            if video_frames <= 0:
                continue
            if max_video_frames is not None and video_frames > int(max_video_frames):
                continue

            utt_id = os.path.splitext(video_rel)[0]
            entries.append({"video_rel": video_rel, "utt_id": utt_id})
            frames.append(video_frames)
            labels.append(all_labels[i])

    if len(entries) == 0:
        logger.warning("No valid entries after filtering by max_video_frames=%s", max_video_frames)
    return root, entries, frames, labels


def _load_video_thwc_to_tchw(path: str) -> torch.Tensor:
    """torchvision.io.read_video → [T, H, W, C] uint8 → permute [T, C, H, W]."""
    import torchvision
    vid, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")
    vid = vid.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
    return vid


class VSRDataset(Dataset):
    """Raw-video VSR dataset (mp4 → torchvision.read_video → VideoTransform)."""

    def __init__(
        self,
        manifest_path: str,
        label_paths: List[str],
        dllm_tokenizer_path: str,
        video_root: Optional[str] = None,         # falls back to manifest's root_dir
        subset: str = "train",                    # train|val|test (selects transform)
        crop_size: int = 88,
        normalize_mean: float = 0.421,
        normalize_std: float = 0.165,
        time_mask_window: int = 10,
        time_mask_stride: int = 25,
        train_crop: str = "random",        # "random" | "center"
        enable_time_mask: bool = True,
        max_video_frames: int = 500,
        modalities: Optional[List[str]] = None,
        # instruction override — None falls back to the default hardcoded string.
        instruction: Optional[str] = None,
    ):
        super().__init__()
        self.modalities = set(modalities) if modalities else {"video"}

        self.root, self.entries, self.frames, self.labels = _load_manifest(
            manifest_path=manifest_path,
            label_paths=label_paths,
            max_video_frames=max_video_frames,
        )
        self.lengths = self.frames

        if len(self.entries) != len(self.labels):
            logger.warning(
                "Manifest entries (%d) and labels (%d) length mismatch; min will be used",
                len(self.entries),
                len(self.labels),
            )
            n = min(len(self.entries), len(self.labels))
            self.entries = self.entries[:n]
            self.labels = self.labels[:n]
            self.frames = self.frames[:n]
            self.lengths = self.frames

        self.video_root = video_root or self.root
        self.video_transform = VideoTransform(
            subset=subset,
            crop_size=crop_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            time_mask_window=time_mask_window,
            time_mask_stride=time_mask_stride,
            train_crop=train_crop,
            enable_time_mask=enable_time_mask,
        )

        self.dllm_tokenizer = AutoTokenizer.from_pretrained(dllm_tokenizer_path, trust_remote_code=True)
        if self.dllm_tokenizer.pad_token_id is None:
            self.dllm_tokenizer.pad_token = self.dllm_tokenizer.eos_token

        # ---- chat template precompute ----
        # Split the chat-templated prefix in two so visual features can be
        # inserted inside the user turn (after instruction, before eot):
        #   user_prefix_ids: <BOS><user_hdr>{instruction}
        #   post_user_ids:   <eot><assistant_hdr>\n\n
        # model.prepare_dllm_inputs_labels stitches visual embeds between them.
        default_instruction = (
            "You are given a silent video of a speaker. "
            "Infer the spoken content from the lip movements and "
            "generate the corresponding transcription."
        )
        instruction = instruction if (instruction is not None) else default_instruction
        self.instruction = instruction
        messages = [{"role": "user", "content": instruction}]
        full_prefix = self.dllm_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).squeeze(0).to(dtype=torch.long)
        # end-of-turn token id (Dream/Qwen2: <|im_end|>=151645).
        eot_id = self.dllm_tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eot_id is None or eot_id == self.dllm_tokenizer.unk_token_id or eot_id < 0:
            raise RuntimeError("Could not find <|im_end|> token in tokenizer.")
        eot_positions = (full_prefix == eot_id).nonzero(as_tuple=False).squeeze(-1)
        if eot_positions.numel() == 0:
            raise RuntimeError(
                f"eot_id={eot_id} not found in chat-templated prefix. "
                f"Template output: {self.dllm_tokenizer.decode(full_prefix)!r}"
            )
        split_idx = int(eot_positions[0].item())  # first eot = user-turn close
        self._user_prefix_ids = full_prefix[:split_idx]      # user header + instruction (no eot)
        self._post_user_ids = full_prefix[split_idx:]        # eot + assistant header

    def __len__(self) -> int:
        return len(self.entries)

    def _load_raw_video(self, video_rel: str) -> torch.Tensor:
        full_path = os.path.join(self.video_root, video_rel)
        vid = _load_video_thwc_to_tchw(full_path)  # [T, C, H, W] uint8
        vid = self.video_transform(vid)            # [T, 1, crop, crop] float
        return vid

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.entries[idx]
        utt_id = e["utt_id"]
        video_rel = e["video_rel"]

        video_features: Optional[torch.Tensor] = None
        if "video" in self.modalities:
            video_features = self._load_raw_video(video_rel)

        transcript = self.labels[idx]
        # post_user_ids is sample-invariant — cached once and shared. Visual
        # features are spliced between input_ids and post_user_ids in
        # model.prepare_dllm_inputs_labels.
        input_ids = self._user_prefix_ids

        eos_id = self.dllm_tokenizer.eos_token_id
        lbl_ids = self.dllm_tokenizer(
            transcript, add_special_tokens=False, return_tensors="pt"
        ).input_ids.squeeze(0)
        labels = torch.cat(
            [lbl_ids, torch.tensor([eos_id], dtype=torch.long, device=lbl_ids.device)],
            dim=0,
        )

        return {
            "input_ids": input_ids,
            "post_user_ids": self._post_user_ids,
            "labels": labels,
            "utt_id": utt_id,
            "video_features": video_features,  # [T, 1, H, W] or None
            "pad_token_id": self.dllm_tokenizer.pad_token_id,
            # K = transcript+EOS length (= labels.numel()). Used by the length predictor.
            "K": int(labels.numel()),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [x["input_ids"] for x in batch]
        labels = [x["labels"] for x in batch]

        video_tensor: Optional[torch.Tensor] = None
        video_padding_mask: Optional[torch.Tensor] = None

        if all(x.get("video_features") is not None for x in batch):
            feats = [x["video_features"] for x in batch]
            B = len(feats)
            T_max = max(f.size(0) for f in feats)
            # raw: [T, C, H, W] → [B, T_max, C, H, W]
            C, H, W = feats[0].size(1), feats[0].size(2), feats[0].size(3)
            out = feats[0].new_zeros(B, T_max, C, H, W)
            pad_mask = torch.ones(B, T_max, dtype=torch.bool)
            for i, f in enumerate(feats):
                T = f.size(0)
                out[i, :T] = f
                pad_mask[i, :T] = False
            video_tensor = out
            video_padding_mask = pad_mask

        return {
            "input_ids": input_ids,
            "post_user_ids": batch[0]["post_user_ids"],   # sample-invariant — shared
            "labels": labels,
            "utt_id": [x["utt_id"] for x in batch],
            "video_features": video_tensor,
            "video_padding_mask": video_padding_mask,
            "pad_token_id": batch[0]["pad_token_id"],
            "K": [x["K"] for x in batch],
        }
