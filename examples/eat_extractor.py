"""
EAT Feature Extractor - Hugging Face Implementation
EAT: Self-Supervised Pre-Training with Efficient Audio Transformer
Authors: Wenxi Chen, Yuzhe Liang, Ziyang Ma, Zhisheng Zheng, Xie Chen
Paper link: https://arxiv.org/pdf/2401.03497

Author: Yucong Zhang
Email: yucong0428@outlook.com

Wraps the EAT model from Hugging Face. Follows official extraction:
- 16 kHz audio -> Kaldi fbank (128 mel, 25ms/10ms)
- Per-utterance normalization
- Pad/trim to target_length=1024 frames
- Forward HF model and take CLS embedding
"""

import os
import sys
from typing import List

import torch
import torchaudio

from siren.core.base_extractor import BaseFeatureExtractor


class FeatureExtractor(BaseFeatureExtractor):
    """EAT feature extractor using Hugging Face model and official preprocessing."""

    def __init__(self, model_name: str = "worstchan/EAT-base_epoch30_pretrain", multi_channel_strategy: str = "concatenate"):
        super().__init__(multi_channel_strategy)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Expected channels for MAFAULDA dataset
        self.expected_channels = 8

        # 16 kHz target
        self.target_sample_rate = 16000
        self.max_seconds = 10
        self.max_samples = self.target_sample_rate * self.max_seconds

        # EAT official settings
        self.num_mel_bins = 128
        self.frame_length_ms = 25
        self.frame_shift_ms = 10
        self.target_length = 1024
        # Official global normalization
        self.norm_mean = -4.268
        self.norm_std = 4.569

        # Import transformers lazily
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError("Transformers is required for EAT HF extractor. Please install: pip install transformers") from e

        # Load HF model (trust remote code for custom forward if needed)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Infer embedding dimension
        hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
        self._embed_dim = int(hidden_size) if hidden_size is not None else 768

    def _segment_waveform(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        segments = []
        total = waveform.shape[-1]
        num_full = total // self.max_samples
        for i in range(num_full):
            start = i * self.max_samples
            end = start + self.max_samples
            segments.append(waveform[..., start:end])
        if num_full * self.max_samples < total:
            start = max(0, total - self.max_samples)
            segments.append(waveform[..., start:])
        if len(segments) == 0:
            segments.append(waveform)
        return segments

    def _waveform_to_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        # Prepare for kaldi.fbank: input [T]
        wave = waveform.squeeze(0)
        mel = torchaudio.compliance.kaldi.fbank(
            wave.unsqueeze(0),
            htk_compat=True,
            sample_frequency=self.target_sample_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=self.frame_shift_ms,
        ).unsqueeze(0)  # [1, T, F]
        # Pad or truncate to target_length
        n_frames = mel.shape[1]
        if n_frames < self.target_length:
            mel = torch.nn.ZeroPad2d((0, 0, 0, self.target_length - n_frames))(mel)
        else:
            mel = mel[:, :self.target_length, :]
        # Global normalization
        mel = (mel - self.norm_mean) / (self.norm_std * 2)
        # Expect [B, 1, T, F]
        return mel.unsqueeze(0)

    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        waveform = signal_tensor
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Remove DC
        waveform = waveform - waveform.mean()

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)(waveform)

        segments = self._segment_waveform(waveform)
        feats = []
        with torch.inference_mode():
            for seg in segments:
                inputs = self._waveform_to_fbank(seg).to(self.device)  # [1, 1, T, F]
                outputs = self.model.extract_features(inputs)
                feats.append(outputs[:, 0])

        return torch.vstack(feats).mean(dim=0).cpu()

    def _get_single_channel_feature_dim(self) -> int:
        return self._embed_dim


if __name__ == "__main__":
    extractor = FeatureExtractor()
    print(f"Feature dimension: {extractor.feature_dim}")
    print("Extractor ready for evaluation!")


