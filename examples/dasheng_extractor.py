"""
Dasheng Feature Extractor - Example Implementation

Author: Yucong Zhang
Email: yucong0428@outlook.com

This wraps the Dasheng models for use with the siren evaluation framework,
following the style of the existing BEATs and CED extractors.

Key points:
- Inherits from BaseFeatureExtractor
- Implements _extract_single_channel_features() and _get_single_channel_feature_dim()
- Handles multi-channel via BaseFeatureExtractor
- Ensures 16 kHz input (resamples if needed)
- Segments long audio into 10s chunks and mean-pools features over time and segments
"""

import os
import sys
from typing import Dict

import torch
import torchaudio

from siren.core.base_extractor import BaseFeatureExtractor

# Ensure local repo is importable - replace with your actual Dasheng implementation path
# Example: sys.path.append("/path/to/your/dasheng/implementation/")
sys.path.append("/path/to/your/dasheng/implementation/")

# Import Dasheng pretrained models
try:
    from Dasheng.dasheng import dasheng_base, dasheng_06B, dasheng_12B
except Exception as e:
    raise ImportError(f"Failed to import Dasheng models. Ensure the path is correct and dependencies are installed. Error: {e}")


class FeatureExtractor(BaseFeatureExtractor):
    """Dasheng feature extractor using pretrained encoders.

    Args:
        model_size: One of {"dasheng_base", "dasheng_06B", "dasheng_12B"}
        multi_channel_strategy: Strategy for multi-channel handling in BaseFeatureExtractor
    """

    def __init__(self, model_size: str = "dasheng_base", multi_channel_strategy: str = "concatenate"):
        super().__init__(multi_channel_strategy)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Expected channels for MAFAULDA dataset
        self.expected_channels = 8

        # Dasheng models are trained for 16 kHz audio
        self.target_sample_rate = 16000
        self.max_samples = 160000  # 10 seconds at 16 kHz

        self.model_size = model_size
        self._feature_dims: Dict[str, int] = {
            "dasheng_base": 768,
            "dasheng_06B": 1280,
            "dasheng_12B": 1536,
        }

        if model_size not in self._feature_dims:
            raise ValueError(
                f"Unknown model size: {model_size}. Choose from {list(self._feature_dims.keys())}"
            )

        # Load pretrained Dasheng model
        self.model = self._load_dasheng_model(model_size).to(self.device)
        self.model.eval()

        print(f"Dasheng model '{model_size}' loaded and moved to {self.device}")
        print(f"Target sampling rate: {self.target_sample_rate} Hz")

    def _load_dasheng_model(self, model_size: str):
        if model_size == "dasheng_base":
            return dasheng_base()
        if model_size == "dasheng_06B":
            return dasheng_06B()
        if model_size == "dasheng_12B":
            return dasheng_12B()
        raise ValueError(f"Unsupported model size: {model_size}")

    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Extract Dasheng embeddings for a single-channel signal.

        The input is guaranteed to be shape [1, samples]. Resamples to 16 kHz when needed,
        chunks into 10-second segments, runs the model, pools over time (mean), and then
        aggregates across segments by mean.
        """
        waveform = signal_tensor

        # Remove DC component
        waveform = waveform - waveform.mean()

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        # Segment into 10s chunks
        segments = []
        num_segments = waveform.shape[-1] // self.max_samples

        # Full segments
        for i in range(num_segments):
            start = i * self.max_samples
            end = start + self.max_samples
            segments.append(waveform[..., start:end])

        # Tail segment (last 10s window)
        if num_segments * self.max_samples < waveform.shape[-1]:
            remaining_start = waveform.shape[-1] - self.max_samples
            segments.append(waveform[..., remaining_start:])

        # Model forward on each segment
        features = []
        with torch.no_grad():
            for segment in segments:
                if segment.dim() == 1:
                    segment = segment.unsqueeze(0)
                segment = segment.to(self.device)

                # Dasheng forward returns [B, T, D]; pool over T
                outputs = self.model(segment)
                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1)
                elif outputs.dim() == 2:
                    # Already [B, D]
                    pass
                else:
                    raise RuntimeError(f"Unexpected Dasheng output shape: {tuple(outputs.shape)}")

                # Convert to [D]
                if outputs.shape[0] == 1:
                    outputs = outputs.squeeze(0)

                features.append(outputs.cpu())

        # Aggregate across segments
        features = torch.stack(features, dim=0).mean(dim=0)
        return features

    def _get_single_channel_feature_dim(self) -> int:
        return self._feature_dims[self.model_size]


# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor(model_size="dasheng_06B")
    print(f"Feature dimension: {extractor.feature_dim}")
    print("Extractor ready for evaluation!")
    print("⚠️  Important: Dasheng requires 16kHz sampling rate - all inputs will be resampled!")


