"""
Mel Spectrogram Feature Extractor - Example Implementation

Author: Yucong Zhang
Email: yucong0428@outlook.com

This is a simplified example showing how to wrap traditional signal processing
features for use with the siren evaluation framework.

Key points for users creating their own extractors:
1. Inherit from BaseFeatureExtractor
2. Implement extract_features() and extract_features_from_signal() methods
3. Set feature_dim property
4. Use load_signal_data() for universal file loading
5. Handle signal preprocessing in extract_features_from_signal()
"""

import torch
import torchaudio
from pathlib import Path

from siren.core.base_extractor import BaseFeatureExtractor


class FeatureExtractor(BaseFeatureExtractor):
    """Mel Spectrogram feature extractor - Example implementation for users."""
    
    def __init__(self, feature_dim: int = 128, pooling_method: str = 'mean', multi_channel_strategy: str = "concatenate"):
        """
        Initialize the feature extractor.
        
        Args:
            feature_dim: Output feature dimension (should match n_mels)
            pooling_method: Temporal pooling method ('mean', 'max')
            multi_channel_strategy: Multi-channel processing strategy
        """
        super().__init__(multi_channel_strategy)
        self.pooling_method = pooling_method
        
        # Set expected channels for MAFAULDA dataset (8 channels)
        self.expected_channels = 8
        
        # Signal processing parameters
        self.target_sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = feature_dim
        
        # Create mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        print(f"Mel Spectrogram extractor initialized")
        print(f"Target sampling rate: {self.target_sample_rate}Hz")
        print(f"Feature dimension: {feature_dim} (n_mels)")
        print(f"Pooling method: {pooling_method}")
    

    
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from a single-channel signal tensor using Mel Spectrogram.
        
        This method is called by the base class for each channel individually.
        The input is guaranteed to be [1, samples] format.
        
        This example shows traditional signal processing approach.
        """
        # Input is guaranteed to be [1, samples] by the base class
        waveform = signal_tensor  # Already in correct format
        
        # Step 2: Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
        
        # Step 3: Extract mel spectrogram
        mel_spec = self.mel_transform(waveform)  # (1, n_mels, time_frames)
        
        # Step 4: Apply temporal pooling (aggregation strategy)
        if self.pooling_method == 'mean':
            features = mel_spec.mean(dim=-1)  # (1, n_mels)
        elif self.pooling_method == 'max':
            features = mel_spec.max(dim=-1)[0]  # (1, n_mels)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # Step 5: Return feature vector
        return features.squeeze(0)  # Remove batch dimension -> (n_mels,)
    
    def _get_single_channel_feature_dim(self) -> int:
        """Return the feature dimension for a single channel."""
        return self.n_mels  # Mel spectrogram output dimension for single channel


# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    print(f"Feature dimension: {extractor.feature_dim}")
    print("Extractor ready for evaluation!")

