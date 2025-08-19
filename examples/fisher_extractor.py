"""
FISHER Feature Extractor - Example Implementation

Author: Yucong Zhang
Email: yucong0428@outlook.com

This is a simplified example showing how to wrap your own model 
for use with the siren evaluation framework.

Key points for users creating their own extractors:
1. Inherit from BaseFeatureExtractor
2. Implement _extract_single_channel_features() method (NOT extract_features_from_signal)
3. Implement _get_single_channel_feature_dim() method
4. Set expected_channels for multi-channel datasets
5. Multi-channel processing is handled automatically by the base class
"""

import torch
import torchaudio
import torch.nn.functional as F
import os
import sys

from siren.core.base_extractor import BandSplitFeatureExtractor

# Add your model path here - replace with your actual FISHER implementation path
# Example: sys.path.append("/path/to/your/fisher/implementation/")
sys.path.append("/path/to/your/fisher/implementation/")
from FISHER.models.fisher import FISHER


class FeatureExtractor(BandSplitFeatureExtractor):
    """FISHER feature extractor - Example implementation for users."""
    
    def __init__(self, multi_channel_strategy: str = "concatenate"):
        # Initialize with FISHER-specific parameters
        super().__init__(multi_channel_strategy, base_feature_dim=384, band_width=50)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set expected channels for MAFAULDA dataset (8 channels)
        self.expected_channels = 8
        
        # Load your model here - replace with your actual model path
        # Example: model_path = "/path/to/your/fisher/checkpoint.pt"
        model_path = "/path/to/your/fisher/checkpoint.pt"
        self.model = FISHER.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Model-specific parameters
        self.norm_mean = 3.017344307886898
        self.norm_std = 2.1531635155379805
        
        print(f"FISHER model loaded from {model_path}")
        print(f"Running on device: {self.device}")
    
    def _calculate_freq_bins(self, sample_rate: int) -> int:
        """Calculate frequency bins for FISHER STFT parameters."""
        n_fft = 25 * sample_rate // 1000  # FISHER uses 25ms window
        return n_fft // 2 + 1
    
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from a single-channel signal tensor using FISHER.
        
        This method is called by the base class for each channel individually.
        The input is guaranteed to be [1, samples] format.
        
        Users should implement their own feature extraction logic here.
        This example shows FISHER-specific processing.
        """
        # Input is guaranteed to be [1, samples] by the base class
        waveform = signal_tensor  # Already in correct format
        waveform = waveform - waveform.mean()  # Remove DC
        
        # Step 2: Convert to spectrogram (model-specific preprocessing)
        stft = torchaudio.transforms.Spectrogram(
            n_fft=25 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            power=1, center=False
        )
        spec = torch.log(torch.abs(stft(waveform)) + 1e-10)
        spec = spec.transpose(-2, -1)  # [1, time, freq]
        spec = (spec + self.norm_mean) / (self.norm_std * 2)
                
        # Step 3: Process in segments if needed,
        input_specs = []
        num_segments = spec.shape[-2] // 1024
        for i in range(num_segments):
            cut_spec = spec[:, i * 1024:(i + 1) * 1024, :]
            if cut_spec.shape[-1] < self.model.cfg.band_width:
                cut_spec = F.pad(cut_spec, (0, self.model.cfg.band_width - cut_spec.shape[-1]))
            input_specs.append(cut_spec)
        if num_segments * 1024 < spec.shape[-2]:
            cut_spec = spec[:, -1024:, :]
            if cut_spec.shape[-1] < self.model.cfg.band_width:
                cut_spec = F.pad(cut_spec, (0, self.model.cfg.band_width - cut_spec.shape[-1]))
            input_specs.append(cut_spec)

        # Step 5: Extract features using your model
        features = []
        for segment_spec in input_specs:
            segment_spec = segment_spec.unsqueeze(1).to(self.device)
            with torch.no_grad():
                feature = self.model.extract_features(segment_spec)
                features.append(feature)

        # Step 6: Aggregate features (mean pooling in this example)
        features = torch.stack(features, dim=0).mean(dim=0)
        return features.cpu().squeeze()
    


# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    print(f"Feature dimension: {extractor.feature_dim}")
    print("Extractor ready for evaluation!")
