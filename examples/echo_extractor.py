"""
ECHO Feature Extractor - Example Implementation

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

# Add your model path here - replace with your actual AudioMAE implementation path
# Example: sys.path.append("/path/to/your/audioMAE/implementation/")
sys.path.append("/path/to/your/ECHO/implementation/")
from ECHO import AudioMAEWithBand

class FeatureExtractor(BandSplitFeatureExtractor):
    """ECHO feature extractor - Example implementation for users."""
    
    def __init__(self, multi_channel_strategy: str = "concatenate"):
        # Initialize with ECHO-specific parameters  
        super().__init__(multi_channel_strategy, base_feature_dim=384, band_width=32)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set expected channels for MAFAULDA dataset (8 channels)
        self.expected_channels = 8
        
        # Model configuration
        self.max_length = 2000
        self.norm_mean = -5.874158
        self.norm_std = 5.223174
        self.band_width = 32
        
        # Load your model here
        model_path = "/path/to/your/ECHO/checkpoint.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError("ECHO model not found. Please download the checkpoint file")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with your configuration
        model_cfg = {
            "spec_len": self.max_length,
            "shift_size": 16,
            "in_chans": 1,
            "embed_dim": 384,
            "encoder_depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "norm_layer": lambda x: torch.nn.LayerNorm(x, eps=1e-6),
            "fix_pos_emb": True,
            "band_width": self.band_width,
            "mask_ratio": 0.75,
            "freq_pos_emb_dim": 384,
        }
        
        self.model = AudioMAEWithBand(**model_cfg)
        
        # Load weights
        if 'encoder' in checkpoint:
            self.model.load_state_dict(checkpoint['encoder'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"ECHO model loaded from {model_path}")
        print(f"Running on device: {self.device}")
    
    def _calculate_freq_bins(self, sample_rate: int) -> int:
        """Calculate frequency bins for ECHO STFT parameters."""
        window_size = int(0.025 * sample_rate)  # ECHO uses 25ms window
        return window_size // 2 + 1
    
    def _calculate_num_bands(self, freq_bins: int) -> int:
        """ECHO uses ceiling division for band calculation."""
        return max(1, (freq_bins + self.band_width - 1) // self.band_width)
    
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from a single-channel signal tensor using ECHO.
        
        This method is called by the base class for each channel individually.
        The input is guaranteed to be [1, samples] format.
        
        Users should implement their own feature extraction logic here.
        This example shows ECHO-specific processing.
        """
        # Input is guaranteed to be [1, samples] by the base class
        waveform = signal_tensor  # Already in correct format
        waveform = waveform - waveform.mean()  # Remove DC
        
        # Step 2: Convert to spectrogram (model-specific preprocessing)
        window_size = int(0.025 * sample_rate)  # 25ms
        hop_size = int(0.01 * sample_rate)  # 10ms
        
        stft = torchaudio.transforms.Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            power=1, center=False
        )
        
        spec = stft(waveform.squeeze(0))
        spec = torch.log(spec + 1e-9)
        spec = (spec - self.norm_mean) / (self.norm_std * 2)
        
        # Step 3: Process in segments if needed
        input_specs = []
        num_segments = spec.shape[-1] // self.max_length
        for i in range(num_segments):
            input_specs.append(spec[..., i * self.max_length:(i + 1) * self.max_length])
        if num_segments * self.max_length < spec.shape[-1]:
            input_specs.append(spec[..., -self.max_length:])
        
        # Step 4: Extract features using your model
        features = []
        for segment_spec in input_specs:
            with torch.no_grad():
                feature = self.model.extract_features(segment_spec.to(self.device), sample_rate).cpu()
                features.append(feature)
        
        # Step 5: Aggregate features (mean pooling in this example)
        features = torch.stack(features, dim=0).mean(dim=0)
        return features.cpu()
    



# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    print(f"Feature dimension: {extractor.feature_dim}")
    print("Extractor ready for evaluation!")

