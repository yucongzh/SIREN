"""
BEATs Feature Extractor - Example Implementation
Beats: Audio pre-training with acoustic tokenizers
Authors: Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, Wanxiang Che, Xiangzhan Yu, Furu Wei
Paper link: https://arxiv.org/pdf/2212.09058

Author: Yucong Zhang
Email: yucong0428@outlook.com

This is a simplified example showing how to wrap your own model 
for use with the siren evaluation framework.

Key points for users creating their own extractors:
1. Inherit from BaseFeatureExtractor
2. Implement extract_features() and extract_features_from_signal() methods
3. Set feature_dim property
4. Use load_signal_data() for universal file loading
5. Handle sampling rate conversion (BEATs requires 16kHz)
"""

import torch
import torchaudio
import torch.nn.functional as F
import os
import sys

from siren.core.base_extractor import BaseFeatureExtractor

# Add your model path here - replace with your actual BEATs implementation path
# Example: sys.path.append("/path/to/your/beats/implementation/")
sys.path.append("/path/to/your/beats/implementation/")
from BEATs.BEATs import BEATs, BEATsConfig


class FeatureExtractor(BaseFeatureExtractor):
    """BEATs feature extractor - Example implementation for users."""
    
    def __init__(self, multi_channel_strategy: str = "concatenate"):
        super().__init__(multi_channel_strategy)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set expected channels for MAFAULDA dataset (8 channels)
        self.expected_channels = 8
        
        # BEATs requires 16kHz sampling rate
        self.target_sample_rate = 16000
        self.max_samples = 160000  # 10 seconds at 16kHz
        
        # Load your model here - replace with your actual model path
        # Example: model_path = "/path/to/your/beats/checkpoint.pt"
        model_path = "/path/to/your/beats/checkpoint.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError("BEATs model not found. Please download BEATs_iter3.pt")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        cfg = BEATsConfig(checkpoint['cfg'])
        
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"BEATs model loaded from {model_path}")
        print(f"Running on device: {self.device}")
        print(f"Target sampling rate: {self.target_sample_rate}Hz")
    

    
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from signal tensor using BEATs.
        
        Users should implement their own feature extraction logic here.
        This example shows BEATs-specific processing.
        
        Important: BEATs only supports 16kHz sampling rate!
        """
        # Input is guaranteed to be [1, samples] by the base class
        waveform = signal_tensor  # Already in correct format
        
        waveform = waveform - waveform.mean()  # Remove DC
        
        # Step 2: CRITICAL - Resample to 16kHz (BEATs requirement)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
            # print(f"Resampled from {sample_rate}Hz to {self.target_sample_rate}Hz for BEATs")
        
        # Step 3: Segment the audio (BEATs max 10 seconds)
        segments = []
        num_segments = waveform.shape[-1] // self.max_samples
        
        # Add complete segments
        for i in range(num_segments):
            start = i * self.max_samples
            end = start + self.max_samples
            segments.append(waveform[..., start:end])
        
        # Handle remaining samples from the end
        if num_segments * self.max_samples < waveform.shape[-1]:
            remaining_start = waveform.shape[-1] - self.max_samples
            segments.append(waveform[..., remaining_start:])
        
        # Step 4: Extract features from each segment using your model
        features = []
        for segment in segments:
            # Prepare input for BEATs
            if segment.dim() == 1:
                segment = segment.unsqueeze(0)
            segment = segment.to(self.device)
            
            # Create padding mask
            padding_mask = torch.zeros(segment.shape[0], segment.shape[1], dtype=torch.bool, device=self.device)
            
            with torch.no_grad():
                segment_features, _ = self.model.extract_features(segment, padding_mask=padding_mask)
                
                # Temporal pooling (mean across time dimension)
                if segment_features.dim() == 3:
                    segment_features = segment_features.mean(dim=1)
                elif segment_features.dim() == 2:
                    segment_features = segment_features.mean(dim=0, keepdim=True)
                
                if segment_features.dim() == 2 and segment_features.shape[0] == 1:
                    segment_features = segment_features.squeeze(0)
                
                features.append(segment_features.cpu())
        
        # Step 5: Aggregate features (mean pooling in this example)
        features = torch.stack(features, dim=0).mean(dim=0)
        return features
    
    def _get_single_channel_feature_dim(self) -> int:
        """Return the feature dimension for a single channel."""
        return 768  # BEATs model output dimension for single channel


# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    print(f"Feature dimension: {extractor.feature_dim}")
    print("Extractor ready for evaluation!")
    print("⚠️  Important: BEATs requires 16kHz sampling rate - all inputs will be resampled!")
