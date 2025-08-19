"""
CED Feature Extractor - Example Implementation

Author: Yucong Zhang
Email: yucong0428@outlook.com

This is a simplified example showing how to wrap the CED model 
for use with the siren evaluation framework.

Based on CED README usage instructions:
- Use Huggingface Transformers (recommended)
- Models work with 16 kHz audio and use 64-dim Mel-spectrograms
- Models expect 16kHz audio input
- Models output classification logits
- CED does NOT use band splitting (unlike ECHO/FISHER)

Key points for users creating their own extractors:
1. Inherit from BaseFeatureExtractor (NOT BandSplitFeatureExtractor)
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

from siren.core.base_extractor import BaseFeatureExtractor

# Try to import transformers for CED models
try:
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    TRANSFORMERS_AVAILABLE = True
    print("Transformers library available for CED models")
except ImportError as e:
    print(f"Warning: Transformers not available: {e}")
    print("Please install transformers: pip install transformers")
    TRANSFORMERS_AVAILABLE = False


class FeatureExtractor(BaseFeatureExtractor):
    """CED feature extractor using Huggingface Transformers (recommended approach)."""
    
    def __init__(self, model_size: str = "ced-small", multi_channel_strategy: str = "concatenate"):
        """
        Initialize the CED feature extractor.
        
        Args:
            model_size: CED model size ('ced-tiny', 'ced-mini', 'ced-small', 'ced-base')
            multi_channel_strategy: Multi-channel processing strategy
        """
        super().__init__(multi_channel_strategy)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for CED models. "
                            "Please install: pip install transformers")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set expected channels for MAFAULDA dataset (8 channels)
        self.expected_channels = 8
        
        # Model configuration
        self.model_size = model_size
        self.target_sample_rate = 16000  # CED models are trained on 16kHz
        self.max_samples = 160000  # 10 seconds at 16kHz (CED default)
        
        # CED model parameters (from README)
        self.model_configs = {
            'ced-tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
            'ced-mini': {'embed_dim': 256, 'depth': 12, 'num_heads': 4},
            'ced-small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
            'ced-base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12}
        }
        
        # Load CED model using Huggingface Transformers
        self.model, self.feature_extractor = self._load_ced_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"{model_size} model loaded successfully via Huggingface")
        print(f"Running on device: {self.device}")
        print(f"Model config: {self.model_configs[model_size]}")
        print(f"Target sampling rate: {self.target_sample_rate}Hz")
    
    def _load_ced_model(self):
        """Load the specified CED model using Huggingface Transformers."""
        if self.model_size not in self.model_configs:
            raise ValueError(f"Unknown model size: {self.model_size}. "
                           f"Available: {list(self.model_configs.keys())}")
        
        # Use Huggingface model names
        model_name = f"mispeech/{self.model_size}"
        
        try:
            # Load feature extractor and model
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
            
            print(f"Successfully loaded {model_name}")
            return model, feature_extractor
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CED model {model_name}: {e}")
    
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from a single-channel signal tensor using CED.
        
        This method is called by the base class for each channel individually.
        The input is guaranteed to be [1, samples] format.
        
        Based on CED README Huggingface usage example.
        """
        # Input is guaranteed to be [1, samples] by the base class
        waveform = signal_tensor  # Already in correct format
        
        # Step 1: Remove DC component
        waveform = waveform - waveform.mean()
        
        # Step 2: CRITICAL - Resample to 16kHz (CED requirement)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
        
        # Step 3: Segment the audio (CED works with 10-second chunks)
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
        
        # Step 4: Extract features from each segment using CED model
        features = []
        for segment in segments:
            # Prepare input for Huggingface model
            # Convert to numpy for feature extractor
            segment_np = segment.squeeze(0).numpy()
            
            # Use feature extractor to prepare inputs (as per README example)
            inputs = self.feature_extractor(
                segment_np, 
                sampling_rate=self.target_sample_rate, 
                return_tensors="pt"
            )
            
            # Extract features using the model
            with torch.no_grad():
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get model outputs
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    segment_features = outputs.hidden_states  # [batch_size, seq_len, hidden_dim]
                
                # Pool features across time dimension
                if segment_features.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                    # Use mean pooling across time dimension
                    segment_features = segment_features.mean(dim=1)  # [batch_size, hidden_dim]
                
                features.append(segment_features.squeeze(0).cpu())
        
        # Step 5: Aggregate features (mean pooling across segments)
        features = torch.stack(features, dim=0).mean(dim=0)
        return features
    
    def _get_single_channel_feature_dim(self) -> int:
        """Return the feature dimension for a single channel."""
        # Return the embedding dimension of the CED model
        return self.model_configs[self.model_size]['embed_dim']


# Example usage
if __name__ == "__main__":
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers library is required for CED models.")
        print("Please install: pip install transformers")
        sys.exit(1)
    
    # Test different model sizes
    for model_size in ['ced-tiny', 'ced-mini', 'ced-small', 'ced-base']:
        try:
            extractor = FeatureExtractor(model_size=model_size)
            print(f"{model_size} feature dimension: {extractor.feature_dim}")
            print(f"Extractor ready for evaluation!")
        except Exception as e:
            print(f"Failed to load {model_size}: {e}")
    
    # Test with default model (ced-mini)
    try:
        extractor = FeatureExtractor()
        print(f"\nDefault {extractor.model_size} feature dimension: {extractor.feature_dim}")
        print("Default extractor ready for evaluation!")
        print("⚠️  Important: CED requires 16kHz sampling rate - all inputs will be resampled!")
    except Exception as e:
        print(f"Failed to load default CED model: {e}")
