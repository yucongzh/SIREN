"""
Base feature extractor interface for DCASE series datasets.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This module provides the base interface for custom feature extractors in the SIREN toolkit.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
import numpy as np


class BaseFeatureExtractor(ABC):
    """
    Base class for custom feature extractors.
    
    Users must implement this interface to provide their own feature extraction logic.
    The feature extractor handles the complete audio processing pipeline:
    audio file -> audio loading -> preprocessing -> multi-channel processing -> feature extraction -> feature vector
    
    The base class provides unified multi-channel processing logic. Users only need to implement
    single-channel feature extraction in _extract_single_channel_features().
    """
    
    def __init__(self, multi_channel_strategy: str = "concatenate"):
        """
        Initialize the feature extractor with multi-channel processing strategy.
        
        Args:
            multi_channel_strategy: Strategy for handling multi-channel signals
                - "concatenate": Extract features from each channel and concatenate (default, recommended)
                - "mean": Average all channels before feature extraction
                - "first": Use only the first channel
                - "last": Use only the last channel
        """
        self.multi_channel_strategy = multi_channel_strategy
        # Reserved flags (no multiprocessing here to keep extractors simple)
        
    def extract_features(self, audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from a single audio file.
        
        This method implements the complete audio processing pipeline:
        1. Load audio file
        2. Preprocess audio (resampling, normalization, etc.)
        3. Extract features using extract_features_from_signal()
        4. Return feature vector
        
        Args:
            audio_path: Path to the audio file
            sample_rate: Sample rate of the audio file
        Returns:
            torch.Tensor: Feature vector of shape (feature_dim,)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is corrupted or invalid
        """
        try:
            # Load signal data using universal loading method
            signal_tensor, actual_sample_rate = self.load_signal_data(audio_path, sample_rate)
            
            # Extract features from loaded signal
            return self.extract_features_from_signal(signal_tensor, actual_sample_rate)
            
        except Exception as e:
            print(f"Failed to extract features from {audio_path}: {e}")
            # Return zero features as fallback
            return torch.zeros(self.feature_dim)
    
    @property
    def feature_dim(self) -> int:
        """
        Return the dimension of extracted features, considering multi-channel strategy.
        
        This property automatically calculates the feature dimension based on the
        multi-channel processing strategy and the single-channel feature dimension.
        For models with sample-rate dependent dimensions, this returns the default (16kHz).
        
        Returns:
            int: Total feature dimension after multi-channel processing
        """
        single_channel_dim = self._get_single_channel_feature_dim()
        
        if self.multi_channel_strategy == "concatenate":
            # Need to know the expected number of channels
            expected_channels = getattr(self, 'expected_channels', 1)
            return single_channel_dim * expected_channels
        else:
            # For mean, first, last strategies, output dimension equals single-channel dimension
            return single_channel_dim
    
    def get_feature_dim_for_sample_rate(self, sample_rate: int) -> int:
        """
        Calculate feature dimension for a specific sample rate.
        
        This method allows calculating the correct feature dimension when the sample rate
        is known, which is useful for models with sample-rate dependent band splitting.
        
        Args:
            sample_rate: The sample rate to calculate dimension for
            
        Returns:
            int: Total feature dimension for the given sample rate
        """
        # Check if the extractor has dynamic sample rate calculation
        if hasattr(self, '_calculate_feature_dim_for_sample_rate'):
            single_channel_dim = self._calculate_feature_dim_for_sample_rate(sample_rate)
        else:
            # Fallback to static dimension
            single_channel_dim = self._get_single_channel_feature_dim()
        
        if self.multi_channel_strategy == "concatenate":
            expected_channels = getattr(self, 'expected_channels', 1)
            return single_channel_dim * expected_channels
        else:
            return single_channel_dim
    
    def extract_features_batch(self, audio_paths: List[str], sample_rate: int = 16000) -> List[torch.Tensor]:
        """
        Extract features from multiple audio files.
        
        This is a convenience method that calls extract_features() for each file.
        Users can override this method to implement batch processing for better efficiency.
        
        Args:
            audio_paths: List of audio file paths
            sample_rate: Sample rate of the audio files
        Returns:
            List[torch.Tensor]: List of feature vectors
        """
        from tqdm import tqdm
        
        features = []
        for audio_path in tqdm(audio_paths, desc="Extracting features", unit="file", ncols=100):
            try:
                feature = self.extract_features(audio_path, sample_rate=sample_rate)
                features.append(feature)
            except Exception as e:
                raise RuntimeError(f"Failed to extract features from {audio_path}: {e}")
        
        return features
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate if an audio file can be processed.
        
        This is an optional method that users can implement to check if an audio file
        is valid before attempting feature extraction.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        import os
        return os.path.exists(audio_path) and os.path.isfile(audio_path)
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported audio file formats.
        
        This is an optional method that users can implement to specify which
        audio file formats their extractor supports.
        
        Returns:
            List[str]: List of supported file extensions (e.g., ['.wav', '.mp3'])
        """
        return ['.wav', '.mp3', '.flac', '.m4a', '.csv', '.mat', '.txt']  # Default supported formats 
    
    def check_signal_type(self, audio_path: str) -> str:
        """
        Check the type of signal the extractor is designed for.
        
        Returns:
            str: Type of signal (e.g., 'audio', 'signal')
        """
        if audio_path.endswith(".wav") or audio_path.endswith(".mp3") or audio_path.endswith(".flac") or audio_path.endswith(".m4a"):
            return 'audio'
        elif audio_path.endswith(".csv") or audio_path.endswith(".mat") or audio_path.endswith(".txt"):
            return 'signal'
        else:
            raise ValueError(f"Unsupported file format: {audio_path}")
    
    def load_signal_data(self, file_path: str, sample_rate: int = 16000) -> Tuple[torch.Tensor, int]:
        """
        Universal signal loading method that handles different file formats.
        
        This method provides a unified interface for loading different types of signal data:
        - Audio files (.wav, .mp3, .flac, .m4a) using torchaudio
        - CSV files (.csv) using pandas (for MAFAULDA-style datasets)
        - MATLAB files (.mat) using scipy.io (for CWRU-style datasets)
        
        Args:
            file_path: Path to the signal file
            sample_rate: Target sample rate (used for resampling if needed)
            
        Returns:
            Tuple[torch.Tensor, int]: (signal_tensor, actual_sample_rate)
            - signal_tensor: Loaded signal as torch tensor, shape [channels, samples]
            - actual_sample_rate: Actual sample rate of the loaded signal
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        import os
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Signal file not found: {file_path}")
        
        signal_type = self.check_signal_type(file_path)
        
        if signal_type == 'audio':
            # Load audio files using torchaudio
            return self._load_audio_file(file_path, sample_rate)
        elif signal_type == 'signal':
            # Load signal data files
            if file_path.endswith('.csv'):
                return self._load_csv_file(file_path, sample_rate)
            elif file_path.endswith('.mat'):
                return self._load_mat_file(file_path, sample_rate)
            else:
                raise ValueError(f"Unsupported signal file format: {file_path}")
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
    
    def _load_audio_file(self, file_path: str, target_sample_rate: int) -> Tuple[torch.Tensor, int]:
        """Load audio files using torchaudio."""
        import torchaudio
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        return waveform, sample_rate
    
    def _load_csv_file(self, file_path: str, sample_rate: int) -> Tuple[torch.Tensor, int]:
        """Load CSV files (MAFAULDA style)."""
        import pandas as pd
        
        # Read CSV file
        data = pd.read_csv(file_path, header=None).values  # shape: [num_samples, channels]
        signal = data.T  # shape: [channels, num_samples]
        
        # Convert to torch tensor
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        
        return signal_tensor, sample_rate
    
    def _load_mat_file(self, file_path: str, sample_rate: int) -> Tuple[torch.Tensor, int]:
        """Load MATLAB files (CWRU style)."""
        from scipy.io import loadmat
        
        try:
            mat_data = loadmat(file_path)
            
            # Find signal data in the .mat file
            signal_data = None
            
            # Try common key patterns for CWRU dataset
            possible_keys = [
                'DE_time', 'FE_time', 'BA_time',  # Standard CWRU keys
                'X097_DE_time', 'X097_FE_time', 'X097_BA_time',  # With prefixes
            ]
            
            # Try exact matches first
            for key in possible_keys:
                if key in mat_data:
                    signal_data = mat_data[key].flatten()
                    break
            
            # If no exact match, try pattern matching
            if signal_data is None:
                for key in mat_data.keys():
                    if (('DE' in key or 'FE' in key or 'BA' in key) and 
                        'time' in key and 
                        not key.startswith('_')):
                        signal_data = mat_data[key].flatten()
                        break
            
            # Fallback: use any non-metadata key
            if signal_data is None:
                for key, value in mat_data.items():
                    if (not key.startswith('_') and 
                        isinstance(value, np.ndarray) and 
                        value.size > 1000):  # Assume signal data has many samples
                        signal_data = value.flatten()
                        break
            
            if signal_data is None:
                raise ValueError(f"Cannot find signal data in {file_path}. "
                               f"Available keys: {list(mat_data.keys())}")
            
            # Convert to torch tensor and add channel dimension
            signal_tensor = torch.tensor(signal_data, dtype=torch.float32).unsqueeze(0)  # [1, samples]
            
            return signal_tensor, sample_rate
            
        except Exception as e:
            raise ValueError(f"Error loading MATLAB file {file_path}: {e}")
    
    def extract_features_from_signal(self, signal_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from a signal tensor with unified multi-channel processing.
        
        This method provides unified multi-channel processing logic. Users should NOT override
        this method. Instead, implement _extract_single_channel_features() for single-channel processing.
        
        Args:
            signal_tensor: Signal tensor of various shapes:
                - [samples]: Single-channel 1D signal (CWRU style)
                - [1, samples]: Single-channel 2D signal 
                - [channels, samples]: Multi-channel signal (MAFAULDA style)
                - [samples, channels]: Multi-channel signal (transposed format)
            sample_rate: Sample rate of the signal
            
        Returns:
            torch.Tensor: Feature vector with shape depending on multi_channel_strategy:
                - concatenate: [channels * single_channel_feature_dim]
                - others: [single_channel_feature_dim]
        """
        # Step 1: Normalize signal shape to [channels, samples]
        normalized_signal = self._normalize_signal_shape(signal_tensor)
        
        # Step 2: Process multi-channel signal according to strategy
        return self._process_multichannel_signal(normalized_signal, sample_rate)
    
    def _normalize_signal_shape(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize signal tensor shape to [channels, samples] format.
        
        This method handles all possible input formats and converts them to a
        standardized [channels, samples] format for consistent processing.
        
        Args:
            signal_tensor: Input signal tensor of various shapes
            
        Returns:
            torch.Tensor: Normalized signal with shape [channels, samples]
        """
        if signal_tensor.dim() == 1:
            # 1D signal: [samples] -> [1, samples]
            return signal_tensor.unsqueeze(0)
        
        elif signal_tensor.dim() == 2:
            dim0, dim1 = signal_tensor.shape
            
            # Heuristic: samples are usually much more than channels
            # Assume channels <= 16 for typical applications
            if dim1 > dim0 and dim0 <= 16:
                # Likely [channels, samples] format
                return signal_tensor
            elif dim0 > dim1 and dim1 <= 16:
                # Likely [samples, channels] format, transpose it
                return signal_tensor.T
            else:
                # Ambiguous case, assume [channels, samples] as default
                return signal_tensor
        else:
            raise ValueError(f"Unsupported signal tensor dimension: {signal_tensor.dim()}. "
                           f"Expected 1D or 2D tensor, got {signal_tensor.shape}")
    
    def _process_multichannel_signal(self, signal_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Process multi-channel signal according to the specified strategy.
        
        Args:
            signal_tensor: Normalized signal tensor with shape [channels, samples]
            sample_rate: Sample rate of the signal
            
        Returns:
            torch.Tensor: Processed feature vector
        """
        channels, samples = signal_tensor.shape
        
        if self.multi_channel_strategy == "concatenate":
            # Extract features from each channel and concatenate
            features_list = []
            for channel_idx in range(channels):
                channel_signal = signal_tensor[channel_idx:channel_idx+1, :]  # [1, samples]
                channel_features = self._extract_single_channel_features(channel_signal, sample_rate)
                features_list.append(channel_features)
            return torch.cat(features_list, dim=0)
        
        elif self.multi_channel_strategy == "mean":
            # Average all channels before feature extraction
            mean_signal = signal_tensor.mean(dim=0, keepdim=True)  # [1, samples]
            return self._extract_single_channel_features(mean_signal, sample_rate)
        
        elif self.multi_channel_strategy == "first":
            # Use only the first channel
            first_signal = signal_tensor[:1, :]  # [1, samples]
            return self._extract_single_channel_features(first_signal, sample_rate)
        
        elif self.multi_channel_strategy == "last":
            # Use only the last channel
            last_signal = signal_tensor[-1:, :]  # [1, samples]
            return self._extract_single_channel_features(last_signal, sample_rate)
        
        else:
            raise ValueError(f"Unknown multi_channel_strategy: '{self.multi_channel_strategy}'. "
                           f"Supported strategies: 'concatenate', 'mean', 'first', 'last'")
    
    @abstractmethod
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Extract features from a single-channel signal tensor.
        
        This is the main method that users need to implement. The input is guaranteed to be
        a single-channel signal with shape [1, samples]. Users don't need to handle
        multi-channel processing or dimension normalization.
        
        Args:
            signal_tensor: Single-channel signal tensor with shape [1, samples]
            sample_rate: Sample rate of the signal
            
        Returns:
            torch.Tensor: Feature vector with shape [single_channel_feature_dim]
        """
        pass
    
    @abstractmethod
    def _get_single_channel_feature_dim(self) -> int:
        """
        Return the feature dimension for a single channel.
        
        This is used to calculate the total feature dimension when using
        the "concatenate" multi-channel strategy.
        
        Returns:
            int: Feature dimension for a single channel
        """
        pass


class BandSplitFeatureExtractor(BaseFeatureExtractor):
    """
    Base class for feature extractors that use band splitting (e.g., FISHER, ECHO).
    
    This class provides common functionality for models that split frequency bands
    and have sample-rate dependent feature dimensions.
    """
    
    def __init__(self, multi_channel_strategy: str = "concatenate", 
                 base_feature_dim: int = 384, band_width: int = 32):
        """
        Initialize the band split feature extractor.
        
        Args:
            multi_channel_strategy: Multi-channel processing strategy
            base_feature_dim: Base feature dimension per band (usually 384)
            band_width: Width of each frequency band
        """
        super().__init__(multi_channel_strategy)
        self.base_feature_dim = base_feature_dim
        self.band_width = band_width
    
    def _get_single_channel_feature_dim(self) -> int:
        """Return the feature dimension for a single channel (default 16kHz)."""
        return self._calculate_feature_dim_for_sample_rate(16000)
    
    def _calculate_feature_dim_for_sample_rate(self, sample_rate: int) -> int:
        """
        Calculate feature dimension for a specific sample rate.
        
        This method can be overridden by subclasses for model-specific calculations.
        
        Args:
            sample_rate: Sample rate to calculate dimension for
            
        Returns:
            int: Feature dimension for single channel at given sample rate
        """
        # Calculate frequency bins based on sample rate and model parameters
        freq_bins = self._calculate_freq_bins(sample_rate)
        
        # Calculate number of bands (can be overridden for different splitting logic)
        num_bands = self._calculate_num_bands(freq_bins)
        
        return self.base_feature_dim * num_bands
    
    def _calculate_freq_bins(self, sample_rate: int) -> int:
        """
        Calculate number of frequency bins for given sample rate.
        
        This should be overridden by subclasses to match their STFT parameters.
        
        Args:
            sample_rate: Sample rate
            
        Returns:
            int: Number of frequency bins
        """
        raise NotImplementedError("Subclasses must implement _calculate_freq_bins()")
    
    def _calculate_num_bands(self, freq_bins: int) -> int:
        """
        Calculate number of bands from frequency bins.
        
        Default implementation uses floor division. Override for ceiling division.
        
        Args:
            freq_bins: Number of frequency bins
            
        Returns:
            int: Number of bands
        """
        num_bands = freq_bins // self.band_width
        return max(1, num_bands)  # At least one band
