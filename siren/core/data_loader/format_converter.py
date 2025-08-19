"""
File format conversion utilities for machine signal datasets.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This module provides utilities for converting between different file formats
commonly used in machine signal datasets, including CSV, WAV, and MAT files.
"""

import numpy as np
import pandas as pd
import wave
import struct
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FormatConverter:
    """
    Utility class for converting between different file formats.
    """
    
    @staticmethod
    def csv_to_wav(csv_file: str, wav_file: str, sampling_rate: int = 16000, 
                   normalize: bool = True, use_all_channels: bool = False):
        """
        Convert CSV file to WAV file.
        
        Args:
            csv_file: Path to input CSV file
            wav_file: Path to output WAV file
            sampling_rate: Target sampling rate
            normalize: Whether to normalize the signal
            use_all_channels: Whether to use all channels or just the first one
        """
        try:
            # Read CSV file
            logger.info(f"Reading CSV file: {csv_file}")
            data = pd.read_csv(csv_file, header=None)
            
            # Check data shape
            logger.info(f"CSV data shape: {data.shape}")
            
            if data.shape[1] != 8:
                logger.warning(f"Expected 8 columns, got {data.shape[1]}")
            
            # Extract vibration data (skip first column which is time)
            vibration_data = data.iloc[:, 1:].values  # Columns 1-7 are vibration sensors
            
            if use_all_channels:
                # Use all channels, average them
                signal = np.mean(vibration_data, axis=1)
            else:
                # Use only the first vibration sensor
                signal = vibration_data[:, 0]
            
            # Normalize if requested
            if normalize:
                signal = signal - np.mean(signal)
                if np.std(signal) > 0:
                    signal = signal / np.std(signal)
            
            # Convert to 16-bit PCM
            signal = np.int16(signal * 32767)
            
            # Write WAV file
            logger.info(f"Writing WAV file: {wav_file}")
            with wave.open(wav_file, 'w') as wav_file_obj:
                wav_file_obj.setnchannels(1)  # Mono
                wav_file_obj.setsampwidth(2)  # 16-bit
                wav_file_obj.setframerate(sampling_rate)
                wav_file_obj.writeframes(signal.tobytes())
            
            logger.info(f"Successfully converted {csv_file} to {wav_file}")
            
        except Exception as e:
            logger.error(f"Error converting {csv_file} to {wav_file}: {e}")
            raise
    
    @staticmethod
    def mat_to_wav(mat_file: str, wav_file: str, sampling_rate: int = 16000):
        """
        Convert MAT file to WAV file.
        
        Args:
            mat_file: Path to input MAT file
            wav_file: Path to output WAV file
            sampling_rate: Target sampling rate
        """
        try:
            # Import scipy only when needed
            from scipy.io import loadmat
            
            logger.info(f"Reading MAT file: {mat_file}")
            mat_data = loadmat(mat_file)
            
            # Find the vibration data in the MAT file
            # This is dataset-specific and may need adjustment
            signal = None
            
            # Try common variable names
            for key in ['DE_time', 'FE_time', 'BA_time', 'time']:
                if key in mat_data:
                    signal = mat_data[key].flatten()
                    logger.info(f"Found signal in variable: {key}")
                    break
            
            if signal is None:
                # Try to find any numeric array
                for key, value in mat_data.items():
                    if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) > 1000:
                        signal = value.flatten()
                        logger.info(f"Using signal from variable: {key}")
                        break
            
            if signal is None:
                raise ValueError("Could not find vibration signal in MAT file")
            
            # Normalize signal
            signal = signal - np.mean(signal)
            if np.std(signal) > 0:
                signal = signal / np.std(signal)
            
            # Convert to 16-bit PCM
            signal = np.int16(signal * 32767)
            
            # Write WAV file
            logger.info(f"Writing WAV file: {wav_file}")
            with wave.open(wav_file, 'w') as wav_file_obj:
                wav_file_obj.setnchannels(1)  # Mono
                wav_file_obj.setsampwidth(2)  # 16-bit
                wav_file_obj.setframerate(sampling_rate)
                wav_file_obj.writeframes(signal.tobytes())
            
            logger.info(f"Successfully converted {mat_file} to {wav_file}")
            
        except Exception as e:
            logger.error(f"Error converting {mat_file} to {wav_file}: {e}")
            raise
    
    @staticmethod
    def batch_convert_csv_to_wav(csv_dir: str, wav_dir: str, sampling_rate: int = 16000,
                                normalize: bool = True, use_all_channels: bool = False):
        """
        Batch convert CSV files to WAV files.
        
        Args:
            csv_dir: Directory containing CSV files
            wav_dir: Directory to save WAV files
            sampling_rate: Target sampling rate
            normalize: Whether to normalize signals
            use_all_channels: Whether to use all channels
        """
        csv_path = Path(csv_dir)
        wav_path = Path(wav_dir)
        wav_path.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(csv_path.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to convert")
        
        for csv_file in csv_files:
            wav_file = wav_path / f"{csv_file.stem}.wav"
            try:
                FormatConverter.csv_to_wav(
                    str(csv_file), str(wav_file), 
                    sampling_rate, normalize, use_all_channels
                )
            except Exception as e:
                logger.error(f"Failed to convert {csv_file}: {e}")
                continue 