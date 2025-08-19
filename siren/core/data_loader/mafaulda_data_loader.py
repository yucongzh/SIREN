"""
MAFAULDA Dataset Data Loader.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This module provides data loading functionality for the MAFAULDA bearing dataset,
supporting multi-channel vibration data with fault classification labels.
"""

import os
import glob
import numpy as np
from typing import List, Tuple, Union, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

from .base_data_loader import BaseDataLoader, DatasetInfo
from .format_converter import FormatConverter

logger = logging.getLogger(__name__)


class MAFAULDADataLoader(BaseDataLoader):
    """Data loader for MAFAULDA Bearing Dataset."""
    
    def __init__(self, dataset_root: str, config: Dict = None):
        """
        Initialize MAFAULDA data loader.
        
        Args:
            dataset_root: Root directory of MAFAULDA dataset
            config: Configuration dictionary with split parameters
        """
        super().__init__(dataset_root, config)
        
        # Default split configuration
        self.test_size = config.get('test_size', 0.2) if config else 0.2
        self.random_state = config.get('random_state', 42) if config else 42
        self.convert_to_wav = config.get('convert_to_wav', True) if config else True
        self.wav_dir = config.get('wav_dir', None) if config else None
        
        # Cache for split results
        self._train_files = None
        self._test_files = None
        self._train_labels = None
        self._test_labels = None
        
        # Convert CSV to WAV if needed
        if self.convert_to_wav and self.wav_dir:
            self._convert_csv_to_wav()
    
    def _get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="mafaulda",
            task_type="fault_classification",
            signal_type="vibration",
            file_format="csv",
            sampling_rate=50000,
            description="MAFAULDA Bearing Fault Dataset"
        )
    
    def _get_all_csv_files(self) -> List[str]:
        """Get all CSV files in the dataset."""
        pattern = str(self.dataset_root / "**/*.csv")
        csv_files = glob.glob(pattern, recursive=True)
        return sorted(csv_files)
    
    def _get_all_files(self) -> List[str]:
        """Get all files in the dataset (generic method for compatibility)."""
        return self._get_all_csv_files()
    
    def _extract_class_from_path(self, file_path: str) -> str:
        """Extract class label from file path."""
        path_parts = Path(file_path).parts
        
        # For overhang and underhang, extract the specific fault type
        if 'overhang' in file_path:
            if 'ball_fault' in file_path:
                return 'overhang-ball_fault'
            elif 'cage_fault' in file_path:
                return 'overhang-cage_fault'
            elif 'outer_race' in file_path:
                return 'overhang-outer_race'
            else:
                return 'overhang'  # fallback
        elif 'underhang' in file_path:
            if 'ball_fault' in file_path:
                return 'underhang-ball_fault'
            elif 'cage_fault' in file_path:
                return 'underhang-cage_fault'
            elif 'outer_race' in file_path:
                return 'underhang-outer_race'
            else:
                return 'underhang'  # fallback
        
        # For other classes, keep the original behavior
        for part in path_parts:
            if part in ['normal', 'horizontal-misalignment', 'imbalance', 'vertical-misalignment']:
                return part
        
        # If not found, try to extract from full path
        if 'normal' in file_path:
            return 'normal'
        elif 'horizontal-misalignment' in file_path:
            return 'horizontal-misalignment'
        elif 'imbalance' in file_path:
            return 'imbalance'
        elif 'vertical-misalignment' in file_path:
            return 'vertical-misalignment'
        else:
            return 'unknown'
    
    def _convert_csv_to_wav(self):
        """Convert CSV files to WAV files."""
        if not self.wav_dir:
            logger.warning("No WAV directory specified, skipping conversion")
            return
        
        wav_path = Path(self.wav_dir)
        wav_path.mkdir(parents=True, exist_ok=True)
        
        # Get all CSV files
        csv_files = self._get_all_csv_files()
        logger.info(f"Converting {len(csv_files)} CSV files to WAV")
        
        for csv_file in csv_files:
            # Create corresponding WAV path
            csv_path = Path(csv_file)
            relative_path = csv_path.relative_to(self.dataset_root)
            wav_file = wav_path / relative_path.with_suffix('.wav')
            
            # Create directory if needed
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert if WAV file doesn't exist
            if not wav_file.exists():
                try:
                    FormatConverter.csv_to_wav(
                        str(csv_file), str(wav_file),
                        sampling_rate=16000, normalize=True, use_all_channels=False
                    )
                except Exception as e:
                    logger.error(f"Failed to convert {csv_file}: {e}")
    
    def _create_train_test_split(self):
        """Create train/test split using sklearn."""
        # Get all files and their labels
        all_files = self._get_all_csv_files()
        all_labels = [self._extract_class_from_path(f) for f in all_files]
        
        # Filter out unknown classes
        valid_indices = [i for i, label in enumerate(all_labels) if label != 'unknown']
        valid_files = [all_files[i] for i in valid_indices]
        valid_labels = [all_labels[i] for i in valid_indices]
        
        logger.info(f"Found {len(valid_files)} valid files")
        logger.info(f"Class distribution: {dict(zip(*np.unique(valid_labels, return_counts=True)))}")
        
        # Create stratified split
        train_files, test_files, train_labels, test_labels = train_test_split(
            valid_files, valid_labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=valid_labels
        )
        
        # Cache results
        self._train_files = train_files
        self._test_files = test_files
        self._train_labels = train_labels
        self._test_labels = test_labels
        
        logger.info(f"Train set: {len(train_files)} files")
        logger.info(f"Test set: {len(test_files)} files")
        
        # Log class distribution
        train_classes, train_counts = np.unique(train_labels, return_counts=True)
        test_classes, test_counts = np.unique(test_labels, return_counts=True)
        
        logger.info("Train class distribution:")
        for cls, count in zip(train_classes, train_counts):
            logger.info(f"  {cls}: {count}")
        
        logger.info("Test class distribution:")
        for cls, count in zip(test_classes, test_counts):
            logger.info(f"  {cls}: {count}")
    
    def get_train_files(self, **kwargs) -> List[str]:
        """Get training file paths."""
        if self._train_files is None:
            self._create_train_test_split()
        
        # Convert to WAV paths if needed
        if self.convert_to_wav and self.wav_dir:
            return [self._convert_to_wav_path(f) for f in self._train_files]
        else:
            return self._train_files
    
    def get_test_files(self, **kwargs) -> List[Tuple[str, Union[int, str]]]:
        """Get test file paths with labels."""
        if self._test_files is None:
            self._create_train_test_split()
        
        # Convert to WAV paths if needed
        if self.convert_to_wav and self.wav_dir:
            test_files = [self._convert_to_wav_path(f) for f in self._test_files]
        else:
            test_files = self._test_files
        
        # Convert string labels to integers
        label_to_int = {cls: i for i, cls in enumerate(self.get_classes())}
        test_labels = [label_to_int[label] for label in self._test_labels]
        
        return list(zip(test_files, test_labels))
    
    def get_classes(self) -> List[str]:
        """Get list of class names."""
        return ['normal', 'horizontal-misalignment', 'imbalance', 'vertical-misalignment', 
                'overhang-ball_fault', 'overhang-cage_fault', 'overhang-outer_race',
                'underhang-ball_fault', 'underhang-cage_fault', 'underhang-outer_race']
    
    def _convert_to_wav_path(self, csv_path: str) -> str:
        """Convert CSV path to WAV path."""
        if not self.wav_dir:
            return csv_path
        
        csv_path_obj = Path(csv_path)
        relative_path = csv_path_obj.relative_to(self.dataset_root)
        wav_path = Path(self.wav_dir) / relative_path.with_suffix('.wav')
        return str(wav_path)
    
    def get_dataset_stats(self) -> Dict:
        """Get detailed dataset statistics."""
        stats = super().get_dataset_stats()
        
        # Add MAFAULDA-specific stats
        if self._train_files is not None:
            train_classes, train_counts = np.unique(self._train_labels, return_counts=True)
            test_classes, test_counts = np.unique(self._test_labels, return_counts=True)
            
            stats.update({
                'train_class_distribution': dict(zip(train_classes, train_counts)),
                'test_class_distribution': dict(zip(test_classes, test_counts)),
                'test_size': self.test_size,
                'random_state': self.random_state,
                'convert_to_wav': self.convert_to_wav
            })
        
        return stats
    
    def get_sample_rate(self) -> int:
        """Get sample rate of the dataset."""
        return 50000

    def get_channel_names(self) -> List[str]:
        """Return human-readable names for the 8 channels.

        Channel mapping (according to official documentation):
        - Channel 0: Tachometer signal (rotation frequency)
        - Channels 1-3: Underhang bearing accelerometer (axial, radial, tangential)
        - Channels 4-6: Overhang bearing accelerometer (axial, radial, tangential)
        - Channel 7: Microphone (audio signal)
        """
        return [
            'tachometer',
            'underhang-axial',
            'underhang-radial',
            'underhang-tangential',
            'overhang-axial',
            'overhang-radial',
            'overhang-tangential',
            'microphone',
        ]

    def load_signal(self, file_path: str):
        """
        Load an 8-channel signal from a MAFAULDA CSV file.
        
        Channel mapping (according to official documentation):
        - Channel 0: Tachometer signal (rotation frequency)
        - Channels 1-3: Underhang bearing accelerometer (axial, radial, tangential)
        - Channels 4-6: Overhang bearing accelerometer (axial, radial, tangential)  
        - Channel 7: Microphone (audio signal)
        
        Returns:
            signal: torch.Tensor, shape [8, num_samples]
            sampling_rate: int, always 50000
        """
        import pandas as pd
        import torch
        data = pd.read_csv(file_path, header=None).values  # shape: [num_samples, 8]
        signal = data.T  # shape: [8, num_samples]
        sampling_rate = 50000
        return torch.tensor(signal, dtype=torch.float32), sampling_rate
