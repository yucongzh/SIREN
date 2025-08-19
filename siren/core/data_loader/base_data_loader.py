"""
Base data loader interface for machine signal datasets.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This module provides the base interface for data loaders in the SIREN framework,
supporting various machine signal datasets with unified loading and preprocessing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    task_type: str  # 'fault_classification', 'lifetime_prediction', 'anomaly_detection'
    signal_type: str  # 'audio', 'vibration'
    file_format: str  # 'wav', 'mat', 'csv'
    sampling_rate: Optional[int] = None
    num_channels: Optional[int] = None
    description: Optional[str] = None


class BaseDataLoader(ABC):
    """
    Base class for machine signal data loaders.
    
    This class provides a unified interface for loading different types of machine signal datasets,
    including audio signals (WAV) and vibration signals (MAT, CSV).
    """
    
    def __init__(self, dataset_root: str, config: Optional[Dict] = None):
        """
        Initialize the data loader.
        
        Args:
            dataset_root: Root directory of the dataset
            config: Optional configuration dictionary
        """
        self.dataset_root = Path(dataset_root)
        self.config = config or {}
        self.dataset_info = self._get_dataset_info()
        
        # Validate dataset exists
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        
        logger.info(f"Initialized {self.__class__.__name__} for {self.dataset_info.name}")
    
    @abstractmethod
    def _get_dataset_info(self) -> DatasetInfo:
        """
        Get dataset information. Including:
        - name: Name of the dataset
        - task_type: Task type of the dataset
        - signal_type: Signal type of the dataset
        - file_format: File format of the dataset
        - sampling_rate: Sampling rate of the dataset
        - description: Description of the dataset
        
        Returns:
            DatasetInfo object containing dataset metadata
        """
        pass
    
    @abstractmethod
    def get_train_files(self, **kwargs) -> List[str]:
        """
        Get training file paths.
        
        Args:
            **kwargs: Additional arguments specific to the dataset
            
        Returns:
            List of training file paths
        """
        pass
    
    @abstractmethod
    def get_test_files(self, **kwargs) -> List[Tuple[str, Union[int, str]]]:
        """
        Get test file paths with labels.
        
        Args:
            **kwargs: Additional arguments specific to the dataset
            
        Returns:
            List of (file_path, label) tuples
        """
        pass
    
    @abstractmethod
    def get_classes(self) -> List[str]:
        """
        Get list of class names.
        
        Returns:
            List of class names
        """
        pass

    def get_sample_rate(self) -> int:
        """
        Get sample rate of the dataset.
        
        Returns:
            Sample rate of the dataset
        """
        return self.dataset_info.sampling_rate
    
    def get_files_with_sampling_rates(self) -> List[Tuple[str, int, str]]:
        """
        Get files with their corresponding sampling rates and class labels.
        This method provides dynamic sampling rate support for datasets with mixed sampling rates.
        
        Returns:
            List of (file_path, sampling_rate, class_label) tuples
            
        Note:
            Default implementation falls back to fixed sampling rate behavior.
            Subclasses can override this for dynamic sampling rate support.
        """
        # Default implementation: use fixed sampling rate for all files
        train_files = self.get_train_files()
        test_files = self.get_test_files()
        
        # Handle test files which might be tuples (file, label)
        if test_files and isinstance(test_files[0], tuple):
            test_files = [item[0] for item in test_files]
        
        all_files = train_files + test_files
        fixed_sample_rate = self.get_sample_rate()
        
        files_with_rates = []
        for file_path in all_files:
            class_label = self._extract_class_from_path(file_path)
            files_with_rates.append((file_path, fixed_sample_rate, class_label))
            
        return files_with_rates
    
    def get_dataset_stats(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        train_files = self.get_train_files()
        test_files = self.get_test_files()
        
        stats = {
            'name': self.dataset_info.name,
            'task_type': self.dataset_info.task_type,
            'signal_type': self.dataset_info.signal_type,
            'file_format': self.dataset_info.file_format,
            'num_train_files': len(train_files),
            'num_test_files': len(test_files),
            'classes': self.get_classes(),
            'sampling_rate': self.dataset_info.sampling_rate,
            'num_channels': self.dataset_info.num_channels,
            'description': self.dataset_info.description
        }
        
        return stats
    
    def validate_file_format(self, file_path: str) -> bool:
        """
        Validate if a file has the expected format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file format is valid, False otherwise
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False
        
        expected_extension = self.dataset_info.file_format.lower()
        actual_extension = file_path.suffix.lower()
        
        if expected_extension == 'wav':
            return actual_extension == '.wav'
        elif expected_extension == 'mat':
            return actual_extension == '.mat'
        elif expected_extension == 'csv':
            return actual_extension == '.csv'
        else:
            return True  # Unknown format, assume valid
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return [f'.{self.dataset_info.file_format}']
    
    def get_task_type(self) -> str:
        """
        Get the task type of this dataset.
        
        Returns:
            Task type string
        """
        return self.dataset_info.task_type
    
    def get_signal_type(self) -> str:
        """
        Get the signal type of this dataset.
        
        Returns:
            Signal type string ('audio' or 'vibration')
        """
        return self.dataset_info.signal_type
    
    def is_audio_dataset(self) -> bool:
        """
        Check if this is an audio dataset.
        
        Returns:
            True if audio dataset, False otherwise
        """
        return self.dataset_info.signal_type == 'audio'
    
    def is_vibration_dataset(self) -> bool:
        """
        Check if this is a vibration dataset.
        
        Returns:
            True if vibration dataset, False otherwise
        """
        return self.dataset_info.signal_type == 'vibration'
    
    def requires_conversion(self) -> bool:
        """
        Check if this dataset requires format conversion.
        
        Returns:
            True if conversion is needed, False otherwise
        """
        return self.dataset_info.file_format in ['mat', 'csv']
    
    def __str__(self) -> str:
        """String representation of the data loader."""
        stats = self.get_dataset_stats()
        return (f"{self.__class__.__name__}({stats['name']}, "
                f"task={stats['task_type']}, signal={stats['signal_type']}, "
                f"format={stats['file_format']})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__() 