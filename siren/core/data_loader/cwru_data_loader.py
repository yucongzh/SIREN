"""
Unified CWRU Dataset DataLoader

Author: Yucong Zhang
Email: yucong0428@outlook.com

Supports all sampling rates and channels (DE/FE) with dynamic sampling rate detection.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
import logging
import numpy as np
from collections import Counter

from .base_data_loader import BaseDataLoader, DatasetInfo

logger = logging.getLogger(__name__)


class CWRUDataLoader(BaseDataLoader):
    """
    Unified CWRU bearing dataset data loader.
    
    Supports all variants:
    - Normal data 
    - Drive End (DE) faults: 12kHz and 48kHz
    - Fan End (FE) faults: 12kHz
    - All fault types: IR (Inner Race), B (Ball), OC (Outer Race) with positions @3, @6, @12
    
    Class labels: normal, DE_IR, DE_B, DE_OC@3, DE_OC@6, DE_OC@12, FE_IR, FE_B, FE_OC@3, FE_OC@6, FE_OC@12
    """
    
    def __init__(self, dataset_root: str, config: Dict = None):
        """
        Initialize CWRU data loader.
        
        Args:
            dataset_root: Path to CWRU dataset root directory
            config: Configuration dictionary
        """
        super().__init__(dataset_root, config)
        
        # Sampling rate mapping based on directory names
        self.sampling_rate_mapping = {
            '48k_Drive_End_Bearing_Fault_Data': 48000,
            '12k_Drive_End_Bearing_Fault_Data': 12000,
            '12k_Fan_End_Bearing_Fault_Data': 12000,
            'Normal': 12000  # Normal data - treat as 12kHz to be consistent
        }
        
        # Channel mapping
        self.channel_mapping = {
            '48k_Drive_End_Bearing_Fault_Data': 'DE',
            '12k_Drive_End_Bearing_Fault_Data': 'DE', 
            '12k_Fan_End_Bearing_Fault_Data': 'FE',
            'Normal': 'Normal'
        }
        
        # Cache for file analysis
        self._all_files_cache = None
        self._files_with_metadata_cache = None
        
        logger.info(f"Initialized unified CWRU data loader for {dataset_root}")
    
    def _get_dataset_info(self) -> DatasetInfo:
        """Get dataset information."""
        return DatasetInfo(
            name='cwru',
            task_type='fault_classification',
            signal_type='vibration',
            file_format='mat',
            sampling_rate=12000,  # Default, actual rates are dynamic
            num_channels=2,  # DE and FE channels available
            description='CWRU Bearing Dataset - Unified loader for all sampling rates and channels'
        )
    
    def get_classes(self) -> List[str]:
        """
        Get the list of fault classes for unified CWRU dataset.
        
        Returns:
            List of class names: normal + DE_* + FE_*
        """
        return [
            # "normal",
            "DE_IR", "DE_B", "DE_OC@3", "DE_OC@6", "DE_OC@12",
            "FE_IR", "FE_B", "FE_OC@3", "FE_OC@6", "FE_OC@12"
        ]
    
    def _scan_all_files(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Scan all .mat files and extract metadata.
        Only include 12kHz data to avoid feature dimension inconsistency.
        
        Returns:
            List of (file_path, metadata) tuples
        """
        if self._files_with_metadata_cache is not None:
            return self._files_with_metadata_cache
        
        logger.info("Scanning CWRU dataset files (12kHz only)...")
        files_with_metadata = []
        
        # Scan all .mat files
        for root, dirs, files in os.walk(self.dataset_root):
            if "Normal" in root: continue
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    metadata = self._extract_metadata_from_path(file_path)
                    
                    # Only include 12kHz data to avoid feature dimension issues
                    if metadata['sampling_rate'] == 12000:
                        files_with_metadata.append((file_path, metadata))
        
        # Cache results
        self._files_with_metadata_cache = files_with_metadata
        
        logger.info(f"Found {len(files_with_metadata)} .mat files (12kHz only)")
        
        # Log class distribution
        class_counts = Counter([meta['class'] for _, meta in files_with_metadata])
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        return files_with_metadata
    
    def _extract_metadata_from_path(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from file path including sampling rate and class label.
        
        Args:
            file_path: Path to .mat file
            
        Returns:
            Metadata dictionary with class, sampling_rate, channel, etc.
        """
        path_parts = file_path.replace(str(self.dataset_root), '').split(os.sep)
        
        metadata = {
            'sampling_rate': 12000,  # Default
            'class': 'unknown',
            'channel': 'DE',
            'data_source': None,
            'fault_type': None,
            'fault_position': None
        }
        
        # Identify data source directory and extract sampling rate
        for part in path_parts:
            if part in self.sampling_rate_mapping:
                metadata['sampling_rate'] = self.sampling_rate_mapping[part]
                metadata['channel'] = self.channel_mapping[part]
                metadata['data_source'] = part
                break
        
        # Extract class label based on directory structure
        if 'Normal' in path_parts:
            metadata['class'] = 'normal'
            metadata['fault_type'] = 'normal'
        else:
            # Determine channel prefix
            if metadata['channel'] == 'DE':
                channel_prefix = 'DE_'
            elif metadata['channel'] == 'FE':
                channel_prefix = 'FE_'
            else:
                channel_prefix = ''
            
            # Identify fault type
            if '/B/' in file_path or '\\B\\' in file_path:
                metadata['fault_type'] = 'B'
                metadata['class'] = f"{channel_prefix}B"
            elif '/IR/' in file_path or '\\IR\\' in file_path:
                metadata['fault_type'] = 'IR'
                metadata['class'] = f"{channel_prefix}IR"
            elif '/OR/' in file_path or '\\OR\\' in file_path:
                metadata['fault_type'] = 'OC'
                # Extract position for outer race faults
                if '@3' in file_path:
                    metadata['fault_position'] = '@3'
                    metadata['class'] = f"{channel_prefix}OC@3"
                elif '@6' in file_path:
                    metadata['fault_position'] = '@6' 
                    metadata['class'] = f"{channel_prefix}OC@6"
                elif '@12' in file_path:
                    metadata['fault_position'] = '@12'
                    metadata['class'] = f"{channel_prefix}OC@12"
                else:
                    # Default to @6 if position unclear
                    metadata['fault_position'] = '@6'
                    metadata['class'] = f"{channel_prefix}OC@6"
        
        return metadata
    
    def _get_all_files(self) -> List[str]:
        """Get all .mat files in the dataset."""
        if self._all_files_cache is not None:
            return self._all_files_cache
        
        files_with_metadata = self._scan_all_files()
        all_files = [file_path for file_path, _ in files_with_metadata]
        
        # Cache results
        self._all_files_cache = all_files
        
        return all_files
    
    def get_files_with_sampling_rates(self) -> List[Tuple[str, int, str]]:
        """
        Get files with their corresponding sampling rates and class labels.
        This is the key method for dynamic sampling rate support.
        
        Returns:
            List of (file_path, sampling_rate, class_label) tuples
        """
        files_with_metadata = self._scan_all_files()
        
        files_with_rates = []
        for file_path, metadata in files_with_metadata:
            if metadata['class'] != 'unknown':  # Skip unknown classes
                files_with_rates.append((
                    file_path,
                    metadata['sampling_rate'],
                    metadata['class']
                ))
        
        return files_with_rates
    
    def _extract_class_from_path(self, file_path: str) -> str:
        """
        Extract class label from file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Class label string
        """
        metadata = self._extract_metadata_from_path(file_path)
        return metadata['class']
    
    def get_train_files(self, **kwargs) -> List[str]:
        """
        Get training files.
        
        For unified CWRU loader, we use a simple split strategy.
        
        Returns:
            List of training file paths
        """
        all_files = self._get_all_files()
        
        # Simple split: use 80% for training
        # Group by class to ensure balanced split
        files_with_metadata = self._scan_all_files()
        class_files = {}
        
        for file_path, metadata in files_with_metadata:
            class_label = metadata['class']
            if class_label != 'unknown':
                if class_label not in class_files:
                    class_files[class_label] = []
                class_files[class_label].append(file_path)
        
        train_files = []
        for class_label, files in class_files.items():
            # Sort for reproducibility
            files = sorted(files)
            # Take first 80% for training
            train_count = max(1, int(len(files) * 0.8))
            train_files.extend(files[:train_count])
        
        return sorted(train_files)
    
    def get_test_files(self, **kwargs) -> List[Tuple[str, Union[int, str]]]:
        """
        Get test files with labels.
        
        Returns:
            List of (file_path, class_index) tuples
        """
        all_files = self._get_all_files()
        train_files = set(self.get_train_files())
        
        # Test files are those not in training set
        test_files = [f for f in all_files if f not in train_files]
        
        # Get class labels and convert to indices
        classes = self.get_classes()
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        test_files_with_labels = []
        for file_path in test_files:
            class_label = self._extract_class_from_path(file_path)
            if class_label in class_to_idx:
                class_idx = class_to_idx[class_label]
                test_files_with_labels.append((file_path, class_idx))
        
        return test_files_with_labels
    
    def get_sample_rate(self) -> int:
        """
        Get default sample rate.
        Note: Actual sample rates are dynamic, this is just a fallback.
        """
        return 12000  # Default, but actual rates are provided via get_files_with_sampling_rates()
    
    def get_dataset_stats(self) -> Dict:
        """Get comprehensive dataset statistics."""
        files_with_metadata = self._scan_all_files()
        
        # Basic stats
        total_files = len(files_with_metadata)
        class_distribution = Counter([meta['class'] for _, meta in files_with_metadata])
        sampling_rate_distribution = Counter([meta['sampling_rate'] for _, meta in files_with_metadata])
        channel_distribution = Counter([meta['channel'] for _, meta in files_with_metadata])
        
        # Train/test split stats
        train_files = self.get_train_files()
        test_files = self.get_test_files()
        
        train_class_dist = Counter()
        test_class_dist = Counter()
        
        train_files_set = set(train_files)
        for file_path, metadata in files_with_metadata:
            class_label = metadata['class']
            if class_label != 'unknown':
                if file_path in train_files_set:
                    train_class_dist[class_label] += 1
                else:
                    test_class_dist[class_label] += 1
        
        stats = {
            'name': 'cwru',
            'task_type': 'fault_classification',
            'signal_type': 'vibration', 
            'file_format': 'mat',
            'total_files': total_files,
            'num_train_files': len(train_files),
            'num_test_files': len(test_files),
            'classes': self.get_classes(),
            'sampling_rates': list(sampling_rate_distribution.keys()),
            'channels': list(channel_distribution.keys()),
            'class_distribution': dict(class_distribution),
            'sampling_rate_distribution': dict(sampling_rate_distribution),
            'channel_distribution': dict(channel_distribution),
            'train_class_distribution': dict(train_class_dist),
            'test_class_distribution': dict(test_class_dist),
            'test_size': len(test_files) / total_files if total_files > 0 else 0,
            'random_state': 1,
            'description': 'CWRU Bearing Dataset - Unified loader supporting all sampling rates and channels'
        }
        
        return stats