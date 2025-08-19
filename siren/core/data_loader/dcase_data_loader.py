"""
Data loader utilities for DCASE series datasets.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This module provides data loading utilities for DCASE series datasets,
supporting various file formats and evaluation configurations.
"""

import os
import glob
import csv
import re
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset evaluation."""
    name: str
    machine_types_getter: Callable
    test_files_getter: Callable
    train_files_getter: Callable
    cache_prefix: str
    results_suffix: str
    memory_bank_getter: Callable


class DCASEDataLoader:
    """Data loader for DCASE series datasets."""
    
    # File pattern constants
    NORMAL_PATTERN = "normal_"  # DCASE 2020 format
    ANOMALY_PATTERN = "anomaly_"  # DCASE 2020 format
    EVAL_CSV_PATTERN = "eval_data_list_*_converted.csv"
    
    # DCASE 2021-2025 format (same for all years)
    DCASE2021_2025_NORMAL_PATTERN = "*_normal_"
    DCASE2021_2025_ANOMALY_PATTERN = "*_anomaly_"
    
    def __init__(self, dataset_root: str):
        """
        Initialize DCASE data loader.
        
        Args:
            dataset_root: Root directory of DCASE dataset
        """
        self.dataset_root = Path(dataset_root)
        self.dev_dir = self.dataset_root / "development"
        self.eval_dir = self.dataset_root / "evaluation"
        
        self._initialize_dataset_configs()
    
    def get_dataset_config(self, dataset_type: str) -> DatasetConfig:
        """
        Get dataset configuration for a specific dataset type.
        
        Args:
            dataset_type: 'dev' or 'eval'
            
        Returns:
            DatasetConfig object
        """
        if dataset_type not in self.dataset_configs:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'dev' or 'eval'")
        return self.dataset_configs[dataset_type]
    
    def _initialize_dataset_configs(self):
        """Initialize dataset configurations for dev and eval sets."""
        self.dataset_configs = {
            'dev': DatasetConfig(
                name='dev',
                machine_types_getter=self.get_machine_types,
                test_files_getter=self.get_dev_test_files,
                train_files_getter=self.get_dev_train_files,
                cache_prefix='dev',
                results_suffix='dev',
                memory_bank_getter=None  # Will be set by tester
            ),
            'eval': DatasetConfig(
                name='eval',
                machine_types_getter=self.get_eval_machine_types,
                test_files_getter=self.get_eval_test_files,
                train_files_getter=self.get_eval_train_files,
                cache_prefix='eval',
                results_suffix='eval',
                memory_bank_getter=None  # Will be set by tester
            )
        }
    
    def get_machine_types(self) -> List[str]:
        """
        Get all machine types in the dataset.
        
        Returns:
            List of machine type names
        """
        if self.dev_dir.exists():
            machine_dirs = [d for d in self.dev_dir.iterdir() if d.is_dir()]
            return sorted([d.name for d in machine_dirs])
        return []
    
    def get_eval_machine_types(self) -> List[str]:
        """
        Get all machine types in the eval dataset.
        
        Returns:
            List of eval machine type names
        """
        if self.eval_dir.exists():
            machine_dirs = [d for d in self.eval_dir.iterdir() if d.is_dir()]
            return sorted([d.name for d in machine_dirs])
        return []
    
    def get_dev_train_files(self, machine_type: str) -> List[str]:
        """
        Get training files (normal only) for dev set.
        
        Args:
            machine_type: Machine type name
            
        Returns:
            List of training file paths
        """
        train_dir = self.dev_dir / machine_type / "train"
        if not train_dir.exists():
            return []
        
        # Try DCASE 2021-2025 format first
        pattern = str(train_dir / f"{self.DCASE2021_2025_NORMAL_PATTERN}*.wav")
        files = glob.glob(pattern)
        
        # If no files found, try standard format
        if not files:
            pattern = str(train_dir / f"{self.NORMAL_PATTERN}*.wav")
            files = glob.glob(pattern)
        
        return sorted(files)
    
    def get_dev_test_files(self, machine_type: str) -> List[Tuple[str, int]]:
        """
        Get test files for dev set with labels.
        
        Args:
            machine_type: Machine type name
            
        Returns:
            List of (file_path, label) tuples where label is 0 for normal, 1 for anomaly
        """
        files_with_labels = []
        
        # Check for DCASE 2021 format (source_test and target_test)
        source_test_dir = self.dev_dir / machine_type / "source_test"
        target_test_dir = self.dev_dir / machine_type / "target_test"
        
        if source_test_dir.exists() and target_test_dir.exists():
            # DCASE 2021 format: combine source_test and target_test
            print(f"  Found DCASE 2021 format for {machine_type}: source_test and target_test")
            
            # Process source_test
            source_normal_pattern = str(source_test_dir / f"{self.DCASE2021_2025_NORMAL_PATTERN}*.wav")
            source_normal_files = glob.glob(source_normal_pattern)
            files_with_labels.extend([(f, 0) for f in source_normal_files])
            
            source_anomaly_pattern = str(source_test_dir / f"{self.DCASE2021_2025_ANOMALY_PATTERN}*.wav")
            source_anomaly_files = glob.glob(source_anomaly_pattern)
            files_with_labels.extend([(f, 1) for f in source_anomaly_files])
            
            # Process target_test
            target_normal_pattern = str(target_test_dir / f"{self.DCASE2021_2025_NORMAL_PATTERN}*.wav")
            target_normal_files = glob.glob(target_normal_pattern)
            files_with_labels.extend([(f, 0) for f in target_normal_files])
            
            target_anomaly_pattern = str(target_test_dir / f"{self.DCASE2021_2025_ANOMALY_PATTERN}*.wav")
            target_anomaly_files = glob.glob(target_anomaly_pattern)
            files_with_labels.extend([(f, 1) for f in target_anomaly_files])
            
        else:
            # Standard format: single test directory
            test_dir = self.dev_dir / machine_type / "test"
            if not test_dir.exists():
                return []
            
            # Try DCASE 2021-2025 format first
            normal_pattern = str(test_dir / f"{self.DCASE2021_2025_NORMAL_PATTERN}*.wav")
            normal_files = glob.glob(normal_pattern)
            files_with_labels.extend([(f, 0) for f in normal_files])
            
            anomaly_pattern = str(test_dir / f"{self.DCASE2021_2025_ANOMALY_PATTERN}*.wav")
            anomaly_files = glob.glob(anomaly_pattern)
            files_with_labels.extend([(f, 1) for f in anomaly_files])
            
            # If no files found, try standard format
            if not files_with_labels:
                normal_pattern = str(test_dir / f"{self.NORMAL_PATTERN}*.wav")
                normal_files = glob.glob(normal_pattern)
                files_with_labels.extend([(f, 0) for f in normal_files])
                
                anomaly_pattern = str(test_dir / f"{self.ANOMALY_PATTERN}*.wav")
                anomaly_files = glob.glob(anomaly_pattern)
                files_with_labels.extend([(f, 1) for f in anomaly_files])
        
        return sorted(files_with_labels, key=lambda x: x[0])
    
    def get_eval_train_files(self, machine_type: str) -> List[str]:
        """
        Get training files (normal only) for eval set.
        
        Args:
            machine_type: Machine type name
            
        Returns:
            List of training file paths
        """
        # Try eval directory first
        train_dir = self.eval_dir / machine_type / "train"
        if not train_dir.exists():
            # Try additional_train directory
            additional_train_dir = self.dataset_root / "additional_train" / machine_type / "train"
            if additional_train_dir.exists():
                train_dir = additional_train_dir
            else:
                return []
        
        # Try DCASE 2021-2025 format first
        pattern = str(train_dir / f"{self.DCASE2021_2025_NORMAL_PATTERN}*.wav")
        files = glob.glob(pattern)
        
        # If no files found, try standard format
        if not files:
            pattern = str(train_dir / f"{self.NORMAL_PATTERN}*.wav")
            files = glob.glob(pattern)
        
        return sorted(files)
    
    def get_eval_test_files(self, machine_type: str) -> List[Tuple[str, int]]:
        """
        Get test files for eval set with labels from CSV mapping.
        
        Args:
            machine_type: Machine type name
            
        Returns:
            List of (file_path, label) tuples where label is 0 for normal, 1 for anomaly
        """
        # Check for source_test and target_test directories (DCASE 2021 format)
        source_test_dir = self.eval_dir / machine_type / "source_test"
        target_test_dir = self.eval_dir / machine_type / "target_test"
        
        if not source_test_dir.exists() and not target_test_dir.exists():
            # Try legacy test directory
            test_dir = self.eval_dir / machine_type / "test"
            if not test_dir.exists():
                return []
            test_dirs = [test_dir]
        else:
            test_dirs = []
            if source_test_dir.exists():
                test_dirs.append(source_test_dir)
            if target_test_dir.exists():
                test_dirs.append(target_test_dir)
        
        # Find eval CSV file
        eval_csv_pattern = str(self.eval_dir / self.EVAL_CSV_PATTERN)
        eval_csv_files = glob.glob(eval_csv_pattern)
        
        if not eval_csv_files:
            return []
        
        # Read label mapping from CSV
        label_mapping = self._read_eval_label_mapping(eval_csv_files[0], machine_type)
        
        # Debug: Print CSV file and label mapping info
        # print(f"DEBUG: Reading labels from {eval_csv_files[0]}")
        # print(f"DEBUG: Found {len(label_mapping)} labels for {machine_type}")
        
        # Get test files from all test directories and map to labels
        test_files = []
        for test_dir in test_dirs:
            pattern = str(test_dir / "*.wav")
            test_files.extend(glob.glob(pattern))
        
        # print(f"DEBUG: Found {len(test_files)} test files for {machine_type}")
        
        files_with_labels = []
        for file_path in test_files:
            filename = os.path.basename(file_path)
            if filename in label_mapping:
                label = label_mapping[filename]
                files_with_labels.append((file_path, label))
        
        # print(f"DEBUG: Mapped {len(files_with_labels)} files with labels for {machine_type}")
        
        return sorted(files_with_labels, key=lambda x: x[0])
    
    def _read_eval_label_mapping(self, csv_path: str, machine_type: str) -> Dict[str, int]:
        """
        Read label mapping for a specific machine type from the evaluation CSV file.

        Args:
            csv_path: Path to the evaluation CSV file.
            machine_type: The machine type to extract labels for.

        Returns:
            A dictionary mapping filename to label (0=normal, 1=anomaly).
        """
        label_mapping: Dict[str, int] = {}
        reference_mapping: Dict[str, str] = {}  # Store reference_filename mapping
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            in_target_section = False
            for row in reader:
                # Section header: machine type name as a single column
                if len(row) == 1:
                    if row[0] == machine_type:
                        in_target_section = True
                        continue
                    if in_target_section:
                        # Reached the next machine type section, stop reading
                        break
                    continue
                # Only process rows within the target machine type section
                if in_target_section and len(row) >= 3:
                    filename = row[0]
                    reference_filename = row[1]
                    try:
                        label = int(row[2])
                    except ValueError:
                        continue  # Skip rows with invalid label
                    label_mapping[filename] = label
                    reference_mapping[filename] = reference_filename
        
        # Store reference_mapping for later use (accumulate instead of overwrite)
        if not hasattr(self, '_reference_mapping'):
            self._reference_mapping = {}
        self._reference_mapping.update(reference_mapping)
        return label_mapping
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dictionary containing dataset statistics
        """
        machine_types = self.get_machine_types()
        dataset_info = {}
        
        for machine_type in machine_types:
            dev_train_count = len(self.get_dev_train_files(machine_type))
            dev_test_files = self.get_dev_test_files(machine_type)
            dev_test_normal = len([f for f, l in dev_test_files if l == 0])
            dev_test_anomaly = len([f for f, l in dev_test_files if l == 1])
            
            eval_train_count = len(self.get_eval_train_files(machine_type))
            eval_test_files = self.get_eval_test_files(machine_type)
            eval_test_normal = len([f for f, l in eval_test_files if l == 0])
            eval_test_anomaly = len([f for f, l in eval_test_files if l == 1])
            
            dataset_info[machine_type] = {
                'dev_train': dev_train_count,
                'dev_test_normal': dev_test_normal,
                'dev_test_anomaly': dev_test_anomaly,
                'eval_train': eval_train_count,
                'eval_test_normal': eval_test_normal,
                'eval_test_anomaly': eval_test_anomaly
            }
        
        return dataset_info
    
    def get_reference_filename(self, filename: str) -> str:
        """
        Get reference filename for a given filename.
        
        Args:
            filename: The filename to look up
            
        Returns:
            Reference filename or None if not found
        """
        return getattr(self, '_reference_mapping', {}).get(filename, None) 