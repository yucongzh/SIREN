"""
Main DCASE testing class with support for custom feature extractors.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This module provides the main DCASE evaluation functionality with support for
custom feature extractors, multiprocessing, and comprehensive result analysis.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, Manager
import pickle
from functools import partial
import logging
import time
import yaml
import importlib.util
import re

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Configure logging for cleaner output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

from .data_loader import DCASEDataLoader, DatasetConfig
from .memory_bank import DCASEMemoryBank as MemoryBank
from .evaluator import DCASEEvaluator
from .base_extractor import BaseFeatureExtractor
from .memory_bank.dcase_memory_bank import MemoryBank as DCASEMemoryBank
from .memory_bank.classification_memory_bank import ClassificationMemoryBank

# Constants for multiprocessing configuration
DEFAULT_CACHE_DIR = 'feature_cache'
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MULTIPROCESSING_ENABLED = True
DEFAULT_CACHE_ENABLED = False

def load_extractor(extractor_path: str):
    spec = importlib.util.spec_from_file_location('feature_extractor', extractor_path)
    feature_extractor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature_extractor_module)
    extractor = feature_extractor_module.FeatureExtractor()
    return extractor

def _extract_features_for_machine_type_worker(args_tuple):
    """
    Worker function for multiprocessing feature extraction.
    
    Args:
        args_tuple: Tuple containing (machine_type, dataset_type, train_files, device, config)
        
    Returns:
        dict: Dictionary containing machine_type, memory_bank, and cache_file_path
    """
    machine_type, dataset_type, train_files, device, config, extractor_path = args_tuple
    
    try:
        # Import and initialize custom extractor in this process
        custom_extractor = load_extractor(extractor_path)
        if not isinstance(custom_extractor, BaseFeatureExtractor):
            raise ValueError("custom_extractor must be an instance of BaseFeatureExtractor")
        
        # Create memory bank for this machine type
        dataset_year = config.get('dataset_year', 2020)
        memory_bank = MemoryBank(dataset_year, machine_type)
        
        # Extract features using custom extractor
        features = custom_extractor.extract_features_batch(train_files, sample_rate=config.get('sample_rate', 16000))
        # Standardize features to 1D vectors (average across channels if 2D)
        std_features = []
        for f in features:
            t = f if isinstance(f, torch.Tensor) else torch.tensor(f)
            if t.dim() == 2:
                t = t.mean(dim=0)
            elif t.dim() != 1:
                raise ValueError(f"Unsupported feature shape in DCASE worker: {t.shape}")
            std_features.append(t)
        features_tensor = torch.stack(std_features) # (num_files, feature_dim)
        
        # Group features by domain/section
        domain_groups = {}
        for i, file_path in enumerate(train_files):
            filename = os.path.basename(file_path)
            
            # Extract domain/section based on dataset year
            if dataset_year == 2020:
                # DCASE 2020: machine ID as domain (id_00, id_01, etc.)
                import re
                match = re.search(r'id_(\d+)', filename)
                if match:
                    domain_key = f"id_{match.group(1)}"
                else:
                    domain_key = 'default'  # Use 'default' instead of 'unknown'
            elif dataset_year in [2021, 2022, 2023, 2024, 2025]:
                # DCASE 2021-2025: section as domain (section_00, section_01, etc.)
                import re
                match = re.search(r'section_(\d+)', filename)
                if match:
                    domain_key = f"section_{match.group(1)}"
                else:
                    domain_key = 'default'  # Use 'default' instead of 'unknown'
            else:
                domain_key = 'default'  # Use 'default' instead of 'unknown'
            
            if domain_key not in domain_groups:
                domain_groups[domain_key] = []
            domain_groups[domain_key].append(i)
        
        # Add features to memory bank by domain/section
        for domain_key, indices in domain_groups.items():
            domain_features = features_tensor[indices]
            
            if dataset_year == 2020:
                # DCASE 2020: Keep original structure
                memory_bank.add_features(domain_key, domain_features)
            else:
                # DCASE 2021-2025: Use new unified structure
                # Determine if this is source or target based on filename
                source_indices = []
                target_indices = []
                
                for idx in indices:
                    filename = os.path.basename(train_files[idx])
                    if '_source_' in filename:
                        source_indices.append(idx)
                    elif '_target_' in filename:
                        target_indices.append(idx)
                
                # Add source features
                if source_indices:
                    source_features = features_tensor[source_indices]
                    memory_bank.add_features(domain_key, source_features, 'source')
                    # print(f"DEBUG: Section {domain_key} - Source samples: {len(source_indices)}")
                
                # Add target features
                if target_indices:
                    target_features = features_tensor[target_indices]
                    memory_bank.add_features(domain_key, target_features, 'target')
                    # print(f"DEBUG: Section {domain_key} - Target samples: {len(target_indices)}")
                
                # Debug: Print memory bank structure
                # if domain_key != 'default':
                #     print(f"DEBUG: Memory bank for {domain_key}: {memory_bank.memory_bank.get(domain_key, 'Not found')}")
        
        # Create cache file path
        cache_file_path = os.path.join(
            config.get('feature_cache', {}).get('cache_dir', DEFAULT_CACHE_DIR),
            f"{dataset_type}_train_{machine_type}.pth"
        )
        
        return {
            'machine_type': machine_type,
            'memory_bank': memory_bank,
            'cache_file_path': cache_file_path
        }
        
    except Exception as e:
        print(f"Error in worker for {machine_type}: {e}")
        return {
            'machine_type': machine_type,
            'memory_bank': None,
            'cache_file_path': None,
            'error': str(e)
        }


def _evaluate_machine_type_worker(args_tuple):
    """
    Worker function for evaluating a single machine type.
    
    Args:
        args_tuple: Tuple containing (machine_type, test_files_with_labels, dataset_config, 
                    device, config, cache_dir, cache_enabled, extractor_path)
        
    Returns:
        Tuple: (machine_type, results_dict_entry, scores_with_filenames)
    """
    (machine_type, test_files_with_labels, dataset_config, 
     device, config, cache_dir, cache_enabled, extractor_path) = args_tuple
    
    try:
        print(f"\nEvaluating {dataset_config.name} {machine_type}...")
        
        # Import and initialize custom extractor in this process
        custom_extractor = load_extractor(extractor_path)
        if not isinstance(custom_extractor, BaseFeatureExtractor):
            raise ValueError("custom_extractor must be an instance of BaseFeatureExtractor")
        
        # Get test files and labels
        test_files, labels = zip(*test_files_with_labels)
        labels = np.array(labels)
        
        # Check if test features are cached
        test_cache_file = os.path.join(cache_dir, f"{dataset_config.cache_prefix}_{machine_type}_test.pth")
        if cache_enabled and os.path.exists(test_cache_file):
            print(f"  ✓ Loading cached test features for {machine_type}...")
            test_features_tensor = torch.load(test_cache_file)
        else:
            print(f"  ⚠ Extracting test features for {machine_type}...")
            # Extract features using custom extractor
            test_features = custom_extractor.extract_features_batch(test_files)
            # Standardize to 1D vectors (average across channels if needed)
            std_features = []
            for f in test_features:
                t = f if isinstance(f, torch.Tensor) else torch.tensor(f)
                if t.dim() == 2:
                    t = t.mean(dim=0)
                elif t.dim() != 1:
                    raise ValueError(f"Unsupported feature shape in DCASE eval worker: {t.shape}")
                std_features.append(t)
            test_features_tensor = torch.stack(std_features)
            
            # Save test features to cache
            if cache_enabled and not os.path.exists(test_cache_file):
                print(f"  ✓ Saving test features for {machine_type} to cache...")
                torch.save(test_features_tensor, test_cache_file)
            elif cache_enabled and os.path.exists(test_cache_file):
                print(f"  ✓ Test cache for {machine_type} already exists, skipping save")
        
        # Prepare results
        results_dict_entry = {
            'y_true': labels,
            'test_features_tensor': test_features_tensor,
            'file_paths': test_files
        }
        
        return machine_type, results_dict_entry, None
        
    except Exception as e:
        print(f"Error evaluating {machine_type}: {e}")
        return machine_type, None, None


from .base_tester import BaseTester

class DCASETester(BaseTester):
    """Main DCASE testing class with support for custom feature extractors."""
    
    def __init__(self, dataset_root: str, year: Union[int, str], 
                 extractor_path: str,
                 results_dir: str = "results",
                 device: str = None,
                 multiprocessing_enabled: bool = True,
                 cache_enabled: bool = False,
                 cache_dir: str = DEFAULT_CACHE_DIR):
        # Initialize base class
        config = {
            'dataset_root': dataset_root,
            'year': year,
            'extractor_path': extractor_path,
            'results_dir': results_dir,
            'device': device,
            'multiprocessing_enabled': multiprocessing_enabled,
            'cache_enabled': cache_enabled,
            'cache_dir': cache_dir
        }
        super().__init__(config)
        """
        Initialize DCASE tester.
        
        Args:
            dataset_root: Root directory containing DCASE datasets
            year: Dataset year (2020-2025) or "all" for all years
            custom_extractor: User-defined feature extractor
            results_dir: Directory to save results
            device: Device to use ('cuda' or 'cpu')
            multiprocessing_enabled: Whether to use multiprocessing
            cache_enabled: Whether to cache features
            cache_dir: Directory for feature cache
        """
        self.dataset_root = Path(dataset_root)
        self.year = year
        self.extractor_path = extractor_path
        self.results_dir = Path(results_dir)
        self.device = device or DEFAULT_DEVICE
        self.multiprocessing_enabled = multiprocessing_enabled
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        
        # Validate custom extractor
        if not os.path.exists(self.extractor_path):
            raise ValueError(f"Extractor path {self.extractor_path} does not exist")
        
        # Initialize components
        self.evaluator = DCASEEvaluator()
        
        # Results storage
        self.dev_results = {}
        self.eval_results = {}
        self.final_scores = {}
        
        # Check datasets
        self._check_datasets()
        
        # Determine years to evaluate
        self.years_to_evaluate = self._get_years_to_evaluate()
        
        logging.info(f"Initialized DCASETester for years: {self.years_to_evaluate}")
    
    def _check_datasets(self):
        """Check if required datasets exist."""
        if self.year == "all":
            # Check all years
            for year in range(2020, 2026):
                dataset_path = self.dataset_root / f"dcase{year}_t2"
                if not dataset_path.exists():
                    logging.warning(f"Dataset for year {year} not found at {dataset_path}")
        else:
            # Check specific year
            dataset_path = self.dataset_root / f"dcase{self.year}_t2"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset for year {self.year} not found at {dataset_path}")
    
    def _get_years_to_evaluate(self) -> List[int]:
        """Get list of years to evaluate."""
        if self.year == "all":
            # Find all available years
            years = []
            for year in range(2020, 2026):
                dataset_path = self.dataset_root / f"dcase{year}_t2"
                if dataset_path.exists():
                    years.append(year)
            return years
        else:
            return [self.year]
    
    def _detect_dataset_year(self, dataset_path: Path) -> int:
        """Detect dataset year from path."""
        dataset_name = dataset_path.name
        if 'dcase2020_t2' in dataset_name:
            return 2020
        elif 'dcase2021_t2' in dataset_name:
            return 2021
        elif 'dcase2022_t2' in dataset_name:
            return 2022
        elif 'dcase2023_t2' in dataset_name:
            return 2023
        elif 'dcase2024_t2' in dataset_name:
            return 2024
        elif 'dcase2025_t2' in dataset_name:
            return 2025
        else:
            raise ValueError(f"Cannot detect year from dataset path: {dataset_path}")
    
    def _extract_domain_or_section(self, file_path: str, dataset_year: int) -> str:
        """
        Extract domain or section information from file path.
        
        Args:
            file_path: Path to the file
            dataset_year: Dataset year
            
        Returns:
            Domain or section identifier
        """
        import re
        filename = os.path.basename(file_path)
        
        if dataset_year == 2020:
            # DCASE 2020: machine ID as domain (id_00, id_01, etc.)
            match = re.search(r'id_(\d+)', filename)
            if match:
                return f"id_{match.group(1)}"
        
        elif dataset_year in [2021, 2022]:
            # DCASE 2021-2022: section as domain (section_00, section_01, etc.)
            match = re.search(r'section_(\d+)', filename)
            if match:
                return f"section_{match.group(1)}"
        
        elif dataset_year in [2023, 2024, 2025]:
            # DCASE 2023-2025: source/target as domain
            if '_source_' in filename:
                return 'source'
            elif '_target_' in filename:
                return 'target'
        
        # Default fallback
        return 'unknown'
    
    def _create_config_for_year(self, year: int) -> Dict:
        """Create configuration for a specific year."""
        dataset_path = self.dataset_root / f"dcase{year}_t2"
        
        # Base memory bank configuration
        memory_bank_config = {
            'similarity_type': 'cosine',
            'aggregation': 'max',
            'k': 1
        }
        
        # Add new parameters for DCASE 2021-2025
        if year >= 2021:
            memory_bank_config.update({
                'domain_strategy': 'min',
                'enable_kmeans_optimization': False,
                'kmeans_threshold': 100,
                'kmeans_n_clusters': 16
            })
        
        return {
            'dataset_root': str(dataset_path),
            'dataset_year': year,
            'results_dir': str(self.results_dir / f"dcase{year}_t2"),
            'device': self.device,
            'multiprocessing': {
                'enabled': self.multiprocessing_enabled,
                'num_processes': mp.cpu_count()
            },
            'feature_cache': {
                'enabled': self.cache_enabled,
                'cache_dir': str(self.cache_dir / f"dcase{year}_t2")
            },
            'memory_bank': memory_bank_config,
            'evaluation': {
                'max_fpr': 0.1,
                'save_detailed_results': True,
                'save_summary': True
            }
        }
    
    def evaluate_year(self, year: int):
        """Evaluate a specific year."""
        logging.info(f"Starting evaluation for DCASE {year}")
        
        # Create config for this year
        config = self._create_config_for_year(year)
        
        # Initialize data loader
        data_loader = DCASEDataLoader(config['dataset_root'])
        
        # Memory banks will be created per machine type
        memory_banks = {
            "dev": {},
            "eval": {}
        }
        
        # Build memory banks
        self._build_memory_banks_for_year(year, config, data_loader, memory_banks)
        
        # Evaluate dev set
        self._evaluate_dataset_for_year(year, config, data_loader, memory_banks, 'dev')
        
        # Evaluate eval set
        self._evaluate_dataset_for_year(year, config, data_loader, memory_banks, 'eval')

        # Evaluate existing scores
        self._evaluate_existing_scores(Path(config['results_dir']).parent, year)
        
        logging.info(f"Completed evaluation for DCASE {year}")
    
    def _evaluate_existing_scores(self, scores_root_dir: str, year: int = None):
        """
        Evaluate existing score files using the official DCASE evaluation method.
        
        Args:
            scores_root_dir: Root directory containing score files (e.g., 'test_results/fisher_extractor')
            year: Specific year to evaluate (e.g., 2023). If None, will auto-detect.
        """
        # Auto-detect year if not specified
        if year is None:
            year = self._detect_dcase_year(scores_root_dir)
        
        logging.info(f"Starting evaluation of existing scores from: {scores_root_dir}")
        logging.info(f"Detected/Using DCASE year: {year}")
        
        # 1. Load score files (dev and eval separately)
        dev_scores, eval_scores = self._load_score_files(scores_root_dir, year)
        
        # 2. Build evaluation mapping (dev and eval separately)
        dev_mapping = self._build_evaluation_mapping(dev_scores, 'dev', year)
        eval_mapping = self._build_evaluation_mapping(eval_scores, 'eval', year)
        
        # 3. Group and evaluate (dev and eval separately)
        dev_grouped = self._group_and_evaluate(dev_mapping, year)
        eval_grouped = self._group_and_evaluate(eval_mapping, year)
        
        # 4. Aggregate final scores (dev and eval separately produce overall)
        dev_overall = self._aggregate_final_scores(dev_grouped, year)
        eval_overall = self._aggregate_final_scores(eval_grouped, year)
        
        # Calculate overall_year score (combines dev and eval)
        overall_year = self._calculate_overall_year_score(dev_grouped, eval_grouped, year)
        
        # 5. Combine results
        results = {
            'year': year,
            'dev': {
                'mapping_count': len(dev_mapping),
                'grouped': dev_grouped,
                'overall': dev_overall
            },
            'eval': {
                'mapping_count': len(eval_mapping),
                'grouped': eval_grouped,
                'overall': eval_overall
            },
            'summary': {
                'dev_overall_score': dev_overall['overall_score'],
                'eval_overall_score': eval_overall['overall_score'],
                'overall_year_score': overall_year['overall_year_score'],
                'aggregation_method': dev_overall['aggregation_method'],
                'overall_year_details': overall_year
            }
        }
        
        logging.info(f"✅ Evaluation completed for DCASE {year}")
        logging.info(f"   Dev overall score: {dev_overall['overall_score']:.6f}")
        logging.info(f"   Eval overall score: {eval_overall['overall_score']:.6f}")
        logging.info(f"   Overall year score: {overall_year['overall_year_score']:.6f}")
        logging.info(f"   Aggregation method: {dev_overall['aggregation_method']}")
        
        # Save results to CSV (TODO: implement)
        self._save_evaluation_results_to_csv(results, scores_root_dir, year)
        
        return results
    
    def _detect_dcase_year(self, scores_root_dir: str) -> int:
        """
        Auto-detect DCASE year from directory structure.
        
        Args:
            scores_root_dir: Root directory containing score files
            
        Returns:
            Detected DCASE year (2020-2025)
            
        Raises:
            ValueError: If no valid DCASE year directory found
        """
        import re
        import os
        
        # Look for directories like dcase2020_t2, dcase2021_t2, etc.
        found_years = []
        
        if not os.path.exists(scores_root_dir):
            raise ValueError(f"Scores root directory not found: {scores_root_dir}")
        
        for item in os.listdir(scores_root_dir):
            item_path = os.path.join(scores_root_dir, item)
            if os.path.isdir(item_path):
                match = re.match(r'dcase(\d{4})_t2', item)
                if match:
                    year = int(match.group(1))
                    if 2020 <= year <= 2025:  # Valid DCASE years
                        found_years.append(year)
        
        if not found_years:
            raise ValueError(f"No valid DCASE year directories found in: {scores_root_dir}")
        
        if len(found_years) > 1:
            # Multiple years found, use the latest one
            latest_year = max(found_years)
            logging.warning(f"Multiple DCASE years found {found_years}, using latest: {latest_year}")
            return latest_year
        
        detected_year = found_years[0]
        logging.info(f"Auto-detected DCASE year: {detected_year}")
        return detected_year
    
    def _load_score_files(self, scores_root_dir: str, year: int) -> tuple:
        """
        Load dev and eval score files for a specific year.
        
        Args:
            scores_root_dir: Root directory containing score files
            year: DCASE year
            
        Returns:
            Tuple of (dev_scores, eval_scores) where each is a dict:
            {machine_type: {filename: score}}
        """
        import os
        
        year_dir = os.path.join(scores_root_dir, f"dcase{year}_t2")
        if not os.path.exists(year_dir):
            raise ValueError(f"Year directory not found: {year_dir}")
        
        dev_dir = os.path.join(year_dir, "dev")
        eval_dir = os.path.join(year_dir, "eval")
        
        dev_scores = {}
        eval_scores = {}
        
        # Load dev scores
        if os.path.exists(dev_dir):
            for filename in os.listdir(dev_dir):
                if filename.startswith("score_") and filename.endswith("_dev.csv"):
                    # Extract machine type from filename: score_ToyCar_dev.csv -> ToyCar
                    machine_type = filename.replace("score_", "").replace("_dev.csv", "")
                    score_file = os.path.join(dev_dir, filename)
                    
                    machine_scores = {}
                    with open(score_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split(',')
                                if len(parts) >= 2:
                                    audio_filename = parts[0]
                                    try:
                                        score = float(parts[1])
                                        machine_scores[audio_filename] = score
                                    except ValueError:
                                        logging.warning(f"Invalid score in {score_file}: {line}")
                                        continue
                    
                    if machine_scores:
                        dev_scores[machine_type] = machine_scores
                        logging.info(f"Loaded {len(machine_scores)} dev scores for {machine_type}")
        
                # Load eval scores - load ALL eval machine types, not just those in dev
        if os.path.exists(eval_dir):
            for filename in os.listdir(eval_dir):
                if filename.startswith("score_") and filename.endswith("_eval.csv"):
                    # Extract machine type from filename: score_ToyCar_eval.csv -> ToyCar
                    machine_type = filename.replace("score_", "").replace("_eval.csv", "")
                    score_file = os.path.join(eval_dir, filename)
                    
                    machine_scores = {}
                    with open(score_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split(',')
                                if len(parts) >= 2:
                                    audio_filename = parts[0]
                                    try:
                                        score = float(parts[1])
                                        machine_scores[audio_filename] = score
                                    except ValueError:
                                        logging.warning(f"Invalid score in {score_file}: {line}")
                                        continue
                    
                    if machine_scores:
                        eval_scores[machine_type] = machine_scores
                        logging.info(f"Loaded {len(machine_scores)} eval scores for {machine_type}")
        
        logging.info(f"Loaded scores for {len(dev_scores)} dev machine types and {len(eval_scores)} eval machine types")
        return dev_scores, eval_scores
    
    def _build_evaluation_mapping(self, scores: dict, dataset_type: str, year: int) -> list:
        """
        Build evaluation mapping from test set files, reversely inferring section and domain information.
        
        Args:
            scores: Dict of {machine_type: {filename: score}}
            dataset_type: 'dev' or 'eval'
            year: DCASE year
            
        Returns:
            List of mappings:
            For DCASE2020: [filename, y_score, y_true, machine_type, id]
            For DCASE2021-2025: [filename, y_score, y_true, section_id, domain, machine_type]
        """
        import re
        from siren.core.data_loader.dcase_data_loader import DCASEDataLoader
        
        # Create config and data loader for this year
        config = self._create_config_for_year(year)
        data_loader = DCASEDataLoader(config['dataset_root'])
        
        mapping = []
        
        for machine_type, machine_scores in scores.items():
            logging.info(f"Building mapping for {machine_type} {dataset_type} set ({len(machine_scores)} files)")
            
            # Get test files with labels from data loader
            if dataset_type == 'dev':
                test_files = data_loader.get_dev_test_files(machine_type)
            else:  # eval
                test_files = data_loader.get_eval_test_files(machine_type)
            
            if not test_files:
                logging.warning(f"No test files found for {machine_type} {dataset_type}")
                continue
            
            # Create lookup dict for quick access
            test_files_dict = {}
            for file_path, label in test_files:
                filename = os.path.basename(file_path)
                test_files_dict[filename] = {'label': label, 'path': file_path}
            
            # Process each scored file
            for filename, score in machine_scores.items():
                if filename not in test_files_dict:
                    logging.warning(f"Score file {filename} not found in test files for {machine_type}")
                    continue
                
                y_score = score
                y_true = test_files_dict[filename]['label']
                
                # Extract metadata based on year
                if year == 2020:
                    # DCASE2020 format: id_XX_YYYYYYYY.wav or anomaly_id_XX_YYYYYYYY.wav
                    id_match = re.search(r'id_(\d+)', filename)
                    if id_match:
                        id_num = f"id_{id_match.group(1)}"
                        mapping.append([filename, y_score, y_true, machine_type, id_num])
                    else:
                        logging.warning(f"Could not extract ID from DCASE2020 filename: {filename}")
                        
                else:
                    # DCASE2021-2025 format: section_XX_source/target_test_...
                    section_match = re.search(r'section_(\d+)', filename)
                    if not section_match:
                        logging.warning(f"Could not extract section from filename: {filename}")
                        continue
                    
                    section_id = f"section_{section_match.group(1)}"
                    
                    # Extract domain (source/target)
                    if dataset_type == 'dev':
                        # For dev files, domain is in filename
                        if 'source' in filename:
                            domain = 'source'
                        elif 'target' in filename:
                            domain = 'target'
                        else:
                            logging.warning(f"Could not extract domain from dev filename: {filename}")
                            continue
                    else:
                        # For eval files, get domain from reference filename via data loader
                        ref_filename = data_loader.get_reference_filename(filename)
                        if ref_filename and 'target' in ref_filename:
                            domain = 'target'
                        else:
                            domain = 'source'  # Default to source
                    
                    # Include machine_type in mapping for DCASE2021-2025
                    mapping.append([filename, y_score, y_true, section_id, domain, machine_type])
        
        logging.info(f"Built mapping with {len(mapping)} entries for {dataset_type} set")
        return mapping
    
    def _group_and_evaluate(self, mapping: list, year: int) -> dict:
        """
        Group mapping data by year-specific strategy and calculate AUC/pAUC for each group.
        
        Args:
            mapping: List mapping entries:
                     DCASE2020: [filename, y_score, y_true, machine_type, id]
                     DCASE2021-2025: [filename, y_score, y_true, section_id, domain, machine_type]
            year: DCASE year
            
        Returns:
            Dictionary containing grouped evaluation results:
            {
                'auc_groups': {group_key: {'auc': value, 'scores': [...], 'labels': [...]}},
                'pauc_groups': {group_key: {'pauc': value, 'scores': [...], 'labels': [...]}},
                'strategy': 'description of grouping strategy used'
            }
        """
        from sklearn import metrics
        import numpy as np
        from collections import defaultdict
        
        if not mapping:
            return {'auc_groups': {}, 'pauc_groups': {}, 'strategy': 'no_data'}
        
        logging.info(f"Grouping and evaluating {len(mapping)} entries using DCASE{year} strategy")
        
        auc_groups = defaultdict(lambda: {'scores': [], 'labels': []})
        pauc_groups = defaultdict(lambda: {'scores': [], 'labels': []})
        
        if year == 2020:
            # DCASE2020: Both AUC and pAUC use machine_type + id grouping
            strategy = "DCASE2020: machine_type + id for both AUC and pAUC"
            
            for entry in mapping:
                filename, y_score, y_true, machine_type, id_num = entry
                group_key = f"{machine_type}_{id_num}"
                
                # Both AUC and pAUC use same grouping
                auc_groups[group_key]['scores'].append(y_score)
                auc_groups[group_key]['labels'].append(y_true)
                pauc_groups[group_key]['scores'].append(y_score)
                pauc_groups[group_key]['labels'].append(y_true)
        
        elif year == 2021:
            # DCASE2021: Both AUC and pAUC use section + domain + machine grouping
            strategy = "DCASE2021: section + domain + machine for both AUC and pAUC"
            
            for entry in mapping:
                filename, y_score, y_true, section_id, domain, machine_type = entry
                group_key = f"{machine_type}_{section_id}_{domain}"
                
                # Both AUC and pAUC use same grouping
                auc_groups[group_key]['scores'].append(y_score)
                auc_groups[group_key]['labels'].append(y_true)
                pauc_groups[group_key]['scores'].append(y_score)
                pauc_groups[group_key]['labels'].append(y_true)
        
        else:
            # DCASE2022-2025: AUC uses section+domain+machine, pAUC uses section+machine only
            strategy = f"DCASE{year}: AUC uses section+domain+machine, pAUC uses section+machine only"
            
            for entry in mapping:
                filename, y_score, y_true, section_id, domain, machine_type = entry
                
                # AUC grouping: section + domain + machine
                auc_group_key = f"{machine_type}_{section_id}_{domain}"
                auc_groups[auc_group_key]['scores'].append(y_score)
                auc_groups[auc_group_key]['labels'].append(y_true)
                
                # pAUC grouping: section + machine (no domain)
                pauc_group_key = f"{machine_type}_{section_id}"
                pauc_groups[pauc_group_key]['scores'].append(y_score)
                pauc_groups[pauc_group_key]['labels'].append(y_true)
        
        # Calculate AUC for each group
        auc_results = {}
        for group_key, group_data in auc_groups.items():
            scores = np.array(group_data['scores'])
            labels = np.array(group_data['labels'])
            
            if len(scores) > 0 and len(np.unique(labels)) > 1:
                try:
                    auc_value = metrics.roc_auc_score(labels, scores)
                    auc_results[group_key] = {
                        'auc': auc_value,
                        'scores': scores.tolist(),
                        'labels': labels.tolist(),
                        'count': len(scores)
                    }
                except Exception as e:
                    logging.warning(f"Failed to calculate AUC for group {group_key}: {e}")
                    auc_results[group_key] = {
                        'auc': 0.0,
                        'scores': scores.tolist(),
                        'labels': labels.tolist(),
                        'count': len(scores)
                    }
            else:
                auc_results[group_key] = {
                    'auc': 0.0,
                    'scores': scores.tolist(),
                    'labels': labels.tolist(),
                    'count': len(scores)
                }
        
        # Calculate pAUC for each group
        pauc_results = {}
        for group_key, group_data in pauc_groups.items():
            scores = np.array(group_data['scores'])
            labels = np.array(group_data['labels'])
            
            if len(scores) > 0 and len(np.unique(labels)) > 1:
                try:
                    pauc_value = metrics.roc_auc_score(labels, scores, max_fpr=0.1)
                    pauc_results[group_key] = {
                        'pauc': pauc_value,
                        'scores': scores.tolist(),
                        'labels': labels.tolist(),
                        'count': len(scores)
                    }
                except Exception as e:
                    logging.warning(f"Failed to calculate pAUC for group {group_key}: {e}")
                    pauc_results[group_key] = {
                        'pauc': 0.0,
                        'scores': scores.tolist(),
                        'labels': labels.tolist(),
                        'count': len(scores)
                    }
            else:
                pauc_results[group_key] = {
                    'pauc': 0.0,
                    'scores': scores.tolist(),
                    'labels': labels.tolist(),
                    'count': len(scores)
                }
        
        logging.info(f"Calculated metrics for {len(auc_results)} AUC groups and {len(pauc_results)} pAUC groups")
        
        return {
            'auc_groups': auc_results,
            'pauc_groups': pauc_results,
            'strategy': strategy
        }
    
    def _aggregate_final_scores(self, grouped_results: dict, year: int) -> dict:
        """
        Aggregate final scores from grouped results using year-specific strategy.
        
        Args:
            grouped_results: Results from _group_and_evaluate()
            year: DCASE year
            
        Returns:
            Dictionary containing final aggregated scores:
            {
                'overall_score': float,
                'auc_scores': [list of AUC values],
                'pauc_scores': [list of pAUC values],
                'aggregation_method': 'harmonic_mean' or 'arithmetic_mean',
                'details': {...}
            }
        """
        import numpy as np
        from scipy import stats
        
        auc_groups = grouped_results.get('auc_groups', {})
        pauc_groups = grouped_results.get('pauc_groups', {})
        
        if not auc_groups and not pauc_groups:
            return {
                'overall_score': 0.0,
                'auc_scores': [],
                'pauc_scores': [],
                'aggregation_method': 'none',
                'details': {
                    'num_auc_groups': 0,
                    'num_pauc_groups': 0,
                    'auc_group_details': {},
                    'pauc_group_details': {}
                }
            }
        
        # Extract AUC scores
        auc_scores = []
        for group_key, group_data in auc_groups.items():
            auc_scores.append(group_data['auc'])
        
        # Extract pAUC scores
        pauc_scores = []
        for group_key, group_data in pauc_groups.items():
            pauc_scores.append(group_data['pauc'])
        
        # Determine aggregation method based on year
        if year == 2020:
            # DCASE2020: Use arithmetic mean
            aggregation_method = 'arithmetic_mean'
            
            # Combine all scores
            all_scores = auc_scores + pauc_scores
            if all_scores:
                overall_score = np.mean(all_scores)
            else:
                overall_score = 0.0
                
            logging.info(f"DCASE2020 aggregation: arithmetic mean of {len(all_scores)} scores = {overall_score:.6f}")
            
        else:
            # DCASE2021-2025: Use harmonic mean  
            aggregation_method = 'harmonic_mean'
            
            # Combine all scores
            all_scores = auc_scores + pauc_scores
            if all_scores:
                # Replace any zero or negative values with a small positive value for harmonic mean
                all_scores_safe = np.maximum(all_scores, np.finfo(float).eps)
                overall_score = stats.hmean(all_scores_safe)
            else:
                overall_score = 0.0
                
            logging.info(f"DCASE{year} aggregation: harmonic mean of {len(all_scores)} scores = {overall_score:.6f}")
        
        # Create detailed breakdown
        details = {
            'num_auc_groups': len(auc_groups),
            'num_pauc_groups': len(pauc_groups),
            'auc_group_details': {},
            'pauc_group_details': {}
        }
        
        # Add AUC group details
        for group_key, group_data in auc_groups.items():
            details['auc_group_details'][group_key] = {
                'auc': group_data['auc'],
                'count': group_data['count']
            }
        
        # Add pAUC group details  
        for group_key, group_data in pauc_groups.items():
            details['pauc_group_details'][group_key] = {
                'pauc': group_data['pauc'],
                'count': group_data['count']
            }
        
        # Special handling for DCASE2021 (separate source/target pAUC tracking)
        if year == 2021:
            source_aucs = []
            target_aucs = []
            source_paucs = []
            target_paucs = []
            
            for group_key, group_data in auc_groups.items():
                if '_source' in group_key:
                    source_aucs.append(group_data['auc'])
                elif '_target' in group_key:
                    target_aucs.append(group_data['auc'])
                    
            for group_key, group_data in pauc_groups.items():
                if '_source' in group_key:
                    source_paucs.append(group_data['pauc'])
                elif '_target' in group_key:
                    target_paucs.append(group_data['pauc'])
            
            details['dcase2021_breakdown'] = {
                'source_aucs': source_aucs,
                'target_aucs': target_aucs,
                'source_paucs': source_paucs,
                'target_paucs': target_paucs
            }
        
        return {
            'overall_score': overall_score,
            'auc_scores': auc_scores,
            'pauc_scores': pauc_scores,
            'aggregation_method': aggregation_method,
            'details': details
        }
    
    def _calculate_overall_year_score(self, dev_grouped: dict, eval_grouped: dict, year: int) -> dict:
        """
        Calculate overall_year score by combining dev and eval AUC/pAUC values.
        
        Args:
            dev_grouped: Dev grouped results from _group_and_evaluate()
            eval_grouped: Eval grouped results from _group_and_evaluate()
            year: DCASE year
            
        Returns:
            Dictionary containing overall_year score and details:
            {
                'overall_year_score': float,
                'aggregation_method': 'harmonic_mean' or 'arithmetic_mean',
                'total_values_count': int,
                'dev_values_count': int,
                'eval_values_count': int,
                'breakdown': {...}
            }
        """
        import numpy as np
        from scipy import stats
        
        # Extract all AUC and pAUC values from dev and eval
        dev_auc_values = [group_data['auc'] for group_data in dev_grouped.get('auc_groups', {}).values()]
        dev_pauc_values = [group_data['pauc'] for group_data in dev_grouped.get('pauc_groups', {}).values()]
        eval_auc_values = [group_data['auc'] for group_data in eval_grouped.get('auc_groups', {}).values()]  
        eval_pauc_values = [group_data['pauc'] for group_data in eval_grouped.get('pauc_groups', {}).values()]
        
        # Combine all values
        all_values = dev_auc_values + dev_pauc_values + eval_auc_values + eval_pauc_values
        
        # Calculate overall_year score based on year
        if year == 2020:
            # DCASE2020: Arithmetic mean
            aggregation_method = 'arithmetic_mean'
            if all_values:
                overall_year_score = np.mean(all_values)
            else:
                overall_year_score = 0.0
        else:
            # DCASE2021-2025: Harmonic mean
            aggregation_method = 'harmonic_mean'
            if all_values:
                # Replace any zero or negative values with a small positive value for harmonic mean
                all_values_safe = np.maximum(all_values, np.finfo(float).eps)
                overall_year_score = stats.hmean(all_values_safe)
            else:
                overall_year_score = 0.0
        
        # Create detailed breakdown
        breakdown = {
            'dev_auc_count': len(dev_auc_values),
            'dev_pauc_count': len(dev_pauc_values),
            'eval_auc_count': len(eval_auc_values),
            'eval_pauc_count': len(eval_pauc_values),
            'dev_auc_values': dev_auc_values,
            'dev_pauc_values': dev_pauc_values,
            'eval_auc_values': eval_auc_values,
            'eval_pauc_values': eval_pauc_values
        }
        
        logging.info(f"Overall year calculation: {aggregation_method} of {len(all_values)} values = {overall_year_score:.6f}")
        logging.info(f"  Dev: {len(dev_auc_values)} AUC + {len(dev_pauc_values)} pAUC = {len(dev_auc_values) + len(dev_pauc_values)} values")
        logging.info(f"  Eval: {len(eval_auc_values)} AUC + {len(eval_pauc_values)} pAUC = {len(eval_auc_values) + len(eval_pauc_values)} values")
        
        return {
            'overall_year_score': overall_year_score,
            'aggregation_method': aggregation_method,
            'total_values_count': len(all_values),
            'dev_values_count': len(dev_auc_values) + len(dev_pauc_values),
            'eval_values_count': len(eval_auc_values) + len(eval_pauc_values),
            'breakdown': breakdown
        }
    
    def _save_evaluation_results_to_csv(self, results: dict, scores_root_dir: str, year: int):
        """
        Save evaluation results to CSV files in the scores directory.
        
        Args:
            results: Complete evaluation results dictionary
            scores_root_dir: Root directory containing score files
            year: DCASE year
        """
        import os
        import pandas as pd
        
        # Create output directory structure
        year_dir = os.path.join(scores_root_dir, f"dcase{year}_t2")
        csv_output_dir = os.path.join(year_dir, "evaluation_results")
        os.makedirs(csv_output_dir, exist_ok=True)
        
        # 1. Save overall summary results
        self._save_overall_summary_csv(results, csv_output_dir, year)
        
        # 2. Save detailed group results
        self._save_detailed_group_results_csv(results, csv_output_dir, year)
        
        # 3. Save raw values breakdown
        self._save_raw_values_breakdown_csv(results, csv_output_dir, year)
        
        logging.info(f"📊 CSV results saved to: {csv_output_dir}")
        
    def _save_overall_summary_csv(self, results: dict, output_dir: str, year: int):
        """Save overall summary metrics to CSV."""
        import pandas as pd
        import os
        
        summary = results['summary']
        overall_year_details = summary['overall_year_details']
        
        # Create summary data
        summary_data = [
            {
                'metric_type': 'Dev Overall',
                'score': summary['dev_overall_score'],
                'description': f'DCASE{year} Dev dataset overall score',
                'aggregation_method': summary['aggregation_method'],
                'values_count': results['dev']['overall']['details']['num_auc_groups'] + results['dev']['overall']['details']['num_pauc_groups']
            },
            {
                'metric_type': 'Eval Overall', 
                'score': summary['eval_overall_score'],
                'description': f'DCASE{year} Eval dataset overall score',
                'aggregation_method': summary['aggregation_method'],
                'values_count': results['eval']['overall']['details']['num_auc_groups'] + results['eval']['overall']['details']['num_pauc_groups']
            },
            {
                'metric_type': 'Overall Year',
                'score': summary['overall_year_score'],
                'description': f'DCASE{year} combined Dev+Eval overall score',
                'aggregation_method': overall_year_details['aggregation_method'],
                'values_count': overall_year_details['total_values_count']
            }
        ]
        
        # Add breakdown details for overall_year
        breakdown = overall_year_details['breakdown']
        summary_data.extend([
            {
                'metric_type': 'Dev Values Breakdown',
                'score': '',
                'description': f"Dev AUC: {breakdown['dev_auc_count']}, Dev pAUC: {breakdown['dev_pauc_count']}",
                'aggregation_method': '',
                'values_count': overall_year_details['dev_values_count']
            },
            {
                'metric_type': 'Eval Values Breakdown', 
                'score': '',
                'description': f"Eval AUC: {breakdown['eval_auc_count']}, Eval pAUC: {breakdown['eval_pauc_count']}",
                'aggregation_method': '',
                'values_count': overall_year_details['eval_values_count']
            }
        ])
        
        # Save to CSV
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, f"dcase{year}_overall_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        logging.info(f"📄 Overall summary saved: {summary_path}")
        
    def _save_detailed_group_results_csv(self, results: dict, output_dir: str, year: int):
        """Save detailed group-level results to CSV."""
        import pandas as pd
        import os
        
        detailed_data = []
        
        # Process Dev results
        dev_grouped = results['dev']['grouped']
        self._add_group_data_to_list(detailed_data, dev_grouped, 'dev', year)
        
        # Process Eval results  
        eval_grouped = results['eval']['grouped']
        self._add_group_data_to_list(detailed_data, eval_grouped, 'eval', year)
        
        # Save to CSV
        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            detailed_path = os.path.join(output_dir, f"dcase{year}_detailed_groups.csv")
            df_detailed.to_csv(detailed_path, index=False)
            logging.info(f"📄 Detailed groups saved: {detailed_path}")
            
    def _add_group_data_to_list(self, data_list: list, grouped_results: dict, dataset_type: str, year: int):
        """Add group data to the data list for CSV export."""
        
        # Add AUC groups
        for group_key, group_data in grouped_results.get('auc_groups', {}).items():
            # Parse group key based on year
            if year == 2020:
                # Format: "machine_id_XX"
                parts = group_key.split('_')
                machine_type = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                section_id = parts[-2] + '_' + parts[-1] if len(parts) > 2 else 'N/A'
                domain = 'N/A'
            else:
                # Format: "machine_section_XX_domain" 
                parts = group_key.split('_')
                if len(parts) >= 4:
                    machine_type = '_'.join(parts[:-3])
                    section_id = parts[-3] + '_' + parts[-2]
                    domain = parts[-1]
                else:
                    machine_type = group_key
                    section_id = 'N/A'
                    domain = 'N/A'
            
            data_list.append({
                'dataset_type': dataset_type,
                'metric_type': 'AUC',
                'machine_type': machine_type,
                'section_id': section_id,
                'domain': domain,
                'group_key': group_key,
                'score': group_data['auc'],
                'sample_count': group_data['count'],
                'description': f"AUC for {group_key} ({group_data['count']} samples)"
            })
        
        # Add pAUC groups
        for group_key, group_data in grouped_results.get('pauc_groups', {}).items():
            # Parse group key based on year
            if year == 2020:
                # Format: "machine_id_XX"
                parts = group_key.split('_')
                machine_type = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                section_id = parts[-2] + '_' + parts[-1] if len(parts) > 2 else 'N/A'
                domain = 'N/A'
            else:
                # Parse for DCASE2021+ which may have different formats
                parts = group_key.split('_')
                if len(parts) >= 4:  # machine_section_XX_domain
                    machine_type = '_'.join(parts[:-3])
                    section_id = parts[-3] + '_' + parts[-2]
                    domain = parts[-1]
                elif len(parts) >= 3:  # machine_section_XX (for DCASE2022+)
                    machine_type = '_'.join(parts[:-2])
                    section_id = parts[-2] + '_' + parts[-1]
                    domain = 'combined'  # pAUC combines source+target for 2022+
                else:
                    machine_type = group_key
                    section_id = 'N/A'
                    domain = 'N/A'
            
            data_list.append({
                'dataset_type': dataset_type,
                'metric_type': 'pAUC',
                'machine_type': machine_type,
                'section_id': section_id,
                'domain': domain,
                'group_key': group_key,
                'score': group_data['pauc'],
                'sample_count': group_data['count'],
                'description': f"pAUC for {group_key} ({group_data['count']} samples)"
            })
            
    def _save_raw_values_breakdown_csv(self, results: dict, output_dir: str, year: int):
        """Save raw values breakdown for overall_year calculation."""
        import pandas as pd
        import os
        
        overall_year_details = results['summary']['overall_year_details']
        breakdown = overall_year_details['breakdown']
        
        raw_data = []
        
        # Add dev AUC values
        for i, value in enumerate(breakdown['dev_auc_values']):
            raw_data.append({
                'dataset_type': 'dev',
                'metric_type': 'AUC',
                'index': i,
                'value': value,
                'description': f'Dev AUC value #{i+1}'
            })
        
        # Add dev pAUC values
        for i, value in enumerate(breakdown['dev_pauc_values']):
            raw_data.append({
                'dataset_type': 'dev',
                'metric_type': 'pAUC', 
                'index': i,
                'value': value,
                'description': f'Dev pAUC value #{i+1}'
            })
        
        # Add eval AUC values
        for i, value in enumerate(breakdown['eval_auc_values']):
            raw_data.append({
                'dataset_type': 'eval',
                'metric_type': 'AUC',
                'index': i,
                'value': value,
                'description': f'Eval AUC value #{i+1}'
            })
        
        # Add eval pAUC values
        for i, value in enumerate(breakdown['eval_pauc_values']):
            raw_data.append({
                'dataset_type': 'eval',
                'metric_type': 'pAUC',
                'index': i,
                'value': value,
                'description': f'Eval pAUC value #{i+1}'
            })
        
        # Save to CSV
        if raw_data:
            df_raw = pd.DataFrame(raw_data)
            raw_path = os.path.join(output_dir, f"dcase{year}_raw_values.csv")
            df_raw.to_csv(raw_path, index=False)
            logging.info(f"📄 Raw values saved: {raw_path}")


    def _build_memory_banks_for_year(self, year: int, config: Dict, data_loader: DCASEDataLoader, 
                                    memory_banks: Dict):
        """Build memory banks for a specific year."""
        logging.info(f"Building memory banks for DCASE {year}")
        
        # Check cache configuration
        cache_config = config.get('feature_cache', {})
        cache_enabled = cache_config.get('enabled', DEFAULT_CACHE_ENABLED)
        cache_dir = cache_config.get('cache_dir', DEFAULT_CACHE_DIR)
        
        # Check multiprocessing configuration
        multiprocessing_enabled = config.get('multiprocessing', {}).get('enabled', DEFAULT_MULTIPROCESSING_ENABLED)
        
        # Step 1: Process dev train data
        logging.info("Step 1: Processing dev train data...")
        dev_machine_types = data_loader.get_machine_types()
        if multiprocessing_enabled:
            self._build_memory_bank_for_dataset_multiprocessing('dev', dev_machine_types, cache_dir, cache_enabled, config, data_loader, memory_banks)
        else:
            self._build_memory_bank_for_dataset('dev', dev_machine_types, cache_dir, cache_enabled, config, data_loader, memory_banks)

        # Step 2: Process eval train data
        logging.info("Step 2: Processing eval train data...")
        eval_machine_types = data_loader.get_eval_machine_types()
        if multiprocessing_enabled:
            self._build_memory_bank_for_dataset_multiprocessing('eval', eval_machine_types, cache_dir, cache_enabled, config, data_loader, memory_banks)
        else:
            self._build_memory_bank_for_dataset('eval', eval_machine_types, cache_dir, cache_enabled, config, data_loader, memory_banks)
        
        logging.info(f"✓ Memory banks built for DCASE {year}")
    
    def _build_memory_bank_for_dataset_multiprocessing(self, dataset_type: str, machine_types: List[str], 
                                                      cache_dir: str, cache_enabled: bool, config: Dict,
                                                      data_loader: DCASEDataLoader, memory_banks: Dict):
        """Build memory bank using multiprocessing."""
        dataset_config = data_loader.get_dataset_config(dataset_type)
        multiprocessing_args = []
        
        for machine_type in machine_types:
            cache_file = os.path.join(cache_dir, f"{dataset_type}_train_{machine_type}.pth")
            
            if cache_enabled and os.path.exists(cache_file):
                print(f"\n✓ Loading cached train features for {machine_type}...")
                memory_bank = MemoryBank(config['dataset_year'], machine_type)
                memory_bank.load(cache_file)
                memory_banks[dataset_type][machine_type] = memory_bank
                continue
            else:
                print(f"[CACHE MISS] {dataset_type} {machine_type} -> {cache_file}")
            
            train_files = dataset_config.train_files_getter(machine_type)
            if train_files:
                logging.info(f"  {dataset_type.capitalize()} train files: {len(train_files)}")
                multiprocessing_args.append((
                    machine_type, dataset_type, train_files, 
                    self.device, config, self.extractor_path
                ))
            else:
                logging.warning(f"No training files found for {machine_type}")
        
        # Process machine types in parallel if there are any to process
        if len(multiprocessing_args) > 0:
            self._process_machine_types_in_parallel(multiprocessing_args, dataset_type, cache_enabled, memory_banks)
        else:
            logging.info(f"✓ All {dataset_type} features loaded from cache")
    
    def _process_machine_types_in_parallel(self, multiprocessing_args: List[Tuple], dataset_type: str, cache_enabled: bool, memory_banks: Dict):
        """
        Process machine types in parallel using multiprocessing.
        
        Args:
            multiprocessing_args: List of argument tuples for worker functions
            dataset_type: 'dev' or 'eval'
            cache_enabled: Whether caching is enabled
            memory_banks: Dictionary to store memory banks
        """
        # Determine number of processes (use CPU count but cap at number of machine types)
        num_processes = min(mp.cpu_count(), len(multiprocessing_args))
        # num_processes = min(mp.cpu_count(), 4)
        logging.info(f"Processing {len(multiprocessing_args)} machine types using {num_processes} processes...")
        
        start_time = time.time()
        
        # Use multiprocessing to extract features
        # Set start method to 'spawn' to avoid issues with fork
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(_extract_features_for_machine_type_worker, multiprocessing_args)
            pool.close()
            pool.join()
        
        # Process results and add memory banks
        for result in results:
            machine_type = result['machine_type']
            memory_bank = result['memory_bank']
            cache_file_path = result['cache_file_path']
            
            logging.info(f"✓ Completed feature extraction for {machine_type} ({memory_bank.get_feature_count()} features)")
            
            # Save memory bank to cache
            if not os.path.exists(cache_file_path):
                memory_bank.save(cache_file_path)
                logging.info(f"✓ Cache saved for {machine_type}")
            
            # Add memory bank to main tester
            memory_banks[dataset_type][machine_type] = memory_bank
        
        elapsed_time = time.time() - start_time
        logging.info(f"✓ All {len(multiprocessing_args)} machine types processed in {elapsed_time:.2f} seconds")
    
    def _build_memory_bank_for_dataset(self, dataset_type: str, machine_types: List[str], 
                                      cache_dir: str, cache_enabled: bool, config: Dict,
                                      data_loader: DCASEDataLoader, memory_banks: Dict):
        """Build memory bank for a specific dataset type."""
        dataset_config = data_loader.get_dataset_config(dataset_type)
        custom_extractor = load_extractor(self.extractor_path)
        if not isinstance(custom_extractor, BaseFeatureExtractor):
            raise ValueError("custom_extractor must be an instance of BaseFeatureExtractor")

        print(f"Building {dataset_type} memory banks...")

        # Create cache directory if it doesn't exist
        if cache_enabled:
            os.makedirs(cache_dir, exist_ok=True)
        
        for machine_type in machine_types:
            cache_file = os.path.join(cache_dir, f"{dataset_type}_train_{machine_type}.pth")
            
            # Check cache first
            if cache_enabled and os.path.exists(cache_file):
                print(f"\n✓ Loading cached train features for {machine_type}...")
                memory_bank = MemoryBank(config['dataset_year'], machine_type)
                memory_bank.load(cache_file)
                memory_banks[dataset_type][machine_type] = memory_bank
                continue
            
            print(f"\nProcessing {dataset_type} {machine_type}...")
            train_files = dataset_config.train_files_getter(machine_type)
            if train_files:
                print(f"  {dataset_type.capitalize()} train files: {len(train_files)}")
                
                # Create memory bank for this machine type
                memory_bank = MemoryBank(config['dataset_year'], machine_type)
                
                # Extract features using custom extractor
                features = custom_extractor.extract_features_batch(train_files)
                # Standardize to 1D vectors (average across channels if needed)
                std_features = []
                for f in features:
                    t = f if isinstance(f, torch.Tensor) else torch.tensor(f)
                    if t.dim() == 2:
                        t = t.mean(dim=0)
                    elif t.dim() != 1:
                        raise ValueError(f"Unsupported feature shape in DCASE build: {t.shape}")
                    std_features.append(t)
                features_tensor = torch.stack(std_features)
                
                # Group features by domain/section
                domain_groups = {}
                for i, file_path in enumerate(train_files):
                    domain_or_section = self._extract_domain_or_section(file_path, config['dataset_year'])
                    if domain_or_section not in domain_groups:
                        domain_groups[domain_or_section] = []
                    domain_groups[domain_or_section].append(i)
                
                # Add features to memory bank by domain/section
                for domain_or_section, indices in domain_groups.items():
                    domain_features = features_tensor[indices]
                    memory_bank.add_features(domain_or_section, domain_features)
                
                print(f"  Added {len(train_files)} {dataset_type} train files to memory bank for {machine_type}")
                
                # Save cache
                if cache_enabled and not os.path.exists(cache_file):
                    print(f"  Saving {dataset_type} train features for {machine_type} to cache...")
                    memory_bank.save(cache_file)
                
                # Add memory bank to main tester
                memory_banks[dataset_type][machine_type] = memory_bank
    
    def _evaluate_dataset_for_year(self, year: int, config: Dict, data_loader: DCASEDataLoader, 
                                  memory_banks: Dict, dataset_type: str):
        """Evaluate a specific dataset for a year."""
        logging.info(f"Evaluating {dataset_type} set for DCASE {year}")
        
        # Get dataset configuration
        dataset_config = data_loader.get_dataset_config(dataset_type)
        
        # Get machine types
        machine_types = dataset_config.machine_types_getter()
        
        # Check cache configuration
        cache_config = config.get('feature_cache', {})
        cache_enabled = cache_config.get('enabled', False)
        cache_dir = cache_config.get('cache_dir', DEFAULT_CACHE_DIR)
        
        # Check multiprocessing configuration
        multiprocessing_config = config.get('multiprocessing', {})
        multiprocessing_enabled = multiprocessing_config.get('enabled', True)
        num_processes = multiprocessing_config.get('num_processes', mp.cpu_count())
        
        # Results dictionary
        results_dict = {}
        
        if multiprocessing_enabled and len(machine_types) > 1:
            print(f"Using multiprocessing with {num_processes} processes for evaluation...")
            self._evaluate_machine_types_in_parallel(
                machine_types, dataset_config, cache_dir, cache_enabled, 
                results_dict, num_processes, dataset_type, config, memory_banks
            )
        else:
            print("Using sequential evaluation...")
            self._evaluate_machine_types_sequential(
                machine_types, dataset_config, cache_dir, cache_enabled, 
                results_dict, dataset_type, config, memory_banks
            )
        
        # Store results
        if dataset_type == 'dev':
            self.dev_results[year] = results_dict
        else:
            self.eval_results[year] = results_dict
        
        print(f"{dataset_type.capitalize()} evaluation completed for DCASE {year}!")
    
    def _evaluate_machine_types_sequential(self, machine_types: List[str], dataset_config: DatasetConfig,
                                         cache_dir: str, cache_enabled: bool, results_dict: Dict, 
                                         dataset_type: str, config: Dict, memory_banks: Dict):
        """Evaluate machine types sequentially."""
        custom_extractor = load_extractor(self.extractor_path)
        if not isinstance(custom_extractor, BaseFeatureExtractor):
            raise ValueError("custom_extractor must be an instance of BaseFeatureExtractor")

        for machine_type in machine_types:
            print(f"\nEvaluating {dataset_config.name} {machine_type}...")
            

            
            # Get test files and labels
            test_files_with_labels = dataset_config.test_files_getter(machine_type)
            if not test_files_with_labels:

                continue
                
            test_files, labels = zip(*test_files_with_labels)
            labels = np.array(labels)
            

            
            # Check if test features are cached
            test_cache_file = os.path.join(cache_dir, f"{dataset_config.cache_prefix}_{machine_type}_test.pth")
            if cache_enabled and os.path.exists(test_cache_file):
                print(f"  ✓ Loading cached test features for {machine_type}...")
                test_features_tensor = torch.load(test_cache_file)
            else:
                print(f"  ⚠ Extracting test features for {machine_type}...")
                # Extract features using custom extractor
                test_features = custom_extractor.extract_features_batch(test_files)
                # Standardize to 1D vectors (average across channels if needed)
                std_features = []
                for f in test_features:
                    t = f if isinstance(f, torch.Tensor) else torch.tensor(f)
                    if t.dim() == 2:
                        t = t.mean(dim=0)
                    elif t.dim() != 1:
                        raise ValueError(f"Unsupported feature shape in DCASE seq eval: {t.shape}")
                    std_features.append(t)
                test_features_tensor = torch.stack(std_features)
                
                # Save test features to cache
                if cache_enabled and not os.path.exists(test_cache_file):
                    print(f"  ✓ Saving test features for {machine_type} to cache...")
                    torch.save(test_features_tensor, test_cache_file)
                elif cache_enabled and os.path.exists(test_cache_file):
                    print(f"  ✓ Test cache for {machine_type} already exists, skipping save")
            
            # Get memory bank for this machine type
            if machine_type not in memory_banks[dataset_type]:

                continue
            
            memory_bank = memory_banks[dataset_type][machine_type]
            
            
            # Compute anomaly scores
            memory_bank_config = config.get('memory_bank', {})
            similarity_type = memory_bank_config.get('similarity_type', 'cosine')
            aggregation = memory_bank_config.get('aggregation', 'max')
            k = memory_bank_config.get('k', 1)
            
            # Get new parameters for DCASE 2021-2025
            domain_strategy = memory_bank_config.get('domain_strategy', 'min')
            enable_kmeans_optimization = memory_bank_config.get('enable_kmeans_optimization', False)
            kmeans_threshold = memory_bank_config.get('kmeans_threshold', 100)
            kmeans_n_clusters = memory_bank_config.get('kmeans_n_clusters', 16)
            
            anomaly_scores = memory_bank.compute_anomaly_scores(
                test_features_tensor, test_files,
                similarity_type=similarity_type, aggregation=aggregation, k=k,
                domain_strategy=domain_strategy, enable_kmeans_optimization=enable_kmeans_optimization,
                kmeans_threshold=kmeans_threshold, kmeans_n_clusters=kmeans_n_clusters
            )
            
            # Store results
            results_dict[machine_type] = {
                'y_true': labels,
                'y_score': anomaly_scores,
                'file_paths': test_files
            }
            

            
            # Save scores to CSV
            scores_with_filenames = [
                (os.path.basename(f), score) for f, score in zip(test_files, anomaly_scores)
            ]
            score_file_path = os.path.join(
                config['results_dir'], dataset_config.name, f'score_{machine_type}_{dataset_config.results_suffix}.csv'
            )
            self.evaluator.save_scores_to_csv(scores_with_filenames, score_file_path)
    
    def _evaluate_machine_types_in_parallel(self, machine_types: List[str], dataset_config: DatasetConfig,
                                          cache_dir: str, cache_enabled: bool, results_dict: Dict, 
                                          num_processes: int, dataset_type: str, config: Dict, memory_banks: Dict):
        """Evaluate machine types in parallel using multiprocessing."""
        # Prepare arguments for multiprocessing
        multiprocessing_args = []
        for machine_type in machine_types:
            test_files_with_labels = dataset_config.test_files_getter(machine_type)
            if not test_files_with_labels:
                continue
                
            args_tuple = (machine_type, test_files_with_labels, dataset_config, 
                         self.device, config, cache_dir, cache_enabled, self.extractor_path)
            multiprocessing_args.append(args_tuple)
        
        if not multiprocessing_args:
            print("No valid machine types to evaluate.")
            return
        
        # Use multiprocessing to evaluate machine types in parallel
        num_processes = min(num_processes, len(multiprocessing_args))
        
        # Set start method to 'spawn' to avoid issues with fork
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(_evaluate_machine_type_worker, multiprocessing_args),
                total=len(multiprocessing_args),
                desc=f"Evaluating {dataset_config.name} machine types",
                ncols=100
            ))
            pool.close()
            pool.join()
        
        # Process results and compute anomaly scores in main process
        for machine_type, results_dict_entry, _ in results:
            if results_dict_entry is not None:
                # Get memory bank for this machine type
                if machine_type not in memory_banks[dataset_type]:
                    print(f"Warning: No memory bank found for {machine_type}, skipping...")
                    continue
                
                memory_bank = memory_banks[dataset_type][machine_type]
                
                # Compute anomaly scores using memory bank in main process
                test_features_tensor = results_dict_entry['test_features_tensor']
                test_files = results_dict_entry['file_paths']
                
                memory_bank_config = config.get('memory_bank', {})
                similarity_type = memory_bank_config.get('similarity_type', 'cosine')
                aggregation = memory_bank_config.get('aggregation', 'max')
                k = memory_bank_config.get('k', 1)
                
                # Get new parameters for DCASE 2021-2025
                domain_strategy = memory_bank_config.get('domain_strategy', 'min')
                enable_kmeans_optimization = memory_bank_config.get('enable_kmeans_optimization', False)
                kmeans_threshold = memory_bank_config.get('kmeans_threshold', 100)
                kmeans_n_clusters = memory_bank_config.get('kmeans_n_clusters', 16)
                
                # Compute anomaly scores
                anomaly_scores = memory_bank.compute_anomaly_scores(
                    test_features_tensor, test_files,
                    similarity_type=similarity_type, aggregation=aggregation, k=k,
                    domain_strategy=domain_strategy, enable_kmeans_optimization=enable_kmeans_optimization,
                    kmeans_threshold=kmeans_threshold, kmeans_n_clusters=kmeans_n_clusters
                )
                
                # Update results dict with scores
                results_dict_entry['y_score'] = anomaly_scores
                del results_dict_entry['test_features_tensor']  # Remove features to save memory
                
                # Save scores to CSV (same as sequential processing)
                scores_with_filenames = [
                    (os.path.basename(f), score) for f, score in zip(test_files, anomaly_scores)
                ]
                score_file_path = os.path.join(
                    config['results_dir'], dataset_config.name, f'score_{machine_type}_{dataset_config.results_suffix}.csv'
                )
                self.evaluator.save_scores_to_csv(scores_with_filenames, score_file_path)
                
                # Debug: Print label and score statistics
                y_true = results_dict_entry['y_true']
                # print(f"DEBUG: {machine_type} - Labels: {np.bincount(y_true)}, Score range: [{anomaly_scores.min():.6f}, {anomaly_scores.max():.6f}], Mean: {anomaly_scores.mean():.6f}")
                # print(f"DEBUG: {machine_type} - Normal scores (mean): {anomaly_scores[y_true == 0].mean():.6f}, Anomaly scores (mean): {anomaly_scores[y_true == 1].mean():.6f}")
                
                results_dict[machine_type] = results_dict_entry
    
        return 
    

    
    def run_evaluation(self):
        """Run complete evaluation pipeline for all specified years."""
        logging.info("Starting DCASE evaluation...")
        
        for year in self.years_to_evaluate:
            try:
                self.evaluate_year(year)
            except Exception as e:
                logging.error(f"Error evaluating DCASE {year}: {e}")
                continue
        
        logging.info("DCASE evaluation completed!")
    
 

    def _extract_granularity_for_scoring(self, filename: str, dataset_year: int, metric_type: str, reference_filename: str = None, machine_type: str = None) -> str:
        """
        Extract granularity information for scoring based on year and metric type.
        
        Args:
            filename: Filename to parse
            dataset_year: Dataset year (2020-2025)
            metric_type: 'auc' or 'pauc'
            reference_filename: Reference filename for eval data (DCASE 2022-2025)
            machine_type: Machine type (fan, pump, valve, etc.) for DCASE 2022-2025
        
        Returns:
            Granularity identifier for grouping
        """
        import re
        
        if dataset_year == 2020:
            # DCASE 2020: all files in one group (machine level scoring)
            return "all_machines"
            
        elif dataset_year == 2021:
            # DCASE 2021: machine_type + section + source/target
            # Format: section_XX_source_test_YYYY.wav or section_XX_target_test_YYYY.wav
            match = re.search(r'section_(\d+)_(source|target)_test_(\d+)', filename)
            if match:
                section = match.group(1)
                source_target = match.group(2)
                return f"section_{section}_{source_target}"
            else:
                # Fallback for dev files: section_XX_source_test_anomaly_YYYY.wav
                match = re.search(r'section_(\d+)_(source|target)_test_(anomaly|normal)_(\d+)', filename)
                if match:
                    section = match.group(1)
                    source_target = match.group(2)
                    return f"section_{section}_{source_target}"
                else:
                    return "unknown"
                    
        elif dataset_year in [2022, 2023, 2024, 2025]:
            # DCASE 2022-2025: 
            # AUC: machine_type + section + source/target
            # pAUC: machine_type + section (ignore source/target)
            if metric_type == 'auc':
                # For eval data, extract source/target from reference_filename
                if reference_filename:
                    # Extract section and source/target from reference_filename
                    # Support all formats: section_XX_source/target_...
                    match = re.search(r'section_(\d+)_(source|target)', reference_filename)
                    if match:
                        section = match.group(1)
                        source_target = match.group(2)
                        # Add machine_type prefix if available
                        if machine_type:
                            return f"{machine_type}_section_{section}_{source_target}"
                        else:
                            return f"section_{section}_{source_target}"
                    else:
                        # print(f"reference. Warning: No match found for {reference_filename}")
                        return "unknown"
                else:
                    # Extract section and source/target for dev files
                    match = re.search(r'section_(\d+)_(source|target)_test_(\d+)', filename)
                    if match:
                        section = match.group(1)
                        source_target = match.group(2)
                        # Add machine_type prefix if available
                        if machine_type:
                            return f"{machine_type}_section_{section}_{source_target}"
                        else:
                            return f"section_{section}_{source_target}"
                    else:
                        # Fallback for dev files
                        match = re.search(r'section_(\d+)_(source|target)_test_(anomaly|normal)_(\d+)', filename)
                        if match:
                            section = match.group(1)
                            source_target = match.group(2)
                            # Add machine_type prefix if available
                            if machine_type:
                                return f"{machine_type}_section_{section}_{source_target}"
                            else:
                                return f"section_{section}_{source_target}"
                        else:
                            # print(f"Not reference. Warning: No match found for {filename}")
                            return "unknown"
            else:  # metric_type == 'pauc'
                # Extract only section for pAUC (ignore source/target)
                match = re.search(r'section_(\d+)', filename)
                if match:
                    section = match.group(1)
                    # Add machine_type prefix if available
                    if machine_type:
                        return f"{machine_type}_section_{section}"
                    else:
                        return f"section_{section}"
                else:
                    return "unknown"
        else:
            return "unknown"

    def _regroup_and_score_by_year(self, year: int, results: Dict, dataset_type: str, data_loader: DCASEDataLoader = None) -> Dict:
        """
        Regroup and score by year according to specific scoring logic.
        
        Args:
            year: Dataset year (2020-2025)
            results: Original results dictionary {machine_type: {y_true, y_score, file_paths}}
            dataset_type: 'dev' or 'eval'
            data_loader: Data loader instance for accessing reference_filename info
        
        Returns:
            Regrouped results dictionary
        """
        regrouped_results = {}
        
        # Collect all files and scores with their machine types
        all_files = []
        all_scores = []
        all_labels = []
        all_reference_filenames = []  # For eval data
        all_machine_types = []  # Track machine type for each file
        
        for machine_type, data in results.items():
            file_paths = data['file_paths']
            scores = data['y_score']
            labels = data['y_true']
            
            for i, file_path in enumerate(file_paths):
                filename = os.path.basename(file_path)
                all_files.append(filename)
                all_scores.append(scores[i])
                all_labels.append(labels[i])
                all_machine_types.append(machine_type)  # Store machine type for this file
                
                # For eval data, we need to get reference_filename from the data loader
                if dataset_type == 'eval' and year in [2022, 2023, 2024, 2025]:
                    all_reference_filenames.append(None)
                else:
                    all_reference_filenames.append(None)
        
        # Group by granularity
        if year == 2020:
            # DCASE 2020: all files in one group
            regrouped_results['all_machines'] = {
                'y_true': np.array(all_labels),
                'y_score': np.array(all_scores)
            }
            
        elif year == 2021:
            # DCASE 2021: group by machine_type + section + source/target
            groups = {}
            
            # Keep track of which file belongs to which machine type
            # Use full file path as key to avoid filename conflicts between machine types
            file_to_machine_type = {}
            for machine_type, data in results.items():
                file_paths = data['file_paths']
                for file_path in file_paths:
                    file_to_machine_type[file_path] = machine_type
            
            for i, filename in enumerate(all_files):
                # Get machine type for this file
                machine_type = all_machine_types[i]  # Use stored machine type
                
                # For eval data, get reference_filename if available
                reference_filename = None
                if dataset_type == 'eval' and data_loader:
                    reference_filename = data_loader.get_reference_filename(filename)
                
                granularity = self._extract_granularity_for_scoring(filename, year, 'auc', reference_filename, machine_type)
                if granularity not in groups:
                    groups[granularity] = {'y_true': [], 'y_score': []}
                groups[granularity]['y_true'].append(all_labels[i])
                groups[granularity]['y_score'].append(all_scores[i])
            
            # Convert to numpy arrays
            for granularity, data in groups.items():
                regrouped_results[granularity] = {
                    'y_true': np.array(data['y_true']),
                    'y_score': np.array(data['y_score'])
                }
                
        elif year in [2022, 2023, 2024, 2025]:
            # DCASE 2022-2025: different grouping for AUC and pAUC
            auc_groups = {}
            pauc_groups = {}
            
            # Keep track of which file belongs to which machine type
            # Use full file path as key to avoid filename conflicts between machine types
            file_to_machine_type = {}
            for machine_type, data in results.items():
                file_paths = data['file_paths']
                for file_path in file_paths:
                    file_to_machine_type[file_path] = machine_type
            
            for i, filename in enumerate(all_files):
                # Get machine type for this file
                machine_type = all_machine_types[i]  # Use stored machine type
                
                # For eval data, get reference_filename if available
                reference_filename = None
                if dataset_type == 'eval' and data_loader:
                    reference_filename = data_loader.get_reference_filename(filename)
                
                # Group for AUC
                auc_granularity = self._extract_granularity_for_scoring(filename, year, 'auc', reference_filename, machine_type)
                if auc_granularity not in auc_groups:
                    auc_groups[auc_granularity] = {'y_true': [], 'y_score': []}
                auc_groups[auc_granularity]['y_true'].append(all_labels[i])
                auc_groups[auc_granularity]['y_score'].append(all_scores[i])
                
                # Group for pAUC
                pauc_granularity = self._extract_granularity_for_scoring(filename, year, 'pauc', reference_filename, machine_type)
                if pauc_granularity not in pauc_groups:
                    pauc_groups[pauc_granularity] = {'y_true': [], 'y_score': []}
                pauc_groups[pauc_granularity]['y_true'].append(all_labels[i])
                pauc_groups[pauc_granularity]['y_score'].append(all_scores[i])
            
            # Convert to numpy arrays and add to results
            for granularity, data in auc_groups.items():
                regrouped_results[f"{granularity}_auc"] = {
                    'y_true': np.array(data['y_true']),
                    'y_score': np.array(data['y_score'])
                }
            
            for granularity, data in pauc_groups.items():
                regrouped_results[f"{granularity}_pauc"] = {
                    'y_true': np.array(data['y_true']),
                    'y_score': np.array(data['y_score'])
                }
        
        return regrouped_results

    def evaluate(self, y_true: List, y_pred: List) -> Dict:
        """
        Evaluate DCASE anomaly detection predictions.
        
        Args:
            y_true: Ground truth labels (0 for normal, 1 for anomaly)
            y_pred: Predicted anomaly scores
            
        Returns:
            Dict containing DCASE evaluation metrics (AUC, pAUC, etc.)
        """
        from .evaluator import DCASEEvaluator
        
        # Convert scores to binary predictions if needed
        if len(y_pred) > 0 and not isinstance(y_pred[0], (int, bool)):
            # Assume y_pred contains anomaly scores, convert to binary
            threshold = 0.5  # Default threshold, could be made configurable
            y_pred_binary = [1 if score > threshold else 0 for score in y_pred]
        else:
            y_pred_binary = y_pred
        
        # Use DCASE evaluator for metrics
        evaluator = DCASEEvaluator()
        results = evaluator.evaluate(y_true, y_pred_binary, y_pred)
        
        return results
    
    def log_results(self, results: Dict) -> None:
        """
        Log DCASE evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        print("=== DCASE Evaluation Results ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print("================================")
    
 