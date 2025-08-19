"""
Fault Classification Tester for SIREN framework.

Author: Yucong Zhang
Email: yucong0428@outlook.com

Supports multiple evaluation modes including k-fold cross validation,
train-test split, and leave-one-out cross validation.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import json
import pickle

from .base_tester import BaseTester
from .data_loader.base_data_loader import BaseDataLoader
from .data_loader.mafaulda_data_loader import MAFAULDADataLoader
from .data_loader.cwru_data_loader import CWRUDataLoader
from .data_loader.idmt_compressed_air_data_loader import IDMTCompressedAirDataLoader
from .data_loader.idmt_electric_engine_data_loader import IDMTElectricEngineDataLoader
from .base_extractor import BaseFeatureExtractor
from .memory_bank.classification_memory_bank import ClassificationMemoryBank

# Configure logging
logger = logging.getLogger(__name__)


class FaultClassificationTester(BaseTester):
    """
    Tester for fault classification tasks using KNN algorithm.
    
    This tester implements the complete fault classification pipeline with k-fold optimization:
    1. For k-fold: Extract features for ALL samples once, then reuse across folds
    2. For train-test: Load training/test data and extract features separately  
    3. Store features and labels in memory bank
    4. Use KNN to classify test samples
    5. Evaluate classification performance with comprehensive metrics
    
    Key optimization: In k-fold mode, features are extracted only once for all samples
    and cached, significantly reducing computation time for multiple fold evaluations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the fault classification tester.
        
        Args:
            config: Configuration dictionary containing:
                - dataset_root: Path to dataset
                - dataset_type: Type of dataset (mafaulda, cwru)
                - feature_extractor: BaseFeatureExtractor instance
                - k: Number of neighbors for KNN (default: 5)
                - metric: Distance metric for KNN (default: 'euclidean')
                - use_kfold: Whether to use k-fold cross validation (default: True)
                - n_splits: Number of folds for k-fold cross validation (default: 5)
                - test_size: Test split ratio for train/test split mode (default: 0.2)
                - random_state: Random seed (default: 42)
                - cache_dir: Directory to cache extracted features (default: None)
                - use_cache: Whether to use cached features (default: True)
                - chunk_size: Size of chunks for KNN processing (default: 1000)
        """
        super().__init__(config)
        
        # Configuration
        self.dataset_root = config.get('dataset_root', '')
        self.dataset_type = config.get('dataset_type', 'mafaulda')
        self.feature_extractor = config.get('feature_extractor')
        self.extractor_name = config.get('extractor_name', 'unknown')
        self.k = config.get('k', 5)
        self.metric = config.get('metric', 'euclidean')
        self.use_kfold = config.get('use_kfold', True)
        self.n_splits = config.get('n_splits', 5)
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.chunk_size = config.get('chunk_size', 5000)  # 新增：分块大小配置
        
        # Validate required components
        if not self.dataset_root:
            raise ValueError("dataset_root must be specified in config")
        if not self.feature_extractor:
            raise ValueError("feature_extractor must be specified in config")
        if not isinstance(self.feature_extractor, BaseFeatureExtractor):
            raise ValueError("feature_extractor must be an instance of BaseFeatureExtractor")
        
        # Initialize components
        self.data_loader = self._create_data_loader()
        self.memory_bank = ClassificationMemoryBank()
        
        # Parallel extraction config (file-level, keeps extractor simple)
        # Per-channel KNN outputs flag
        self.return_per_channel_knn = bool(config.get('return_per_channel_knn', False))
        try:
            cpu_cnt = os.cpu_count() or 4
        except Exception:
            cpu_cnt = 4
        self.num_workers = config.get('num_workers', min(8, cpu_cnt))
        
        # Initialize k-fold if needed
        if self.use_kfold:
            self.kfold = StratifiedKFold(
                n_splits=self.n_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
        
        eval_mode = f"k-fold (n_splits={self.n_splits})" if self.use_kfold else f"train-test split (test_size={self.test_size})"
        logger.info(f"Initialized FaultClassificationTester with k={self.k}, metric={self.metric}, chunk_size={self.chunk_size}, eval_mode={eval_mode}")
    
    def _create_data_loader(self) -> BaseDataLoader:
        """
        Create data loader based on dataset type.
        
        Returns:
            BaseDataLoader: Configured data loader instance
        """
        # Prepare dataset-specific config (exclude framework-specific parameters)
        dataset_config = {k: v for k, v in self.config.items() 
                         if k not in ['dataset_type', 'feature_extractor', 'extractor_name']}
        
        # Map dataset types to their corresponding DataLoader classes
        dataset_mapping = {
            'mafaulda': MAFAULDADataLoader,
            'cwru': CWRUDataLoader,  # Unified CWRU loader supporting all sampling rates and channels
            'iica': IDMTCompressedAirDataLoader,
            'iiee': IDMTElectricEngineDataLoader,
        }
        
        if self.dataset_type not in dataset_mapping:
            available_types = list(dataset_mapping.keys())
            raise ValueError(f"Dataset type '{self.dataset_type}' not supported. "
                           f"Available types: {available_types}")
        
        loader_class = dataset_mapping[self.dataset_type]
        data_loader = loader_class(self.dataset_root, dataset_config)
        logger.info(f"Successfully loaded dataset type: {self.dataset_type}")
        
        return data_loader
    
    def run_evaluation(self) -> Dict:
        """
        Run the complete fault classification evaluation.
        
        Returns:
            Dict containing evaluation results
        """
        # Check if LOOCV is requested
        if self.config.get('use_loocv', False):
            return self._run_loocv_evaluation()
        elif self.use_kfold:
            return self._run_kfold_evaluation()
        else:
            return self._run_train_test_evaluation()
    
    def _run_kfold_evaluation(self) -> Dict:
        """
        Run k-fold cross validation evaluation with optimized feature extraction.
        
        Returns:
            Dict containing aggregated evaluation results across all folds
        """
        logger.info(f"Starting {self.n_splits}-fold cross validation evaluation...")
        
        # Get all data files and labels for k-fold split
        all_files, all_labels = self._get_all_data()
        
        # Step 1: Extract features for all files once (optimization!)
        logger.info("Extracting features for all files (this may take a while but only done once)...")
        feature_start_time = time.time()
        all_features = self._extract_all_features_with_cache(all_files)
        feature_extraction_time = time.time() - feature_start_time
        logger.info(f"Feature extraction completed in {feature_extraction_time:.2f} seconds")
        
        # Store results for each fold
        fold_results = []
        fold_predictions = []
        fold_true_labels = []
        
        # Run k-fold cross validation
        kfold_start_time = time.time()
        for fold_idx, (train_idx, test_idx) in enumerate(self.kfold.split(all_files, all_labels)):
            fold_start_time = time.time()
            logger.info(f"Running fold {fold_idx + 1}/{self.n_splits}...")
            
            # Split data indices for this fold
            train_features = [all_features[i] for i in train_idx]
            test_features = [all_features[i] for i in test_idx] 
            train_labels = [all_labels[i] for i in train_idx]
            test_labels = [all_labels[i] for i in test_idx]
            
            # Run evaluation for this fold using pre-extracted features
            fold_result = self._run_single_fold_with_features(
                fold_idx, train_features, test_features, train_labels, test_labels
            )
            
            fold_results.append(fold_result)
            fold_predictions.extend(fold_result['y_pred'])
            fold_true_labels.extend(fold_result['y_true'])
            
            fold_time = time.time() - fold_start_time
            logger.info(f"Fold {fold_idx + 1} accuracy: {fold_result['accuracy']:.4f} (completed in {fold_time:.2f}s)")
        
        total_kfold_time = time.time() - kfold_start_time
        logger.info(f"All {self.n_splits} folds completed in {total_kfold_time:.2f} seconds")
        
        # Aggregate results across all folds
        logger.info("Aggregating results across all folds...")
        aggregated_results = self._aggregate_fold_results(fold_results, fold_predictions, fold_true_labels)
        
        # Log and save aggregated results
        self.log_results(aggregated_results)
        self.save_results(aggregated_results)
        
        return aggregated_results
    
    def _run_train_test_evaluation(self) -> Dict:
        """
        Run traditional train-test split evaluation with feature caching support.
        
        Returns:
            Dict containing evaluation results
        """
        logger.info("Starting train-test split evaluation...")

        # External test path: train on train_cut, test on test (IIEE-specific but generic-safe)
        if self.config.get('use_external_test', False):
            logger.info("External test mode detected: training on train_cut, testing on test")
            # Collect files
            train_files = self.data_loader.get_train_files()
            test_files_with_labels = self.data_loader.get_test_files()
            test_files = [fp for fp, _ in test_files_with_labels]

            # Union list defines extraction order (cache path already contains _external suffix)
            all_files = list(train_files) + list(test_files)
            logger.info("Extracting features for train_cut + test (with cache support)...")
            all_features = self._extract_all_features_with_cache(all_files)

            # Split back
            num_train = len(train_files)
            train_features = all_features[:num_train]
            test_features = all_features[num_train:]

            # Labels
            label_to_int = {c: i for i, c in enumerate(self.data_loader.get_classes())}
            train_labels = []
            for f in train_files:
                cls = getattr(self.data_loader, '_extract_class_from_path')(f)
                if cls in label_to_int:
                    train_labels.append(label_to_int[cls])
            y_true = [lbl for _, lbl in test_files_with_labels]
        else:
            # Step 1: Get all files in the same order as k-fold mode (for cache compatibility)
            all_files, all_labels = self._get_all_data()

            # Get train/test split from data loader
            train_files = self.data_loader.get_train_files()
            test_files_with_labels = self.data_loader.get_test_files()
            test_files = [item[0] if isinstance(item, tuple) else item for item in test_files_with_labels]

            # Create mapping for efficient file index lookup
            file_to_index = {file_path: idx for idx, file_path in enumerate(all_files)}
            train_indices = [file_to_index[f] for f in train_files if f in file_to_index]
            test_indices = [file_to_index[f] for f in test_files if f in file_to_index]

            # Extract features for all files (with caching support)
            logger.info("Extracting features for all files (with cache support)...")
            all_features = self._extract_all_features_with_cache(all_files)

            # Split features back into train and test using indices
            train_features = [all_features[i] for i in train_indices]
            test_features = [all_features[i] for i in test_indices]

            # Get labels using indices (all_labels is already integer labels)
            train_labels = [all_labels[i] for i in train_indices]
            y_true = [lbl for _, lbl in test_files_with_labels]
        
        # Step 2: Store features in memory bank
        logger.info("Storing features in memory bank...")
        self.memory_bank.clear()
        # Standardize shapes to [channels, feature_dim]
        train_features = [self._standardize_feature_shape(f) for f in train_features]
        test_features = [self._standardize_feature_shape(f) for f in test_features]
        self.memory_bank.batch_add(train_features, train_labels)
        
        # Step 3: Get test labels (prepared above)
        logger.info("Preparing test labels...")
        
        # Step 4: Classify test samples using KNN
        logger.info("Classifying test samples using KNN...")
        y_pred = []
        for feature in test_features:
            # Get K nearest neighbors with unified multi-channel handling
            nearest_labels, distances = self.memory_bank.knn_query(
                self._standardize_feature_shape(feature), k=self.k, metric=self.metric
            )
            
            # Simple majority voting
            from collections import Counter
            label_counts = Counter(nearest_labels)
            predicted_label = label_counts.most_common(1)[0][0]
            y_pred.append(predicted_label)
        
        # Step 5: Evaluate results
        logger.info("Evaluating classification results...")
        results = self.evaluate(y_true, y_pred)
        
        # Step 6: Log results
        self.log_results(results)
        
        # Step 7: Save results to file
        self.save_results(results)
        
        # Optional: group by condition (IIEE external test)
        if self.config.get('group_by_condition', False) and hasattr(self.data_loader, 'get_test_conditions'):
            try:
                condition_map = self.data_loader.get_test_conditions()
                # Build per-condition metrics
                cond_to_indices = {}
                for idx, f in enumerate(test_files):
                    cond = condition_map.get(f, 'unknown')
                    cond_to_indices.setdefault(cond, []).append(idx)
                per_condition = {}
                for cond, idxs in cond_to_indices.items():
                    y_true_c = [y_true[i] for i in idxs]
                    y_pred_c = [y_pred[i] for i in idxs]
                    metrics_c = self.evaluate(y_true_c, y_pred_c)
                    per_condition[cond] = {
                        'accuracy': metrics_c['accuracy'],
                        'precision': metrics_c['precision'],
                        'recall': metrics_c['recall'],
                        'f1_score': metrics_c['f1_score'],
                        'support': metrics_c['support'],
                    }
                results['per_condition'] = per_condition
                # Save CSV summary
                import pandas as pd
                rows = []
                for cond, m in per_condition.items():
                    rows.append({
                        'condition': cond,
                        'accuracy': m['accuracy'],
                        'precision': m['precision'],
                        'recall': m['recall'],
                        'f1_score': m['f1_score'],
                        'support': m['support']
                    })
                if rows:
                    extractor_name = self.extractor_name
                    dataset_name = self.data_loader._get_dataset_info().name
                    results_dir = os.path.join('test_results', extractor_name, dataset_name)
                    os.makedirs(results_dir, exist_ok=True)
                    pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'per_condition_summary.csv'), index=False)
                # Save updated results (with per_condition) again
                self.save_results(results)
            except Exception as e:
                logger.warning(f"Failed to compute per-condition summary: {e}")

        return results
    
    def _run_loocv_evaluation(self) -> Dict:
        """
        Run Leave-One-Out Cross Validation evaluation.
        Ideal for small datasets where every sample is valuable for testing.
        
        Returns:
            Dict containing LOOCV evaluation results
        """
        logger.info("Starting Leave-One-Out Cross Validation (LOOCV) evaluation...")
        
        # Get all data files and labels
        all_files, all_labels = self._get_all_data()
        n_samples = len(all_files)
        
        logger.info(f"Running LOOCV with {n_samples} samples (each sample tested once)")
        
        # Extract features for all files once (optimization!)
        logger.info("Extracting features for all files (this may take a while but only done once)...")
        feature_start_time = time.time()
        all_features = self._extract_all_features_with_cache(all_files)
        feature_extraction_time = time.time() - feature_start_time
        logger.info(f"Feature extraction completed in {feature_extraction_time:.2f} seconds")
        
        # Vectorized LOOCV in one shot using batch KNN with leave-one-out masking
        logger.info("Preparing batch structures for vectorized LOOCV...")
        loocv_start_time = time.time()
        # print("DEBUG | shape of the extracted features:", all_features[0].shape)
        # Standardize to [C,D] and stack to [B,C,D]
        features_std = [self._standardize_feature_shape(f) for f in all_features]
        X = np.stack([f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in features_std])  # [B,C,D]
        # Build a single memory bank containing all samples
        vector_mb = ClassificationMemoryBank()
        vector_mb.batch_add(features_std, all_labels)
        # Batch KNN with leave-one-out (diagonal masked)
        from collections import Counter
        if self.return_per_channel_knn:
            # 使用分块KNN查询，控制内存使用
            logger.info(f"Running chunked KNN query with chunk_size={self.chunk_size} for LOOCV ({n_samples} samples)")
            topk_labels, _topk_dists, topk_labels_ch, _topk_dists_ch = vector_mb.knn_query_batch_chunked(
                X, k=self.k, metric=self.metric, leave_one_out=True, 
                return_per_channel=True, chunk_size=self.chunk_size
            )
        else:
            # 使用分块KNN查询，控制内存使用
            logger.info(f"Running chunked KNN query with chunk_size={self.chunk_size} for LOOCV ({n_samples} samples)")
            topk_labels, _topk_dists = vector_mb.knn_query_batch_chunked(
                X, k=self.k, metric=self.metric, leave_one_out=True, 
                chunk_size=self.chunk_size
            )
            topk_labels_ch = None
        all_predictions = [Counter(lbls).most_common(1)[0][0] for lbls in topk_labels]
        all_true_labels = all_labels
        # If requested, compute per-channel predictions and summary metrics
        per_channel_summary = None
        if self.return_per_channel_knn and topk_labels_ch is not None:
            # topk_labels_ch: [B, C, k] -> per-channel predictions by majority vote over k
            B = len(all_true_labels)
            C = len(topk_labels_ch[0]) if B > 0 else 0
            per_channel_preds: List[List[int]] = [[] for _ in range(C)]
            for i in range(B):
                for c in range(C):
                    votes = Counter(topk_labels_ch[i][c])
                    pred_c = votes.most_common(1)[0][0]
                    per_channel_preds[c].append(pred_c)
            # Build per-channel summary with metrics
            per_channel_summary = []
            try:
                channel_names = getattr(self.data_loader, 'get_channel_names', None)
                channel_names = channel_names() if callable(channel_names) else None
            except Exception:
                channel_names = None
            for c in range(C):
                y_pred_c = per_channel_preds[c]
                acc_c = accuracy_score(all_true_labels, y_pred_c)
                prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
                    all_true_labels, y_pred_c, average='weighted', zero_division=0
                )
                per_channel_summary.append({
                    'channel_index': c,
                    'channel_name': channel_names[c] if channel_names and c < len(channel_names) else f'channel-{c}',
                    'accuracy': float(acc_c),
                    'precision': float(prec_c),
                    'recall': float(rec_c),
                    'f1_score': float(f1_c),
                })
        # Compose per-iteration summary (no distances to save big memory/time)
        iteration_results = []
        for i in range(n_samples):
            item = {
                'iteration': i,
                'test_file': all_files[i],
                'true_label': all_labels[i],
                'predicted_label': all_predictions[i],
                'correct': all_predictions[i] == all_labels[i],
                'num_train': n_samples - 1,
            }
            if self.return_per_channel_knn and topk_labels_ch is not None:
                item['per_channel_topk_labels'] = topk_labels_ch[i]  # [C][k]
            iteration_results.append(item)
        total_loocv_time = time.time() - loocv_start_time
        logger.info(f"LOOCV (vectorized) completed in {total_loocv_time:.2f} seconds")
        
        # Calculate final metrics
        logger.info("Calculating final LOOCV metrics...")
        final_results = self.evaluate(all_true_labels, all_predictions)
        
        # Add LOOCV-specific information
        final_results.update({
            'evaluation_mode': 'loocv',
            'n_samples': n_samples,
            'iteration_results': iteration_results,
            'per_channel_summary': per_channel_summary,
            'loocv_statistics': {
                'total_iterations': n_samples,
                'correct_predictions': sum(r['correct'] for r in iteration_results),
                'final_accuracy': sum(r['correct'] for r in iteration_results) / n_samples,
                'feature_extraction_time': feature_extraction_time,
                'total_evaluation_time': total_loocv_time
            },
            'config': {
                'k': self.k,
                'metric': self.metric,
                'use_loocv': True,
                'random_state': self.random_state
            }
        })
        
        # Log and save results
        self.log_results(final_results)
        self.save_results(final_results)
        
        return final_results
    
    def _get_all_data(self) -> Tuple[List[str], List[int]]:
        """
        Get all data files and labels for k-fold split.
        
        Returns:
            Tuple of (all_files, all_labels) where labels are integers
        """
        # Try new dynamic sampling rate method first
        if hasattr(self.data_loader, 'get_files_with_sampling_rates'):
            files_with_rates = self.data_loader.get_files_with_sampling_rates()
            all_files = [item[0] for item in files_with_rates]  # file_path
            all_labels_str = [item[2] for item in files_with_rates]  # class_label
            # Store sampling rates for later use
            self._file_sampling_rates = {item[0]: item[1] for item in files_with_rates}
        else:
            # Fallback to legacy method
            if hasattr(self.data_loader, '_get_all_files'):
                all_files = self.data_loader._get_all_files()
            elif hasattr(self.data_loader, '_get_all_csv_files'):
                # Fallback for MAFAULDA compatibility
                all_files = self.data_loader._get_all_csv_files()
            else:
                raise AttributeError(f"Data loader {type(self.data_loader)} must implement '_get_all_files()' or 'get_files_with_sampling_rates()' method")
            
            all_labels_str = [self.data_loader._extract_class_from_path(f) for f in all_files]
            # Use fixed sampling rate for legacy loaders
            fixed_rate = self.data_loader.get_sample_rate()
            self._file_sampling_rates = {f: fixed_rate for f in all_files}
        
        # Filter out unknown classes
        valid_indices = [i for i, label in enumerate(all_labels_str) if label != 'unknown']
        valid_files = [all_files[i] for i in valid_indices]
        valid_labels_str = [all_labels_str[i] for i in valid_indices]
        
        # Convert string labels to integers
        label_to_int = {cls: i for i, cls in enumerate(self.data_loader.get_classes())}
        valid_labels = [label_to_int[label] for label in valid_labels_str]
        
        logger.info(f"Found {len(valid_files)} valid files for k-fold evaluation")
        logger.info(f"Class distribution: {dict(zip(*np.unique(valid_labels_str, return_counts=True)))}")
        
        return valid_files, valid_labels
    
    def _extract_all_features_with_cache(self, all_files: List[str]) -> List[torch.Tensor]:
        """
        Extract features for all files with caching support.
        
        Args:
            all_files: List of all file paths
            
        Returns:
            List of extracted features corresponding to all_files
        """
        # Try to load cached features
        kfold_cache_path = self._get_features_cache_path()
        
        if kfold_cache_path and os.path.exists(kfold_cache_path):
            logger.info(f"Loading cached k-fold features from {kfold_cache_path}")
            cached_data = np.load(kfold_cache_path, allow_pickle=True)
            cached_files = list(cached_data['files'])
            
            # Handle both old format (features array) and new format (individual features)
            if 'features' in cached_data:
                # Old format: features are in a single array
                cached_features = [torch.tensor(f) for f in cached_data['features']]
            else:
                # New format: features are stored individually
                cached_features = []
                for i in range(len(cached_files)):
                    feature_key = f'feature_{i}'
                    if feature_key in cached_data:
                        cached_features.append(torch.tensor(cached_data[feature_key]))
            
            # Check if cached files match current files
            if cached_files == all_files and len(cached_features) == len(all_files):
                # Standardize feature shapes to [channels, feature_dim]
                standardized = [self._standardize_feature_shape(f) for f in cached_features]
                logger.info(f"Cache hit! Loaded {len(standardized)} pre-extracted features")
                return standardized
            else:
                logger.info("Cache miss: file list has changed, will re-extract features")
        
        # Extract features for all files using dynamic sampling rates (parallel)
        logger.info(f"Extracting features for {len(all_files)} files with {self.num_workers} workers...")
        all_features = self._extract_features_with_dynamic_rates_parallel(all_files, max_workers=self.num_workers)
        # Standardize shapes immediately
        all_features = [self._standardize_feature_shape(f) for f in all_features]
        
        # Save features to cache for future use
        if kfold_cache_path:
            logger.info(f"Saving features to cache: {kfold_cache_path}")
            try:
                features_np = [f.cpu().numpy() for f in all_features]
                np.savez(kfold_cache_path, files=all_files, features=features_np)
            except ValueError as e:
                if "inhomogeneous shape" in str(e):
                    logger.warning(f"Features have inconsistent shapes, saving individually: {e}")
                    # Save features individually to handle inconsistent shapes
                    features_dict = {f'feature_{i}': f.cpu().numpy() for i, f in enumerate(all_features)}
                    features_dict['files'] = all_files
                    np.savez(kfold_cache_path, **features_dict)
                else:
                    raise e
        
        return all_features

    def _extract_features_with_dynamic_rates_parallel(self, files: List[str], max_workers: int = 4) -> List[torch.Tensor]:
        """
        Parallel file-level feature extraction using a thread pool to reuse the already-loaded extractor.
        Preserves file order in the returned list.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        features: List[Optional[torch.Tensor]] = [None] * len(files)
        
        # Build sampling rate mapping if available
        file_to_sr = getattr(self, '_file_sampling_rates', {})
        default_sr = self.data_loader.get_sample_rate()
        
        def _task(idx: int, file_path: str):
            sr = getattr(self, '_file_sampling_rates', {}).get(file_path, self.data_loader.get_sample_rate())
            feat = self.feature_extractor.extract_features(file_path, sample_rate=sr)
            # print(feat.shape, sr)
            return idx, feat
        
        # Thread pool keeps single CUDA context; adjust workers if needed
        max_workers = max(1, int(max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_task, i, fp) for i, fp in enumerate(files)]
            for fut in as_completed(futures):
                idx, feat = fut.result()
                features[idx] = feat
        
        # Replace any None with zeros to be safe (should not happen unless exception suppressed)
        for i, f in enumerate(features):
            if f is None:
                logger.warning(f"Feature for index {i} is None, filling zeros")
                features[i] = torch.zeros(self.feature_extractor.feature_dim)
        
        return features  # type: ignore

    def _standardize_feature_shape(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Ensure feature shape is [channels, feature_dim].
        - If 1D: reshape to [expected_channels, feature_dim_per_channel] when divisible; else [1, -1]
        - If 2D: return as-is
        """
        # Convert numpy to torch if needed
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature)
        if feature.dim() == 2:
            return feature
        if feature.dim() == 1:
            total_dim = feature.shape[0]
            expected_channels = getattr(self.feature_extractor, 'expected_channels', 1)
            if expected_channels > 1 and total_dim % expected_channels == 0:
                per_channel_dim = total_dim // expected_channels
                return feature.view(expected_channels, per_channel_dim)
            else:
                return feature.view(1, total_dim)
        raise ValueError(f"Unsupported feature tensor dimension: {feature.dim()}")
    
    def _get_features_cache_path(self) -> str:
        """
        Get cache path for storing all extracted features.
        Will try to find existing cache files with different naming conventions.
        
        Returns:
            str: Features cache file path
        """
        extractor_name = self.extractor_name
        dataset_name = self.data_loader._get_dataset_info().name
        suffix = "_external" if self.config.get('use_external_test', False) else ""
        task_name = "fault_classification"
        
        # Create feature cache directory
        cache_dir = os.path.join('feature_cache', extractor_name, dataset_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try legacy kfold cache first (for backward compatibility)
        legacy_cache_path = os.path.join(cache_dir, f"{task_name}_{dataset_name}_kfold_all_features.npz")
        if os.path.exists(legacy_cache_path):
            return legacy_cache_path
        
        # Return new cache path
        return os.path.join(cache_dir, f"{task_name}_{dataset_name}{suffix}_all_features.npz")
    
    def _run_single_fold_with_features(self, fold_idx: int, train_features: List[torch.Tensor], 
                                     test_features: List[torch.Tensor], train_labels: List[int], 
                                     test_labels: List[int]) -> Dict:
        """
        Run evaluation for a single fold using pre-extracted features.
        
        Args:
            fold_idx: Fold index
            train_features: Pre-extracted training features
            test_features: Pre-extracted test features
            train_labels: Training labels (integers)
            test_labels: Test labels (integers)
            
        Returns:
            Dict containing fold evaluation results
        """
        # Set up memory bank for this fold
        # Standardize all train/test feature shapes to [channels, feature_dim]
        train_features = [self._standardize_feature_shape(f) for f in train_features]
        test_features = [self._standardize_feature_shape(f) for f in test_features]
        fold_memory_bank = ClassificationMemoryBank()
        fold_memory_bank.batch_add(train_features, train_labels)
        
        # Classify test samples using batch KNN (vectorized)
        # Prepare batch tensor [B,C,D]
        import numpy as np
        from collections import Counter
        batch_array = []
        for feature in test_features:
            feat_std = self._standardize_feature_shape(feature)
            # Convert to numpy and ensure shape [C,D]
            if isinstance(feat_std, torch.Tensor):
                feat_np = feat_std.cpu().numpy()
            else:
                feat_np = np.array(feat_std)
            batch_array.append(feat_np)
        X = np.stack(batch_array, axis=0)  # [B,C,D]

        # 使用分块KNN查询，控制内存使用
        logger.info(f"Running chunked KNN query with chunk_size={self.chunk_size} for {len(test_features)} test samples")
        topk_labels, _topk_dists = fold_memory_bank.knn_query_batch_chunked(
            X, k=self.k, metric=self.metric, chunk_size=self.chunk_size
        )
        # Majority vote per row
        y_pred = [Counter(row).most_common(1)[0][0] for row in topk_labels]
        
        # Evaluate results for this fold
        fold_results = self.evaluate(test_labels, y_pred)
        fold_results['fold_idx'] = fold_idx
        fold_results['y_true'] = test_labels
        fold_results['y_pred'] = y_pred
        fold_results['num_train'] = len(train_features)
        fold_results['num_test'] = len(test_features)
        
        return fold_results
    
    def _convert_to_json_serializable(self, obj):
        """
        Recursively convert numpy types to JSON serializable Python types.
        
        Args:
            obj: Object to convert (can be dict, list, numpy types, etc.)
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj
    
    def _aggregate_fold_results(self, fold_results: List[Dict], all_predictions: List[int], all_true_labels: List[int]) -> Dict:
        """
        Aggregate results across all folds.
        
        Args:
            fold_results: List of results from each fold
            all_predictions: All predictions across folds
            all_true_labels: All true labels across folds
            
        Returns:
            Dict containing aggregated results
        """
        # Calculate overall metrics using all predictions
        overall_results = self.evaluate(all_true_labels, all_predictions)
        
        # Calculate statistics across folds
        fold_accuracies = [fold['accuracy'] for fold in fold_results]
        fold_precisions = [fold['precision'] for fold in fold_results]
        fold_recalls = [fold['recall'] for fold in fold_results]
        fold_f1s = [fold['f1_score'] for fold in fold_results]
        
        # Add k-fold specific metrics
        overall_results.update({
            'evaluation_mode': 'k-fold',
            'n_splits': self.n_splits,
            'fold_results': fold_results,
            'fold_statistics': {
                'accuracy_mean': np.mean(fold_accuracies),
                'accuracy_std': np.std(fold_accuracies),
                'accuracy_min': np.min(fold_accuracies),
                'accuracy_max': np.max(fold_accuracies),
                'precision_mean': np.mean(fold_precisions),
                'precision_std': np.std(fold_precisions),
                'recall_mean': np.mean(fold_recalls),
                'recall_std': np.std(fold_recalls),
                'f1_mean': np.mean(fold_f1s),
                'f1_std': np.std(fold_f1s)
            },
            'config': {
                'k': self.k,
                'metric': self.metric,
                'n_splits': self.n_splits,
                'random_state': self.random_state,
                'use_kfold': self.use_kfold
            }
        })
        
        return overall_results
    
    def _get_memory_bank_path(self) -> str:
        """
        Get memory bank path based on extractor and dataset.
        
        Returns:
            str: Memory bank file path
        """
        extractor_name = self.extractor_name
        dataset_name = self.data_loader._get_dataset_info().name
        suffix = "_external" if self.config.get('use_external_test', False) else ""
        task_name = "fault_classification"
        
        # Create feature cache directory
        cache_dir = os.path.join('feature_cache', extractor_name, dataset_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        return os.path.join(cache_dir, f"{task_name}_{dataset_name}{suffix}_memory_bank.npz")
    
    def extract_features(self, files: List[str], sample_rate: int = 16000) -> List[torch.Tensor]:
        """
        Extract features from files using the configured feature extractor.
        
        Args:
            files: List of file paths
            sample_rate: Sample rate of the audio files
            
        Returns:
            List of extracted features
        """
        logger.info(f"Extracting features from {len(files)} files...")
        
        # Use the feature extractor's batch processing (all format handling is now in BaseFeatureExtractor)
        return self.feature_extractor.extract_features_batch(files, sample_rate=sample_rate)
    
    def extract_features_with_dynamic_rates(self, files: List[str]) -> List[torch.Tensor]:
        """
        Extract features from files using dynamic sampling rates.
        
        Args:
            files: List of file paths
            
        Returns:
            List of extracted features
        """
        from tqdm import tqdm
        
        logger.info(f"Extracting features from {len(files)} files with dynamic sampling rates...")
        
        features = []
        for file_path in tqdm(files, desc="Extracting features", unit="file", ncols=100):
            try:
                # Get the specific sampling rate for this file
                sample_rate = getattr(self, '_file_sampling_rates', {}).get(file_path, self.data_loader.get_sample_rate())
                
                # Extract features with file-specific sampling rate
                feature = self.feature_extractor.extract_features(file_path, sample_rate=sample_rate)
                features.append(feature)
            except Exception as e:
                logger.error(f"Failed to extract features from {file_path}: {e}")
                raise RuntimeError(f"Failed to extract features from {file_path}: {e}")
        
        return features
    
    def evaluate(self, y_true: List, y_pred: List) -> Dict:
        """
        Evaluate classification predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dict containing evaluation metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Create confusion matrix
        classes = self.data_loader.get_classes()
        # Convert integer labels back to strings for confusion matrix
        int_to_label = {i: cls for i, cls in enumerate(classes)}
        y_true_str = [int_to_label[label] for label in y_true]
        y_pred_str = [int_to_label[label] for label in y_pred]
        
        cm = confusion_matrix(y_true_str, y_pred_str, labels=classes)
        
        # Generate classification report
        report = classification_report(y_true_str, y_pred_str, labels=classes, output_dict=True, zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'precision_per_class': dict(zip(classes, precision_per_class)),
            'recall_per_class': dict(zip(classes, recall_per_class)),
            'f1_per_class': dict(zip(classes, f1_per_class)),
            'support_per_class': dict(zip(classes, support_per_class)),
            'confusion_matrix': cm.tolist(),
            'classes': classes,
            'classification_report': report,
            'config': {
                'k': self.k,
                'metric': self.metric,
                'test_size': self.test_size,
                'random_state': self.random_state
            }
        }
        
        return results
    
    def log_results(self, results: Dict) -> None:
        """
        Log and output evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        logger.info("=" * 60)
        logger.info("FAULT CLASSIFICATION EVALUATION RESULTS")
        logger.info("=" * 60)
        
        # Check evaluation mode and log accordingly
        eval_mode = results.get('evaluation_mode')
        if eval_mode == 'k-fold':
            self._log_kfold_results(results)
        elif eval_mode == 'loocv':
            self._log_loocv_results(results)
        else:
            self._log_standard_results(results)
        
        logger.info("=" * 60)
    
    def _log_standard_results(self, results: Dict) -> None:
        """Log results for standard train-test split."""
        # Overall metrics
        logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Overall Precision: {results['precision']:.4f}")
        logger.info(f"Overall Recall: {results['recall']:.4f}")
        logger.info(f"Overall F1-Score: {results['f1_score']:.4f}")
        logger.info(f"Total Support: {results['support']}")
        
        # Per-class metrics
        logger.info("\nPer-Class Metrics:")
        logger.info("-" * 60)
        for class_name in results['classes']:
            precision = results['precision_per_class'][class_name]
            recall = results['recall_per_class'][class_name]
            f1 = results['f1_per_class'][class_name]
            support = results['support_per_class'][class_name]
            logger.info(f"{class_name:20s}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
        
        # Configuration
        logger.info(f"\nConfiguration:")
        logger.info(f"K: {results['config']['k']}")
        logger.info(f"Metric: {results['config']['metric']}")
        logger.info(f"Test Size: {results['config']['test_size']}")
        logger.info(f"Random State: {results['config']['random_state']}")
        
    def _log_kfold_results(self, results: Dict) -> None:
        """Log results for k-fold cross validation."""
        # Overall metrics (aggregated across all folds)
        logger.info(f"K-Fold Cross Validation Results (n_splits={results['n_splits']}):")
        logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Overall Precision: {results['precision']:.4f}")
        logger.info(f"Overall Recall: {results['recall']:.4f}")
        logger.info(f"Overall F1-Score: {results['f1_score']:.4f}")
        logger.info(f"Total Support: {results['support']}")
        
        # Fold statistics
        fold_stats = results['fold_statistics']
        logger.info(f"\nFold Statistics:")
        logger.info(f"Accuracy:  {fold_stats['accuracy_mean']:.4f} ± {fold_stats['accuracy_std']:.4f} (min: {fold_stats['accuracy_min']:.4f}, max: {fold_stats['accuracy_max']:.4f})")
        logger.info(f"Precision: {fold_stats['precision_mean']:.4f} ± {fold_stats['precision_std']:.4f}")
        logger.info(f"Recall:    {fold_stats['recall_mean']:.4f} ± {fold_stats['recall_std']:.4f}")
        logger.info(f"F1-Score:  {fold_stats['f1_mean']:.4f} ± {fold_stats['f1_std']:.4f}")
        
        # Individual fold results
        logger.info(f"\nIndividual Fold Results:")
        logger.info("-" * 80)
        logger.info(f"{'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Train':<8} {'Test':<8}")
        logger.info("-" * 80)
        for fold_result in results['fold_results']:
            fold_idx = fold_result['fold_idx'] + 1
            acc = fold_result['accuracy']
            prec = fold_result['precision']
            rec = fold_result['recall']
            f1 = fold_result['f1_score']
            num_train = fold_result['num_train']
            num_test = fold_result['num_test']
            logger.info(f"{fold_idx:<6} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {num_train:<8} {num_test:<8}")
        
        # Per-class metrics (aggregated)
        logger.info("\nPer-Class Metrics (Aggregated):")
        logger.info("-" * 60)
        for class_name in results['classes']:
            precision = results['precision_per_class'][class_name]
            recall = results['recall_per_class'][class_name]
            f1 = results['f1_per_class'][class_name]
            support = results['support_per_class'][class_name]
            logger.info(f"{class_name:20s}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
        
        # Configuration
        logger.info(f"\nConfiguration:")
        logger.info(f"K: {results['config']['k']}")
        logger.info(f"Metric: {results['config']['metric']}")
        logger.info(f"N Splits: {results['config']['n_splits']}")
        logger.info(f"Random State: {results['config']['random_state']}")
        logger.info(f"Use K-Fold: {results['config']['use_kfold']}")
    
    def _log_loocv_results(self, results: Dict) -> None:
        """Log results for Leave-One-Out Cross Validation."""
        loocv_stats = results['loocv_statistics']
        
        # Overall metrics
        logger.info(f"Leave-One-Out Cross Validation Results:")
        logger.info(f"Total Samples: {results['n_samples']}")
        logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Overall Precision: {results['precision']:.4f}")
        logger.info(f"Overall Recall: {results['recall']:.4f}")
        logger.info(f"Overall F1-Score: {results['f1_score']:.4f}")
        logger.info(f"Total Support: {results['support']}")
        
        # LOOCV statistics
        logger.info(f"\nLOOCV Statistics:")
        logger.info(f"Total Iterations: {loocv_stats['total_iterations']}")
        logger.info(f"Correct Predictions: {loocv_stats['correct_predictions']}")
        logger.info(f"Final Accuracy: {loocv_stats['final_accuracy']:.4f}")
        logger.info(f"Feature Extraction Time: {loocv_stats['feature_extraction_time']:.2f}s")
        logger.info(f"Total Evaluation Time: {loocv_stats['total_evaluation_time']:.2f}s")
        
        # Per-class metrics
        logger.info("\nPer-Class Metrics:")
        logger.info("-" * 60)
        for class_name in results['classes']:
            if class_name in results['precision_per_class']:
                precision = results['precision_per_class'][class_name]
                recall = results['recall_per_class'][class_name]
                f1 = results['f1_per_class'][class_name]
                support = results['support_per_class'][class_name]
                logger.info(f"{class_name:20s}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
            else:
                logger.info(f"{class_name:20s}: No samples in test set")

        # Optional per-channel summary (when --per_channel_knn enabled in LOOCV)
        if results.get('per_channel_summary'):
            logger.info("\nPer-Channel Summary:")
            logger.info("-" * 60)
            for ch in results['per_channel_summary']:
                logger.info(
                    f"[{ch['channel_index']}] {ch.get('channel_name','channel')}: "
                    f"Acc={ch['accuracy']:.4f}, Prec={ch['precision']:.4f}, "
                    f"Rec={ch['recall']:.4f}, F1={ch['f1_score']:.4f}"
                )
        
        # Show some example misclassifications (first 5)
        misclassified = [r for r in results['iteration_results'] if not r['correct']]
        if misclassified:
            logger.info(f"\nMisclassification Examples (showing first 5 of {len(misclassified)}):")
            logger.info("-" * 80)
            for i, miss in enumerate(misclassified[:5]):
                file_name = os.path.basename(miss['test_file'])
                true_class = results['classes'][miss['true_label']]
                pred_class = results['classes'][miss['predicted_label']]
                logger.info(f"  {i+1}. {file_name}: True={true_class}, Predicted={pred_class}")
        
        # Configuration
        logger.info(f"\nConfiguration:")
        logger.info(f"K: {results['config']['k']}")
        logger.info(f"Metric: {results['config']['metric']}")
        logger.info(f"Use LOOCV: {results['config']['use_loocv']}")
        logger.info(f"Random State: {results['config']['random_state']}")
    
    def get_dataset_stats(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dict containing dataset statistics
        """
        return self.data_loader.get_dataset_stats()
    
    def save_memory_bank(self, path: str) -> None:
        """
        Save the memory bank to disk.
        
        Args:
            path: Path to save the memory bank
        """
        self.memory_bank.save(path)
        logger.info(f"Memory bank saved to {path}")
    
    def load_memory_bank(self, path: str) -> None:
        """
        Load the memory bank from disk.
        
        Args:
            path: Path to load the memory bank from
        """
        self.memory_bank.load(path)
        logger.info(f"Memory bank loaded from {path}")
    
    def save_results(self, results: Dict) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
        """
        import json
        import os
        from datetime import datetime
        
        # Create results directory structure: test_results/{extractor_name}/{dataset_name}/
        extractor_name = self.extractor_name
        dataset_name = self.data_loader._get_dataset_info().name
        results_dir = os.path.join('test_results', extractor_name, dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename without timestamp (user request)
        task_name = "fault_classification"  # Task-specific name
        suffix = "_external" if self.config.get('use_external_test', False) else ""
        results_file = os.path.join(results_dir, f"{task_name}_{dataset_name}{suffix}.json")
        
        # Prepare results for saving (convert numpy arrays to lists)
        save_results = self._convert_to_json_serializable(results)
        
        # Add metadata
        metadata = {
            'dataset': dataset_name,
            'feature_extractor': self.feature_extractor.__class__.__name__,
            'k': self.k,
            'metric': self.metric,
            'random_state': self.random_state,
            'feature_dim': getattr(self.feature_extractor, '_feature_dim', 'unknown')
        }
        
        # Add evaluation mode specific metadata
        if results.get('evaluation_mode') == 'k-fold':
            metadata.update({
                'evaluation_mode': 'k-fold',
                'n_splits': self.n_splits,
                'use_kfold': self.use_kfold,
                'total_files': sum(fold['num_train'] + fold['num_test'] for fold in results.get('fold_results', [])),
                'avg_train_files_per_fold': float(np.mean([fold['num_train'] for fold in results.get('fold_results', [])])),
                'avg_test_files_per_fold': float(np.mean([fold['num_test'] for fold in results.get('fold_results', [])])),
                'optimization_note': 'Features extracted once and reused across all folds'
            })
        else:
            metadata.update({
                'evaluation_mode': 'train-test',
                'test_size': self.test_size,
                'num_train_files': len(self.memory_bank.features),
                'num_test_files': len(results.get('y_true', []))
            })
        
        # Convert metadata to JSON serializable format
        save_results['metadata'] = self._convert_to_json_serializable(metadata)
        
        # Save to JSON file
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(save_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {results_file}")
        except TypeError as e:
            logger.warning(f"Failed to save JSON results: {e}")
            # Fallback: save only basic metrics
            basic_results = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'classes': results['classes'],
                'metadata': self._convert_to_json_serializable(metadata)
            }
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(basic_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Basic results saved to {results_file}")
        
        logger.info(f"Results saved to {results_file}")
        
        # Also save a summary text file
        summary_file = os.path.join(results_dir, f"{task_name}_{dataset_name}{suffix}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("FAULT CLASSIFICATION EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Feature Extractor: {self.feature_extractor.__class__.__name__}\n")
            f.write(f"K: {self.k}\n")
            f.write(f"Metric: {self.metric}\n")
            f.write(f"Random State: {self.random_state}\n")
            f.write(f"Number of Classes: {len(results['classes'])}\n")
            f.write(f"Classes: {results['classes']}\n")
            
            # Add evaluation mode specific info
            if results.get('evaluation_mode') == 'k-fold':
                f.write(f"Evaluation Mode: K-Fold Cross Validation\n")
                f.write(f"N Splits: {self.n_splits}\n")
                fold_stats = results['fold_statistics']
                f.write(f"Total Files: {save_results['metadata']['total_files']}\n")
                f.write(f"Avg Train Files per Fold: {save_results['metadata']['avg_train_files_per_fold']:.1f}\n")
                f.write(f"Avg Test Files per Fold: {save_results['metadata']['avg_test_files_per_fold']:.1f}\n\n")
                
                f.write("OVERALL METRICS (Aggregated across all folds):\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"F1-Score: {results['f1_score']:.4f}\n\n")
                
                f.write("FOLD STATISTICS:\n")
                f.write(f"Accuracy:  {fold_stats['accuracy_mean']:.4f} ± {fold_stats['accuracy_std']:.4f} (min: {fold_stats['accuracy_min']:.4f}, max: {fold_stats['accuracy_max']:.4f})\n")
                f.write(f"Precision: {fold_stats['precision_mean']:.4f} ± {fold_stats['precision_std']:.4f}\n")
                f.write(f"Recall:    {fold_stats['recall_mean']:.4f} ± {fold_stats['recall_std']:.4f}\n")
                f.write(f"F1-Score:  {fold_stats['f1_mean']:.4f} ± {fold_stats['f1_std']:.4f}\n\n")
                
                f.write("INDIVIDUAL FOLD RESULTS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Train':<8} {'Test':<8}\n")
                f.write("-" * 80 + "\n")
                for fold_result in results['fold_results']:
                    fold_idx = fold_result['fold_idx'] + 1
                    acc = fold_result['accuracy']
                    prec = fold_result['precision']
                    rec = fold_result['recall']
                    f1 = fold_result['f1_score']
                    num_train = fold_result['num_train']
                    num_test = fold_result['num_test']
                    f.write(f"{fold_idx:<6} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {num_train:<8} {num_test:<8}\n")
                f.write("\n")
            else:
                f.write(f"Evaluation Mode: Train-Test Split\n")
                f.write(f"Test Size: {self.test_size}\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 60 + "\n")
            for class_name in results['classes']:
                precision = results['precision_per_class'][class_name]
                recall = results['recall_per_class'][class_name]
                f1 = results['f1_per_class'][class_name]
                support = results['support_per_class'][class_name]
                f.write(f"{class_name:20s}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}\n")

            # If available, add per-channel summary (for LOOCV with --per_channel_knn)
            if results.get('per_channel_summary'):
                f.write("\nPER-CHANNEL SUMMARY (MAFAULDA):\n")
                f.write("-" * 60 + "\n")
                for ch in results['per_channel_summary']:
                    f.write(
                        f"[{ch['channel_index']}] {ch.get('channel_name','channel')}: "
                        f"Acc={ch['accuracy']:.4f}, Prec={ch['precision']:.4f}, "
                        f"Rec={ch['recall']:.4f}, F1={ch['f1_score']:.4f}\n"
                    )
        
        logger.info(f"Summary saved to {summary_file}")
