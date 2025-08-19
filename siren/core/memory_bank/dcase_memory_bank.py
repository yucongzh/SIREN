#!/usr/bin/env python3
"""
Simplified Memory Bank implementation for DCASE series datasets.

Author: Yucong Zhang
Email: yucong0428@outlook.com

Each memory bank handles only one machine type.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Constants for memory bank configuration
DEFAULT_DOMAIN_STRATEGY = 'min'
DEFAULT_KMEANS_THRESHOLD = 100
DEFAULT_KMEANS_N_CLUSTERS = 16
DEFAULT_ENABLE_KMEANS_OPTIMIZATION = False


from .base_memory_bank import BaseMemoryBank

class MemoryBank(BaseMemoryBank):
    """
    Memory Bank for a single machine type.
    
    Supports:
    - DCASE 2020: id_XX -> features (unchanged)
    - DCASE 2021-2025: section_XX -> {'source': features, 'target': features}
    """
    
    def __init__(self, dataset_year: int, machine_type: str):
        """
        Initialize Memory Bank for a single machine type.
        
        Args:
            dataset_year: DCASE dataset year (2020-2025)
            machine_type: Machine type (e.g., 'fan', 'pump')
        """
        self.dataset_year = dataset_year
        self.machine_type = machine_type
        self.memory_bank = {}  # section_id/domain -> features
        
        # Default clustering parameters for DCASE 2023-2025
        self.default_n_clusters = 16
        
        logger.info(f"Initialized MemoryBank for {machine_type} (DCASE {dataset_year})")
    
    def add(self, feature, label=None):
        """BaseMemoryBank interface - not supported for DCASE MemoryBank"""
        raise NotImplementedError("DCASEMemoryBank: 请使用 add_features(domain_or_section, features, domain_type) 方法")
    
    def batch_add(self, features, labels=None):
        """BaseMemoryBank interface - not supported for DCASE MemoryBank"""
        raise NotImplementedError("DCASEMemoryBank: 请使用 add_features(domain_or_section, features, domain_type) 方法")
    
    def add_features(self, domain_or_section: str, features: Union[List[torch.Tensor], np.ndarray, torch.Tensor], 
                    domain_type: Optional[str] = None):
        """
        Add features to memory bank.
        
        Args:
            domain_or_section: For 2020: id_XX; For 2021-2025: section_XX
            features: Feature tensors or arrays
            domain_type: For 2021-2025: 'source' or 'target'; For 2020: None
        """
        # Convert features to numpy array if needed
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        elif isinstance(features, list):
            features = torch.stack(features).cpu().numpy()
        
        if self.dataset_year == 2020:
            # DCASE 2020: Keep original structure
            self.memory_bank[domain_or_section] = features
        else:
            # DCASE 2021-2025: Use new unified structure
            if domain_or_section not in self.memory_bank:
                self.memory_bank[domain_or_section] = {'source': None, 'target': None}
            
            if domain_type in ['source', 'target']:
                self.memory_bank[domain_or_section][domain_type] = features
            else:
                raise ValueError(f"domain_type must be 'source' or 'target' for DCASE {self.dataset_year}")

    
    def _extract_domain_from_filename(self, filename: str) -> str:
        """
        Extract domain/section from filename based on dataset year.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Domain/section string
        """
        import re
        
        if self.dataset_year == 2020:
            # DCASE 2020: machine ID as domain (id_00, id_01, etc.)
            match = re.search(r'id_(\d+)', filename)
            if match:
                return f"id_{match.group(1)}"
        
        elif self.dataset_year in [2021, 2022, 2023, 2024, 2025]:
            # DCASE 2021-2025: section as domain (section_00, section_01, etc.)
            match = re.search(r'section_(\d+)', filename)
            if match:
                return f"section_{match.group(1)}"
        
        return 'unknown'
    
    def compute_anomaly_scores(self, test_features: torch.Tensor, test_files: List[str], 
                              similarity_type: str = 'cosine', 
                              aggregation: str = 'max', k: int = 1,
                              domain_strategy: str = DEFAULT_DOMAIN_STRATEGY,
                              enable_kmeans_optimization: bool = DEFAULT_ENABLE_KMEANS_OPTIMIZATION,
                              kmeans_threshold: int = DEFAULT_KMEANS_THRESHOLD,
                              kmeans_n_clusters: int = DEFAULT_KMEANS_N_CLUSTERS) -> np.ndarray:
        """
        Compute anomaly scores for test features with domain-aware batching.
        
        Args:
            test_features: Test feature tensor
            test_files: List of test file paths (for domain matching if needed)
            similarity_type: Similarity metric ('cosine', 'euclidean')
            aggregation: Aggregation method ('max', 'mean', 'min', 'knn')
            k: Number of nearest neighbors to consider
            domain_strategy: Strategy for combining source/target scores ('min', 'max', 'mean')
            enable_kmeans_optimization: Whether to use kmeans optimization
            kmeans_threshold: Sample threshold for enabling kmeans
            kmeans_n_clusters: Number of kmeans clusters
            
        Returns:
            np.ndarray: Anomaly scores for test samples
        """
        if self.dataset_year == 2020:
            # DCASE 2020: Use original logic unchanged
            return self._compute_anomaly_scores_2020(test_features, test_files, 
                                                   similarity_type, aggregation, k)
        else:
            # DCASE 2021-2025: Use new unified logic
            return self._compute_anomaly_scores_2021_2025(test_features, test_files,
                                                        similarity_type, aggregation, k,
                                                        domain_strategy, enable_kmeans_optimization,
                                                        kmeans_threshold, kmeans_n_clusters)

    def _compute_anomaly_scores_2020(self, test_features: torch.Tensor, test_files: List[str],
                                    similarity_type: str, aggregation: str, k: int) -> np.ndarray:
        """
        Original anomaly score computation for DCASE 2020.
        """
        # Convert to numpy for consistency
        if isinstance(test_features, torch.Tensor):
            test_np = test_features.cpu().numpy()
        else:
            test_np = np.array(test_features)
        
        # Step 1: Group test features by domain
        domain_groups = {}  # domain -> (indices, features)
        for i, file_path in enumerate(test_files):
            domain = self._extract_domain_from_filename(file_path)
            if domain not in self.memory_bank:
                continue
            
            if domain not in domain_groups:
                domain_groups[domain] = {'indices': [], 'features': []}
            
            domain_groups[domain]['indices'].append(i)
            domain_groups[domain]['features'].append(test_np[i])
        
        # Step 2: Initialize scores array
        scores = np.zeros(len(test_files))
        
        # Step 3: Process each domain batch
        if len(domain_groups) == 0:
            print("No domain found in memory bank")
            print("Use all domains in memory bank")
            memory_features = np.concatenate([self.memory_bank[domain] for domain in self.memory_bank.keys()], axis=0)
            batch_similarities = self._compute_similarities_batch(
                test_np, memory_features, similarity_type
            )
            batch_scores = self._aggregate_similarities_batch(
                batch_similarities, aggregation, k
            )
            scores = batch_scores
        else:
            for domain, group_data in domain_groups.items():
                indices = group_data['indices']
                domain_features = np.array(group_data['features'])  # (batch_size, feature_dim)
                memory_features = self.memory_bank[domain]  # (memory_size, feature_dim)
                
                # Batch compute similarities for this domain
                batch_similarities = self._compute_similarities_batch(
                    domain_features, memory_features, similarity_type
                )
                
                # Batch compute anomaly scores
                batch_scores = self._aggregate_similarities_batch(
                    batch_similarities, aggregation, k
                )
                
                # Store scores in original positions
                scores[indices] = batch_scores

        return scores

    def _compute_anomaly_scores_2021_2025(self, test_features: torch.Tensor, test_files: List[str],
                                         similarity_type: str, aggregation: str, k: int,
                                         domain_strategy: str, enable_kmeans_optimization: bool,
                                         kmeans_threshold: int, kmeans_n_clusters: int) -> np.ndarray:
        """
        New unified anomaly score computation for DCASE 2021-2025.
        """
        # Convert to numpy for consistency
        if isinstance(test_features, torch.Tensor):
            test_np = test_features.cpu().numpy()
        else:
            test_np = np.array(test_features)
        
        # Step 1: Group test features by section
        section_groups = {}  # section -> (indices, features)
        for i, file_path in enumerate(test_files):
            section = self._extract_domain_from_filename(file_path)
            if section not in self.memory_bank:
                # print(f"DEBUG: Section {section} not found in memory bank. Available sections: {list(self.memory_bank.keys())}")
                continue
            
            if section not in section_groups:
                section_groups[section] = {'indices': [], 'features': []}
            
            section_groups[section]['indices'].append(i)
            section_groups[section]['features'].append(test_np[i])
        
        # print(f"DEBUG: Found {len(section_groups)} sections in test files")
        # for section, group_data in section_groups.items():
        #     print(f"DEBUG: Section {section} has {len(group_data['indices'])} test samples")
        
        # Step 2: Initialize scores array
        scores = np.zeros(len(test_files))
        
        # Step 3: Process each section batch
        if len(section_groups) == 0:
            print("No section found in memory bank")
            print("Use all sections in memory bank")
            # Fallback: use all available data
            all_source_features = []
            all_target_features = []
            for section_data in self.memory_bank.values():
                if section_data['source'] is not None:
                    all_source_features.append(section_data['source'])
                if section_data['target'] is not None:
                    all_target_features.append(section_data['target'])
            
            if all_source_features:
                all_source_features = np.concatenate(all_source_features, axis=0)
            if all_target_features:
                all_target_features = np.concatenate(all_target_features, axis=0)
            
            batch_scores = self._compute_unified_scores(
                test_np, all_source_features, all_target_features,
                similarity_type, aggregation, k, domain_strategy,
                enable_kmeans_optimization, kmeans_threshold, kmeans_n_clusters
            )
            scores = batch_scores
        else:
            for section, group_data in section_groups.items():
                indices = group_data['indices']
                section_features = np.array(group_data['features'])
                
                # Get source and target features for this section
                source_features = self.memory_bank[section]['source']
                target_features = self.memory_bank[section]['target']
                
                # Debug: Print memory bank info
                # print(f"DEBUG: Memory bank for {section} - Source: {source_features.shape if source_features is not None else 'None'}, Target: {target_features.shape if target_features is not None else 'None'}")
                if source_features is not None:
                    # print(f"DEBUG: Source features - Mean: {source_features.mean():.6f}, Std: {source_features.std():.6f}")
                    pass
                if target_features is not None:
                    # print(f"DEBUG: Target features - Mean: {target_features.mean():.6f}, Std: {target_features.std():.6f}")
                    pass
                # print(f"DEBUG: Test features - Mean: {section_features.mean():.6f}, Std: {section_features.std():.6f}")
                
                # Check if section has any data
                if source_features is None and target_features is None:
                    raise ValueError(f"Section {section} has no source or target data for {self.machine_type}")
                
                # Compute scores for this section
                batch_scores = self._compute_unified_scores(
                    section_features, source_features, target_features,
                    similarity_type, aggregation, k, domain_strategy,
                    enable_kmeans_optimization, kmeans_threshold, kmeans_n_clusters
                )
                
                # Store scores in original positions
                scores[indices] = batch_scores

        return scores

    def _compute_unified_scores(self, test_features: np.ndarray, source_features: Optional[np.ndarray],
                               target_features: Optional[np.ndarray], similarity_type: str, aggregation: str,
                               k: int, domain_strategy: str, enable_kmeans_optimization: bool,
                               kmeans_threshold: int, kmeans_n_clusters: int) -> np.ndarray:
        """
        Compute unified scores for source and target domains.
        """
        scores = []
        
        # Compute source scores
        if source_features is not None:
            source_score = self._compute_domain_score(
                test_features, source_features, similarity_type, aggregation, k,
                enable_kmeans_optimization, kmeans_threshold, kmeans_n_clusters
            )
            scores.append(source_score)
            # print(f"DEBUG: Source score - Mean: {source_score.mean():.8f}, Min: {source_score.min():.8f}, Max: {source_score.max():.8f}")
        
        # Compute target scores
        if target_features is not None:
            target_score = self._compute_domain_score(
                test_features, target_features, similarity_type, aggregation, k,
                enable_kmeans_optimization, kmeans_threshold, kmeans_n_clusters
            )
            scores.append(target_score)
            # print(f"DEBUG: Target score - Mean: {target_score.mean():.8f}, Min: {target_score.min():.8f}, Max: {target_score.max():.8f}")
        
        # Combine scores based on strategy
        if domain_strategy == 'min':
            final_score = np.minimum(scores[0], scores[1]) if len(scores) == 2 else scores[0]
        elif domain_strategy == 'max':
            final_score = np.maximum(scores[0], scores[1]) if len(scores) == 2 else scores[0]
        elif domain_strategy == 'mean':
            final_score = np.mean(scores, axis=0) if len(scores) > 1 else scores[0]
        else:
            raise ValueError(f"Unknown domain_strategy: {domain_strategy}")
        
        # print(f"DEBUG: Final score ({domain_strategy}) - Mean: {final_score.mean():.8f}, Min: {final_score.min():.8f}, Max: {final_score.max():.8f}")
        return final_score

    def _compute_domain_score(self, test_features: np.ndarray, domain_features: np.ndarray,
                             similarity_type: str, aggregation: str, k: int,
                             enable_kmeans_optimization: bool, kmeans_threshold: int,
                             kmeans_n_clusters: int) -> np.ndarray:
        """
        Compute scores for a single domain with optional kmeans optimization.
        """
        if enable_kmeans_optimization and len(domain_features) > kmeans_threshold:
            # print(f"DEBUG: Using kmeans optimization for {len(domain_features)} samples")
            # Use kmeans clustering for optimization
            cluster_centers = self._kmeans_cluster(domain_features, kmeans_n_clusters)
            similarities = self._compute_similarities_batch(test_features, cluster_centers, similarity_type)
        else:
            # Direct computation
            similarities = self._compute_similarities_batch(test_features, domain_features, similarity_type)
        
        return self._aggregate_similarities_batch(similarities, aggregation, k)

    def _kmeans_cluster(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Perform kmeans clustering on features.
        """
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        cluster_centers = kmeans.cluster_centers_
        
        return cluster_centers

    def _compute_similarities_batch(self, test_batch: np.ndarray, memory_features: np.ndarray, 
                                  similarity_type: str) -> np.ndarray:
        """
        Compute similarities for a batch of test features against memory features.
        
        Args:
            test_batch: (batch_size, feature_dim)
            memory_features: (memory_size, feature_dim)
            similarity_type: 'cosine' or 'euclidean'
        
        Returns:
            similarities: (batch_size, memory_size)
        """
        if similarity_type == 'cosine':
            # Normalize vectors for cosine similarity
            test_norm = test_batch / (np.linalg.norm(test_batch, axis=1, keepdims=True) + 1e-10)
            memory_norm = memory_features / (np.linalg.norm(memory_features, axis=1, keepdims=True) + 1e-10)
            
            # Compute cosine similarity: (batch_size, memory_size)
            similarities = np.dot(test_norm, memory_norm.T)
            
        elif similarity_type == 'euclidean':
            # Compute negative Euclidean distance
            # Reshape for broadcasting: (batch_size, 1, feature_dim) - (1, memory_size, feature_dim)
            distances = np.linalg.norm(
                test_batch[:, np.newaxis, :] - memory_features[np.newaxis, :, :], 
                axis=2
            )
            similarities = -distances  # Higher = more similar
            
        else:
            raise ValueError(f"Unsupported similarity type: {similarity_type}")
        
        return similarities

    def _aggregate_similarities_batch(self, similarities: np.ndarray, aggregation: str, k: int) -> np.ndarray:
        """
        Aggregate similarities for a batch and convert to anomaly scores.
        
        Args:
            similarities: (batch_size, memory_size)
            aggregation: 'max', 'mean', 'min', 'knn'
            k: number of nearest neighbors
        
        Returns:
            scores: (batch_size,) - anomaly scores
        """
        if aggregation == 'knn':
            k = min(k, similarities.shape[1])  # Ensure k doesn't exceed available samples
            if k == 1:
                # k=1 is equivalent to max aggregation
                scores = 1.0 - np.max(similarities, axis=1)
            else:
                # Find top-k similarities for each sample
                top_k_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
                top_k_similarities = np.take_along_axis(similarities, top_k_indices, axis=1)
                # Convert to cosine distances and average
                cosine_distances = 1.0 - top_k_similarities
                scores = np.mean(cosine_distances, axis=1)

        elif aggregation == 'max':
            scores = 1.0 - np.max(similarities, axis=1)
        elif aggregation == 'mean':
            scores = 1.0 - np.mean(similarities, axis=1)
        elif aggregation == 'min':
            scores = 1.0 - np.min(similarities, axis=1)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        return scores
    
    def get_domains_or_sections(self) -> List[str]:
        """Get list of domains or sections in memory bank."""
        return list(self.memory_bank.keys())
    
    def has_domain_or_section(self, domain_or_section: str) -> bool:
        """Check if domain or section exists in memory bank."""
        return domain_or_section in self.memory_bank
    
    def get_feature_count(self) -> int:
        """Get total number of features for this machine type."""
        if self.dataset_year == 2020:
            return sum(len(features) for features in self.memory_bank.values())
        else:
            total_count = 0
            for section_data in self.memory_bank.values():
                if section_data['source'] is not None:
                    total_count += len(section_data['source'])
                if section_data['target'] is not None:
                    total_count += len(section_data['target'])
            return total_count
    
    def save(self, path: str):
        """Save memory bank to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.memory_bank, path)
        logger.info(f"Memory bank for {self.machine_type} saved to {path}")
    
    def load(self, path: str):
        """Load memory bank from file."""
        if os.path.exists(path):
            self.memory_bank = torch.load(path)
            logger.info(f"Memory bank for {self.machine_type} loaded from {path}")
        else:
            logger.warning(f"Memory bank file not found: {path}")
    
    def clear(self):
        """Clear all data from memory bank."""
        self.memory_bank = {}
        logger.info(f"Memory bank for {self.machine_type} cleared")
    
    def __len__(self) -> int:
        """Get number of domains/sections in memory bank."""
        return len(self.memory_bank)
    
    def __contains__(self, domain_or_section: str) -> bool:
        """Check if domain or section exists."""
        return domain_or_section in self.memory_bank 