"""
Evaluation calculation utilities for DCASE series datasets.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This module provides evaluation utilities for calculating AUC, pAUC, and other metrics
for DCASE series datasets with proper grouping and aggregation strategies.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import hmean
from typing import Dict, List, Tuple, Optional
import csv
import os
from pathlib import Path


class DCASEEvaluator:
    """Evaluator for DCASE series datasets."""
    
    def __init__(self):
        """Initialize DCASE evaluator."""
        pass
        
    def calculate_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Calculate ROC-AUC score.
        
        Args:
            y_true: Ground truth labels (0=normal, 1=anomaly)
            y_score: Anomaly scores (higher = more anomalous)
            
        Returns:
            ROC-AUC score
        """
        if len(np.unique(y_true)) < 2:
            return 0.5  # Default score for single class
            
        return roc_auc_score(y_true, y_score)
        
    def calculate_pauc(self, y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.1) -> float:
        """
        Calculate partial AUC score.
        
        Args:
            y_true: Ground truth labels (0=normal, 1=anomaly)
            y_score: Anomaly scores (higher = more anomalous)
            max_fpr: Maximum false positive rate
            
        Returns:
            Partial AUC score
        """
        if len(np.unique(y_true)) < 2:
            return 0.5  # Default score for single class
            
        return roc_auc_score(y_true, y_score, max_fpr=max_fpr)
        
    def calculate_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            y_true: Ground truth labels (0=normal, 1=anomaly)
            y_pred: Predicted labels (0=normal, 1=anomaly)
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if len(np.unique(y_true)) < 2:
            return 0.0, 0.0, 0.0
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score
        
    def calculate_harmonic_mean(self, scores: List[float]) -> float:
        """
        Calculate harmonic mean of scores.
        
        Args:
            scores: List of scores
            
        Returns:
            Harmonic mean
        """
        if not scores:
            return 0.0
            
        # Filter out invalid scores
        valid_scores = [s for s in scores if not np.isnan(s) and s > 0]
        
        if not valid_scores:
            return 0.0
            
        return hmean(valid_scores)
        
    def calculate_arithmetic_mean(self, scores: List[float]) -> float:
        """
        Calculate arithmetic mean of scores.
        
        Args:
            scores: List of scores
            
        Returns:
            Arithmetic mean
        """
        if not scores:
            return 0.0
            
        # Filter out invalid scores
        valid_scores = [s for s in scores if not np.isnan(s)]
        
        if not valid_scores:
            return 0.0
            
        return np.mean(valid_scores)
        
    def evaluate_machine_type(self, y_true: np.ndarray, y_score: np.ndarray, 
                            y_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate performance for a single machine type.
        
        Args:
            y_true: Ground truth labels (0=normal, 1=anomaly)
            y_score: Anomaly scores (higher = more anomalous)
            y_pred: Predicted labels (optional, for precision/recall/F1)
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            'auc': self.calculate_auc(y_true, y_score),
            'pauc': self.calculate_pauc(y_true, y_score)
        }
        
        if y_pred is not None:
            precision, recall, f1 = self.calculate_precision_recall_f1(y_true, y_pred)
            results.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
        return results
        
    def evaluate_all_machines(self, results_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Evaluate performance across all machine types.
        
        Args:
            results_dict: Dictionary with machine_type -> {y_true, y_score, y_pred}
            
        Returns:
            Dictionary with machine_type -> evaluation_metrics
        """
        all_results = {}
        
        for machine_type, data in results_dict.items():
            y_true = data['y_true']
            y_score = data['y_score']
            y_pred = data.get('y_pred', None)
            
            metrics = self.evaluate_machine_type(y_true, y_score, y_pred)
            all_results[machine_type] = metrics
            
        return all_results
        
    def calculate_final_scores(self, dev_results: Dict[str, Dict], 
                             eval_results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate final evaluation scores.
        
        Args:
            dev_results: Dev evaluation results by machine type
            eval_results: Eval evaluation results by machine type
            
        Returns:
            Dictionary with final scores
        """
        # Extract AUC and pAUC scores
        dev_aucs = [results['auc'] for results in dev_results.values()]
        eval_aucs = [results['auc'] for results in eval_results.values()]
        dev_paucs = [results['pauc'] for results in dev_results.values()]
        eval_paucs = [results['pauc'] for results in eval_results.values()]
        
        # Calculate harmonic means for AUC
        dev_hmean = self.calculate_harmonic_mean(dev_aucs)
        eval_hmean = self.calculate_harmonic_mean(eval_aucs)
        
        # Calculate overall harmonic mean for AUC
        all_aucs = dev_aucs + eval_aucs
        overall_hmean = self.calculate_harmonic_mean(all_aucs)
        
        # Calculate arithmetic means for AUC
        dev_amean = self.calculate_arithmetic_mean(dev_aucs)
        eval_amean = self.calculate_arithmetic_mean(eval_aucs)
        overall_amean = self.calculate_arithmetic_mean(all_aucs)
        
        # Calculate harmonic means for pAUC
        dev_pauc_hmean = self.calculate_harmonic_mean(dev_paucs)
        eval_pauc_hmean = self.calculate_harmonic_mean(eval_paucs)
        
        # Calculate overall harmonic mean for pAUC
        all_paucs = dev_paucs + eval_paucs
        overall_pauc_hmean = self.calculate_harmonic_mean(all_paucs)
        
        # Calculate arithmetic means for pAUC
        dev_pauc_amean = self.calculate_arithmetic_mean(dev_paucs)
        eval_pauc_amean = self.calculate_arithmetic_mean(eval_paucs)
        overall_pauc_amean = self.calculate_arithmetic_mean(all_paucs)
        
        return {
            'dev_harmonic_mean': dev_hmean,
            'eval_harmonic_mean': eval_hmean,
            'overall_harmonic_mean': overall_hmean,
            'dev_arithmetic_mean': dev_amean,
            'eval_arithmetic_mean': eval_amean,
            'overall_arithmetic_mean': overall_amean,
            'dev_pauc_harmonic_mean': dev_pauc_hmean,
            'eval_pauc_harmonic_mean': eval_pauc_hmean,
            'overall_pauc_harmonic_mean': overall_pauc_hmean,
            'dev_pauc_arithmetic_mean': dev_pauc_amean,
            'eval_pauc_arithmetic_mean': eval_pauc_amean,
            'overall_pauc_arithmetic_mean': overall_pauc_amean
        }
        
    def save_scores_to_csv(self, scores: List[Tuple[str, float]], file_path: str):
        """
        Save anomaly scores to CSV file.
        
        Args:
            scores: List of (filename, score) tuples
            file_path: Path to save CSV file
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for filename, score in scores:
                writer.writerow([filename, score])
                

        
 