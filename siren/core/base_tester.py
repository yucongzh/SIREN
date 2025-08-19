"""
Base Tester abstract class for SIREN framework.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This module provides the abstract base class for all testers in the SIREN framework,
defining the common interface and functionality that all testers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseTester(ABC):
    """
    Abstract base class for all testers in SIREN framework.
    
    This class defines the common interface that all testers must implement,
    including DCASE testers, fault classification testers, and future testers.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the base tester.
        
        Args:
            config: Configuration dictionary for the tester
        """
        self.config = config or {}
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def run_evaluation(self) -> Dict:
        """
        Run the complete evaluation process.
        
        Returns:
            Dict containing evaluation results
        """
        pass
    
    @abstractmethod
    def evaluate(self, y_true: List, y_pred: List) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dict containing evaluation metrics
        """
        pass
    
    @abstractmethod
    def log_results(self, results: Dict) -> None:
        """
        Log and output evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        pass
    
    def extract_features(self, files: List[str]) -> List:
        """
        Extract features from files (optional implementation).
        
        Args:
            files: List of file paths
            
        Returns:
            List of extracted features
        """
        raise NotImplementedError(f"Feature extraction not implemented in {self.__class__.__name__}")
    
    def get_dataset_stats(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dict containing dataset statistics
        """
        raise NotImplementedError(f"Dataset stats not implemented in {self.__class__.__name__}")
    
    def __str__(self) -> str:
        """String representation of the tester."""
        return f"{self.__class__.__name__}(config={self.config})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__() 