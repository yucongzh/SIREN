"""
SIREN - Signal Representation Evaluation for Machines

A comprehensive evaluation framework for DCASE series datasets with custom feature extractors.
"""

from .core.base_extractor import BaseFeatureExtractor
from .core.data_loader import DCASEDataLoader, DatasetConfig
from .core.evaluator import DCASEEvaluator
from .core.memory_bank import DCASEMemoryBank as MemoryBank
from .core.dcase_tester import DCASETester
from .core.fault_classification_tester import FaultClassificationTester

# Main classes for users
__all__ = [
    'BaseFeatureExtractor',
    'DCASEDataLoader', 
    'DatasetConfig',
    'DCASEEvaluator',
    'MemoryBank',
    'DCASETester'
    'FaultClassificationTester',
]

# Version info
__version__ = "0.1.0"
__author__ = "Yucong Zhang"
__email__ = "yucong0428@outlook.com"
