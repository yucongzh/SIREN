"""
Base Memory Bank interface for SIREN framework.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This module provides the base interface for memory banks in the SIREN framework,
supporting feature storage and retrieval for various evaluation tasks.
"""

from abc import ABC, abstractmethod

class BaseMemoryBank(ABC):
    @abstractmethod
    def add(self, feature, label=None):
        """Add a single feature (with optional label)"""
        pass

    @abstractmethod
    def batch_add(self, features, labels=None):
        """Add features in batch (with optional labels)"""
        pass

    @abstractmethod
    def clear(self):
        """Clear all stored features and labels"""
        pass

    @abstractmethod
    def save(self, path):
        """Save memory bank to file"""
        pass

    @abstractmethod
    def load(self, path):
        """Load memory bank from file"""
        pass