"""
Data loader package for SIREN framework.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This package contains various data loaders for different machine signal datasets.
"""

from .base_data_loader import BaseDataLoader
from .dcase_data_loader import DCASEDataLoader, DatasetConfig

# Machine fault classification data loaders
from .cwru_data_loader import CWRUDataLoader
from .mafaulda_data_loader import MAFAULDADataLoader
from .idmt_compressed_air_data_loader import IDMTCompressedAirDataLoader
from .idmt_electric_engine_data_loader import IDMTElectricEngineDataLoader
# TODO: Implement other data loaders
# from .paderborn_data_loader import PaderbornDataLoader
# from .jnu_data_loader import JNUDataLoader
# from .xjtu_data_loader import XJTUDataLoader
# from .mfpt_data_loader import MFPTDataLoader
# from .southeastu_data_loader import SoutheastUDataLoader
# from .uconn_data_loader import UConnDataLoader
# from .whu_data_loader import WHUDataLoader
# from .idmt_electric_engine_data_loader import IDMTElectricEngineDataLoader

# Lifetime prediction data loaders
# from .nasa_data_loader import NASADataLoader
# from .phm2023_data_loader import PHM2023DataLoader

# Anomaly detection data loaders
# from .idmt_compressed_air_data_loader import IDMTCompressedAirDataLoader

# Format conversion utilities
from .format_converter import FormatConverter
from ..memory_bank.base_memory_bank import BaseMemoryBank
from ..memory_bank.dcase_memory_bank import MemoryBank as DCASEMemoryBank
from ..memory_bank.classification_memory_bank import ClassificationMemoryBank

__all__ = [
    # Base classes
    'BaseDataLoader',
    
    # DCASE data loader
    'DCASEDataLoader',
    'DatasetConfig',
    
    # Machine fault classification data loaders
    'CWRUDataLoader',
    'MAFAULDADataLoader',
    'IDMTCompressedAirDataLoader',
    'IDMTElectricEngineDataLoader',
    # TODO: Add other data loaders when implemented
    # 'PaderbornDataLoader',
    # 'JNUDataLoader',
    # 'XJTUDataLoader',
    # 'MFPTDataLoader',
    # 'SoutheastUDataLoader',
    # 'UConnDataLoader',
    # 'WHUDataLoader',
    'IDMTElectricEngineDataLoader',
    
    # Lifetime prediction data loaders
    # 'NASADataLoader',
    # 'PHM2023DataLoader',
    
    # Anomaly detection data loaders
    'IDMTCompressedAirDataLoader',
    
    # Utilities
    'FormatConverter'
] 