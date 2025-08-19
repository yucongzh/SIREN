"""
IDMT-ISA Compressed Air (IICA) Dataset Data Loader.

Author: Yucong Zhang
Email: yucong0428@outlook.com

Structure (observed):
  raw/{tubeleak, ventleak}/{hydr, hydr_low, lab, work, work_low}/{1,2,3}/*.wav

Filename pattern (from README): S_L_N_M_.wav where
  - S: session {1,2,3}
  - L: annotation (iO = no leak, niO = leak present)
  - N: knob rotations
  - M: mic configuration

Default classes (3-class):
  - no_leak := iO (in either leak-type root)
  - tubeleak := niO under raw/tubeleak/**
  - ventleak := niO under raw/ventleak/**
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .base_data_loader import BaseDataLoader, DatasetInfo


class IDMTCompressedAirDataLoader(BaseDataLoader):
    """Data loader for IDMT-ISA Compressed Air (IICA)."""

    def __init__(self, dataset_root: str, config: Dict | None = None):
        super().__init__(dataset_root, config)

        # Config
        self.num_folds: int = int(self.config.get('n_splits', 5))
        self.random_state: int = int(self.config.get('random_state', 42))
        self.use_kfold: bool = bool(self.config.get('use_kfold', True))

        # Cache
        self._all_files: List[str] | None = None
        self._all_labels: List[str] | None = None
        self._split_indices: List[Tuple[np.ndarray, np.ndarray]] | None = None
        self._current_fold: int = 0

    # ---- Base info ----
    def _get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="iica",
            task_type="fault_classification",
            signal_type="audio",
            file_format="wav",
            sampling_rate=48000,
            num_channels=1,
            description="IDMT-ISA Compressed Air leak dataset (24-bit mono 48kHz wav)"
        )

    # ---- Helpers ----
    def _list_all_wavs(self) -> List[str]:
        if self._all_files is not None:
            return self._all_files
        patterns = [
            str(self.dataset_root / 'raw' / 'tubeleak' / '**' / '*.wav'),
            str(self.dataset_root / 'raw' / 'ventleak' / '**' / '*.wav'),
        ]
        files: List[str] = []
        for p in patterns:
            files.extend(glob.glob(p, recursive=True))
        self._all_files = sorted(files)
        return self._all_files

    def _extract_class_from_path(self, file_path: str) -> str:
        p = file_path.replace('\\', '/').lower()
        # parse iO / niO
        # we normalize to lowercase and search '/_io_' and '/_nio_'
        fname = os.path.basename(p)
        lbl = 'unknown'
        if '_io_' in fname:
            lbl = 'no_leak'
        elif '_nio_' in fname:
            # leak present; decide type by folder
            if '/tubeleak/' in p:
                lbl = 'tubeleak'
            elif '/ventleak/' in p:
                lbl = 'ventleak'
        return lbl

    def _get_all_labels(self) -> List[str]:
        if self._all_labels is not None:
            return self._all_labels
        files = self._list_all_wavs()
        labels = [self._extract_class_from_path(f) for f in files]
        # filter unknowns
        filtered = [(f, y) for f, y in zip(files, labels) if y != 'unknown']
        self._all_files = [f for f, _ in filtered]
        self._all_labels = [y for _, y in filtered]
        return self._all_labels

    def _ensure_kfold(self):
        if self._split_indices is not None:
            return
        labels = np.array(self._get_all_labels())
        indices = np.arange(len(labels))
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
        self._split_indices = list(skf.split(indices, labels))
        self._current_fold = 0

    # ---- Public API ----
    def get_classes(self) -> List[str]:
        return ['no_leak', 'tubeleak', 'ventleak']

    def _get_all_files(self) -> List[str]:
        return self._list_all_wavs()

    def get_train_files(self, **kwargs) -> List[str]:
        if not self.use_kfold:
            # use all files as train in non-kfold mode
            return self._list_all_wavs()
        self._ensure_kfold()
        train_idx, test_idx = self._split_indices[self._current_fold]
        files = self._list_all_wavs()
        return [files[i] for i in train_idx]

    def get_test_files(self, **kwargs) -> List[Tuple[str, Union[int, str]]]:
        if not self.use_kfold:
            # default non-kfold: return all files with integer labels
            files = self._list_all_wavs()
            labels = [self._extract_class_from_path(f) for f in files]
            label_to_int = {c: i for i, c in enumerate(self.get_classes())}
            return [(f, label_to_int[y]) for f, y in zip(files, labels) if y in label_to_int]
        self._ensure_kfold()
        train_idx, test_idx = self._split_indices[self._current_fold]
        files = self._list_all_wavs()
        labels = [self._extract_class_from_path(f) for f in files]
        label_to_int = {c: i for i, c in enumerate(self.get_classes())}
        return [(files[i], label_to_int[labels[i]]) for i in test_idx if labels[i] in label_to_int]

    def next_fold(self):
        """Advance to next fold (if using k-fold)."""
        if self._split_indices is None:
            return
        self._current_fold = (self._current_fold + 1) % len(self._split_indices)

    def get_files_with_sampling_rates(self) -> List[Tuple[str, int, str]]:
        files = self._list_all_wavs()
        labels = [self._extract_class_from_path(f) for f in files]
        return [(f, 48000, y) for f, y in zip(files, labels) if y != 'unknown']

