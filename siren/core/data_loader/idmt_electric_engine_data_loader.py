"""
IDMT-ISA Electric Engine (IIEE) Dataset Data Loader.

Author: Yucong Zhang
Email: yucong0428@outlook.com

Structure (observed):
  - train/{engine1_good, engine2_broken, engine3_heavyload}/pure.wav (long recordings)
  - train_cut/{engine1_good, engine2_broken, engine3_heavyload}/pure_*.wav (3s clips)
  - test/{engine1_good, engine2_broken, engine3_heavyload}/{talking_*.wav, atmo_*.wav, ...}

Default evaluation:
  - Use `train_cut/**` with stratified k-fold (k=5 by default)
Optional mode:
  - External test: train on `train_cut/**`, test on `test/**`
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .base_data_loader import BaseDataLoader, DatasetInfo


class IDMTElectricEngineDataLoader(BaseDataLoader):
    """Data loader for IDMT-ISA Electric Engine (IIEE)."""

    def __init__(self, dataset_root: str, config: Dict | None = None):
        super().__init__(dataset_root, config)

        # Config
        self.num_folds: int = int(self.config.get('n_splits', 5))
        self.random_state: int = int(self.config.get('random_state', 42))
        self.use_kfold: bool = bool(self.config.get('use_kfold', True))
        self.use_external_test: bool = bool(self.config.get('use_external_test', False))
        # In external mode, prefer using test_cut clips by default (train_cut vs test_cut symmetry)
        self.external_use_test_cut: bool = bool(self.config.get('external_use_test_cut', True))

        # Cache
        self._train_files: List[str] | None = None
        self._train_labels: List[str] | None = None
        self._split_indices: List[Tuple[np.ndarray, np.ndarray]] | None = None
        self._current_fold: int = 0

    # ---- Base info ----
    def _get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="iiee",
            task_type="fault_classification",
            signal_type="audio",
            file_format="wav",
            sampling_rate=44100,
            num_channels=1,
            description="IDMT-ISA Electric Engine dataset (mono 44.1kHz wav; 3s train_cut clips)"
        )

    # ---- Helpers ----
    def _classes(self) -> List[str]:
        return ['engine1_good', 'engine2_broken', 'engine3_heavyload']

    def _list_traincut_wavs(self) -> List[str]:
        pattern = str(self.dataset_root / 'train_cut' / '**' / '*.wav')
        files = glob.glob(pattern, recursive=True)
        return sorted(files)

    def _list_train_wavs(self) -> List[str]:
        """List long clean training recordings under train/** (pure.wav)."""
        pattern = str(self.dataset_root / 'train' / '**' / 'pure*.wav')
        files = glob.glob(pattern, recursive=True)
        return sorted(files)

    def _list_test_wavs(self) -> List[str]:
        pattern = str(self.dataset_root / 'test' / '**' / '*.wav')
        files = glob.glob(pattern, recursive=True)
        return sorted(files)

    def _list_testcut_wavs(self) -> List[str]:
        pattern = str(self.dataset_root / 'test_cut' / '**' / '*.wav')
        files = glob.glob(pattern, recursive=True)
        return sorted(files)

    def _extract_class_from_path(self, file_path: str) -> str:
        p = Path(file_path)
        parts = [x.lower() for x in p.parts]
        for c in self._classes():
            if c in parts:
                return c
        return 'unknown'

    def _ensure_kfold(self):
        if self._split_indices is not None:
            return
        train_files = self._list_traincut_wavs()
        train_labels = [self._extract_class_from_path(f) for f in train_files]
        # filter unknowns
        pairs = [(f, y) for f, y in zip(train_files, train_labels) if y in self._classes()]
        self._train_files = [f for f, _ in pairs]
        self._train_labels = [y for _, y in pairs]
        labels = np.array(self._train_labels)
        indices = np.arange(len(labels))
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
        self._split_indices = list(skf.split(indices, labels))
        self._current_fold = 0

    # ---- Public API ----
    def get_classes(self) -> List[str]:
        return self._classes()

    def _get_all_files(self) -> List[str]:
        # expose train_cut files for compatibility
        return self._list_traincut_wavs()

    def get_train_files(self, **kwargs) -> List[str]:
        if self.use_external_test:
            # external test mode: train on 3s clips from train_cut/** (avoid leakage from long recording)
            return self._list_traincut_wavs()
        if not self.use_kfold:
            return self._list_traincut_wavs()
        self._ensure_kfold()
        train_idx, test_idx = self._split_indices[self._current_fold]
        return [self._train_files[i] for i in train_idx]  # type: ignore

    def get_test_files(self, **kwargs) -> List[Tuple[str, Union[int, str]]]:
        label_to_int = {c: i for i, c in enumerate(self.get_classes())}
        if self.use_external_test:
            # Prefer test_cut if enabled and available, else fallback to test
            test_cut_dir = self.dataset_root / 'test_cut'
            if self.external_use_test_cut and test_cut_dir.exists():
                test_files = self._list_testcut_wavs()
            else:
                test_files = self._list_test_wavs()
            labels = [self._extract_class_from_path(f) for f in test_files]
            return [(f, label_to_int[y]) for f, y in zip(test_files, labels) if y in label_to_int]
        if not self.use_kfold:
            files = self._list_traincut_wavs()
            labels = [self._extract_class_from_path(f) for f in files]
            return [(f, label_to_int[y]) for f, y in zip(files, labels) if y in label_to_int]
        self._ensure_kfold()
        train_idx, test_idx = self._split_indices[self._current_fold]
        return [(self._train_files[i], label_to_int[self._train_labels[i]])  # type: ignore
                for i in test_idx]

    def next_fold(self):
        if self._split_indices is None:
            return
        self._current_fold = (self._current_fold + 1) % len(self._split_indices)

    def get_files_with_sampling_rates(self) -> List[Tuple[str, int, str]]:
        # Use train_cut files by default
        files = self._list_traincut_wavs()
        labels = [self._extract_class_from_path(f) for f in files]
        return [(f, 44100, y) for f, y in zip(files, labels) if y in self._classes()]

    def get_test_conditions(self) -> Dict[str, str]:
        """Return mapping from test file path to noise/scene condition name.

        Conditions parsed from filename:
          - talking_*.wav -> "talking"
          - atmo_low.wav / atmo_medium.wav / atmo_high.wav -> same literal
          - whitenoise_low.wav -> "whitenoise_low"
          - stresstest.wav -> "stresstest"
        Unknown patterns -> "unknown"
        """
        # Use whichever list is used in external mode; default to test
        test_files = self._list_test_wavs()
        if self.use_external_test and self.external_use_test_cut and (self.dataset_root / 'test_cut').exists():
            test_files = self._list_testcut_wavs()
        mapping: Dict[str, str] = {}
        for fp in test_files:
            name = Path(fp).name.lower()
            cond = 'unknown'
            if name.startswith('talking_'):
                cond = 'talking'
            elif name.startswith('atmo_'):
                # keep full atmo_* literal
                cond = name.replace('.wav', '')
            elif name.startswith('whitenoise_low'):
                cond = 'whitenoise_low'
            elif name.startswith('stresstest'):
                cond = 'stresstest'
            mapping[fp] = cond
        return mapping

