"""Lazy HDF5 dataset for memory-efficient ECG data loading."""

import torch
import numpy as np
from torch.utils.data import Dataset
import h5py


class LazyECGDataset(Dataset):
    """Memory-efficient ECG dataset that reads traces lazily from HDF5 on disk.

    Only metadata (ages, weights, sex labels) is stored in RAM. ECG traces are
    read from the HDF5 file on-demand per sample, enabling training on datasets
    that exceed available memory.

    Args:
        h5_path: Path to the HDF5 file containing ECG traces.
        dset_name: Name of the dataset within the HDF5 file.
        ages: Array of patient ages.
        weights: Array of per-sample loss weights.
        sex_labels: Optional array of sex labels (0=female, 1=male).
        mask: Optional boolean mask selecting a subset of samples.
        age_scale: If provided, ages are divided by this value for normalization.
    """

    def __init__(self, h5_path, dset_name, ages, weights, sex_labels=None, mask=None, age_scale=None):
        self.h5_path = h5_path
        self.dset_name = dset_name
        self.ages = ages if age_scale is None else (ages / age_scale)
        self.weights = weights
        self.sex_labels = sex_labels

        if mask is not None:
            self.valid_indices = np.nonzero(mask)[0]
        else:
            self.valid_indices = np.arange(len(ages))

        # Opened lazily inside worker processes for multiprocessing safety
        self.h5_file = None
        self.traces_dset = None

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest')
            self.traces_dset = self.h5_file[self.dset_name]

        real_idx = self.valid_indices[idx]
        trace = self.traces_dset[real_idx]
        age = self.ages[real_idx]
        weight = self.weights[real_idx]

        trace_t = torch.tensor(trace, dtype=torch.float32)
        age_t = torch.tensor(age, dtype=torch.float32)
        weight_t = torch.tensor(weight, dtype=torch.float32)

        if self.sex_labels is not None:
            sex_t = torch.tensor(self.sex_labels[real_idx], dtype=torch.long)
            return trace_t, age_t, weight_t, sex_t

        return trace_t, age_t, weight_t