""" Modified from https://github.com/clabrugere/pytorch-scarf/blob/master/scarf/dataset.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class AEDataset(Dataset):
    def __init__(self, data, target, columns=None, weights=None, task_type=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.columns = columns
        if weights is not None:
            self.weights = np.array(weights)
        else:
            self.weights = np.ones(self.data.shape[0])
        self.task_type = task_type

    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        if self.task_type == "classification":
            y = torch.tensor(self.target[index], dtype=torch.long)
        else:
            y = torch.tensor(self.target[index], dtype=torch.float32)
        weights = torch.tensor(self.weights[index], dtype=torch.float32)
        return x, y, weights

    def __len__(self):
        return len(self.data)
