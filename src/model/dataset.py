"""
Torch Dataset wrapper around the pre-built feature matrix.
"""

import torch
from torch.utils.data import Dataset


class GHGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
