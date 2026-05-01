"""
GHG regression MLP: BatchNorm + ReLU + Dropout blocks ending in a scalar head.
"""

import torch.nn as nn

from src.config import DROPOUT, HIDDEN_DIMS


class GHGNet(nn.Module):
    def __init__(self, input_dim: int, hidden=None, drop=DROPOUT):
        super().__init__()
        if hidden is None:
            hidden = HIDDEN_DIMS

        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
