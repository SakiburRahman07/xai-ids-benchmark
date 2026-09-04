"""1D-CNN for tabular IDS features.

Each flow's numeric feature vector is treated as a 1-channel 1D sequence,
matching the most common DL-IDS architecture in the literature (see
XAI_IDS_Gap_Analysis.md Theme B). Outputs a binary logit.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_channels: tuple[int, ...] = (128, 64),
        kernel_size: int = 3,
        dropout: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()
        self.n_features = n_features
        # Input shape: (B, 1, n_features)
        layers = []
        in_ch = 1
        pad = kernel_size // 2
        for ch in hidden_channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size, padding=pad))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_ch, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, 1, n_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.features(x)            # (B, C, L)
        h = self.global_pool(h).squeeze(-1)  # (B, C)
        return self.head(h)             # (B, output_dim) logits

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
