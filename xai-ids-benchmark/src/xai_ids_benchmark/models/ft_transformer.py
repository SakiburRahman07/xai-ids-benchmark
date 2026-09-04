"""FT-Transformer for tabular IDS features (Gorishniy et al., 2021).

Each numeric feature is tokenized into an embedding via a learned linear
projection; Transformer blocks then attend across feature tokens. We use the
`rtdl` package if available, else a minimal local implementation, to avoid
hard dependency on an old PyPI build.

This implementation is intentionally compact (2 blocks, d_token=64) to fit a
single GPU workstation and to be matched-budget-trainable against the 1D-CNN.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class _Tokenizer(nn.Module):
    """Numeric-feature tokenizer: one embedding token per feature."""
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.projection = nn.Linear(1, d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, n_features, d_token)
        return self.projection(x.unsqueeze(-1))


class _TransformerBlock(nn.Module):
    def __init__(self, d_token: int, ffn_d_hidden: int, attention_dropout: float, ffn_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, num_heads=4, dropout=attention_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden), nn.GELU(), nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, d_token),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual.
        a = self.norm1(x)
        a, _ = self.attn(a, a, a)
        x = x + a
        f = self.norm2(x)
        f = self.ffn(f)
        return x + f


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_blocks: int = 2,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        ffn_d_hidden: int = 128,
        output_dim: int = 1,
    ):
        super().__init__()
        self.n_features = n_features
        self.tokenizer = _Tokenizer(n_features, d_token)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(d_token, ffn_d_hidden, attention_dropout, ffn_dropout)
             for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, 64), nn.GELU(), nn.Linear(64, output_dim)
        )
        # CLS token (learnable) prepended to pool.
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features)
        tokens = self.tokenizer(x)                      # (B, n_features, d_token)
        B = tokens.size(0)
        cls = self.cls.expand(B, -1, -1)                # (B, 1, d_token)
        x = torch.cat([cls, tokens], dim=1)              # (B, 1+n_features, d_token)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                        # CLS logits

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
