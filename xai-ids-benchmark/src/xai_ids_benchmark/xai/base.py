"""Common Explainer interface.

Every XAI method returns a per-instance attribution array of shape
(n_instances, n_features) with a sign convention: positive = pushes toward
the 'attack' class, negative = pushes toward 'benign'. The metric code uses
this uniform interface so all four methods are scored identically.
"""
from __future__ import annotations

import abc
from typing import Any

import numpy as np


class Explainer(abc.ABC):
    """Base class for XAI explainers."""

    name: str = "base"
    family: str = "abstract"

    def __init__(self, model: Any, train_x: np.ndarray, train_y: np.ndarray,
                 feature_names: list[str], device: str = "cpu", **kwargs):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.feature_names = feature_names
        self.device = device
        self.kwargs = kwargs

    @abc.abstractmethod
    def explain(self, X: np.ndarray) -> np.ndarray:
        """Return attributions of shape (n_instances, n_features)."""
        raise NotImplementedError

    # Optional: return per-instance coverage flag (1 = explanation generated,
    # 0 = failed). Default assumes full coverage.
    def coverage(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X), dtype=int)
