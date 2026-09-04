"""Proxy utility metric: simulated rank-based triage.

Replaces the (IRB-requiring) human analyst triage task per the proposal §3.
Given an alert queue, explanation-guided reordering improves top-k attack-
family recall and time-to-critical-alert vs (a) no-explanation and
(b) random-reorder baselines.

This is a *proxy*; it does not involve human subjects (no IRB needed).
"""
from __future__ import annotations

import numpy as np


class ProxyUtility:
    def __init__(self, model, X: np.ndarray, y: np.ndarray, fam: list[str],
                 attributions: np.ndarray | None = None,
                 k_values: list[int] = (10, 50, 100), device: str = "cpu"):
        self.model = model
        self.X = X
        self.y = y
        self.fam = np.array(fam)
        self.attr = attributions
        self.k_values = list(k_values)
        self.device = device

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        import torch
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32, device=self.device))
            return torch.sigmoid(logits).cpu().numpy().ravel()

    def evaluate(self) -> dict:
        if self.attr is None:
            raise ValueError("ProxyUtility requires attributions")
        # Score each alert by sum of positive attributions over critical features.
        scores = np.abs(self.attr).sum(axis=1)
        res = {}
        # No-explanation baseline: order by model confidence (P attack) descending.
        p = self._predict_proba(self.X)
        rng = np.random.default_rng(20260827)
        # Critical families = non-benign attacks.
        is_attack = (self.fam != "benign")
        for k in self.k_values:
            res[f"top{k}_recall_explain"] = _topk_recall(scores, is_attack, k)
            res[f"top{k}_recall_noexplain"] = _topk_recall(p, is_attack, k)
            res[f"top{k}_recall_random"] = _topk_recall(rng.permutation(len(self.X)), is_attack, k)
            res[f"top{k}_time_to_critical_explain"] = _time_to_critical(scores, is_attack)
            res[f"top{k}_time_to_critical_noexplain"] = _time_to_critical(p, is_attack)
        return res


def _topk_recall(scores: np.ndarray, is_attack: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)  # descending
    top = order[:k]
    if is_attack.sum() == 0:
        return 1.0
    return float(is_attack[top].sum() / is_attack.sum())


def _time_to_critical(scores: np.ndarray, is_attack: np.ndarray) -> int:
    """Index (1-based) of the first true-attack alert in the reordered queue."""
    order = np.argsort(-scores)
    for rank, idx in enumerate(order, start=1):
        if is_attack[idx]:
            return rank
    return len(scores)
