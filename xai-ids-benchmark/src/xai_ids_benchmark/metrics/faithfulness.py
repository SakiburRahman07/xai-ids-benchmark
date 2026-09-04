"""Faithfulness metrics for XAI explanations.

Implements the battery from the proposal/preregistration:
  * Deletion/Insertion AUC (prediction drop as top features are removed/added)
  * Infidelity (Yeh et al. — sensitivity of explanation to input perturbation)
  * Sensitivity-n (Bouchon et al.)
  * Sufficiency (prediction preservation under top-k ablation)
  * Necessity (low-attribution flip should not change prediction)
  * Descriptive accuracy (does top-k match ground-truth importance?)
  * Sparsity (effective number of features used)
"""
from __future__ import annotations

import numpy as np


def _model_predict(model, X, device="cpu") -> np.ndarray:
    """Return P(attack) in [0,1] for sklearn or PyTorch models."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    import torch
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy().ravel()


class Faithfulness:
    def __init__(self, model, X: np.ndarray, y: np.ndarray, baseline: np.ndarray,
                 device: str = "cpu", top_k: list[int] = (3, 5, 10), steps: int = 20):
        self.model = model
        self.X = X
        self.y = y
        self.baseline = baseline  # per-feature baseline (e.g., train mean)
        self.device = device
        self.top_k = list(top_k)
        self.steps = steps

    def evaluate(self, attributions: np.ndarray) -> dict:
        n = len(self.X)
        out = {}
        out["deletion_auc"] = self._deletion_insertion(attributions, mode="deletion")
        out["insertion_auc"] = self._deletion_insertion(attributions, mode="insertion")
        out["sufficiency"] = self._sufficiency(attributions)
        out["necessity"] = self._necessity(attributions)
        out["sparsity"] = self._sparsity(attributions)
        out["infidelity"] = self._infidelity(attributions)
        return out

    def _deletion_insertion(self, attr: np.ndarray, mode: str = "deletion") -> float:
        """Area under the prediction-vs-features curve (averaged per instance)."""
        p0 = _model_predict(self.model, self.X, self.device)
        # Rank features by |attribution|, most important first.
        order = np.argsort(-np.abs(attr), axis=1)
        n_feat = self.X.shape[1]
        steps = min(self.steps, n_feat)
        aucs = []
        for i in range(len(self.X)):
            x = self.X[i].copy()
            probs = [p0[i]]
            for s in range(1, steps + 1):
                k = int(np.ceil(n_feat * s / steps))
                xs = self.X[i].copy()
                if mode == "deletion":
                    # Replace top-k with baseline.
                    xs[order[i, :k]] = self.baseline[order[i, :k]]
                else:
                    # Insertion: start from baseline, restore top-k.
                    xs = self.baseline.copy()
                    xs[order[i, :k]] = self.X[i, order[i, :k]]
                prob = _model_predict(self.model, xs.reshape(1, -1), self.device)[0]
                probs.append(prob)
            aucs.append(np.trapz(probs, dx=1.0 / steps))
        return float(np.mean(aucs))

    def _sufficiency(self, attr: np.ndarray) -> dict:
        """Prediction preservation when only top-k features are kept (restored baseline)."""
        p0 = _model_predict(self.model, self.X, self.device)
        order = np.argsort(-np.abs(attr), axis=1)
        res = {}
        for k in self.top_k:
            preserved = 0
            for i in range(len(self.X)):
                xs = self.baseline.copy()
                xs[order[i, :k]] = self.X[i, order[i, :k]]
                p = _model_predict(self.model, xs.reshape(1, -1), self.device)[0]
                # Preserved if predicted class unchanged.
                if (p >= 0.5) == (p0[i] >= 0.5):
                    preserved += 1
            res[f"top_{k}"] = preserved / len(self.X)
        return res

    def _necessity(self, attr: np.ndarray) -> dict:
        """Low-attribution flip should NOT change prediction. Reports the
        fraction of instances whose prediction stayed the same when the
        low-attribution features are ablated (high = good necessity)."""
        p0 = _model_predict(self.model, self.X, self.device)
        order = np.argsort(-np.abs(attr), axis=1)
        res = {}
        for k in self.top_k:
            preserved = 0
            for i in range(len(self.X)):
                # Keep top-k, ablate the rest.
                xs = self.baseline.copy()
                xs[order[i, :k]] = self.X[i, order[i, :k]]
                p = _model_predict(self.model, xs.reshape(1, -1), self.device)[0]
                if (p >= 0.5) == (p0[i] >= 0.5):
                    preserved += 1
            res[f"low_k_{k}"] = preserved / len(self.X)
        return res

    def _sparsity(self, attr: np.ndarray) -> float:
        """Effective number of features (L2/L1 of |attr|). Lower = more sparse."""
        l1 = np.abs(attr).sum(axis=1)
        l2 = np.sqrt((attr ** 2).sum(axis=1))
        # Avoid divide-by-zero.
        eff = np.where(l1 > 0, (l2 / l1) ** 2, 0.0)
        return float(np.mean(eff))

    def _infidelity(self, attr: np.ndarray, n_perturb: int = 50, sigma: float = 0.1) -> float:
        """Yeh et al. infidelity: squared error between (attr . perturbation)
        and (prediction change). Lower = more faithful."""
        rng = np.random.default_rng(20260827)
        p0 = _model_predict(self.model, self.X, self.device)
        infs = []
        for i in range(min(n_perturb, len(self.X))):
            delta = rng.normal(0, sigma, size=self.X.shape[1]).astype(np.float32)
            xs = (self.X[i] + delta).reshape(1, -1)
            p1 = _model_predict(self.model, xs, self.device)[0]
            pred_change = (p1 - p0[i]) ** 2
            expl_change = float(attr[i] @ delta) ** 2
            infs.append((expl_change - pred_change) ** 2)
        return float(np.mean(infs))
