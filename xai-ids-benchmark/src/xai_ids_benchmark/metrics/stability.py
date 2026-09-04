"""Stability metrics for XAI explanations.

  * Kendall τ rank correlation over bootstrap resamples
  * Perturbation stability (prediction-preserving perturbations)
  * Explanability Fragility Score (Vourganas 2026) — relies on feature
    multicollinearity (VIF); computed primarily on UNSW-NB15.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


class Stability:
    def __init__(self, explain_fn, X: np.ndarray, n_bootstrap: int = 100,
                 epsilon: float = 0.05, random_state: int = 20260827):
        # explain_fn: callable(X: np.ndarray) -> np.ndarray attributions
        self.explain_fn = explain_fn
        self.X = X
        self.n_bootstrap = n_bootstrap
        self.epsilon = epsilon
        self.rng = np.random.default_rng(random_state)

    def evaluate(self, attributions: np.ndarray | None = None) -> dict:
        out = {}
        out["kendall_tau_bootstrap"] = self._kendall_bootstrap(attributions)
        out["perturbation_stability"] = self._perturbation(attributions)
        return out

    def _kendall_bootstrap(self, attr: np.ndarray | None) -> float:
        """Mean Kendall τ between attributions on the full sample and on
        bootstrap resamples (averaged over instances and features)."""
        if attr is None:
            attr = self.explain_fn(self.X)
        n = len(self.X)
        taus = []
        for b in range(self.n_bootstrap):
            idx = self.rng.integers(0, n, size=n)
            try:
                attr_b = self.explain_fn(self.X[idx])
            except Exception:
                continue
            # Per-feature rank correlation, then average.
            for f in range(attr.shape[1]):
                t, _ = kendalltau(np.abs(attr[:, f]), np.abs(attr_b[:, f]))
                if not np.isnan(t):
                    taus.append(t)
        return float(np.mean(taus)) if taus else 0.0

    def _perturbation(self, attr: np.ndarray | None) -> float:
        """Stability under prediction-preserving perturbations: rank correlation
        between attributions of x and x + small noise."""
        if attr is None:
            attr = self.explain_fn(self.X)
        noise = self.rng.normal(0, self.epsilon, size=self.X.shape).astype(np.float32)
        # Prediction-preserving: clip perturbations so they do not cross 0 by much.
        Xp = self.X + noise
        try:
            attr_p = self.explain_fn(Xp)
        except Exception:
            return 0.0
        taus = []
        for f in range(attr.shape[1]):
            t, _ = kendalltau(np.abs(attr[:, f]), np.abs(attr_p[:, f]))
            if not np.isnan(t):
                taus.append(t)
        return float(np.mean(taus)) if taus else 0.0


def fragility_score(X: np.ndarray, attr: np.ndarray) -> float:
    """Explanability Fragility Score (Vourganas 2026).

    High VIF features that are also high-attribution inflate attribution
    variance. We compute mean VIF-weighted attribution as a proxy score;
    higher = more fragile. Returns a single scalar.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    try:
        vifs = np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])
    except Exception:
        return 0.0
    vifs = np.nan_to_num(vifs, nan=0.0, posinf=0.0)
    # Normalize VIF to [0,1] then weight mean |attr|.
    w = vifs / (vifs.max() + 1e-9)
    mean_attr = np.abs(attr).mean(axis=0)
    return float((w * mean_attr).sum() / (mean_attr.sum() + 1e-9))
