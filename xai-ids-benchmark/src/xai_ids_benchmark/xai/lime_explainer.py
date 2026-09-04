"""LIME explainer wrapper (lime.lime_tabular.LimeTabularExplainer)."""
from __future__ import annotations

import numpy as np

from .base import Explainer


class LimeExplainer(Explainer):
    name = "LIME"
    family = "post_hoc_local"

    def __init__(self, model, train_x, train_y, feature_names, device="cpu",
                 num_features: int = 10, num_samples: int = 1000,
                 random_state: int = 20260827, mode: str = "classification", **kw):
        super().__init__(model, train_x, train_y, feature_names, device, **kw)
        self.num_features = num_features
        self.num_samples = num_samples
        self.random_state = random_state
        self.mode = mode
        import lime
        import lime.lime_tabular
        self._lime = lime.lime_tabular.LimeTabularExplainer(
            train_x, feature_names=feature_names, class_names=["benign", "attack"],
            discretize_continuous=False, mode=mode, random_state=random_state,
        )

    def _predict_fn(self, x: np.ndarray) -> np.ndarray:
        # Return probabilities for class 1 (attack) in shape (n, 2).
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(x)
            return p
        # PyTorch model.
        import torch
        with torch.no_grad():
            logits = self.model(torch.tensor(x, dtype=torch.float32, device=self.device))
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        return np.stack([1 - probs, probs], axis=1)

    def explain(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), len(self.feature_names)), dtype=np.float32)
        for i, row in enumerate(X):
            try:
                exp = self._lime.explain_instance(
                    row, self._predict_fn, num_features=len(self.feature_names),
                    num_samples=self.num_samples, labels=(1,)
                )
                # exp.as_map()[1] is [(feat_idx, weight), ...]
                for feat_idx, w in exp.as_map()[1]:
                    out[i, feat_idx] = w
            except Exception:
                # Leave zeros on failure (coverage tracked separately if needed).
                pass
        return out
