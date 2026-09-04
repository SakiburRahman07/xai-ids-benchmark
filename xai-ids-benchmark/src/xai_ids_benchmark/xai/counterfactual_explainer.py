"""Counterfactual explainer wrapper (dice-ml).

Returns an attribution matrix compatible with the others by using
counterfactual feature *change magnitudes* as the attribution signal:
features that must change to flip the prediction are treated as
high-attribution. Coverage (1=CF found, 0=failed) is reported per instance.
"""
from __future__ import annotations

import numpy as np

from .base import Explainer


class CounterfactualExplainer(Explainer):
    name = "Counterfactual"
    family = "post_hoc_actionable"

    def __init__(self, model, train_x, train_y, feature_names, device="cpu",
                 total_CFs: int = 3, desired_class: int = 0,
                 proximity_weight: float = 0.5, sparsity_weight: float = 0.5,
                 diversity_weight: float = 1.0, max_iter: int = 200, **kw):
        super().__init__(model, train_x, train_y, feature_names, device, **kw)
        self.total_CFs = total_CFs
        self.desired_class = desired_class
        self.max_iter = max_iter
        self._coverage: np.ndarray | None = None
        self._build_dice()

    def _build_dice(self):
        import dice_ml
        import pandas as pd
        cols = list(self.feature_names) + ["label"]
        arr = np.concatenate([self.train_x, self.train_y.reshape(-1, 1)], axis=1)
        d = pd.DataFrame(arr, columns=cols)
        d["label"] = d["label"].astype(int)
        self._dice_data = dice_ml.Data(dataframe=d, continuous_features=list(self.feature_names),
                                      outcome_name="label")
        # Sklearn-backed model (we use a sklearn surrogate if the DL model is PyTorch,
        # because dice's PyTorch backend requires specific model interfaces).
        from sklearn.ensemble import GradientBoostingClassifier
        self._surrogate = GradientBoostingClassifier(random_state=42).fit(self.train_x, self.train_y)
        self._dice_model = dice_ml.Model(model=self._surrogate, backend="sklearn")
        self._dice = dice_ml.Dice(self._dice_data, self._dice_model)

    def explain(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd
        out = np.zeros((len(X), len(self.feature_names)), dtype=np.float32)
        cov = np.ones(len(X), dtype=int)
        df = pd.DataFrame(X, columns=list(self.feature_names))
        for i in range(len(X)):
            try:
                cf = self._dice.generate_counterfactuals(
                    df.iloc[[i]], total_CFs=self.total_CFs,
                    desired_class=self.desired_class,
                )
                if cf is None or len(cf.cf_examples_list[0].final_cfs_df) == 0:
                    cov[i] = 0
                    continue
                cfs = cf.cf_examples_list[0].final_cfs_df.to_numpy(dtype=np.float32)
                # Attribution = mean absolute change across the CFs.
                delta = np.abs(cfs - X[i]).mean(axis=0)
                out[i] = delta
            except Exception:
                cov[i] = 0
        self._coverage = cov
        return out

    def coverage(self, X: np.ndarray) -> np.ndarray:
        if self._coverage is None:
            self.explain(X)
        return self._coverage
