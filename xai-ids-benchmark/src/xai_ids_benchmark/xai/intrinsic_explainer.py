"""Intrinsic reference explainer: a small decision tree whose path provides
intrinsic (non-post-hoc) explanations. Bounding post-hoc vs intrinsic
faithfulness is part of the proposal design (§5.3).
"""
from __future__ import annotations

import numpy as np

from .base import Explainer
from sklearn.tree import DecisionTreeClassifier


class IntrinsicExplainer(Explainer):
    name = "Intrinsic reference"
    family = "intrinsic"

    def __init__(self, model, train_x, train_y, feature_names, device="cpu",
                 max_depth: int = 6, random_state: int = 20260827, **kw):
        super().__init__(model, train_x, train_y, feature_names, device, **kw)
        # The intrinsic reference is itself an interpretable model trained on
        # the same data, independent of the DL model being explained. Its
        # feature_importances_ serves as the explanation signal.
        self.tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.tree.fit(train_x, train_y)

    def explain(self, X: np.ndarray) -> np.ndarray:
        # Per-instance attribution: feature_importances_ tiled per instance
        # plus a path-based component via the decision path.
        fi = self.tree.feature_importances_.astype(np.float32)
        out = np.tile(fi, (len(X), 1))
        # Add path-based indicator: 1 for features used in the leaf path.
        try:
            paths = self.tree.decision_path(X).toarray()
            # decision_path shape: (n, n_nodes). Map nodes to features.
            node_feature = self.tree.tree_.feature  # feature index per node
            feat_path = np.zeros_like(out)
            for i in range(len(X)):
                used_nodes = np.where(paths[i] > 0)[0]
                used_feats = set(node_feature[n] for n in used_nodes if node_feature[n] >= 0)
                for f in used_feats:
                    feat_path[i, f] = 1.0
            out = out * feat_path  # zero out unused features; keep importances on used
        except Exception:
            pass
        return out
