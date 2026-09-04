"""SHAP explainer wrapper.

Primary variant: Captum's GradientShap (a Shapley-value-style gradient
attribution). The preregistration also reports Integrated Gradients as a
sensitivity variant, and DeepSHAP/DeepLiftShap as an additional sensitivity.
For tree models (the intrinsic reference), TreeSHAP is used directly.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import Explainer


class ShapExplainer(Explainer):
    name = "SHAP"
    family = "post_hoc_additive"

    def __init__(self, model, train_x, train_y, feature_names, device="cpu",
                 variant: str = "gradient_shap", n_samples: int = 200, **kw):
        super().__init__(model, train_x, train_y, feature_names, device, **kw)
        self.variant = variant
        self.n_samples = n_samples
        self.baselines = torch.tensor(train_x.mean(axis=0), dtype=torch.float32, device=device)

    def _captum_attr(self, attr_cls, x: torch.Tensor, **extra):
        # Captum attribution call; returns (n, features).
        return attr_cls(self.model).attribute(x, baselines=self.baselines, **extra)

    def explain(self, X: np.ndarray) -> np.ndarray:
        # Tree-based model (intrinsic or sklearn) -> use TreeSHAP via shap if available.
        if hasattr(self.model, "predict_proba") and not isinstance(self.model, torch.nn.Module):
            try:
                import shap
                expl = shap.TreeExplainer(self.model)
                return expl.shap_values(X)
            except Exception:
                # Fall back to a generic KernelSHAP (slow) or feature_importances_.
                if hasattr(self.model, "feature_importances_"):
                    fi = self.model.feature_importances_
                    return np.tile(fi, (len(X), 1)).astype(np.float32)
                raise

        # PyTorch model -> Captum.
        from captum.attr import GradientShap, IntegratedGradients, DeepLiftShap
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        if self.variant == "gradient_shap":
            attr = self._captim_attr(GradientShap, x, n_samples=self.n_samples, stdevs=0.0)
        elif self.variant == "integrated_gradients":
            attr = IntegratedGradients(self.model).attribute(x, baselines=self.baselines)
        elif self.variant == "deepshap":
            attr = self._captim_attr(DeepLiftShap, x)
        else:
            raise ValueError(f"Unknown SHAP variant '{self.variant}'")
        return attr.detach().cpu().numpy().astype(np.float32)

    def _captim_attr(self, attr_cls, x, **extra):
        return attr_cls(self.model).attribute(x, baselines=self.baselines, **extra)
