"""Cost metrics: explanation generation latency + GPU memory + scaling curve.
Extends Tolay 2026 (TreeSHAP 700.8s vs 1.47s for XGBoost) to the DL-IDS setting.
"""
from __future__ import annotations

import time

import numpy as np


class Cost:
    def __init__(self, explain_fn, X: np.ndarray, repeats: int = 5,
                 warmup: int = 10, sample_sizes: list[int] = (1000, 5000, 10000, 50000)):
        self.explain_fn = explain_fn   # callable(X) -> attributions
        self.X = X
        self.repeats = repeats
        self.warmup = warmup
        self.sample_sizes = list(sample_sizes)

    def evaluate(self) -> dict:
        out = {}
        out["latency_per_instance_ms"] = self._latency()
        out["scaling_curve_ms"] = self._scaling()
        out["gpu_memory_mb"] = self._gpu_memory()
        return out

    def _latency(self) -> float:
        # Warmup
        n = min(self.warmup, len(self.X))
        for _ in range(3):
            self.explain_fn(self.X[:n])
        # Measure
        times = []
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            self.explain_fn(self.X[:max(n, 100)])
            times.append((time.perf_counter() - t0) * 1000.0)
        per_inst = np.mean(times) / max(n, 100)
        return float(per_inst)

    def _scaling(self) -> dict:
        res = {}
        for sz in self.sample_sizes:
            n = min(sz, len(self.X))
            try:
                t0 = time.perf_counter()
                self.explain_fn(self.X[:n])
                res[sz] = (time.perf_counter() - t0) * 1000.0
            except Exception:
                res[sz] = None
        return res

    def _gpu_memory(self) -> float | None:
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            torch.cuda.reset_peak_memory_stats()
            self.explain_fn(self.X)
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        except Exception:
            return None
