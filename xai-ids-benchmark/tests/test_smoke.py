"""Smoke test: run the benchmark on tiny synthetic data to verify the
full pipeline wiring (no real datasets required).

Run:  python -m pytest tests/test_smoke.py -q
   or: python tests/test_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _make_synthetic_ds(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, 8)).astype(np.float32)
    # Attack = nonlinear rule on features 0,1,2.
    p_attack = 1 / (1 + np.exp(-(x[:, 0] + 0.5 * x[:, 1] - x[:, 2])))
    y = (p_attack > 0.5).astype(int)
    fam = np.where(y == 0, "benign", "dos")
    cols = [f"f{i}" for i in range(8)] + ["label"]
    df = pd.DataFrame(np.concatenate([x, y.reshape(-1, 1)], axis=1), columns=cols)
    df["label"] = df["label"].map({0: "benign", 1: "dos"})
    df["family"] = fam
    return df, cols[:-1]


def test_pipeline():
    from xai_ids_benchmark.data.splits import dedup_and_partition
    from xai_ids_benchmark.data.preprocessing import preprocess
    from xai_ids_benchmark.models.cnn1d import CNN1D
    from xai_ids_benchmark.models.ft_transformer import FTTransformer
    from xai_ids_benchmark.runner import train_model

    df, feats = _make_synthetic_ds()
    train, val, test, _ = dedup_and_partition(df, feats, "label", test_size=0.3, val_size=0.2)
    pp = preprocess(train, val, test, "label", {"benign": ["benign"], "dos": ["dos"]})
    Xtr, ytr = pp["X_train"], pp["y_train"]
    Xva, yva = pp["X_val"], pp["y_val"]

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for Model in (CNN1D, FTTransformer):
        m = Model(Xtr.shape[1])
        rep = train_model(m, Xtr, ytr, Xva, yva, max_epochs=3, device=dev)
        assert rep["best_macro_f1"] >= 0.0
    print("smoke test OK")


if __name__ == "__main__":
    test_pipeline()
