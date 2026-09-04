"""Preprocessing: categorical encoding, scaling, feature alignment across
datasets. Produces (X, y, family, feature_names) ready for the models.

Important: all fitting (scaler/encoder) is done on TRAIN ONLY, then applied to
val/test, to respect the leakage-control protocol.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _select_numeric_features(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c.lower() not in ("label", "attack_cat")]


def _onehot_encode(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """One-hot encode categorical columns; fit categories on train only."""
    if not cat_cols:
        return train, val, test, []
    # Combine with a category fit on train.
    cats = {}
    for c in cat_cols:
        if c in train.columns:
            cats[c] = sorted(train[c].astype(str).unique().tolist())
    def enc(df: pd.DataFrame) -> pd.DataFrame:
        for c, levels in cats.items():
            for lvl in levels:
                df[f"{c}__{lvl}"] = (df[c].astype(str) == lvl).astype(float)
        return df.drop(columns=list(cats.keys()))
    train, val, test = enc(train.copy()), enc(val.copy()), enc(test.copy())
    # Align columns (val/test may miss some levels; add zero cols).
    cols = train.columns.tolist()
    for d in (val, test):
        for c in cols:
            if c not in d.columns:
                d[c] = 0.0
        # Remove unseen categories in val/test (avoid leakage).
        for c in list(d.columns):
            if c not in cols:
                d = d.drop(columns=[c])
    return train[cols], val[cols], test[cols], cols


def preprocess(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    families_cfg: dict,
    cat_cols: list[str] | None = None,
) -> dict:
    """Produce the standardized benchmark tensors.

    Returns a dict with keys:
      X_train, X_val, X_test : np.ndarray (float32)
      y_train, y_val, y_test : np.ndarray (int, 0=benign/other, 1=attack for OvR)
      fam_train, fam_val, fam_test : list[str] (attack family per row)
      feature_names : list[str]
      scaler : fitted StandardScaler
    For multi-class family classification the runner builds per-family OvR
    targets from the family labels directly; here we provide a binary
    benign-vs-attack target aligned with the proposal's 'binary OvR' framing.
    """
    cat_cols = cat_cols or []
    feature_cols = _select_numeric_features(train)
    # One-hot encode categoricals (if any) — fit on train.
    train_e, val_e, test_e, onehot_cols = _onehot_encode(
        train, val, test, [c for c in cat_cols if c in train.columns]
    )
    # Keep numeric feature columns only (one-hot cols are already numeric);
    # exclude the target and any leftover non-numeric columns (e.g., 'family').
    feature_cols = [c for c in train_e.columns
                    if c != target_col and c not in cat_cols
                    and pd.api.types.is_numeric_dtype(train_e[c])]
    Xtr = train_e[feature_cols].astype(np.float32)
    Xva = val_e[feature_cols].astype(np.float32)
    Xte = test_e[feature_cols].astype(np.float32)

    # Replace inf/NaN with 0 (some IDS datasets have inf values).
    for d in (Xtr, Xva, Xte):
        d.replace([np.inf, -np.inf], np.nan, inplace=True)
        d.fillna(0.0, inplace=True)

    scaler = StandardScaler().fit(Xtr.to_numpy())
    Xtr = scaler.transform(Xtr.to_numpy()).astype(np.float32)
    Xva = scaler.transform(Xva.to_numpy()).astype(np.float32)
    Xte = scaler.transform(Xte.to_numpy()).astype(np.float32)

    # Binary target: benign vs attack (OvR family handled by runner per family).
    def _bin_y(df: pd.DataFrame) -> np.ndarray:
        y = df[target_col].astype(str).str.lower().str.strip()
        benign = families_cfg.get("benign", ["benign", "normal"])
        return np.array([0 if any(y.iloc[i] == b or y.iloc[i].startswith(b) for b in benign) else 1
                         for i in range(len(y))], dtype=np.int64)

    ytr = _bin_y(train)
    yva = _bin_y(val)
    yte = _bin_y(test)

    def _fam(df: pd.DataFrame) -> list[str]:
        from .loaders import family_label
        return [family_label(v, families_cfg) for v in df[target_col].values]

    return dict(
        X_train=Xtr, X_val=Xva, X_test=Xte,
        y_train=ytr, y_val=yva, y_test=yte,
        fam_train=_fam(train), fam_val=_fam(val), fam_test=_fam(test),
        feature_names=feature_cols, scaler=scaler,
    )
