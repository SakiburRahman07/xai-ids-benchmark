"""Leakage-controlled splits for IDS benchmarks.

Implements the protocols cited in the proposal:
  * Tolay 2026 (arXiv:2608.10349): exact-feature hash dedup + deterministic
    hash-level partition; label-collision removal.
  * Bouke 2026 (arXiv:2606.29797): temporal split where timestamps exist;
    validation-only checkpoint selection and threshold calibration.

These re-splits prevent the train/test leakage that inflates the "99%+ accuracy"
claims common in the IDS literature (see XAI_IDS_Gap_Analysis.md Gap 4/6).
"""
from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _row_hash(values: np.ndarray) -> np.ndarray:
    """Deterministic per-row hash over numeric feature values.

    Uses blake2b of the packed bytes. Stable across runs and machines
    (hashlib is deterministic; unlike Python's built-in hash).
    """
    # Convert to a contiguous float array with NaN handled as a sentinel.
    arr = np.ascontiguousarray(values, dtype=np.float64)
    # Treat NaN as a distinct sentinel so rows differing only by NaN differ.
    arr = np.nan_to_num(arr, nan=-1.0e9, posinf=1.0e9, neginf=-1.0e9)
    out = np.empty(arr.shape[0], dtype=object)
    for i in range(arr.shape[0]):
        out[i] = hashlib.blake2b(arr[i].tobytes(), digest_size=16).hexdigest()
    return out


def dedup_and_partition(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.3,
    val_size: float = 0.2,
    random_state: int = 20260827,
    remove_label_collisions: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Hash-based dedup + deterministic hash-level partition (Tolay 2026).

    Returns (train, val, test, split_report).
    """
    report: dict = {"n_before": int(len(df))}

    # 1) Exact-feature dedup: drop duplicate feature vectors.
    hashes = _row_hash(df[feature_cols].to_numpy())
    df = df.assign(_hash=hashes)
    dup_mask = df.duplicated(subset=["_hash"], keep="first")
    df = df[~dup_mask].copy()
    report["n_after_dedup"] = int(len(df))
    report["n_duplicates_removed"] = int(dup_mask.sum())

    # 2) Label-collision removal: identical features with conflicting labels.
    if remove_label_collisions:
        lbl_counts = df.groupby("_hash")[target_col].nunique()
        colliding = lbl_counts[lbl_counts > 1].index
        n_before_collision = int(len(df))
        df = df[~df["_hash"].isin(colliding)].copy()
        report["n_label_collisions_removed"] = n_before_collision - int(len(df))
    else:
        report["n_label_collisions_removed"] = 0

    df = df.drop(columns=["_hash"])

    # 3) Deterministic hash-level partition: hash -> fold assignment, so the
    # same row never appears in two folds regardless of reshuffling.
    partition_hashes = np.array(
        [int(hashlib.blake2b(row.tobytes(), digest_size=8).hexdigest(), 16)
         for row in df[feature_cols].to_numpy()]
    )
    # Scale to [0,1) and assign folds deterministically.
    norm = (partition_hashes % 10000) / 10000.0
    df = df.assign(_fold_score=norm)

    test_mask = df["_fold_score"] < test_size
    remainder = df[~test_mask].copy()
    # val_size is a fraction of the remainder.
    val_threshold = test_size + (1 - test_size) * val_size
    val_mask = (remainder["_fold_score"] >= test_size) & (remainder["_fold_score"] < val_threshold)
    train_mask = remainder["_fold_score"] >= val_threshold

    test = df[test_mask].drop(columns=["_fold_score"])
    val = remainder[val_mask].drop(columns=["_fold_score"])
    train = remainder[train_mask].drop(columns=["_fold_score"])

    report.update(
        n_train=int(len(train)), n_val=int(len(val)), n_test=int(len(test)),
        random_state=random_state,
    )
    return train, val, test, report


def temporal_split(
    df: pd.DataFrame,
    time_col: str,
    test_size: float = 0.3,
    val_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Temporal split (Bouke 2026): train < val < test in time order.

    Requires a parseable timestamp column. Falls back to raise ValueError
    if unavailable so the caller can downgrade to hash_dedup.
    """
    if time_col not in df.columns:
        raise ValueError(f"temporal split requested but '{time_col}' not present")
    times = pd.to_datetime(df[time_col], errors="coerce")
    if times.isna().any():
        # If many missing, raise so caller decides; else drop NaNs.
        raise ValueError(f"temporal split: {int(times.isna().sum())} unparseable timestamps")
    df = df.assign(_t=times).sort_values("_t")
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * (1 - test_size) * val_size)
    test = df.iloc[n - n_test:].drop(columns=["_t"])
    val = df.iloc[n - n_test - n_val : n - n_test].drop(columns=["_t"])
    train = df.iloc[: n - n_test - n_val].drop(columns=["_t"])
    report = {"n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)), "split": "temporal"}
    return train, val, test, report


def train_val_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    protocol: str = "hash_dedup",
    time_col: str | None = None,
    test_size: float = 0.3,
    val_size: float = 0.2,
    random_state: int = 20260827,
    remove_label_collisions: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if protocol == "temporal" and time_col is not None:
        try:
            return temporal_split(df, time_col, test_size, val_size)
        except ValueError:
            # Graceful fallback to hash dedup if timestamps unusable.
            pass
    return dedup_and_partition(
        df, feature_cols, target_col, test_size, val_size, random_state,
        remove_label_collisions,
    )
