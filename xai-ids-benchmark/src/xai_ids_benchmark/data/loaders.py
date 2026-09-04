"""Dataset loaders for CICIDS2017, UNSW-NB15, CICIoT2023.

All three datasets require a manual download step (UNB/UNSW host them behind a
form). The loaders auto-detect files placed under data_raw/<dataset>/ and
concatenate/normalize them into a single DataFrame with a consistent
`(X, y, family)` interface.

If files are missing, the loader prints the manual_instructions from the
config and raises a FileNotFoundError (so the Colab notebook can catch and
guide the user).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Tuple

import pandas as pd


def _find_files(glob_pattern: str, base: str | os.PathLike = ".") -> list[str]:
    # Patterns in config are relative to repo root.
    full = os.path.join(base, glob_pattern)
    files = sorted(glob.glob(full, recursive=True))
    return files


def _normalize_labels(series: pd.Series, how: str = "lower_strip") -> pd.Series:
    if how == "lower_strip":
        return series.astype(str).str.strip().str.lower()
    return series.astype(str).str.strip()


def load_cicids2017(cfg: dict, base: str | os.PathLike = ".") -> pd.DataFrame:
    files = _find_files(cfg["files_glob"], base)
    if not files:
        raise FileNotFoundError(
            f"No CICIDS2017 CSVs found under {cfg['files_glob']}.\n{cfg.get('manual_instructions','')}"
        )
    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        # Some CICIDS2017 CSVs have a leading space in column names.
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    target = cfg["target_column"]
    # Normalize target (e.g., 'BENIGN' -> 'benign')
    df[target] = _normalize_labels(df[target], cfg.get("label_normalize", "lower_strip"))
    # Drop configured leakage columns if present.
    for c in cfg.get("drop_columns", []):
        if c in df.columns:
            df = df.drop(columns=[c])
    # Coerce numeric columns; non-numeric (e.g., categorical) left for preprocessing.
    return df


def load_unsw_nb15(cfg: dict, base: str | os.PathLike = ".") -> pd.DataFrame:
    files = _find_files(cfg["files_glob"], base)
    if not files:
        raise FileNotFoundError(
            f"No UNSW-NB15 CSVs found under {cfg['files_glob']}.\n{cfg.get('manual_instructions','')}"
        )
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    target = cfg["target_column"].lower()
    df = df.rename(columns={df.columns[df.columns.str.contains(target, na=False)][0]: "attack_cat"})
    for c in cfg.get("drop_columns", []):
        if c in df.columns:
            df = df.drop(columns=[c])
    df["attack_cat"] = _normalize_labels(df["attack_cat"], "lower_strip")
    return df


def load_ciciot2023(cfg: dict, base: str | os.PathLike = ".") -> pd.DataFrame:
    files = _find_files(cfg["files_glob"], base)
    if not files:
        raise FileNotFoundError(
            f"No CICIoT2023 CSVs found under {cfg['files_glob']}.\n{cfg.get('manual_instructions','')}"
        )
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    target = cfg["target_column"].lower()
    df[target] = _normalize_labels(df[target], "lower_strip")
    for c in cfg.get("drop_columns", []):
        if c in df.columns:
            df = df.drop(columns=[c])
    return df


_LOADERS = {
    "cicids2017": load_cicids2017,
    "unsw_nb15": load_unsw_nb15,
    "ciciot2023": load_ciciot2023,
}


def load_dataset(name: str, cfg: dict, base: str | os.PathLike = ".") -> pd.DataFrame:
    """Dispatch to the right loader by dataset key."""
    if name not in _LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(_LOADERS)}")
    return _LOADERS[name](cfg, base)


def family_label(row_label: str, families: dict) -> str:
    """Map a raw label to an attack-family bucket (e.g., 'benign', 'dos', ...)."""
    rl = str(row_label).strip().lower()
    for fam, members in families.items():
        if any(rl.startswith(m) or rl == m for m in members):
            return fam
    return "other"
