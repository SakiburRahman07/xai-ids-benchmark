"""Main benchmark runner: full factorial 4 (XAI) × 2 (DL) × 3 (dataset).

Orchestrates: data load → leakage-controlled split → preprocess → train DL
models to a matched performance budget → run each XAI method → evaluate all
four metric families → log raw metric traces → run Friedman/Nemenyi.

Usage (CLI):
    python scripts/run_benchmark.py --config-dir config --quick

In Colab:
    See notebooks/run_in_colab.ipynb.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .utils import set_seed, load_configs, get_logger, ensure_dir
from .data.loaders import load_dataset
from .data.splits import train_val_test_split
from .data.preprocessing import preprocess
from .models.cnn1d import CNN1D
from .models.ft_transformer import FTTransformer
from .xai.shap_explainer import ShapExplainer
from .xai.lime_explainer import LimeExplainer
from .xai.counterfactual_explainer import CounterfactualExplainer
from .xai.intrinsic_explainer import IntrinsicExplainer
from .metrics.faithfulness import Faithfulness
from .metrics.stability import Stability
from .metrics.cost import Cost
from .metrics.proxy_utility import ProxyUtility
from .analysis.stats import friedman_nemenyi, h2_wilcoxon


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model: torch.nn.Module, Xtr, ytr, Xva, yva, max_epochs=50,
                patience=8, batch_size=1024, lr=1e-3, device="cpu") -> dict:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device).reshape(-1, 1)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    yv = torch.tensor(yva, dtype=torch.float32, device=device).reshape(-1, 1)
    best, best_ep, no_improve = -1.0, 0, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xv)).cpu().numpy().ravel()
        from sklearn.metrics import f1_score
        f1 = f1_score(yva, (probs >= 0.5).astype(int), average="macro")
        if f1 > best:
            best, best_ep, no_improve = f1, ep, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return {"best_macro_f1": float(best), "best_epoch": best_ep}


def build_explainer(name: str, model, train_x, train_y, feat_names, device, cfg) -> Any:
    if name == "shap":
        return ShapExplainer(model, train_x, train_y, feat_names, device, **cfg["params"])
    if name == "lime":
        return LimeExplainer(model, train_x, train_y, feat_names, device, **cfg["params"])
    if name == "counterfactual":
        return CounterfactualExplainer(model, train_x, train_y, feat_names, device, **cfg["params"])
    if name == "intrinsic":
        return IntrinsicExplainer(model, train_x, train_y, feat_names, device, **cfg["params"])
    raise ValueError(f"Unknown XAI method '{name}'")


def run_one_cell(dataset_name: str, ds_cfg: dict, model_name: str, model_cfg: dict,
                 xai_cfgs: dict, metric_cfgs: dict, base: str, quick: bool,
                 logger) -> dict:
    seed = ds_cfg["split"]["random_state"]
    set_seed(seed)
    dev = _device()
    logger.info(f"[{dataset_name} | {model_name}] loading data...")
    try:
        df = load_dataset(dataset_name, ds_cfg, base=base)
    except FileNotFoundError as e:
        logger.error(f"Dataset {dataset_name} not available: {e}")
        return {"dataset": dataset_name, "model": model_name, "error": str(e)}

    if quick:
        df = df.sample(min(2000, len(df)), random_state=seed).reset_index(drop=True)
        logger.info(f"  quick mode: subsampled to {len(df)} rows")

    feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = ds_cfg["target_column"] if ds_cfg["target_column"] in df.columns else "attack_cat"
    if target_col in feat_cols:
        feat_cols.remove(target_col)

    logger.info(f"  splitting (leakage-controlled)...")
    train, val, test, split_report = train_val_test_split(
        df, feat_cols, target_col, protocol=ds_cfg["split"]["protocol"],
        test_size=ds_cfg["split"]["test_size"], val_size=ds_cfg["split"]["val_size"],
        random_state=seed, remove_label_collisions=ds_cfg["split"]["remove_label_collisions"],
    )
    logger.info(f"  split: {split_report}")

    pp = preprocess(train, val, test, target_col, ds_cfg.get("families", {}),
                    cat_cols=ds_cfg.get("proto_category_columns", []))
    Xtr, Xva, Xte = pp["X_train"], pp["X_val"], pp["X_test"]
    ytr, yva, yte = pp["y_train"], pp["y_val"], pp["y_test"]
    feat_names = pp["feature_names"]

    # Build model.
    n_feat = Xtr.shape[1]
    if model_name == "cnn1d":
        model = CNN1D(n_feat, **{k: v for k, v in model_cfg["params"].items()
                                if k in ("hidden_channels", "kernel_size", "dropout", "output_dim")})
    elif model_name == "ft_transformer":
        model = FTTransformer(n_feat, **{k: v for k, v in model_cfg["params"].items()
                                         if k in ("d_token", "n_blocks", "attention_dropout",
                                                  "ffn_dropout", "ffn_d_hidden", "output_dim")})
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    logger.info(f"  training {model_cfg['name']} to matched budget...")
    train_report = train_model(model, Xtr, ytr, Xva, yva, device=dev,
                               max_epochs=20 if quick else 50)
    logger.info(f"  train report: {train_report}")

    # Per-XAI evaluation.
    rows = []
    baseline = Xtr.mean(axis=0)
    for method_key, mcfg in xai_cfgs.items():
        if not mcfg.get("enabled", False):
            continue
        logger.info(f"  XAI method: {mcfg['name']}")
        t0 = time.perf_counter()
        try:
            expl = build_explainer(method_key, model, Xtr, ytr, feat_names, dev, mcfg)
            # Use a subset for stability/cost to keep runtime bounded.
            n_eval = min(500 if quick else 2000, len(Xte))
            X_eval = Xte[:n_eval]
            y_eval = yte[:n_eval]
            fam_eval = pp["fam_test"][:n_eval]
            attr = expl.explain(X_eval)
            cov = expl.coverage(X_eval)
        except Exception as e:
            logger.error(f"  XAI {mcfg['name']} failed: {e}")
            rows.append({"method": mcfg["name"], "model": model_cfg["name"],
                         "dataset": ds_cfg["name"], "error": str(e)})
            continue
        explain_fn = lambda X: expl.explain(X)
        # Metric families
        try:
            faith = Faithfulness(model, X_eval, y_eval, baseline, device=dev).evaluate(attr)
        except Exception as e:
            logger.warning(f"  faithfulness error: {e}")
            faith = {}
        try:
            stab = Stability(explain_fn, X_eval[:100]).evaluate(attr)
        except Exception as e:
            logger.warning(f"  stability error: {e}")
            stab = {}
        try:
            cost = Cost(explain_fn, X_eval).evaluate()
        except Exception as e:
            logger.warning(f"  cost error: {e}")
            cost = {}
        try:
            proxy = ProxyUtility(model, X_eval, y_eval, fam_eval, attr, device=dev).evaluate()
        except Exception as e:
            logger.warning(f"  proxy_utility error: {e}")
            proxy = {}
        rows.append({
            "method": mcfg["name"], "model": model_cfg["name"], "dataset": ds_cfg["name"],
            "vintage": ds_cfg["vintage"], "macro_f1": train_report.get("best_macro_f1"),
            "coverage": float(cov.mean()), "explain_time_s": time.perf_counter() - t0,
            "faithfulness": faith, "stability": stab, "cost": cost, "proxy_utility": proxy,
            "split_report": split_report,
        })
    return rows


def run_benchmark(config_dir: str, base: str = ".", out_dir: str = "results",
                  quick: bool = False, datasets: list[str] | None = None,
                  models: list[str] | None = None, methods: list[str] | None = None) -> Path:
    cfgs = load_configs(config_dir)
    logger = get_logger("xai_ids_benchmark", log_dir=out_dir)
    set_seed(20260827)
    out_dir_p = ensure_dir(out_dir)

    all_rows = []
    ds_cfgs = cfgs["datasets"]["datasets"]
    model_cfgs = cfgs["models"]["models"]
    xai_cfgs = {k: v for k, v in cfgs["xai"]["xai_methods"].items()
                if (methods is None or k in methods) and v.get("enabled", False)}

    ds_keys = datasets or list(ds_cfgs.keys())
    model_keys = models or list(model_cfgs.keys())

    for ds in ds_keys:
        for m in model_keys:
            try:
                rows = run_one_cell(ds, ds_cfgs[ds], m, model_cfgs[m], xai_cfgs,
                                    cfgs["metrics"]["metrics"], base, quick, logger)
                all_rows.extend(rows)
            except Exception as e:
                logger.error(f"Cell [{ds}|{m}] crashed: {e}")

    # Save raw traces
    raw_path = out_dir_p / "raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, default=str)
    logger.info(f"Saved raw results -> {raw_path}")

    # Build a tidy metric table for analysis
    flat = []
    for r in all_rows:
        if "error" in r:
            continue
        for fam in ("faithfulness", "stability", "cost", "proxy_utility"):
            fam_v = r.get(fam, {}) or {}
            for mk, mv in _flatten(fam_v).items():
                flat.append({
                    "dataset": r["dataset"], "vintage": r["vintage"],
                    "model": r["model"], "method": r["method"],
                    "metric_family": fam, "metric": mk, "value": mv,
                    "macro_f1": r.get("macro_f1"), "coverage": r.get("coverage"),
                })
    if flat:
        df = pd.DataFrame(flat)
        df.to_csv(out_dir_p / "tidy_metrics.csv", index=False)
        logger.info(f"Saved tidy metrics -> {out_dir_p / 'tidy_metrics.csv'}")
        _run_analysis(df, out_dir_p, logger)
    return out_dir_p


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _run_analysis(df: pd.DataFrame, out_dir: Path, logger) -> None:
    """Run Friedman/Nemenyi per metric and H2 Wilcoxon on stability."""
    from .analysis.stats import friedman_nemenyi, h2_wilcoxon
    results = {}
    # For each scalar metric, build (dataset_block, method) matrix.
    for metric, sub in df.groupby("metric"):
        if sub["value"].isna().all():
            continue
        pivot = sub.pivot_table(index="dataset", columns="method", values="value", aggfunc="mean")
        pivot = pivot.dropna()
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            continue
        # For metrics where higher is worse (e.g., latency, infidelity), negate.
        negate = any(t in metric.lower() for t in ("latency", "infidelity", "cost", "time_to_critical", "sparsity"))
        vals = -pivot.to_numpy() if negate else pivot.to_numpy()
        try:
            fn = friedman_nemenyi(vals, pivot.columns.tolist())
            results[metric] = fn
        except Exception as e:
            logger.warning(f"Friedman failed for {metric}: {e}")
    with open(out_dir / "friedman_nemenyi.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved Friedman/Nemenyi -> {out_dir / 'friedman_nemenyi.json'}")

    # H2: stability degradation on UNSW-NB15 (per-method stability scores)
    stab = df[df["metric_family"] == "stability"]
    per_ds = {}
    for ds, sub in stab.groupby("dataset"):
        per_ds[ds] = {m: sub[sub["method"] == m]["value"].dropna().to_numpy()
                      for m in sub["method"].unique()}
    try:
        h2 = h2_wilcoxon(per_ds)
        with open(out_dir / "h2_wilcoxon.json", "w", encoding="utf-8") as f:
            json.dump(h2, f, indent=2, default=str)
        logger.info(f"Saved H2 Wilcoxon -> {out_dir / 'h2_wilcoxon.json'}")
    except Exception as e:
        logger.warning(f"H2 Wilcoxon failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="XAI-IDS benchmark runner")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--base", default=".")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--quick", action="store_true", help="smoke run: subsample to 2k rows, 20 epochs")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()
    run_benchmark(args.config_dir, args.base, args.out_dir, args.quick,
                  args.datasets, args.models, args.methods)


if __name__ == "__main__":
    main()
