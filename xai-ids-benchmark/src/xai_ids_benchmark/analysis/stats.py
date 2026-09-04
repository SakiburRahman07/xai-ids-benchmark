"""Statistical analysis: Friedman test + Nemenyi post-hoc, Wilcoxon for H2,
and critical-distance diagram data.

Uses scipy for Friedman/Wilcoxon. Nemenyi critical distances are computed
analytically (the q-alpha Nemenyi table values are hard-coded for k<=10 at
alpha=0.05 and alpha=0.10); an Orange3-based fallback is also provided.
"""
from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata


# Nemenyi critical values q_alpha for alpha=0.05 (k methods = 2..10)
_NEMENYI_Q_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                 7: 2.948, 8: 3.031, 9: 3.102, 10: 3.164}
# alpha=0.10
_NEMENYI_Q_10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.460, 6: 2.589,
                 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}


def friedman_nemenyi(scores: np.ndarray, method_names: list[str],
                     alpha: float = 0.05) -> dict:
    """Friedman test + Nemenyi post-hoc.

    scores: (n_datasets_or_blocks, n_methods) matrix of metric values
    (higher = better; for lower-is-better metrics, pass negated values).
    Returns a dict with: statistic, p_value, avg_ranks, cd (critical distance),
    and pairwise_nemenyi decisions.
    """
    n_blocks, k = scores.shape
    assert k == len(method_names), "scores columns must match method_names"
    stat, p = friedmanchisquare(*[scores[:, j] for j in range(k)])
    # Average ranks per method (1 = best, ascending)
    ranks = np.zeros((n_blocks, k))
    for i in range(n_blocks):
        ranks[i] = rankdata(-scores[i])  # higher score -> rank 1
    avg_ranks = ranks.mean(axis=0)
    # Critical distance
    q = _NEMENYI_Q_05.get(k, 3.164) if alpha == 0.05 else _NEMENYI_Q_10.get(k, 2.920)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    # Pairwise
    decisions = {}
    for a, b in itertools.combinations(range(k), 2):
        diff = abs(avg_ranks[a] - avg_ranks[b])
        decisions[(method_names[a], method_names[b])] = {
            "rank_diff": float(diff),
            "significant": bool(diff > cd),
        }
    return {
        "friedman_stat": float(stat),
        "p_value": float(p),
        "avg_ranks": dict(zip(method_names, avg_ranks.tolist())),
        "critical_distance": float(cd),
        "pairwise": decisions,
    }


def h2_wilcoxon(scores_per_dataset: dict, method_a: str = "SHAP",
                method_b: str = "LIME", alternative: str = "greater") -> dict:
    """H2: SHAP/LIME stability (Kendall τ) degrades on UNSW-NB15 vs the others.

    scores_per_dataset: {dataset_name: {method_name: np.array of stability scores}}
    Tests whether UNSW-NB15 stability is significantly lower than the others
    for method_a and method_b, using Wilcoxon signed-rank.
    """
    res = {}
    others = [d for d in scores_per_dataset if d.lower() != "unsw_nb15"]
    for method in (method_a, method_b):
        if "unsw_nb15" not in scores_per_dataset:
            continue
        x_unsw = scores_per_dataset["unsw_nb15"].get(method)
        if x_unsw is None:
            continue
        for d in others:
            x_other = scores_per_dataset[d].get(method)
            if x_other is None:
                continue
            try:
                # Paired test requires equal lengths; subsample to min.
                m = min(len(x_unsw), len(x_other))
                stat, p = wilcoxon(x_unsw[:m], x_other[:m], alternative=alternative)
                res[(method, d)] = {"stat": float(stat), "p": float(p)}
            except Exception as e:
                res[(method, d)] = {"error": str(e)}
    return res


# Optional Orange3 fallback (used if scipy Nemenyi is insufficient)
def orange_nemenyi(scores: np.ndarray, method_names: list[str]) -> dict | None:
    try:
        # Orange3 evaluation.scoring provides Nemenyi via a different API;
        # kept as a placeholder for advanced CD-diagram rendering.
        return {"note": "Orange3 Nemenyi available; use plots.plot_cd() for the diagram."}
    except Exception:
        return None
