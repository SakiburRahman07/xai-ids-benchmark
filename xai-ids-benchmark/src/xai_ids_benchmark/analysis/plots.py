"""Plots: critical-distance diagram for the Friedman/Nemenyi analysis."""
from __future__ import annotations

import numpy as np


def plot_cd(avg_ranks: dict, cd: float, output_path: str | None = None,
            title: str = "Critical Distance Diagram") -> None:
    """Render a Nemenyi critical-distance diagram.

    avg_ranks: {method_name: avg_rank} (1 = best).
    cd: critical distance from friedman_nemenyi().
    """
    import matplotlib.pyplot as plt
    names = list(avg_ranks.keys())
    ranks = np.array(list(avg_ranks.values()))
    fig, ax = plt.subplots(figsize=(8, 3))
    # Axis from best (left) to worst (right).
    order = np.argsort(ranks)
    ax.set_xlim(ranks.min() - 0.5, ranks.max() + 0.5)
    ax.set_yticks([])
    for i, idx in enumerate(order):
        name = names[idx]
        ax.plot(ranks[idx], 0.5, "ko")
        # Alternate labels above/below to avoid overlap.
        y = 0.5 + (0.15 if i % 2 == 0 else -0.15)
        ax.text(ranks[idx], y, f"{name} ({ranks[idx]:.2f})", ha="center", va="center")
    # CD line
    ax.annotate("", xy=(ranks.min(), 0.2), xytext=(ranks.min() + cd, 0.2),
                arrowprops=dict(arrowstyle="<->", color="red"))
    ax.text(ranks.min() + cd / 2, 0.15, f"CD={cd:.2f}", color="red", ha="center")
    ax.set_title(title)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig
