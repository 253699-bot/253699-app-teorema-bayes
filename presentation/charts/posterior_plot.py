"""Posterior probability chart component."""

from __future__ import annotations

import matplotlib.pyplot as plt

from domain.models import ProbabilityComputation


def build_posterior_chart(prob: ProbabilityComputation) -> plt.Figure:
    """Build bar chart comparing prior vs posterior."""
    labels = ["P(A)", "P(A|B)"]
    values = [prob.p_a, prob.p_a_given_b]

    fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=110, facecolor="#F4F9E9")
    ax.set_facecolor("#F4F9E9")
    bars = ax.bar(labels, values, color=["#B4B8AB", "#153243"])
    ax.set_ylim(0, 1)
    ax.set_title("Comparación probabilística")
    ax.set_ylabel("Probabilidad")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.2%}", ha="center")

    fig.tight_layout()
    return fig
