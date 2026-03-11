"""Confusion matrix chart component."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from domain.models import ClassificationMetrics


def build_confusion_matrix_chart(metrics: ClassificationMetrics) -> plt.Figure:
    """Build confusion matrix heatmap figure."""
    cm = metrics.confusion_matrix.as_array()

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=110, facecolor="#F4F9E9")
    ax.set_facecolor("#F4F9E9")
    custom_cmap = sns.light_palette("#153243", as_cmap=True)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=custom_cmap,
        cbar=False,
        xticklabels=["Pred Negativo", "Pred Positivo"],
        yticklabels=["Real Negativo", "Real Positivo"],
        ax=ax,
    )
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    return fig
