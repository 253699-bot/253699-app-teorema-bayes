"""Histogram chart component."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def build_histogram(df: pd.DataFrame, column: str, bins: int = 30) -> plt.Figure:
    """Build histogram for numeric column."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=110, facecolor="#F4F9E9")
    ax.set_facecolor("#F4F9E9")
    sns.histplot(df[column].dropna(), bins=bins, kde=True, color="#284B63", ax=ax)
    ax.set_title(f"Distribución: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig
