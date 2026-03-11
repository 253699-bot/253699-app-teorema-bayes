"""Time series chart component."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_time_series(df: pd.DataFrame, datetime_col: str, target_col: str) -> plt.Figure:
    """Build temporal trend chart for anomaly ratio."""
    temp = df[[datetime_col, target_col]].copy()
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not infer format.*")
        temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
        
    temp = temp.dropna(subset=[datetime_col])

    temp["period"] = temp[datetime_col].dt.to_period("M").dt.to_timestamp()
    grouped = temp.groupby("period", as_index=False)[target_col].mean()
    grouped = grouped.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=110, facecolor="#F4F9E9")
    ax.set_facecolor("#F4F9E9")
    sns.lineplot(data=grouped, x="period", y=target_col, marker="o", color="#153243", ax=ax)
    ax.set_title("Evolución temporal de la tasa de evento")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Tasa de evento")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig
