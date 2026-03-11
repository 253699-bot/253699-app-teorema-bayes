"""Validation helpers for inputs and user actions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_csv_path(path: str) -> None:
    """Validate CSV path existence and extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    if p.suffix.lower() != ".csv":
        raise ValueError("El archivo debe tener extensión .csv")


def validate_non_empty_dataframe(df: pd.DataFrame) -> None:
    """Ensure dataframe has rows and columns."""
    if df.empty:
        raise ValueError("El CSV no contiene filas para procesar")
    if len(df.columns) == 0:
        raise ValueError("El CSV no contiene columnas")
