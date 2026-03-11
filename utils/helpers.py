"""Shared utility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BINARY_TRUE_SET = {1, "1", True, "true", "yes", "y", "si", "sí", "t"}
BINARY_FALSE_SET = {0, "0", False, "false", "no", "n", "f"}


def normalize_binary_value(value: Any) -> int | None:
    """Normalize known binary-like values to 0/1."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        candidate = value.strip().lower()
    else:
        candidate = value

    if candidate in BINARY_TRUE_SET:
        return 1
    if candidate in BINARY_FALSE_SET:
        return 0
    return None


def safe_mode(series: pd.Series) -> Any:
    """Return mode safely even for empty columns."""
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else None
