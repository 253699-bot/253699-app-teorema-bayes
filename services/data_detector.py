"""Automatic schema detection service."""

from __future__ import annotations

import pandas as pd

from domain.models import DetectedSchema
from utils.helpers import normalize_binary_value


class DataDetector:
    """Detects numeric, categorical, datetime, and binary columns."""

    def __init__(self, min_numeric_unique: int = 10) -> None:
        self.min_numeric_unique = min_numeric_unique

    def detect_schema(self, df: pd.DataFrame) -> DetectedSchema:
        """Run robust schema detection over dataframe columns."""
        schema = DetectedSchema()

        for column in df.columns:
            series = df[column]
            if self._is_datetime(series):
                schema.datetime_columns.append(column)
            elif self._is_binary(series):
                schema.binary_columns.append(column)
            elif self._is_numeric(series):
                schema.numeric_columns.append(column)
            else:
                schema.categorical_columns.append(column)

        # Avoid overlapping category with binary columns.
        schema.categorical_columns = [
            c for c in schema.categorical_columns if c not in schema.binary_columns
        ]
        return schema

    def _is_numeric(self, series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series):
            return series.nunique(dropna=True) >= self.min_numeric_unique
        converted = pd.to_numeric(series, errors="coerce")
        ratio = converted.notna().mean()
        return ratio >= 0.9 and converted.nunique(dropna=True) >= self.min_numeric_unique

    def _is_datetime(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        sample = series.dropna().astype("string")
        if sample.empty:
            return False
            
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not infer format.*")
            parsed = pd.to_datetime(sample, errors="coerce", utc=False)
            
        parse_ratio = parsed.notna().mean()
        return parse_ratio >= 0.8

    def _is_binary(self, series: pd.Series) -> bool:
        sample = series.dropna()
        if sample.empty:
            return False
        normalized = sample.map(normalize_binary_value)
        return normalized.notna().mean() >= 0.95
