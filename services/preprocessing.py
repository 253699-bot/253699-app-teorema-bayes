"""Preprocessing service for feature/target preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from domain.models import DetectedSchema
from utils.helpers import normalize_binary_value


@dataclass
class SplitData:
    """Container for train/test split."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class PreprocessingService:
    """Prepare datasets for manual Naive Bayes training and evaluation."""

    def make_binary_target(
        self,
        series: pd.Series,
        positive_label: Any | None = None,
    ) -> tuple[pd.Series, Any]:
        """Transform target into binary labels based on selected positive class."""
        normalized = series.map(normalize_binary_value)
        if normalized.notna().mean() >= 0.95:
            y = normalized.fillna(0).astype(int)
            chosen_positive = 1 if positive_label is None else int(bool(positive_label))
            return y, chosen_positive

        non_null_values = series.dropna().unique().tolist()
        if len(non_null_values) < 2:
            raise ValueError("La variable objetivo necesita al menos 2 clases")

        chosen_positive = positive_label if positive_label is not None else non_null_values[0]
        y = (series == chosen_positive).astype(int)
        return y, 1

    def prepare_features(
        self,
        df: pd.DataFrame,
        schema: DetectedSchema,
        target_column: str,
    ) -> pd.DataFrame:
        """Create feature matrix using detected usable columns."""
        usable_columns = [
            c for c in (schema.numeric_columns + schema.binary_columns + schema.categorical_columns)
            if c != target_column
        ]
        if not usable_columns:
            raise ValueError("No hay columnas de características disponibles para entrenar")

        x = df[usable_columns].copy()

        for col in x.columns:
            if col in schema.numeric_columns:
                x[col] = pd.to_numeric(x[col], errors="coerce")
                x[col] = x[col].fillna(x[col].median())
            elif col in schema.binary_columns:
                x[col] = x[col].map(normalize_binary_value).fillna(0).astype(int)
            else:
                x[col] = x[col].astype("string").fillna("<MISSING>")

        return x

    def train_test_split(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        random_state: int,
    ) -> SplitData:
        """Simple deterministic train-test split without sklearn dependency."""
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size debe estar entre 0 y 1")

        data = x.copy()
        data["__target__"] = y.values
        shuffled = data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        split_idx = int(len(shuffled) * (1 - test_size))
        train = shuffled.iloc[:split_idx]
        test = shuffled.iloc[split_idx:]

        x_train = train.drop(columns=["__target__"])
        y_train = train["__target__"]
        x_test = test.drop(columns=["__target__"])
        y_test = test["__target__"]

        return SplitData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
