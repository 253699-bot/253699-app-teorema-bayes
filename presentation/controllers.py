"""Application controller coordinating UI and services."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.config import AppConfig
from domain.bayes_classifier import ManualNaiveBayes
from domain.metrics import classification_metrics
from domain.models import DetectedSchema, TrainArtifacts
from domain.probability_engine import compute_probability_report
from presentation.viewmodels import AnalysisResultVM, DatasetSummaryVM
from services.data_detector import DataDetector
from services.data_loader import DataLoader
from services.insight_engine import InsightEngine
from services.preprocessing import PreprocessingService

LOGGER = logging.getLogger(__name__)


class AppController:
    """Orchestrates data loading, modeling and reporting."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.loader = DataLoader()
        self.detector = DataDetector(min_numeric_unique=config.min_numeric_unique)
        self.preprocessing = PreprocessingService()
        self.insight_engine = InsightEngine()

        self.df: pd.DataFrame | None = None
        self.schema: DetectedSchema | None = None
        self.train_artifacts: TrainArtifacts | None = None
        self.model_df: pd.DataFrame | None = None
        self.last_result: AnalysisResultVM | None = None

    def load_dataset(self, path: str) -> DatasetSummaryVM:
        """Load dataset and detect schema."""
        self.df = self.loader.load_dataset(path)
        self.schema = self.detector.detect_schema(self.df)

        return DatasetSummaryVM(
            rows=len(self.df),
            columns=len(self.df.columns),
            schema=self.schema,
        )

    def analyze(
        self,
        target_column: str,
        positive_label: Any | None,
        evidence_column: str,
        threshold: float | None = None,
    ) -> AnalysisResultVM:
        """Run complete probability + Naive Bayes pipeline."""
        if self.df is None or self.schema is None:
            raise ValueError("Primero debes cargar un dataset")

        y_binary, positive_binary_label = self.preprocessing.make_binary_target(
            self.df[target_column],
            positive_label=positive_label,
        )

        model_df = self.df.copy()
        model_df[target_column] = y_binary
        self.model_df = model_df

        x = self.preprocessing.prepare_features(
            df=model_df,
            schema=self.schema,
            target_column=target_column,
        )

        split = self.preprocessing.train_test_split(
            x=x,
            y=y_binary,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        clf = ManualNaiveBayes(epsilon=self.config.epsilon)
        clf.fit(split.x_train, split.y_train)

        y_prob = clf.predict_proba(split.x_test, positive_label=positive_binary_label)
        y_pred = clf.predict(split.x_test, positive_label=positive_binary_label)

        metrics = classification_metrics(
            y_true=split.y_test,
            y_pred=y_pred,
            positive_label=positive_binary_label,
            epsilon=self.config.epsilon,
        )

        probability = compute_probability_report(
            data=model_df,
            target_column=target_column,
            positive_value=positive_binary_label,
            evidence_column=evidence_column,
            threshold=threshold,
            epsilon=self.config.epsilon,
        )

        feature_signal = self._compute_feature_signal(
            x=split.x_train,
            y=split.y_train,
            positive_label=positive_binary_label,
        )

        insights = self.insight_engine.generate(
            df=model_df,
            target_column=target_column,
            probability_result=probability,
            metrics=metrics,
            feature_importance=feature_signal,
        )

        self.train_artifacts = TrainArtifacts(
            feature_columns=list(split.x_train.columns),
            target_column=target_column,
            positive_label=positive_binary_label,
            y_true=split.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            metrics=metrics,
        )

        LOGGER.info("Analysis complete. Accuracy: %.4f", metrics.accuracy)

        result = AnalysisResultVM(
            probability=probability,
            metrics=metrics,
            insights=insights,
            feature_signal=feature_signal,
        )
        self.last_result = result
        return result

    def _compute_feature_signal(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        positive_label: Any,
    ) -> list[tuple[str, float]]:
        """Approximate feature influence using class separation signal."""
        pos_mask = y == positive_label
        neg_mask = ~pos_mask

        scores: list[tuple[str, float]] = []
        for col in x.columns:
            if pd.api.types.is_numeric_dtype(x[col]):
                pos_mean = float(x.loc[pos_mask, col].mean()) if pos_mask.any() else 0.0
                neg_mean = float(x.loc[neg_mask, col].mean()) if neg_mask.any() else 0.0
                score = abs(pos_mean - neg_mean)
            else:
                pos_mode = x.loc[pos_mask, col].mode(dropna=True)
                neg_mode = x.loc[neg_mask, col].mode(dropna=True)
                pos_v = pos_mode.iloc[0] if not pos_mode.empty else ""
                neg_v = neg_mode.iloc[0] if not neg_mode.empty else ""
                score = 1.0 if pos_v != neg_v else 0.0
            scores.append((col, float(score)))

        scores.sort(key=lambda t: t[1], reverse=True)
        return scores
