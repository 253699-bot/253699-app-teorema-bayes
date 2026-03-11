"""View models exchanged between controller and UI."""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.models import ClassificationMetrics, DetectedSchema, ProbabilityComputation
from services.insight_engine import InsightReport


@dataclass
class DatasetSummaryVM:
    """Dataset summary shown in UI."""

    rows: int
    columns: int
    schema: DetectedSchema


@dataclass
class AnalysisResultVM:
    """Complete analysis payload for rendering."""

    probability: ProbabilityComputation
    metrics: ClassificationMetrics
    insights: InsightReport
    feature_signal: list[tuple[str, float]] = field(default_factory=list)
