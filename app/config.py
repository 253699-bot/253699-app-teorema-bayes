"""Application configuration module."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Global configuration for the application."""

    app_title: str = "APLICACION BAYESIANA (DETECCION DE ANOMALIAS)"
    window_width: int = 1400
    window_height: int = 920
    test_size: float = 0.2
    random_state: int = 42
    min_numeric_unique: int = 10
    default_hist_bins: int = 30
    epsilon: float = 1e-12
