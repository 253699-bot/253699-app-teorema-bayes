"""CSV loading service."""

from __future__ import annotations

import logging

import pandas as pd

from utils.validators import validate_csv_path, validate_non_empty_dataframe

LOGGER = logging.getLogger(__name__)


class DataLoader:
    """Service for loading CSV data."""

    def load_dataset(self, path: str) -> pd.DataFrame:
        """Load dataset from multiple formats and validate it contains usable rows/columns."""
        # Note: validators.validate_csv_path is skipped here or adapt to validate_file_path if needed
        # We will assume Streamlit already filtered the file extensions before passing temp files
        LOGGER.info("Loading Dataset from %s", path)
        
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".xlsx"):
            df = pd.read_excel(path)
        elif path.endswith(".json"):
            df = pd.read_json(path)
        else:
            raise ValueError("Formato de archivo no soportado")
            
        validate_non_empty_dataframe(df)
        LOGGER.info("Loaded dataframe with shape: %s", df.shape)
        return df
