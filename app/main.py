"""Application entrypoint for Streamlit."""

from __future__ import annotations

import logging
import streamlit as st

from presentation.ui import render_ui

def configure_logging() -> None:
    """Configure global logging for the app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def main() -> None:
    """Initialize and run Streamlit UI."""
    configure_logging()
    
    st.set_page_config(
        page_title="Plataforma Bayesiana de Análisis",
        page_icon=":material/query_stats:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    render_ui()

if __name__ == "__main__":
    main()
