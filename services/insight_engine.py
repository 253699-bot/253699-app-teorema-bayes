"""Automated statistical insight generator."""

from __future__ import annotations

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

import pandas as pd

import os
import dotenv
import google.generativeai as genai

from domain.models import ClassificationMetrics, ProbabilityComputation

# Load env variables and configure Gemini
dotenv.load_dotenv()
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
except Exception as e:
    LOGGER.warning("Could not configure Gemini API: %s", e)

@dataclass
class InsightReport:
    """Natural language insight output."""

    highlights: list[str]
    ai_conclusions: str = ""


class InsightEngine:
    """Generate concise data-driven insights from model outputs."""

    def generar_insights_con_ia(self, metricas: dict, objetivo: str, evidencia: str) -> str:
        """Call Gemini to explain metrics and recommend business actions."""
        prompt = (
            f"Eres un analista de datos experto. Basado en un modelo Naive Bayes con exactitud "
            f"de {metricas.get('accuracy', 0):.2%}, y analizando cómo la evidencia '{evidencia}' "
            f"afecta al objetivo '{objetivo}', redacta 3 viñetas breves en español explicando "
            f"qué significa esto y dando una recomendación de negocio."
        )
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"No se pudieron generar insights con IA. Revisa tu API Key o conexion: {e}"

    def generate(
        self,
        df: pd.DataFrame,
        target_column: str,
        probability_result: ProbabilityComputation,
        metrics: ClassificationMetrics,
        feature_importance: list[tuple[str, float]],
    ) -> InsightReport:
        """Generate prioritized insight bullets."""
        highlights: list[str] = []

        base_rate = probability_result.p_a
        posterior = probability_result.p_a_given_b

        highlights.append(
            (
                f"Tasa base de evento ({target_column}) = {base_rate:.2%}; "
                f"con evidencia ({probability_result.evidence_name}) sube a {posterior:.2%}."
            )
        )

        if metrics.sensitivity < 0.65:
            highlights.append(
                "Alerta: sensibilidad baja. El modelo podría estar perdiendo eventos anómalos reales."
            )
        if metrics.specificity < 0.65:
            highlights.append(
                "Alerta: especificidad baja. Existen falsos positivos relevantes."
            )

        top_features = feature_importance[:3]
        if top_features:
            formatted = ", ".join([f"{name} ({score:.3f})" for name, score in top_features])
            highlights.append(f"Variables más influyentes (señal aproximada): {formatted}.")

        n_rows = len(df)
        n_cols = len(df.columns)
        highlights.append(
            f"Dataset analizado: {n_rows} filas y {n_cols} columnas."
        )
        
        # Auto-generate AI insights combining the metrics map to strings
        metrics_dict = {
            "accuracy": metrics.accuracy,
            "sensitivity": metrics.sensitivity,
            "specificity": metrics.specificity
        }
        
        ai_text = self.generar_insights_con_ia(
            metricas=metrics_dict,
            objetivo=target_column,
            evidencia=probability_result.evidence_name
        )

        return InsightReport(highlights=highlights, ai_conclusions=ai_text)
