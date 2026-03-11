"""Streamlit user interface."""

from __future__ import annotations

import os
import tempfile
import streamlit as st

from app.config import AppConfig
from presentation.charts.confusion_matrix_plot import build_confusion_matrix_chart
from presentation.charts.histogram import build_histogram
from presentation.charts.posterior_plot import build_posterior_chart
from presentation.charts.time_series import build_time_series
from presentation.controllers import AppController
from services.pdf_generator import crear_reporte_pdf


def init_session_state() -> None:
    """Initialize session state variables."""
    if "controller" not in st.session_state:
        config = AppConfig()
        st.session_state.controller = AppController(config=config)
    
    if "dataset_summary" not in st.session_state:
        st.session_state.dataset_summary = None
        
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False

@st.dialog("Confirmar Ingesta")
def review_ingest_modal() -> None:
    """Show modal dialog to confirm file uploading."""
    uploaded_file = st.session_state.pending_file
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    valid_extensions = [".csv", ".xlsx", ".json"]
    
    if file_extension not in valid_extensions:
        st.error("Formato no soportado. Por favor, ingresa un archivo CSV, XLSX o JSON.")
        if st.button("Cancelar", use_container_width=True):
            st.session_state.pending_file = None
            st.rerun()
    else:
        st.write("Has seleccionado un archivo para su ingesta al Motor de Inferencia.")
        st.info(f"Nombre: {uploaded_file.name}\nTamaño: {uploaded_file.size} bytes")
        
        col_yes, col_no = st.columns(2)
        if col_yes.button("Sí, Limpiar y Calcular", use_container_width=True):
            controller: AppController = st.session_state.controller
            try:
                # Replicating the tempfile parsing from older structure
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                    
                st.session_state.dataset_summary = controller.load_dataset(tmp_path)
                st.session_state.last_uploaded_name = uploaded_file.name
                st.session_state.analysis_complete = False
                
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                    
            except Exception as exc:
                st.error(f"Ocurrió un error cargando el archivo: {exc}")
            
            # Clear pending to close modal and update visually
            st.session_state.pending_file = None
            st.rerun()
            
        if col_no.button("Cancelar", key="cancel_valid", use_container_width=True):
            st.session_state.pending_file = None
            st.rerun()


def render_ui() -> None:
    """Render the main Streamlit application layout."""
    init_session_state()
    controller: AppController = st.session_state.controller

    # Header & Footer CSS Injection + FontAwesome CDN
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <style>
            .fixed-header {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                background-color: #153243;
                color: #F4F9E9;
                z-index: 999;
                text-align: center;
                padding: 1rem 0;
                font-size: 1.5rem;
                font-weight: bold;
                border-bottom: 3px solid #EEF0EB;
            }
            .fixed-footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #153243;
                color: #B4B8AB;
                z-index: 999;
                text-align: center;
                padding: 0.5rem 0;
                font-size: 0.9rem;
            }
            /* Add padding to the main container so content doesn't hide behind fixed elements */
            .main .block-container {
                padding-top: 5rem;
                padding-bottom: 4rem;
            }
            .fa-icon { color: #284B63; margin-right: 0.4rem; }
        </style>
        <div class="fixed-header">Plataforma Bayesiana - Identificador de Anomalías</div>
        <div class="fixed-footer">&copy; Sistema de Defensa Universitario | Motor de Inferencia Bayesiano</div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("DETECTAR ANOMALÍAS")
        
        uploaded_file = st.file_uploader(
            "Cargar Repositorio de Datos", 
            type=["csv", "xlsx", "json"],
            help="Soporta CSV, Excel y JSON."
        )
        
        if uploaded_file is not None:
            # Check if this is a new file or pending confirmation
            if getattr(st.session_state, "last_uploaded_name", "") != uploaded_file.name:
                st.session_state.pending_file = uploaded_file
            
            # Modal Flow:
            if getattr(st.session_state, "pending_file", None) is not None:
                review_ingest_modal()
                
        # Feature selectors
        if st.session_state.dataset_summary is not None:
            schema = controller.schema
            
            # Filtro estricto: sin datetime
            valid_targets = []
            if schema is not None:
                valid_targets = (
                    schema.categorical_columns + 
                    schema.binary_columns + 
                    schema.numeric_columns
                )
                
            all_cols = controller.df.columns.tolist() if controller.df is not None else []
            
            target_col = st.selectbox(
                "Columna Objetivo (Target)", 
                options=valid_targets if valid_targets else [""],
                help="Variable binaria/categórica para el evento anómalo a predecir"
            )
            
            positive_label = st.text_input(
                "Etiqueta positiva (Opcional)",
                help="Para variables categóricas: ¿Qué valor específico representa tu anomalía/fallo?"
            )
            
            evidence_col = st.selectbox(
                "Columna de Evidencia",
                options=all_cols,
                index=min(1, len(all_cols) - 1) if all_cols else 0
            )
            
            threshold_txt = st.text_input(
                "Umbral evidencia numérica (Opcional)",
                help="Opcional. Si se omite, se usa un umbral automático estadístico"
            )
            
            if st.button("Ejecutar Modelo Bayesiano", type="primary"):
                if target_col and evidence_col:
                    try:
                        threshold_val = float(threshold_txt) if threshold_txt.strip() else None
                        pos_lbl = positive_label.strip() if positive_label.strip() else None
                        
                        with st.spinner("Ejecutando iteraciones de análisis..."):
                            result = controller.analyze(
                                target_column=target_col,
                                positive_label=pos_lbl,
                                evidence_column=evidence_col,
                                threshold=threshold_val,
                            )
                            st.session_state.analysis_complete = True
                            
                        st.success("Analisis de anomalias completado con exito")
                    except Exception as exc:
                        st.error(f"La ejecución falló: {exc}")
                else:
                    st.warning("Por favor, selecciona las variables de Objetivo y de Evidencia.")
                    
            summary = st.session_state.dataset_summary
            st.info(
                f"Archivo cargado exitosamente:\n\n"
                f"- {summary.rows} filas\n\n- {summary.columns} columnas detectadas\n\n"
                f"Variables Numericas: {len(summary.schema.numeric_columns)}\n\n"
                f"Categoricas: {len(summary.schema.categorical_columns)}\n\n"
                f"Fechas: {len(summary.schema.datetime_columns)}\n\n"
                f"Binarias: {len(summary.schema.binary_columns)}"
            )

        else:
            st.info("Carga un dataset CSV para iniciar")


    # Main body content
    if st.session_state.analysis_complete and controller.last_result:
        result = controller.last_result
        
        # Top KPI Metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Exactitud (Accuracy)", f"{result.metrics.accuracy:.2%}")
        col2.metric("Sensibilidad (Recall)", f"{result.metrics.sensitivity:.2%}")
        col3.metric("Especificidad", f"{result.metrics.specificity:.2%}")
        col4.metric("P(A|B) Posterior", f"{result.probability.p_a_given_b:.2%}")
        
        st.divider()
        
        # Insights text flow
        st.subheader("Insights generados automáticamente")
        for highlight in result.insights.highlights:
            st.write(f"- {highlight}")
            
        st.divider()
        
        # Charts section
        st.subheader("Visualización del Modelo")
        try:
            render_charts(controller, target_col)
        except Exception as chart_exc:
            st.warning(f"Error al generar la gráfica, revise las variables seleccionadas. Detalles: {chart_exc}")
    else:
        st.write("Tus insights automáticos aparecerán en esta área tras correr el código.")


def render_charts(controller: AppController, target_col: str) -> None:
    """Generate and display charts in the streamlit columns."""
    df = controller.df
    model_df = controller.model_df
    artifacts = controller.train_artifacts
    last_result = controller.last_result
    schema = controller.schema
    
    if df is None or artifacts is None or schema is None or last_result is None:
        return

    st.write("### Evidencia vs Resultados")
    # Row 1: Distribution vs Classifier Confusion Grid
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("**1. Distribución Numérica (Evidencia)**")
        if schema.numeric_columns:
            numeric_col = schema.numeric_columns[0]
            fig_hist = build_histogram(df=df, column=numeric_col, bins=controller.config.default_hist_bins)
            st.pyplot(fig_hist)
        else:
            st.caption("No existen columnas numéricas en el dataset base.")
            
    with row1_col2:
        st.markdown("**2. Matriz de Confusión (Exactitud Global)**")
        fig_cm = build_confusion_matrix_chart(artifacts.metrics)
        st.pyplot(fig_cm)
            
    # Row 2: Time Series vs Posterior Logic Grid
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.markdown("**3. Impacto en Serie Temporal**")
        if schema.datetime_columns:
            dt_col = schema.datetime_columns[0]
            temp_df = model_df if model_df is not None else df.copy()
            fig_time = build_time_series(df=temp_df, datetime_col=dt_col, target_col=target_col)
            st.pyplot(fig_time)
        else:
            st.caption("No se detectó una columna datetime en el esquema.")
            
    with row2_col2:
        st.markdown("**4. Probabilidad A Priori vs Posterior**")
        probability = last_result.probability
        fig_post = build_posterior_chart(probability)
        st.pyplot(fig_post)
        
    st.divider()
    
    # AI Conclusiones / "Cuadritos Bonitos"
    if hasattr(last_result.insights, "ai_conclusions") and last_result.insights.ai_conclusions:
        st.markdown(
            '<h3><i class="fa-solid fa-lightbulb fa-icon"></i> Conclusiones de la Inteligencia Artificial</h3>',
            unsafe_allow_html=True
        )
        st.info(last_result.insights.ai_conclusions)
        
    st.write("<br/>", unsafe_allow_html=True)
    
    # PDF Generator Bottom
    metricas_dict = {
        "Exactitud": f"{artifacts.metrics.accuracy:.2%}",
        "Sensibilidad": f"{artifacts.metrics.sensitivity:.2%}",
        "Especificidad": f"{artifacts.metrics.specificity:.2%}",
        "P(A|B) Posterior": f"{last_result.probability.p_a_given_b:.2%}"
    }
    
    pdf_bytes = crear_reporte_pdf(
        metricas=metricas_dict, 
        insights_texto=getattr(last_result.insights, "ai_conclusions", "Sin conclusiones de IA disponibles")
    )
    
    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        st.download_button(
            label="Descargar Reporte PDF",
            data=pdf_bytes,
            file_name="Reporte_Anomalias.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
            icon=":material/download:"
        )
