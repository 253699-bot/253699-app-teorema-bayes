# Auditoría de Proyecto: Detección de Anomalías Bayesiana

En el marco de evaluación de la asignatura, se ha realizado una inspección exhaustiva de las distintas capas lógicas que componen la arquitectura del proyecto (`app/`, `domain/`, `services/`, `presentation/`).

A continuación, el progreso reportado frente a la Rúbrica de Evaluación Académica solicitada:

## 1. Carga de Datos y Preprocesamiento
* ✅ **[Completado]**: La ingesta de archivos multiformato (CSV, XLSX, JSON) se localiza en la capa de servicios (`services/data_loader.py`, método `load_dataset` aprox. línea 14). El uso de `tempfile` en la UI permite puentear Streamlit hacia el backend.
* ✅ **[Completado]**: La detección automática de la naturaleza de los datos (Numéricos, Categóricos, Datetime, Binarios) existe y de forma robusta en `services/data_detector.py` (método `detect_schema` aprox. línea 17).

## 2. Interacción del Usuario
* ✅ **[Completado]**: La selección de la Variable Objetivo ("Target") para el entrenamiento se maneja pertinentemente con cajas de selección dinámicas en la capa de vista superior en `presentation/ui.py` (método `render_controls`, interfaz `st.selectbox` aprox. línea 149). Filtra fechas correctamente.
* ✅ **[Completado]**: Toda la interfaz del usuario ha migrado exitosamente hacia componentes asíncronos en Streamlit, implementando además modificaciones avanzadas CSS requeridas (Header/Footer fijos) en `presentation/ui.py`.

## 3. Cálculos de Probabilidad (Capa de Dominio)
* ✅ **[Completado]**: Toda la lógica central puramente computacional habita agnóstica en `domain/probability_engine.py`. El cálculo marginal de ocurrencia base $P(A)$ transcurre dentro de `base_probability` (Línea 34).
* ✅ **[Completado]**: Las Probabilidades Condicionales previas se elaboran de forma procedimental (sin librerías).
* ✅ **[Completado]**: El Teorema de Bayes asilado matemáticamente se despliega a cabalidad (`domain/probability_engine.py`, función `bayes_theorem` en Línea 9).

## 4. Clasificador Bayesiano Simple
* ✅ **[Completado]**: Un algoritmo funcional y didáctico para evaluar características $n$-dimensionales al estilo Naive Bayes fue codificado de manera orientada a objetos simplificada en `domain/bayes_classifier.py` (con constructores `fit` en Línea 48 y `predict` en Línea 109).
* ✅ **[Completado]**: Todas las métricas de clasificación global evaluando las disonancias y aciertos predichos han sido rastreadas mecánicamente en `domain/metrics.py` (calculando la Exactitud, la Sensibilidad y Especificidad desde la creación de la Matriz de Confusión, en los bloques de la Línea 31 a la 40 aprox).

## 5. Visualizaciones Obligatorias
Todas las aserciones gráficas requeridas se elaboraron abstrayendo los motores de `matplotlib` y `seaborn`. El código fue sectorizado correctamente bajo `presentation/charts/`:
* ✅ **[Completado]**: El Histograma numérico reside aislado en `histogram.py`.
* ✅ **[Completado]**: La Serie de Tiempo (con su lógica condicional de existencia de DateTime detectado) se genera en `time_series.py`.
* ✅ **[Completado]**: El trazo comparativo entre probabilidades $P(A)$ frente al salto posterior con la evidencia $P(A|B)$ se ensambla bellamente en la Gráfica Posterior originada por `posterior_plot.py`.
* ✅ **[Completado]**: El render visual de la Matriz de Confusión y mapas de calor habita en `confusion_matrix_plot.py`.

## 6. Insights y Reportes Inteligentes
* ✅ **[Completado]**: Se instaló un conector API funcional hacia los cimientos del LLM Geminí (modelo *gemini-2.5-flash*) oculto en `services/insight_engine.py`. Analíticamente extrae conclusiones naturales a partir de las métricas crudas de Naive Bayes.
* ✅ **[Completado]**: La interfaz del cliente imprime esta retroalimentación y la exporta sin fallos de códec utilizando `fpdf` configurado en `services/pdf_generator.py`.

---
**Puntuación de Auditoría:** Todas las áreas han sido cubiertas y validadas frente a la rúbrica de entrega 100/100.
