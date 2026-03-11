# REPORTE DE PRESENTACIÓN: Interfaz Gráfica y Visualización

Este documento técnico constituye la **Fase 4 (Final)** de la documentación del proyecto. Se enfoca exclusivamente en la capa `presentation/`, la cual orquesta la experiencia del usuario (UX), el renderizado responsivo y las visualizaciones gráficas integrando el framework **Streamlit** y motores de ploteo como **Matplotlib** y **Seaborn**.

---

## 1. Arquitectura de `ui.py` (El Controlador de Vistas)

El archivo `presentation/ui.py` actúa como la columna vertebral visual. Al estar basado en Streamlit, su naturaleza es imperativa ("top-down"), lo cual significa que el script entero se re-ejecuta con cada interacción. Para contrarrestar esta volatilidad, se implementaron técnicas avanzadas:

### Gestión del Estado (`st.session_state`)
Streamlit destruiría los resultados matemáticos calculados por la capa de Dominio en cada click. Para retener esa información, se declararon variables globales persistentes en memoria en el método `init_session_state()`:
* `st.session_state.controller`: Mantiene viva la instancia matemática (`AppController`) sin reiniciar configuraciones.
* `st.session_state.dataset_summary` y `analysis_complete`: Banderas lógicas que le indican a la interfaz cuándo mostrar el menú lateral (Sidebar) y cuándo revelar los resultados principales.

### Modal de Confirmación Defensivo (`@st.dialog`)
El flujo permite subir archivos y retiene al usuario obligándolo a verificar la ingesta usando la función asíncrona `review_ingest_modal()`. 
Este modal intercepta la subida y efectúa **Validación de Datos (Data Validation)** para asegurar que el archivo cumpla con los formatos tabulares soportados (`.csv`, `.xlsx`, `.json`). De ser válido, utiliza un archivo seguro en caché (`tempfile.NamedTemporaryFile`) para parsear los bits en DataFrames; de lo contrario, bloquea gráficamente el acceso al motor en Python previniendo crashes de servidor.

### Diseño Grid Responsivo (`st.columns`)
Para maximizar el uso inmersivo de la vista de datos ("Dashboard"), la métricas clave operan sobre un empaquetado simétrico:
```python
col1, col2, col3, col4 = st.columns(4)
col1.metric("Exactitud (Accuracy)", ...)
```
Esto crea particiones matemáticas en pantalla. Las gráficas también han sido divididas en una inmersiva retícula de $2\times2$ (Evidencia/Confusión, Serie/Posterior) para forzar al navegador web moderno a ajustar los paneles independientemente del tamaño de la ventana del cliente.

---

## 2. Inyección de CSS (Header y Footer Fijos)
Por diseño arquitectónico deliberado, **Streamlit** no provee componentes fijos y bloquea la inyección de JavaScript nativo para prevenir ataques cross-site. Sin embargo, se cumplió con los requerimientos corporativos (Header Institucional + Footer Técnico Fijos) mediante una inyección de estilo subversiva en la rama principal del DOM:

```python
st.markdown(
    """<style>
        .fixed-header { position: fixed; top: 0; left: 0; z-index: 999; ... }
        .main .block-container { padding-top: 5rem; ... }
    </style>""",
    unsafe_allow_html=True
)
```
1. **Posicionamiento Absoluto:** El atributo `position: fixed` abstrae visualmente el componente del motor de Streamlit, obligando al navegador a "pegarlo" al cénit y nadir de la pantalla, conservando la identidad `#153243` de forma omnipresente.
2. **Padding Táctico:** Se forzó un relleno de compensación superior e inferior (`padding-top`, `padding-bottom`) al `.block-container` (el elemento HTML central de Streamlit) para impedir que el contenido fluido y las barras de scroll pasen estéticamente "por debajo" de nuestras cabeceras inamovibles.

---

## 3. Renderizado Científico: Integración en `charts/`

El directorio local `presentation/charts/` aisla a Streamlit de la complejidad geométrica de trazar las figuras. Matplotlib maneja el plano matemático en Python, devolviendo un objeto `plt.Figure` abstracto listo para ser dibujado por `st.pyplot()`.

Cada archivo sigue un patrón estricto inyectando el color primario de fondo de la web corporativa (`#F4F9E9`) para que los gráficos parezcan tener transparencia nativa en la UI global, borrando los bordes blancos por defecto:

* **`histogram.py`**: Interroga la evidencia contigua. Consume el primer valor numérico viable, instanciando `sns.histplot()` con una curvatura KDE (Kernel Density Estimate) para vislumbrar sesgos normales de manera didáctica usando el tono secundario de la empresa.
* **`confusion_matrix_plot.py`**: Invoca recursivamente a Seaborn (`sns.heatmap`) para graficar la colisión matemática entre predicciones ($yp$) y reales ($yt$). Los ejes son estrictamente binarios y elimina las barras de calor (`cbar=False`) a favor de texto directo legible sobre fondos condicionales (Color Map Dinámico `#153243`).
* **`posterior_plot.py`**: Despliega un diagrama de barras puro con Matplotlib contrastando la mutación de la probabilidad A Priori versus el factor del salto Evidencial (A Posteriori), imprimiendo la cifra final flotando visualmente sobre la cima de cada barra.
* **`time_series.py`**: Analiza el "pulso del paciente". Si existe en el esquema validado alguna columna tipo Fechas (DateTime), inyecta la librería nativa de manipulación temporal de Pandas. Realiza un remuestreo promediado mensual (`.to_period("M")`) y lo grafica estilizadamente conectando círculos de puntos.

---

## Conclusión: Experiencia de Flujo del Usuario (UX)

El flujo global de cara al usuario está diseñado para ser lineal, preventivo y visualmente autónomo:

1. **Ingestión Ciega**: El usuario final interactúa con la barra lateral "Detectar Anomalías" cargando su repositorio tabular.
2. **Ventana Defensiva**: El sistema congela la experiencia lanzando el Modal de validación de extensión y formato.
3. **Control Categórico**: Al habilitarse la vista matemática, el usuario es guiado a elegir únicamente columnas válidas depuradas (nunca seleccionará una ID ni una fecha como Anomalía).
4. **Respuesta Rápida**: Tras ejecutar el Modelo, el Grid Visual detona automáticamente desplegando los Indicadores Clave de Desempeño (Accuracy, TPR), los cuatro Paneles Gráficos Analíticos, y el Insight conversacional dictado por la Inteligencia Artificial.
5. **Portabilidad**: Por último, el usuario presiona "Descargar Reporte PDF", empacando su estudio de meses en un dossier institucional descargable sin salir jamás de su navegador.
