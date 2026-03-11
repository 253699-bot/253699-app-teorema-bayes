# Plataforma Bayesiana para Detección de Anomalías

Este es el proyecto final para la asignatura de **Probabilidad y Estadística**. Consiste en un Dashboard interactivo web impulsado por Inteligencia Artificial (Google Gemini) y un motor matemático entrenado puramente a mano que aplica el **Teorema de Bayes** para encontrar fallos, outliers y anomalías en repositorios de datos mixtos.

## Características Principales
* **Carga Agnóstica:** Soporta archivos `.csv`, `.xlsx` y `.json`.
* **Detección Automática:** Clasifica variables en Numéricas, Categóricas, Binarias y Fechas al vuelo.
* **Solución al Underflow:** Clasificador Naive Bayes programado 100% *from-scratch*, resolviendo el underflow aritmético mediante suma de Logaritmos.
* **Insights GenAI:** Emite recomendaciones de negocio instantáneas sobre las métricas matemáticas usando la API de Gemini 2.5 Flash.
* **Exportación:** Generación nativa de reportes `.pdf` in-app.

---

## Guía de Instalación y Ejecución Local

Sigue estos pasos para clonar y levantar el proyecto en tu máquina.

### 1. Clonar el repositorio
```bash
git clone git@github-253699-bot:tu-usuario/tu-repositorio.git
cd bayes_anomaly_app
```

### 2. Configurar Entorno Virtual e Instalar Dependencias
Es altamente recomendado utilizar un entorno virtual de Python.
```bash
# Crear entorno virtual (ej. en Linux/Mac)
python3 -m venv .venv
source .venv/bin/activate

# En Windows:
# python -m venv .venv
# .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configurar Claves de Entorno (API Key)
El sistema requiere una llave de Google Gemini AI para el módulo generador de Insights.
1. Haz una copia del archivo de ejemplo:
   ```bash
   cp .env.example .env
   ```
2. Edita el archivo `.env` que acabas de crear y pega tu API Key real de Google AI Studio:
   ```env
   GEMINI_API_KEY=AIzA...
   ```

### 4. Levantar la Aplicación
Inicia el servidor local de Streamlit ejecutando:
```bash
PYTHONPATH=. streamlit run app/main.py
```
> El framework abrirá automáticamente una ventana en tu navegador web en la dirección `http://localhost:8501`.

---

## Recursos Adicionales
- Para ejemplos de datasets compatibles y pre-etiquetados, visita la carpeta `/sample_data/`.
- La arquitectura técnica, matemática y de diseño está documentada a detalle en el archivo `REPORTE_FINAL_COMPLETO.md` ubicado en la raíz del proyecto.
