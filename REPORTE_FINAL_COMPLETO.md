# Reporte Final Completo: Plataforma Bayesiana para Detección de Anomalías

**Asignatura:** Probabilidad y Estadística  
**Tipo de Trabajo:** Proyecto Final  
**Tecnologías:** Python · Streamlit · Pandas · Matplotlib · Google Gemini AI

---

## 1. Introducción

### 1.1 Contexto del Problema

En los entornos industriales, médicos y de ciencia de datos modernos, la identificación temprana de comportamientos anómalos tiene un impacto directo en la reducción de costos operativos, riesgos de seguridad y pérdidas económicas. Un "evento anómalo" (también llamado outlier, fallo o anomalía) es aquel que ocurre de forma estadísticamente inusual respecto al comportamiento normativo histórico del sistema.

Sin embargo, la mayoría de las herramientas de monitoreo modernas confunden alta complejidad matemática con precisión; dependiendo de librerías de caja negra (como `scikit-learn`) que calculan y deciden sin que el operador comprenda el razonamiento subyacente. Este proyecto plantea que la comprensión matemática profunda del fenómeno probabilístico es inseparable del diseño de una herramienta confiable, transparente y auditable.

### 1.2 Importancia del Análisis Probabilístico

La estadística clásica permite cuantificar qué tan **probable** es que un evento "malo" haya ocurrido dada la información registrada en los sensores. Esta cuantificación transforma la detección de anomalías de una actividad subjetiva (basada en la intuición del técnico) a un razonamiento matemáticamente respaldado, donde cada decisión tiene un valor numérico de certeza asociado.

Las preguntas que este sistema responde de forma objetiva son:
- ¿Cuál es la frecuencia base de fallos en este equipo? $P(A)$
- Si la temperatura supera el umbral, ¿cuánto aumenta la probabilidad de fallo? $P(A|B)$
- ¿Qué tan confiables son estas predicciones? (Accuracy, Recall, Specificity)

### 1.3 Justificación del Uso del Teorema de Bayes

El Teorema de Bayes representa el puente matemático ideal entre la **experiencia histórica** (probabilidad a priori) y la **evidencia nueva** (la variable observada en tiempo real). A diferencia de modelos de caja negra basados en redes neuronales, el enfoque Bayesiano:

- Es **explicable por diseño**: cada paso probabilístico puede ser inspeccionado y defendido.
- Funciona con **datasets pequeños**: no requiere millones de registros para producir inferencias útiles.
- Se adapta a **datos mixtos**: soporta variables numéricas, categóricas y binarias sin tratamientos complejos.
- Mantiene la **coherencia matemática**: los resultados son probabilidades verdaderas, directamente interpretables.

---

## 2. Marco Teórico

### 2.1 Definición de Evento

Un **evento** $A$ en el contexto de la teoría de la probabilidad es cualquier subconjunto del espacio muestral $\Omega$. Formalmente:

$$A \subseteq \Omega$$

En el contexto de detección de anomalías, un evento es la ocurrencia de un fallo (ej. $A = \{\text{Máquina falló}\}$), definido por los datos históricos de la variable objetivo seleccionada por el usuario desde la interfaz de la plataforma.

### 2.2 Probabilidad Clásica

La probabilidad de un evento $A$ en un espacio muestral con resultados igualmente probables es:

$$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{Número de casos favorables}}{\text{Número de casos totales}}$$

En la implementación, la función `base_probability()` de `domain/probability_engine.py` calcula esta frecuencia relativa aprovechando la operación vectorizada de Pandas:

```python
return float((target == positive_value).mean())
```

El operador `.mean()` sobre una máscara booleana equivale exactamente a calcular $P(A)$ de la forma clásica.

### 2.3 Probabilidad Condicional

La probabilidad de que ocurra el evento $A$ dado que ya se sabe que ocurrió el evento $B$ se define como:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

Esta definición es el núcleo de todo sistema detector de anomalías basado en correlaciones: la **evidencia** $B$ (ej. temperatura elevada) actualiza nuestra certeza sobre $A$ (fallo inminente).

### 2.4 Independencia Estadística

Dos eventos $A$ y $B$ son estadísticamente **independientes** si y solo si:

$$P(A \cap B) = P(A) \cdot P(B) \iff P(A|B) = P(A)$$

Cuando esta condición se cumple, la observación de $B$ no aporta ninguna información adicional sobre la ocurrencia de $A$. Esta propiedad es la **suposición fundamental** (y principal limitación) del clasificador Naive Bayes implementado en `domain/bayes_classifier.py`, el cual asume independencia entre todas las características.

### 2.5 Teorema de Bayes

El Teorema de Bayes invierte la condicionalidad, permitiendo calcular $P(A|B)$ a partir de $P(B|A)$:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Donde:
- $P(A)$ es la **probabilidad a priori** del evento (tasa base histórica).
- $P(B|A)$ es la **verosimilitud** (likelihood): qué tan común es la evidencia cuando el evento ocurre.
- $P(B)$ es la **evidencia marginal**: probabilidad total de observar la evidencia.
- $P(A|B)$ es la **probabilidad a posteriori**: nuestra certeza actualizada.

En el código, la función `bayes_theorem()` de `probability_engine.py` implementa esta ecuación de forma directa:

```python
def bayes_theorem(p_b_given_a, p_a, p_b, epsilon=1e-12):
    denominator = max(p_b, epsilon)
    return float((p_b_given_a * p_a) / denominator)
```

El parámetro `epsilon = 1e-12` previene matemáticamente la singularidad de división por cero cuando $P(B) \to 0$.

### 2.6 Variables Aleatorias y Distribuciones Relevantes

**Variable Aleatoria Discreta (Bernoulli):** Para variables binarias (Sí/No, 0/1), el clasificador utiliza la distribución de Bernoulli. La probabilidad de observar el valor $x \in \{0, 1\}$ dado la clase $c$ es:

$$P(x|c) = p_c^x \cdot (1 - p_c)^{1-x}$$

**Variable Aleatoria Continua (Gaussiana):** Para variables numéricas, el clasificador asume distribución Normal con media $\mu_c$ y varianza $\sigma^2_c$ estimadas por clase:

$$f(x|c) = \frac{1}{\sqrt{2\pi\sigma_c^2}} \exp\left(-\frac{(x - \mu_c)^2}{2\sigma_c^2}\right)$$

Su implementación logarítmica directa en `_gaussian_log_pdf()` es:

$$\ln f(x|c) = -\frac{1}{2}\ln(2\pi\sigma_c^2) - \frac{(x-\mu_c)^2}{2\sigma_c^2}$$

**Variable Aleatoria Categórica (Multinomial con Laplace Smoothing):** Para categorías de texto, la probabilidad se estima con suavizamiento de Laplace para evitar probabilidades nulas frente a valores no vistos en entrenamiento:

$$P(v|c) = \frac{\text{count}(v, c) + 1}{\text{count}(c) + |V|}$$

Donde $|V|$ es el tamaño del vocabulario de la variable.

---

## 3. Preparación de Datasets

### 3.1 Identificación del Formato del Archivo

El módulo `services/data_loader.py` implementa la clase `DataLoader`, cuyo método `load_dataset()` detecta dinámicamente el formato del archivo mediante inspección de la extensión de su ruta temporal:

```python
if path.endswith(".csv"):
    df = pd.read_csv(path)
elif path.endswith(".xlsx"):
    df = pd.read_excel(path)
elif path.endswith(".json"):
    df = pd.read_json(path)
else:
    raise ValueError("Formato de archivo no soportado")
```

En el flujo real de la aplicación, la capa `presentation/ui.py` inyecta una validación defensiva previa en el modal de confirmación (`review_ingest_modal`) que verifica la extensión del archivo via `os.path.splitext()` antes de que el objeto llegue al servicio backend. Esto garantiza que nunca se intente parsear un PDF, imagen o archivo binario incompatible.

### 3.2 Extracción de Dimensiones del Dataset

Tras la carga, el `DataLoader` invoca la validación:
```python
validate_non_empty_dataframe(df)
LOGGER.info("Loaded dataframe with shape: %s", df.shape)
```

El atributo `.shape` de un DataFrame de Pandas devuelve la tupla `(filas, columnas)`, que es luego expuesta al usuario en la barra lateral del Dashboard como un bloque informativo que muestra el número de registros, dimensiones detectadas y desglose por tipo de variable.

### 3.3 Clasificación Automática de Variables (`data_detector.py`)

El módulo `services/data_detector.py` implementa la clase `DataDetector` con el método `detect_schema()`, que examina columna por columna el DataFrame recibido y lo clasifica en cuatro categorías:

| Tipo de Columna | Criterio de Clasificación |
|---|---|
| **Datetime** | `is_datetime64_any_dtype()` o $\geq 80\%$ de valores parseables como fecha |
| **Binaria** | $\geq 95\%$ de valores normalizables vía `normalize_binary_value()` (Sí/No, 1/0, True/False) |
| **Numérica** | `is_numeric_dtype()` con $\geq 10$ valores únicos |
| **Categórica** | Todo lo restante |

El clasificador aplica prioridad estricta en este orden, lo que garantiza que una columna de enteros `[0, 1]` sea primero evaluada como binaria antes de ser interpretada incorrectamente como numérica.

### 3.4 Tratamiento de la Variable Objetivo

En `services/preprocessing.py`, el método `make_binary_target()` transforma la columna objetivo elegida por el usuario en un vector binario estándar `{0, 1}` antes del entrenamiento. El proceso es adaptativo:

- Si la columna ya es binaria nativa (Sí/No detectado por el detector de esquema), se normaliza directamente a enteros.
- Si la columna es categórica multiclase, el sistema convierte el valor positivo elegido por el usuario (ej. "Fallo") en clase `1`, y todo lo demás se agrupa como clase `0`.

Este diseño asegura compatibilidad con el clasificador binario de la capa de dominio sin importar la naturaleza original del dataset.

### 3.5 Tratamiento y Bloqueo de Columnas Datetime

Las columnas de fecha y hora reciben un tratamiento especial en dos frentes:

1. **Bloqueo en la UI**: El menú desplegable de la interfaz que permite elegir la "Columna Objetivo" excluye explícitamente las columnas de tipo datetime del `DetectedSchema`, impidiendo que el usuario seleccione por error un timestamp como variable a predecir (lo cual causaría colisiones de claves en Pandas al existir cientos de valores únicos).

2. **Exclusión del entrenamiento**: El método `prepare_features()` de `PreprocessingService` construye la matriz de características $X$ incluyendo únicamente columnas numéricas, binarias y categóricas, ignorando por completo las columnas de fecha del esquema.

---

## 4. Metodología

### 4.1 Limpieza de Datos

Antes de construir la matriz de entrada del clasificador, el `PreprocessingService` aplica una estrategia de imputación diferenciada por tipo de columna:

- **Columnas numéricas**: Los valores faltantes (`NaN`) se reemplazan por la mediana estadística de la columna, preservando la distribución ante valores extremos (outliers).
- **Columnas binarias**: Los valores no mapeables se reemplazan por `0` (ausencia del evento como valor conservador).
- **Columnas categóricas**: Las celdas vacías se rellenan con el valor centinela literal `"<MISSING>"`, que es tratado como una categoría más por el clasificador (evitando su exclusión silenciosa).

### 4.2 Selección de Umbrales para Variables Numéricas

Cuando la columna de evidencia seleccionada es de tipo numérico, el sistema necesita definir un punto de corte para determinar cuándo dicha variable "activa" la evidencia. La función `_default_threshold()` de `probability_engine.py` calcula un umbral estadístico automático:

```python
q75 = float(clean.quantile(0.75))
mean = float(clean.mean())
return max(q75, mean)
```

El umbral es el **máximo entre el percentil 75 y la media aritmética**, lo que garantiza que el umbral automático active la evidencia solo en el cuartil superior de la distribución, donde la señal anómala suele ser más intensa. El usuario también puede sobrescribir este umbral ingresando un valor numérico manual en el campo de la barra lateral.

### 4.3 Cálculo de Probabilidades

La función central `compute_probability_report()` de `probability_engine.py` orquesta el flujo completo de cálculo en cuatro pasos:

1. Extraer la serie objetivo $y$ y la serie de evidencia $b$ del DataFrame.
2. Construir la máscara booleana de evidencia ($b > \text{umbral}$ para numérica, $b = \text{moda}$ para categórica).
3. Calcular $P(A)$, $P(B|A)$ y $P(B)$ mediante operaciones vectorizadas de Pandas:
   $$P(B|A) = \frac{|\{i : b_i = \text{True} \text{ y } y_i = \text{positivo}\}|}{|\{i : y_i = \text{positivo}\}|}$$
4. Calcular $P(A|B)$ invocando `bayes_theorem()`.

Todo el resultado queda encapsulado en un objeto de valor inmutable `ProbabilityComputation` de `domain/models.py`.

### 4.4 Entrenamiento del Clasificador y Manejo del Underflow Aritmético

El clasificador `ManualNaiveBayes` de `domain/bayes_classifier.py` implementa el algoritmo Naive Bayes multidimensional de forma **completamente manual**, sin dependencia de `scikit-learn`. El método `fit()` entrena los parámetros estadísticos de cada columna por clase:

- Para columnas **numéricas**: calcula $\mu_c$ y $\sigma^2_c$ por clase.
- Para columnas **binarias**: calcula $p_c = P(x=1|c)$ por clase.
- Para columnas **categóricas**: calcula el tabla de probabilidades $P(v|c)$ con suavizamiento de Laplace.

#### El Problema del Underflow Aritmético

El modelo canónico de Naive Bayes calcula el posterior mediante la **multiplicación** de todas las verosimilitudes individuales:

$$P(c | x_1, \dots, x_n) \propto P(c) \cdot \prod_{i=1}^{n} P(x_i | c)$$

Si $n$ es grande (ej. 20 columnas) y cada $P(x_i|c)$ es una fracción pequeña (ej. 0.01), el producto puede ser inferior al valor mínimo representable por el tipo `float64` de Python ($\approx 5 \times 10^{-324}$), colapsando silenciosamente a `0.0`. Esto imposibilita comparar las probabilidades entre clases.

#### La Solución: Suma de Logaritmos

Aprovechando la propiedad algebraica del logaritmo de un producto:
$$\ln\left(\prod_{i} p_i\right) = \sum_{i} \ln(p_i)$$

El método `_log_posterior()` opera exclusivamente en el espacio logarítmico, acumulando una **suma de valores negativos** en lugar de un producto de fracciones:

```python
log_prob = log(max(prior, self.epsilon))   # ln(P(c))
for feature in self.feature_columns:
    # ...
    log_prob += log(max(p, self.epsilon))  # += ln(P(xi|c))
return log_prob
```

Para recuperar una probabilidad normalizada entre `[0, 1]` al término, `predict_proba()` aplica la técnica **log-sum-exp** con corrección de overflow:

```python
max_log = max(log_posteriors.values())
exp_scores[cls] = exp(logp - max_log)
```

Restar el `max_log` antes de exponenciar garantiza que al menos un valor sea `exp(0) = 1.0`, eliminando el riesgo de overflow numérico por el camino inverso.

---

## 5. Resultados (Exposición de Resultados en la UI)

La capa `presentation/ui.py` recoge la instancia de resultados calculados por el `AppController` y los despliega en un layout de Dashboard responsivo.

### 5.1 Probabilidad Base P(A)

El `InsightEngine` de `services/insight_engine.py` genera automáticamente la siguiente leyenda como primer bullet del informe:

> *"Tasa base de evento (columna_objetivo) = X.XX%; con evidencia (evidencia_col == valor) sube a Y.YY%."*

Esto contrasta de forma directa e intuitiva el $P(A)$ original (la frecuencia histórica de fallos sin más información) versus el $P(A|B)$ posterior enriquecido por la evidencia seleccionada.

### 5.2 Probabilidades Condicionales

El sistema reporta el vector de características que mayor diferencia estadística generó respecto a la clase objetivo. El `InsightEngine` extrae las tres variables con mayor señal inferida:

```python
top_features = feature_importance[:3]
formatted = ", ".join([f"{name} ({score:.3f})" for name, score in top_features])
highlights.append(f"Variables más influyentes (señal aproximada): {formatted}.")
```

Adicionalmente, la interfaz expone automáticamente el valor de $P(B|A)$ (verosimilitud) y $P(B)$ (marginal de evidencia) como parte del objeto `ProbabilityComputation` desempacado en los Indicadores Clave.

### 5.3 Visualización de Probabilidades Posteriores

La gráfica generada por `presentation/charts/posterior_plot.py` despliega un diagrama de barras comparativo entre:
- La barra izquierda: $P(A)$ — probabilidad a priori (antes de la evidencia).
- La barra derecha: $P(A|B)$ — probabilidad a posteriori (después de considerar la evidencia).

La diferencia visual entre ambas barras representa gráficamente el **valor informativo de la evidencia**: cuanto mayor es el salto, más útil es esa variable para predecir la anomalía.

### 5.4 Métricas de Clasificación

La plataforma expone cuatro indicadores científicos de rendimiento en una fila de métricas KPI en la parte superior del dashboard:

| Métrica | Fórmula | Interpretación |
|---|---|---|
| **Exactitud (Accuracy)** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Porcentaje global de predicciones correctas |
| **Sensibilidad (Recall)** | $\frac{TP}{TP + FN}$ | Capacidad de detectar anomalías reales |
| **Especificidad** | $\frac{TN}{TN + FP}$ | Capacidad de evitar falsas alarmas |
| **P(A\|B) Posterior** | $\frac{P(B|A) \cdot P(A)}{P(B)}$ | Probabilidad actualizada de la anomalía |

La **Matriz de Confusión** se renderiza como un mapa de calor (heatmap) mediante Seaborn, mostrando visualmente los TP, FP, TN y FN en una cuadrícula $2 \times 2$ con escala de color corporativa.

---

## 6. Análisis e Insights (Motor de Inteligencia Artificial)

El módulo `services/insight_engine.py` implementa la clase `InsightEngine` que genera dos tipos de análisis complementarios:

### 6.1 ¿Qué Variable Aporta Más Información?

El campo `feature_importance` es calculado por el `AppController` y pasado al `InsightEngine`. Las variables con mayor diferencial de log-posterior entre la clase positiva y negativa son las más discriminantes. El sistema expone explícitamente las **3 variables más influyentes** con su puntaje de señal relativa.

Matemáticamente, una variable aporta información cuando la distribución $P(x_i|A=1)$ difiere significativamente de $P(x_i|A=0)$, es decir, cuando la observación de su valor cambia considerablemente nuestra estimación del evento.

### 6.2 ¿El Evento es Raro?

La tasa base $P(A)$ responde directamente esta pregunta. Si $P(A) < 0.10$, el evento es estadísticamente infrecuente (un dataset "desbalanceado"), lo que explica por qué la Sensibilidad tiende a bajar: el modelo aprende a predecir siempre la clase mayoritaria como estrategia de minimización del error global.

### 6.3 ¿Las Variables Parecen Independientes?

Esta pregunta es la más profunda y difícil de responder desde la plataforma sin análisis adicional. El sistema aplica la **asunción de Naive Bayes** (independencia condicional entre todas las características dada la clase), que es exactamente la suposición matemática que permite el cálculo del producto de verosimilitudes. Si el modelo obtiene alta exactitud con esa asunción, es una señal indirecta de que las correlaciones entre variables no son determinantes para la predicción.

El usuario puede inferir la correlación empíricamente observando si la **Sensibilidad y Especificidad** son altas simultáneamente: si ambas son elevadas, las variables probablemente proveem señales complementarias e independientes.

### 6.4 ¿Qué Tan Confiable es el Modelo?

La plataforma integra la API de Google Gemini (modelo `gemini-2.5-flash`) para traducir las métricas numéricas a un lenguaje de negocio interpretable. El prompt instruccional que se envía al modelo de lenguaje incluye explícitamente la exactitud, la evidencia y el objetivo:

> *"Eres un analista de datos experto. Basado en un modelo Naive Bayes con exactitud de X%, y analizando cómo la evidencia '...' afecta al objetivo '...', redacta 3 viñetas breves en español explicando qué significa esto y dando una recomendación de negocio."*

Las viñetas generadas por la IA aparecen en la sección "Conclusiones de la Inteligencia Artificial" de la interfaz web, y son incluidas automáticamente en el reporte PDF descargable generado por `services/pdf_generator.py`.

---

## 7. Conclusiones

### 7.1 Limitaciones del Enfoque Actual

La **asunción de independencia condicional** de Naive Bayes es una simplificación que raramente se cumple en datos reales de sistemas industriales o de comportamiento humano. Si variables como "Temperatura" y "Presión" están correlacionadas entre sí (lo cual es físicamente esperado), el modelo sobreestimará la confianza de sus predicciones porque contará la misma señal dos veces.

Otras limitaciones observables:
- **Datasets desbalanceados**: Cuando la clase positiva (anomalía) es una minoría, la sensibilidad del modelo tiende a degradarse sin técnicas de balanceo como oversampling o ajuste de umbrales.
- **Variables numéricas no gaussianas**: La distribución de Gauss asumida para variables continuas puede no representar fielmente distribuciones bimodales, exponenciales o con colas pesadas.
- **Umbral fijo de decisión**: El clasificador usa por defecto un umbral de $P(c|x) \geq 0.5$, que puede no ser óptimo en contextos donde el costo de un falso negativo (anomalía no detectada) es mucho mayor que el de un falso positivo.

### 7.2 Mejoras Futuras

**Corto plazo (Arquitectónicas):**
- Implementar un módulo de análisis de correlación (`domain/correlation_engine.py`) que advierte al usuario cuando dos variables tienen una correlación de Pearson $|r| > 0.7$, comprometiendo la asunción de independencia.
- Agregar soporte para ajuste dinámico del umbral de decisión basado en la curva ROC (Receiver Operating Characteristic).

**Mediano plazo (Modelos Más Complejos):**
- Migrar hacia un **Modelo Gráfico Bayesiano** (Bayesian Network) donde la estructura causal entre variables se representa explícitamente, eliminando la asunción de independencia estricta.
- Considerar un **Clasificador de Árbol de Decisión Bayesiano** que combine la interpretabilidad de los árboles con la robustez probabilística de Bayes para datasets con interacciones complejas.

### 7.3 Relación con la Teoría de la Asignatura

Este proyecto constituye una aplicación directa de los contenidos centrales de la asignatura de Probabilidad y Estadística:

- **Unidad de Probabilidad Básica**: La función `base_probability()` implementa la definición clásica de frecuencia relativa $P(A) = n_A / n$.
- **Unidad de Probabilidad Condicional**: La función `conditional_probability()` es la realización computacional directa de la definición formal $P(A|B)$.
- **Unidad del Teorema de Bayes**: La función `bayes_theorem()` expone fielmente la fórmula canónica del Teorema con su manejo de singularidades.
- **Unidad de Variables Aleatorias**: El clasificador `ManualNaiveBayes` modela tres distribuciones formales: Bernoulli (binaria), Gaussiana (numérica) y Multinomial con suavizamiento de Laplace (categórica).
- **Unidad de Estadística Descriptiva**: Las métricas Accuracy, Sensibilidad, Especificidad y la Matriz de Confusión son herramientas estándar de la estadística aplicada, codificadas manualmente en `domain/metrics.py`.

El conjunto representa la demostración práctica de que la teoría probabilística no es un ejercicio teórico abstracto, sino el fundamento matemático detrás de sistemas de inteligencia artificial reales que impactan decisiones críticas en organizaciones de todo tipo.
