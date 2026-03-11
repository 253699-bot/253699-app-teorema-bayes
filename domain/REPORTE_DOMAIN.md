# REPORTE DE DOMINIO: Motor Matemático y Teorema de Bayes

Este documento técnico constituye la **Fase 2** de la documentación del proyecto. Analiza exhaustivamente la capa `domain/`, la cual ha sido diseñada agnóstica a cualquier framework visual siguiendo el principio de diseño *Separation of Concerns*.

Aquí reside todo el núcleo estadístico y probabilístico de la plataforma.

---

## 1. probability_engine.py: Cálculo de Probabilidades Marginales y Condicionales

Este módulo procedural es el esqueleto de la estadística descriptiva del proyecto.

### Cálculo de P(A) (Probabilidad Marginal Base)
La probabilidad base (A priori) de que ocurra un evento "A" (ej. "La máquina falló") se obtiene dividiendo los eventos positivos sobre el universo total de eventos. En lugar de iterar con `for` loops, el código aprovecha la vectorización de Pandas:

```python
def base_probability(target, positive_value):
    if target.empty:
        return 0.0
    return float((target == positive_value).mean())
```
**Análisis línea por línea:**
* `target == positive_value`: Genera una máscara Booleana (`True`/`False`) de toda la columna objetivo.
* `.mean()`: En Python, `True` vale `1` y `False` vale `0`. Al promediarlos matemáticamente, se obtiene exactamente la proporción (frecuencia relativa) de casos positivos, representando perfectamente $P(A)$.

### Cálculo Condicional y Teorema de Bayes Puro
El Teorema de Bayes dicta que:
$$ P(A|B) = \frac{P(A) \cdot P(B|A)}{P(B)} $$

Para evitar dependencias pesadas, la función fue programada "cruda" en su forma procedimental:
```python
def bayes_theorem(p_b_given_a, p_a, p_b, epsilon=1e-12):
    denominator = max(p_b, epsilon)
    return float((p_b_given_a * p_a) / denominator)
```
**Análisis matemático:**
* `epsilon=1e-12`: Es una constante minúscula de estabilidad numérica. 
* `max(p_b, epsilon)`: Previene la asintótica matemática de una división por cero si el evento evidencia "B" resulta ser imposible en el dataset. El dividendo aplica puramente la multiplicativa de la regla de Bayes.

---

## 2. bayes_classifier.py: Clasificador Predictivo Multidimensional

Mientras que el motor de probabilidad calcula $P(A|B)$ para de variables estáticas 1 a 1, el objeto `ManualNaiveBayes` expande el Teorema en $N$-dimensiones iterando la evidencia en múltiples columnas.

### Solución al "Underflow Aritmético" (Subdesbordamiento)
El Naive Bayes estándar asume que todas las características (predictores) son estadísticamente independientes. La fórmula original multiplica todas las probabilidades condicionales juntas:
$$ P(\text{Clase}) \cdot \prod_{i=1}^{n} P(x_i | \text{Clase}) $$

**El Problema Computacional**: Multiplicar docenas de probabilidades fraccionarias minúsculas (ej. $0.003 \times 0.0014 \times 0.01$) genera un número con demasiados ceros decimales, provocando que los procesadores (CPU) del rango de `float64` pierdan precisión o lo redondeen prematuramente a un Absoluto `0.0`. Esto se conoce como ***Underflow***.

**La Solución en el Código**: Aplicar propiedades de **Logaritmos**.
Sabiendo que el logaritmo de un producto es la suma de los logaritmos ($\log(a \cdot b) = \log(a) + \log(b)$), el código transforma las multiplicaciones en sumas, operando con números negativos grandes en lugar de microscópicos.
Se visualiza claramente en el método `_log_posterior(self, row, cls)`:

```python
prior = self.class_priors.get(cls, self.epsilon)
log_prob = log(max(prior, self.epsilon))  # Inicio del sumatorio

for feature in self.feature_columns:
    # ... Lógica interna según tipo (Numérico/Categórico/Binario)
    
    # En lugar de multiplicar, SUMA
    log_prob += log(max(p, self.epsilon))
```

Finalmente, para devolver una probabilidad legible (0.0 a 1.0) a la interfaz (`predict_proba`), invierte el orden restando el valor máximo y pasándolo por una función exponencial (`exp`) actuando como una pseudo compuerta `Softmax`:
```python
max_log = max(log_posteriors.values())
exp_scores[cls] = exp(logp - max_log)  # Prevención de Overflow en exponenciación
```

---

## 3. metrics.py: Evaluación de Disonancias Predictivas

Una vez que el clasificador Bayesiano iteró todas las inferencias, necesita comparar sus **Predicciones ($yp$)** contra la **Realidad ($yt$)**. 

### Matriz de Confusión
```python
tp = int(((yt == positive_label) & (yp == positive_label)).sum())
fp = int(((yt != positive_label) & (yp == positive_label)).sum())
```
* **True Positives (TP)**: La realidad era Anómala, y el modelo pronosticó Anómalo.
* **False Positives (FP)**: La realidad era Normal, pero el modelo falló ("Falsa Alarma").

### Cálculo de Métricas Finales
La función `classification_metrics()` mapea la Matriz hacia los KPI científicos.
```python
accuracy = (cm.true_positive + cm.true_negative) / max(total, 1)
sensitivity = cm.true_positive / max(cm.true_positive + cm.false_negative, epsilon)
specificity = cm.true_negative / max(cm.true_negative + cm.false_positive, epsilon)
```
* **Exactitud (Accuracy)**: Aciertos globales (TP + TN) sobre todo el dataset.
* **Sensibilidad (Recall)**: De todos los casos que eran **verdaderamente positivos** (ej: piezas averiadas), ¿cuántos logró captar nuestro modelo? (La métrica preferida en Detección de Anomalías severas).
* La directiva `max(..., epsilon)` también rige en los divisores para inmunizar al algoritmo de los errores `ZeroDivisionError` en *slices* muy pequeños o datos sesgados (Imbalanced Datasets).
