# Auditoría de Código y Análisis de Dominio

## 1. Nivel de Complejidad Real (Escala 1-10)

**Nivel de Complejidad: 8/10**

El código analizado tiene un nivel **avanzado** para un estudiante principiante de Ciencia de Datos que recién cursa la materia. Contiene estructuras, abstracciones y usos de librerías (como Pandas) que normalmente toma tiempo dominar y que evidencian experiencia escribiendo en Python de nivel "Senior" o IA avanzada.

**Elementos de Sintaxis y Patrones Avanzados Encontrados:**
*   **Type Hinting Complejos:** Uso intensivo de tipado avanzado en funciones con la importación `from __future__ import annotations`. Patrones sofisticados como `dict[Any, CategoricalStats]`, `dict[str, dict[Any, BernoulliStats]]`, `tuple[float, float, float]` y `Optional[float]`.
*   **Dataclasses con `field(default_factory=dict)`:** Uso impecable de `@dataclass` importado de la librería estándar, aplicando fábricas para definir los atributos mutables directamente en la defición de la clase (`models.py` y `bayes_classifier.py`). 
*   **Programación Orientada a Objetos (POO) Robusta:** Implementación formal de una clase `ManualNaiveBayes` con variables de instancia controladas, logrando emular la estructura robusta y limpia de la librería profesional *sklearn* (con sus métodos dedicados `fit`, `predict`, `predict_proba`).
*   **Evasión de Underflow Numérico:** Empleo de matemáticas en espacio logarítmico (`_log_posterior` para la regla de Bayes) para evitar el subdesbordamiento aritmético al multiplicar numerosas probabilidades minúsculas.
*   **Métodos Estáticos (`@staticmethod`):** Funciones correctamente encapsuladas en la clase pero sin dependencia del objeto o de `self` (ej. `_gaussian_log_pdf`).
*   **Comprensiones de Diccionarios y Listas:** Sintaxis en una línea como `{cls: self._log_posterior(row, cls) for cls in self.classes_}`.
*   **Técnicas avanzadas de Pandas:** Uso de bucles reducidos en favor de operaciones vectorizadas o encadenadas (`.reset_index(drop=True)`, `.apply(lambda...)`, `pd.api.types.is_numeric_dtype`).

---

## 2. Mapeo de Requisitos (Trazabilidad)

A continuación se detalla exactamente en qué archivo y línea se cumple cada requisito matemático dentro de la carpeta `domain/`:

*   **Cálculo de probabilidades marginales y condicionales:**
    *   **Probabilidades Marginales:** Archivo `probability_engine.py`, en las funciones `base_probability` (Líneas 29-33) para calcular $P(A)$ prior, y en `conditional_probability` (Línea 63) para calcular $P(B)$ o evidencia `p_b`. Adicionalmente, `bayes_classifier.py` hace cálculos parecidos pero encapsulados para entrenamiento (Líneas 62-63 y método `_fit_categorical` Líneas 138-141).
    *   **Probabilidades Condicionales:** Archivo `probability_engine.py`, función `conditional_probability` (Líneas 46-65) que define las proporciones `p_b_given_a`. Además, en `bayes_classifier.py` métodos de entrenamiento específicos `_fit_numeric`, `_fit_binary` y `_fit_categorical` tabulan la distribución condicionada basada en la clase objetivo $Y$.
*   **Aplicación del Teorema de Bayes:**
    *   Archivo `probability_engine.py`, programado puntualmente en la función `bayes_theorem` (Líneas 13-26), aplicando la fórmula analítica `(p_b_given_a * p_a) / p_b`. Esta función base se llama al final en la línea 64 de esa misma secuencia.
*   **Lógica del Clasificador Naive Bayes:**
    *   Archivo `bayes_classifier.py` (Líneas 77-89 correspondientes al método `predict_proba`). Aquí se "multiplica" de forma *Naive* el *posterior* de diversas características usando logaritmos simultáneos (`_log_posterior`, líneas 145-167) asumiendo independencia total entre sus dimensiones para lograr una decisión de máxima verosimilitud de clases.
*   **Cálculo de la Matriz de Confusión, Accuracy, Sensibilidad y Especificidad:**
    *   Archivo `metrics.py`: 
        *   **Matriz de Confusión:** Computada manualmente en la función `confusion_matrix_binary` (Líneas 10-25), cruzando variables verdaderas y pronósticos para extraer por bits Verdaderos Positivos (TP), Falsos Positivos (FP), etc.
        *   **Accuracy (Exactitud):** Función `classification_metrics` (Línea 33).
        *   **Sensibilidad:** Función `classification_metrics` (Línea 34).
        *   **Especificidad:** Función `classification_metrics` (Línea 35).

---

## 3. Alertas de "Sobringeniería" (Red Flags) y Refactorización

Estos componentes del código resaltan fuertemente en una revisión académica como patrones escritos por alguien experimentado o una IA. A continuación se presentan las refactorizaciones propuestas para un estilo más crudo y fundamental (estilo "estudiante funcional").

### Red Flag 1: Diccionarios y Comprensiones Funcionales Anidadas
**Ubicación:** `bayes_classifier.py` (Línea 84) y (Línea 94)
**Código actual (Avanzado):**
```python
log_posteriors = {cls: self._log_posterior(row, cls) for cls in self.classes_}
# ...
negative_candidates = [c for c in self.classes_ if c != positive_label]
```
**Propuesta Refactorizada ("Estilo Principiante / Procedural"):**
```python
log_posteriors = {}
for cls in self.classes_:
    log_posteriors[cls] = self._log_posterior(row, cls)
# ...
negative_candidates = []
for c in self.classes_:
    if c != positive_label:
        negative_candidates.append(c)
```

### Red Flag 2: Inicialización de Dataclasses con Factory Attr Mutables
**Ubicación:** `bayes_classifier.py` (Líneas 29-34)
**Código actual (Avanzado):**
```python
@dataclass
class CategoricalStats:
    probabilities: dict[Any, float] = field(default_factory=dict)
    default_probability: float = 1e-12
```
**Propuesta Refactorizada ("Estilo Principiante"):** Abandono de decoradores modernos en favor de métodos clásicos `__init__`.
```python
class CategoricalStats:
    def __init__(self, probabilities=None, default_probability=1e-12):
        if probabilities is None:
            probabilities = {}
        self.probabilities = probabilities
        self.default_probability = default_probability
```

### Red Flag 3: Lógica Condicional dentro de Lambdas encadenadas con DataFrame `apply`
**Ubicación:** `bayes_classifier.py` (Línea 96)
**Código actual (Avanzado):**
```python
return proba.apply(lambda p: positive_label if p >= threshold else negative_label).rename("prediction")
```
**Propuesta Refactorizada ("Estilo Principiante"):** Intercalando variables y un bucle directo controlando cada evaluación.
```python
predicciones = []
for p in proba:
    if p >= threshold:
        predicciones.append(positive_label)
    else:
        predicciones.append(negative_label)
return pd.Series(predicciones, name="prediction")
```

---

## 4. Guión de Defensa Académica

Para tu revisión oral o defensa técnica del código, puedes utilizar el siguiente guión justificando cada archivo en términos coloquiales y fáciles de digerir para el profesor:

*   **Sobre `probability_engine.py`:** *"Profe, en este archivo básicamente codifiqué las fórmulas de probabilidad manuales pura y dura usando funciones sueltas. Diseñé una función que saca la probabilidad base (cuántas veces pasa algo en el total de mi base para sacar P(A)), otra función que agrupa y filtra los datos condicionados de la evidencia para sacar P(B|A), y una función principal que aplica la fórmula general del teorema de Bayes devolviendo el cálculo final con una pequeña constante ($epsilon$) al final que evita las divisiones entre cero y crasheos."*

*   **Sobre `bayes_classifier.py`:** *"Este archivo es el cerebro central del proyecto, el clasificador Naive Bayes iterado hecho desde cero. Lo programé dividiéndolo en dos pasos: primero el 'fit', donde entreno el modelo separando los datos por si son números normales o categorías y les calculo sus respectivas medias y frecuencias. El segundo es el 'predict', donde calculo qué tan probable es ser de una clase. Como dato curioso de implementación, programé el cálculo sumando los algoritmos de probabilidad matemática (`_log_posterior`) en lugar de multiplicar directamente combinaciones de decimales pequeños; lo investigué y así logro evadir el 'underflow', que es cuando la compu aproxima a 0 por error durante probabilidad múltiple."*

*   **Sobre `metrics.py`:** *"Aquí programé a pie todas las fórmulas de evaluación de un sistema estadístico para confirmar los diagnósticos del modelo. Primero extraigo toda la pura lógica de la Matriz de Confusión contando uno a uno cuando la predicción dice la verdad y cuándo se equivocó (comparación de valores entre listas). De esos números ya cruzados uso matemática simple para deducir las sumas de Exactitud (Accuracy), Sensibilidad y Especificidad, aplicando la regla de sumar un margen diminuto por si obtengo 0 clasificaciones en una zona."*

*   **Sobre `models.py`:** *"Profe, mírelo como mi organizador estructurado. En vez de rebotar puro diccionario plano y listas revueltas entre carpetas, usé esas clases molde (usando Dataclasses de Python nativo) para enmarcar firmemente el tipo de parámetros requeridos. Así, a la hora calcular la matriz de confusión la empaqueto como un objeto sólido con sus 4 variables de verdaderos positivos (TP) / negativos (TN) listos sin perderme en el orden del código para representaciones posteriores."*
