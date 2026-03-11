"""Manual Naive Bayes classifier for mixed data types."""

from math import exp, log, pi, sqrt

import pandas as pd

from utils.helpers import normalize_binary_value


class NumericStats:
    """Gaussian parameters for a numeric feature."""

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance


class BernoulliStats:
    """Bernoulli parameter for a binary feature."""

    def __init__(self, p_one):
        self.p_one = p_one


class CategoricalStats:
    """Categorical conditional probabilities with Laplace smoothing."""

    def __init__(self, probabilities=None, default_probability=1e-12):
        if probabilities is None:
            probabilities = {}
        self.probabilities = probabilities
        self.default_probability = default_probability


class ManualNaiveBayes:
    """Naive Bayes model supporting numeric, binary and categorical columns."""

    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.class_priors = {}
        self.classes_ = []
        self.feature_types = {}
        self.numeric_stats = {}
        self.bernoulli_stats = {}
        self.categorical_stats = {}
        self.feature_columns = []

    def fit(self, x, y):
        """Train Naive Bayes parameters from data."""
        df = x.copy()
        
        self.feature_columns = []
        for col in df.columns:
            self.feature_columns.append(col)
            
        y = y.reset_index(drop=True)
        df = df.reset_index(drop=True)
        
        clases_unicas = []
        for valor in y.dropna().unique():
            clases_unicas.append(valor)
        clases_unicas.sort()
        self.classes_ = clases_unicas

        if len(self.classes_) < 2:
            raise ValueError("Se requieren al menos dos clases en el objetivo")

        n_samples = len(y)
        for cls in self.classes_:
            cantidad = (y == cls).sum()
            self.class_priors[cls] = float(cantidad / max(n_samples, 1))

        for col in self.feature_columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                self.feature_types[col] = "numeric"
                self.numeric_stats[col] = self._fit_numeric(series, y)
            elif self._is_binary_like(series):
                self.feature_types[col] = "binary"
                self.bernoulli_stats[col] = self._fit_binary(series, y)
            else:
                self.feature_types[col] = "categorical"
                self.categorical_stats[col] = self._fit_categorical(series, y)

    def predict_proba(self, x, positive_label):
        """Predict posterior probability for the selected positive class."""
        if positive_label not in self.class_priors:
            raise ValueError(f"La clase positiva {positive_label} no existe en el modelo")

        probs = []
        for idx, row in x.iterrows():
            log_posteriors = {}
            for cls in self.classes_:
                log_posteriors[cls] = self._log_posterior(row, cls)
                
            max_log = max(log_posteriors.values())
            
            exp_scores = {}
            for cls in log_posteriors:
                logp = log_posteriors[cls]
                exp_scores[cls] = exp(logp - max_log)
                
            denom = sum(exp_scores.values()) + self.epsilon
            probabilidad_positiva = float(exp_scores[positive_label] / denom)
            probs.append(probabilidad_positiva)
            
        return pd.Series(probs, index=x.index, name="probability")

    def predict(self, x, positive_label, threshold=0.5):
        """Predict class labels using positive class posterior threshold."""
        proba = self.predict_proba(x=x, positive_label=positive_label)
        
        negative_candidates = []
        for c in self.classes_:
            if c != positive_label:
                negative_candidates.append(c)
                
        negative_label = negative_candidates[0]
        
        predicciones = []
        for p in proba:
            if p >= threshold:
                predicciones.append(positive_label)
            else:
                predicciones.append(negative_label)
                
        return pd.Series(predicciones, index=x.index, name="prediction")

    def _fit_numeric(self, series, y):
        stats = {}
        clean = pd.to_numeric(series, errors="coerce")
        if clean.notna().any():
            global_mean = float(clean.mean())
            global_var = float(clean.var())
        else:
            global_mean = 0.0
            global_var = 1.0

        for cls in self.classes_:
            cls_values = clean[y == cls].dropna()
            if not cls_values.empty:
                mean = float(cls_values.mean())
            else:
                mean = global_mean
                
            if len(cls_values) > 1:
                variance = float(cls_values.var())
            else:
                variance = global_var
                
            stats[cls] = NumericStats(mean=mean, variance=max(variance, self.epsilon))
        return stats

    def _fit_binary(self, series, y):
        stats = {}
        
        valores_normalizados = []
        for val in series:
            valores_normalizados.append(normalize_binary_value(val))
        normalized = pd.Series(valores_normalizados, index=series.index)

        for cls in self.classes_:
            cls_values = normalized[y == cls].dropna()
            if cls_values.empty:
                p_one = 0.5
            else:
                p_one = float((cls_values == 1).mean())
            
            p_final = min(max(p_one, self.epsilon), 1.0 - self.epsilon)
            stats[cls] = BernoulliStats(p_one=p_final)
        return stats

    def _fit_categorical(self, series, y):
        stats = {}
        s = series.astype("string").fillna("<MISSING>")
        
        vocab_set = set()
        for valor in s:
            vocab_set.add(valor)
        vocab = list(vocab_set)
        vocab.sort()
        
        vocab_size = max(len(vocab), 1)

        for cls in self.classes_:
            cls_values = s[y == cls]
            total = len(cls_values)
            counts = cls_values.value_counts(dropna=False)
            
            probs = {}
            for value in vocab:
                conteo = counts.get(value, 0)
                probs[value] = float((conteo + 1) / (total + vocab_size))
                
            stats[cls] = CategoricalStats(
                probabilities=probs,
                default_probability=float(1 / (total + vocab_size)),
            )
        return stats

    def _log_posterior(self, row, cls):
        prior = self.class_priors.get(cls, self.epsilon)
        log_prob = log(max(prior, self.epsilon))

        for feature in self.feature_columns:
            f_type = self.feature_types[feature]
            value = row.get(feature)
            
            if f_type == "numeric":
                stats = self.numeric_stats[feature][cls]
                log_prob += self._gaussian_log_pdf(value, stats.mean, stats.variance)
            elif f_type == "binary":
                stats = self.bernoulli_stats[feature][cls]
                normalized = normalize_binary_value(value)
                if normalized is None:
                    continue
                if normalized == 1:
                    p = stats.p_one
                else:
                    p = 1.0 - stats.p_one
                log_prob += log(max(p, self.epsilon))
            else:
                stats = self.categorical_stats[feature][cls]
                if pd.isna(value):
                    normalized_cat = "<MISSING>"
                else:
                    normalized_cat = str(value)
                p = stats.probabilities.get(normalized_cat, stats.default_probability)
                log_prob += log(max(p, self.epsilon))

        return log_prob

    @staticmethod
    def _gaussian_log_pdf(value, mean, variance):
        if pd.isna(value):
            return 0.0
        val = float(value)
        coeff = -0.5 * log(2.0 * pi * variance)
        exponent = -((val - mean) ** 2) / (2.0 * variance)
        return coeff + exponent

    @staticmethod
    def _is_binary_like(series):
        sample = series.dropna()
        if sample.empty:
            return False
            
        validadores = []
        for val in sample:
            validadores.append(normalize_binary_value(val))
            
        validos = pd.Series(validadores).notna()
        return validos.mean() >= 0.95
