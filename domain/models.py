"""Domain models and typed contracts."""

import numpy as np
import pandas as pd


class DetectedSchema:
    """Detected column schema for a dataset."""

    def __init__(self, numeric_columns=None, categorical_columns=None, datetime_columns=None, binary_columns=None):
        if numeric_columns is None:
            numeric_columns = []
        if categorical_columns is None:
            categorical_columns = []
        if datetime_columns is None:
            datetime_columns = []
        if binary_columns is None:
            binary_columns = []
            
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.datetime_columns = datetime_columns
        self.binary_columns = binary_columns


class ProbabilityComputation:
    """Summary of a Bayes probability computation."""

    def __init__(self, event_name, evidence_name, p_a, p_b_given_a, p_b, p_a_given_b):
        self.event_name = event_name
        self.evidence_name = evidence_name
        self.p_a = p_a
        self.p_b_given_a = p_b_given_a
        self.p_b = p_b
        self.p_a_given_b = p_a_given_b


class ConfusionMatrixResult:
    """Confusion matrix components."""

    def __init__(self, true_positive, false_positive, true_negative, false_negative):
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.true_negative = true_negative
        self.false_negative = false_negative

    def as_array(self):
        """Return confusion matrix as 2x2 ndarray."""
        return np.array(
            [
                [self.true_negative, self.false_positive],
                [self.false_negative, self.true_positive],
            ]
        )


class ClassificationMetrics:
    """Binary classification metrics."""

    def __init__(self, accuracy, sensitivity, specificity, precision, f1_score, confusion_matrix):
        self.accuracy = accuracy
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.precision = precision
        self.f1_score = f1_score
        self.confusion_matrix = confusion_matrix


class TrainArtifacts:
    """Trained classifier artifacts for downstream reporting."""

    def __init__(self, feature_columns, target_column, positive_label, y_true, y_pred, y_prob, metrics):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.positive_label = positive_label
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.metrics = metrics
