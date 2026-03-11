"""Manual binary classification metrics."""

import pandas as pd

from domain.models import ClassificationMetrics, ConfusionMatrixResult


def confusion_matrix_binary(y_true, y_pred, positive_label):
    """Compute confusion matrix elements for binary classification."""
    yt = y_true.reset_index(drop=True)
    yp = y_pred.reset_index(drop=True)

    tp = int(((yt == positive_label) & (yp == positive_label)).sum())
    fp = int(((yt != positive_label) & (yp == positive_label)).sum())
    tn = int(((yt != positive_label) & (yp != positive_label)).sum())
    fn = int(((yt == positive_label) & (yp != positive_label)).sum())

    return ConfusionMatrixResult(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def classification_metrics(y_true, y_pred, positive_label, epsilon=1e-12):
    """Compute manual classification metrics from predictions."""
    cm = confusion_matrix_binary(y_true=y_true, y_pred=y_pred, positive_label=positive_label)
    total = cm.true_positive + cm.false_positive + cm.true_negative + cm.false_negative

    accuracy = (cm.true_positive + cm.true_negative) / max(total, 1)
    sensitivity = cm.true_positive / max(cm.true_positive + cm.false_negative, epsilon)
    specificity = cm.true_negative / max(cm.true_negative + cm.false_positive, epsilon)
    precision = cm.true_positive / max(cm.true_positive + cm.false_positive, epsilon)
    f1 = (2.0 * precision * sensitivity) / max(precision + sensitivity, epsilon)

    return ClassificationMetrics(
        accuracy=float(accuracy),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision),
        f1_score=float(f1),
        confusion_matrix=cm,
    )
