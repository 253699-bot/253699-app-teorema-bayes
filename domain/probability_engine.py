"""Probability and Bayes theorem related logic."""

import numpy as np
import pandas as pd

from domain.models import ProbabilityComputation


def bayes_theorem(p_b_given_a, p_a, p_b, epsilon=1e-12):
    """Compute P(A|B) using Bayes theorem."""
    denominator = max(p_b, epsilon)
    return float((p_b_given_a * p_a) / denominator)


def base_probability(target, positive_value):
    """Compute P(A) for the positive target class."""
    if target.empty:
        return 0.0
    return float((target == positive_value).mean())


def _default_threshold(feature):
    """Return automatic threshold for numeric evidence."""
    clean = feature.dropna()
    if clean.empty:
        return 0.0
    q75 = float(clean.quantile(0.75))
    mean = float(clean.mean())
    return max(q75, mean)


def conditional_probability(target, evidence_mask, positive_value, epsilon=1e-12):
    """Compute P(B|A), P(B), and P(A|B)."""
    p_a = base_probability(target, positive_value)

    mask_a = target == positive_value
    evidence_true = evidence_mask.fillna(False)

    if mask_a.sum() == 0:
        p_b_given_a = 0.0
    else:
        p_b_given_a = float(evidence_true[mask_a].mean())

    if len(evidence_true) > 0:
        p_b = float(evidence_true.mean())
    else:
        p_b = 0.0

    p_a_given_b = bayes_theorem(p_b_given_a, p_a, p_b, epsilon=epsilon)
    return p_b_given_a, p_b, p_a_given_b


def compute_probability_report(data, target_column, positive_value, evidence_column, threshold=None, epsilon=1e-12):
    """Compute prior, likelihood and posterior probabilities."""
    target = data[target_column]
    evidence = data[evidence_column]

    if pd.api.types.is_numeric_dtype(evidence):
        if threshold is not None:
            used_threshold = threshold
        else:
            used_threshold = _default_threshold(evidence)
            
        evidence_mask = evidence > used_threshold
        evidence_name = f"{evidence_column} > {used_threshold:.3f}"
    else:
        mode = evidence.mode(dropna=True)
        if not mode.empty:
            modal_value = mode.iloc[0]
        else:
            modal_value = None
            
        evidence_mask = evidence == modal_value
        evidence_name = f"{evidence_column} == {modal_value}"

    p_a = base_probability(target, positive_value)
    
    result = conditional_probability(
        target=target,
        evidence_mask=evidence_mask,
        positive_value=positive_value,
        epsilon=epsilon,
    )
    p_b_given_a = result[0]
    p_b = result[1]
    p_a_given_b = result[2]

    return ProbabilityComputation(
        event_name=f"{target_column}={positive_value}",
        evidence_name=evidence_name,
        p_a=p_a,
        p_b_given_a=p_b_given_a,
        p_b=p_b,
        p_a_given_b=p_a_given_b,
    )
