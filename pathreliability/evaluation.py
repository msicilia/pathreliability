"""
evaluation.py
=============
Predictions of a fitted probe, the metrics of a set of predictions, and the
excess AURC expected from a ranking that carries no information.
"""
from __future__ import annotations

import numpy as np

from .probe import probe_logits
from .scoring import aurc, confidence, ece, excess_aurc, qwk


def probe_predict(probe, Z, n_classes: int):
    """Predicted label and maximum softmax probability of a fitted probe."""
    if not np.array_equal(probe[-1].classes_, np.arange(n_classes)):
        raise RuntimeError("the probe was not fitted on every class")
    logits = probe_logits(probe, Z)
    return logits.argmax(1), confidence(logits)


def selective_metrics(y, pred, conf, n_classes: int | None = None) -> dict:
    """Accuracy, AURC, E-AURC and ECE of predictions with a confidence each.

    The quadratic weighted kappa is added when `n_classes` is given (ordinal labels).
    """
    correct = np.asarray(pred) == np.asarray(y)
    uncertainty = 1.0 - np.asarray(conf, dtype=float)
    out = {"accuracy": float(correct.mean()),
           "aurc": aurc(correct, uncertainty),
           "e_aurc": excess_aurc(correct, uncertainty),
           "ece": ece(conf, correct)}
    if n_classes is not None:
        out["qwk"] = qwk(y, pred, n_classes)
    return out


def random_ranking_excess_aurc(correct, uncertainty) -> float:
    """Expected E-AURC of a ranking that is independent of correctness.

    Under such a ranking the expected selective risk equals the error rate at every
    coverage, so the expected AURC is the error rate; the oracle AURC is recovered
    from the two library metrics as AURC - E-AURC.
    """
    oracle = aurc(correct, uncertainty) - excess_aurc(correct, uncertainty)
    return float(1.0 - np.mean(correct) - oracle)
