"""
probe.py
========
The linear probe used wherever frozen embeddings are classified.

Embeddings are standardised and passed to a multinomial logistic regression.
Its inverse regularisation strength C is selected by three-fold
cross-validation on the training split, over a log-spaced grid from 1e-6 to 1,
by the log-loss of the held-out folds; the test split takes no part in
the selection. When group ids are given
(patient or slide per patch), the folds keep each group on one side, so that C
is not selected on patches from patients or slides the probe was fitted on.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROBE_CS = np.logspace(-6, 0, 7)
CV_FOLDS = 3
CV_SEED = 0
CV_SCORING = "neg_log_loss"
TOL = 1e-6
MAX_ITER = 10000


def fit_probe(Z, y, groups=None, Cs=None, grouped_cv: bool = True, scoring: str = CV_SCORING):
    """Fit the probe and return the fitted scikit-learn pipeline.

    `Cs` overrides the grid; a single value fixes C and no cross-validation is
    run. `grouped_cv=False` ignores
    `groups` when forming the folds, and `scoring` changes the selection criterion;
    both exist for the sensitivity analysis.
    """
    Cs = PROBE_CS if Cs is None else np.atleast_1d(Cs)
    if len(Cs) == 1:
        clf = LogisticRegression(C=float(Cs[0]), tol=TOL, max_iter=MAX_ITER)
        return make_pipeline(StandardScaler(), clf).fit(Z, y)
    if groups is not None and grouped_cv:
        folds = StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=CV_SEED)
        cv = list(folds.split(Z, y, groups))
    else:
        cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=CV_SEED)
    clf = LogisticRegressionCV(Cs=Cs, l1_ratios=(0,), cv=cv, scoring=scoring, tol=TOL,
                               max_iter=MAX_ITER, n_jobs=1, use_legacy_attributes=False)
    return make_pipeline(StandardScaler(), clf).fit(Z, y)


def selected_c(probe) -> float:
    """The C of a fitted probe: selected by cross-validation, or the fixed value."""
    clf = probe[-1]
    return float(clf.C_ if hasattr(clf, "C_") else clf.C)


def probe_logits(probe, Z) -> np.ndarray:
    """Logits of a fitted probe, one column per class.

    For two classes scikit-learn returns one decision value d; the equivalent
    two-class logits are (-d/2, d/2), whose softmax equals `predict_proba`.
    """
    d = probe.decision_function(Z)
    return d if d.ndim == 2 else np.c_[-d / 2.0, d / 2.0]

