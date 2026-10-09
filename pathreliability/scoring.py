"""
scoring.py
==========
Post-hoc novelty scorers and reliability metrics for frozen embeddings.

All functions take plain arrays: embeddings (N x D) and, for the logit-based
scores, the logits of the linear probe. numpy, scipy and scikit-learn only.

Conventions
-----------
* A higher novelty score means more out-of-distribution.
* In detection metrics the out-of-distribution set is the positive class.
* Confidence is the maximum softmax probability; uncertainty is one minus it.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.metrics import cohen_kappa_score, roc_auc_score

SCORERS = ("maha", "knn", "energy", "msp")
KNN_K = 50
ECE_BINS = 15


def l2_normalise(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# --------------------------------------------------------------------------- #
# feature-space scorers
# --------------------------------------------------------------------------- #
class MahalanobisScorer:
    """Squared Mahalanobis distance to the nearest class mean (Lee et al., 2018).

    One covariance matrix is shared by all classes: the pooled within-class
    covariance, estimated with Ledoit-Wolf shrinkage.
    """

    def __init__(self, normalise: bool = False):
        self.normalise = normalise

    def _prepare(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        return l2_normalise(Z) if self.normalise else Z

    def fit(self, Z, y) -> "MahalanobisScorer":
        Z, y = self._prepare(Z), np.asarray(y)
        classes = np.unique(y)
        self.means_ = np.stack([Z[y == c].mean(0) for c in classes])
        centred = Z - self.means_[np.searchsorted(classes, y)]
        self.precision_ = np.linalg.pinv(LedoitWolf().fit(centred).covariance_)
        return self

    def score(self, Z) -> np.ndarray:
        Z = self._prepare(Z)
        dists = []
        for mean in self.means_:
            d = Z - mean
            dists.append(((d @ self.precision_) * d).sum(axis=1))
        return np.stack(dists, axis=1).min(1)


class KNNScorer:
    """Distance to the k-th nearest training embedding (Sun et al., 2022).

    Embeddings are L2-normalised by default, so the Euclidean distance used here
    orders points exactly as the cosine distance does.
    """

    def __init__(self, k: int = KNN_K, normalise: bool = True):
        self.k = k
        self.normalise = normalise

    def _prepare(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        return l2_normalise(Z) if self.normalise else Z

    def fit(self, Z, y=None) -> "KNNScorer":
        self.bank_ = self._prepare(Z)
        self.bank_sq_ = (self.bank_ ** 2).sum(1)
        return self

    def score(self, Z, block: int = 2048) -> np.ndarray:
        Q = self._prepare(Z)
        out = np.empty(len(Q))
        for i in range(0, len(Q), block):
            q = Q[i:i + block]
            d2 = (q ** 2).sum(1)[:, None] + self.bank_sq_[None, :] - 2.0 * (q @ self.bank_.T)
            kth = np.partition(d2, self.k - 1, axis=1)[:, self.k - 1]
            out[i:i + block] = np.sqrt(np.maximum(kth, 0.0))
        return out


# --------------------------------------------------------------------------- #
# logit-based scores
# --------------------------------------------------------------------------- #
def energy_score(logits: np.ndarray) -> np.ndarray:
    """Negative free energy, -logsumexp(logits) (Liu et al., 2020)."""
    return -logsumexp(np.asarray(logits, dtype=np.float64), axis=1)


def confidence(logits: np.ndarray) -> np.ndarray:
    """Maximum softmax probability (Hendrycks and Gimpel, 2017)."""
    return softmax(np.asarray(logits, dtype=np.float64), axis=1).max(1)


def msp_score(logits: np.ndarray) -> np.ndarray:
    """One minus the maximum softmax probability."""
    return 1.0 - confidence(logits)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auroc(s_id: np.ndarray, s_ood: np.ndarray) -> float:
    """AUROC of a novelty score, with the out-of-distribution set as positive."""
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]
    return float(roc_auc_score(y, np.r_[s_id, s_ood]))


def _ranked_correct(correct: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """Correctness ordered from most to least certain. Predictions with equal
    uncertainty share their mean correctness, which is the expectation over a
    random order within the tie."""
    uncertainty = np.asarray(uncertainty, dtype=float)
    order = np.argsort(uncertainty, kind="stable")
    ranked = np.asarray(correct, dtype=float)[order]
    _, tie = np.unique(uncertainty[order], return_inverse=True)
    return (np.bincount(tie, weights=ranked) / np.bincount(tie))[tie]


def risk_coverage_curve(correct: np.ndarray, uncertainty: np.ndarray):
    """Selective risk against coverage, abstaining on the most uncertain first.

    Returns (coverage, risk): after keeping the k most certain predictions,
    coverage is k/n and risk is their error rate.
    """
    ranked = _ranked_correct(correct, uncertainty)
    k = np.arange(1, len(ranked) + 1)
    return k / len(ranked), 1.0 - np.cumsum(ranked) / k


def aurc(correct: np.ndarray, uncertainty: np.ndarray) -> float:
    """Area under the risk-coverage curve: the mean selective risk over all coverages."""
    return float(risk_coverage_curve(correct, uncertainty)[1].mean())


def excess_aurc(correct: np.ndarray, uncertainty: np.ndarray) -> float:
    """AURC minus the AURC of a perfect ranking at the same error rate
    (Geifman et al., 2019). Zero when every error is ranked below every correct
    prediction."""
    correct = np.asarray(correct, dtype=float)
    oracle = aurc(np.sort(correct)[::-1], np.arange(len(correct)))
    return aurc(correct, uncertainty) - oracle


def selective_accuracy(correct: np.ndarray, uncertainty: np.ndarray, coverage: float) -> float:
    """Accuracy on the `coverage` fraction of most certain predictions."""
    ranked = _ranked_correct(correct, uncertainty)
    k = max(1, int(round(coverage * len(ranked))))
    return float(ranked[:k].mean())


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Quadratic weighted kappa over the ordinal labels 0 .. n_classes-1."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                   labels=np.arange(n_classes)))
