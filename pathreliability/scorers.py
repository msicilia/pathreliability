"""
Post-hoc OOD / uncertainty / selective-prediction scorers on frozen embeddings.

Self-contained (numpy + scikit-learn only). Higher score == more anomalous / less
certain. For OOD metrics, OOD is the positive class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, average_precision_score


def _l2(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _as2d(x):
    x = np.asarray(x, dtype=np.float64)
    return x[None, :] if x.ndim == 1 else x


@dataclass
class MahalanobisScorer:
    """Class-conditional Gaussian, shared Ledoit-Wolf covariance. Primary detector."""
    normalize: bool = True
    _means: dict = field(default_factory=dict, repr=False)
    _prec: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, Z, y):
        Z = _l2(_as2d(Z)) if self.normalize else _as2d(Z)
        y = np.asarray(y)
        centred = np.empty_like(Z)
        for c in np.unique(y):
            m = Z[y == c].mean(0)
            self._means[c] = m
            centred[y == c] = Z[y == c] - m
        self._prec = np.linalg.pinv(LedoitWolf().fit(centred).covariance_)
        return self

    def score(self, Z):
        Z = _l2(_as2d(Z)) if self.normalize else _as2d(Z)
        dists = np.stack([((Z - m) @ self._prec * (Z - m)).sum(1)
                          for m in self._means.values()], axis=1)
        return dists.min(1)


@dataclass
class KNNScorer:
    """Distance to the k-th nearest training embedding (cosine). Assumption-free secondary."""
    k: int = 50
    normalize: bool = True
    _bank: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, Z, y=None):
        Z = _as2d(Z)
        self._bank = _l2(Z) if self.normalize else Z
        return self

    def score(self, Z, block=4096):
        Q = _l2(_as2d(Z)) if self.normalize else _as2d(Z)
        k = min(self.k, len(self._bank))
        out = np.empty(len(Q))
        for i in range(0, len(Q), block):
            sims = Q[i:i + block] @ self._bank.T
            out[i:i + block] = 1.0 - np.partition(sims, -k, axis=1)[:, -k]
        return out


def logit_scores(logits, T=1.0):
    logits = _as2d(logits)
    z = logits / T
    z = z - z.max(1, keepdims=True)
    sm = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    energy = -T * (np.log(np.exp(z).sum(1)) + (logits / T).max(1))
    return {"msp": sm.max(1), "energy": energy}


def ood_metrics(s_id, s_ood):
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]
    s = np.r_[s_id, s_ood]
    thr = np.quantile(s_ood, 0.05)
    return {"auroc": float(roc_auc_score(y, s)),
            "aupr": float(average_precision_score(y, s)),
            "fpr@95": float((s_id >= thr).mean()),
            "n_id": int(len(s_id)), "n_ood": int(len(s_ood))}


def risk_coverage_curve(y_true, y_pred, uncertainty):
    order = np.argsort(uncertainty)
    correct = (np.asarray(y_true)[order] == np.asarray(y_pred)[order]).astype(float)
    n = len(correct)
    k = np.arange(1, n + 1)
    return k / n, 1.0 - np.cumsum(correct) / k


def aurc(coverage, risk):
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(risk, coverage))


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)
