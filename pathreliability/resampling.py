"""
resampling.py
=============
Bootstrap resampling of test patches and pooling of the draws over splits.

Patches of a grouped task (patient or slide id per patch) are resampled by
cluster: groups are drawn with replacement and every patch of a drawn group is
taken, as many times as its group is drawn. Patches of an ungrouped task are
drawn individually.
"""
from __future__ import annotations

import numpy as np


def bootstrap_indices(n: int, groups=None, n_boot: int = 300, seed: int = 0):
    """Yield `n_boot` index arrays, each one bootstrap resample of `n` patches.

    With `groups` (one id per patch) the resample is a cluster bootstrap and its
    length varies with the sizes of the drawn groups.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, n)
        return
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError("one group id per patch is required")
    _, code = np.unique(groups, return_inverse=True)
    order = np.argsort(code, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(code))[:-1])
    for _ in range(n_boot):
        drawn = rng.integers(0, len(members), len(members))
        yield np.concatenate([members[g] for g in drawn])


def pooled_interval(draws_per_split, level: float = 95.0) -> tuple[float, float]:
    """Percentile interval of the bootstrap draws pooled over splits."""
    pooled = np.concatenate([np.asarray(d, dtype=float) for d in draws_per_split])
    if not np.isfinite(pooled).all():
        raise ValueError("non-finite bootstrap draw")
    tail = (100.0 - level) / 2.0
    lo, hi = np.percentile(pooled, [tail, 100.0 - tail])
    return float(lo), float(hi)
