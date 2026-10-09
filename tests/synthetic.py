"""
synthetic.py
============
Synthetic embeddings files for the tests: Gaussian classes with a per-group
offset, split as `embed.build_embeddings` splits a patch set. No real data.

Two tasks are generated, named after the tasks of the method paper whose settings
they exercise:

    nct      six classes, three of them named as the paper's non-diagnostic
             classes; group ids; blurred test embeddings
    sicap    four ordered classes; group ids; no blur

and a third, `breakhis`, that only serves as the second dataset of `nct`.
"""
from __future__ import annotations

import numpy as np

from pathreliability.embed import SPLIT_SEED
from pathreliability.splits import split_indices

DIM = 12
BLUR_SIGMAS = (1.0, 2.0, 3.0)
CLASSES = {"nct": ("ADI", "BACK", "DEB", "NORM", "STR", "TUM"),
           "sicap": ("NC", "G3", "G4", "G5"),
           "breakhis": ("benign", "malignant")}
FAR = {"nct": "breakhis", "sicap": "nct"}            # as in the paper
ARTEFACT = {"nct": ("BACK", "DEB", "ADI")}
ORDINAL = ("sicap",)
SEEDS = {"nct": 1, "sicap": 2, "breakhis": 3}


def make(task: str, per_class: int = 150, groups_per_class: int = 10) -> dict:
    """The arrays of the embeddings file of one synthetic task."""
    rng = np.random.default_rng(SEEDS[task])
    classes = CLASSES[task]
    centre = rng.normal(0.0, 3.0, DIM) * (task == "breakhis")
    if task in ORDINAL:             # grades along one direction, neighbours overlapping
        means = np.outer(np.arange(len(classes)), rng.normal(0.0, 0.7, DIM))
    else:
        means = rng.normal(0.0, 1.2, (len(classes), DIM))
    Z, y, groups = [], [], []
    for label in range(len(classes)):
        offsets = rng.normal(0.0, 0.6, (groups_per_class, DIM))
        member = rng.integers(0, groups_per_class, per_class)
        Z.append(centre + means[label] + offsets[member] + rng.normal(0.0, 1.0, (per_class, DIM)))
        y += [label] * per_class
        groups += [f"{classes[label]}_{g}" for g in member]
    Z, y, groups = np.concatenate(Z).astype(np.float32), np.array(y), np.array(groups)
    tr, te = split_indices(y, groups, y, SPLIT_SEED)
    names = np.array([f"{task}_{i}" for i in range(len(y))])
    out = dict(Z_train=Z[tr], y_train=y[tr], paths_train=names[tr], groups_train=groups[tr],
               Z_test=Z[te], y_test=y[te], paths_test=names[te], groups_test=groups[te],
               classes=np.array(classes))
    if task not in ORDINAL:
        for sigma in BLUR_SIGMAS:   # blur: shrink towards the centre and add noise
            noise = rng.normal(0.0, 0.3 * sigma, Z[te].shape)
            out[f"Z_test_blur_s{sigma}"] = (Z[te] * (1.0 - 0.15 * sigma) + noise).astype(np.float32)
    return out
