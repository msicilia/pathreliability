"""
conditions.py
=============
Out-of-distribution conditions of an audit: tiers, condition names and sets.

    tier       condition          out-of-distribution set
    far        far                patches of a second dataset, when one is given
    blur       blur_s<sigma>      the test split under Gaussian blur, one condition
                                  per sigma stored in the embeddings file
    near       holdout_<class>    test patches of one class that is held out (tasks
                                  with three or more classes)
    artefact   artefact           test patches of the classes named as non-diagnostic,
                                  held out jointly

In the far and blur tiers the in-distribution set is the test split and nothing is
removed from the training split. In the near and artefact tiers the held-out
classes are removed from the training and test splits; the in-distribution set is
the remaining test patches.
"""
from __future__ import annotations

import numpy as np

TIERS = ("far", "blur", "near", "artefact")
FAR = "far"
ARTEFACT = "artefact"
BLUR_KEY = "Z_test_blur_s"          # prefix of the blurred test embeddings in a file
NEAR_MIN_CLASSES = 3


def blur_sets(d) -> dict[float, np.ndarray]:
    """sigma -> embeddings of the blurred test split, for every sigma stored in `d`."""
    sigmas = sorted(float(k[len(BLUR_KEY):]) for k in d.keys() if k.startswith(BLUR_KEY))
    n_test = len(d["y_test"])
    out = {}
    for sigma in sigmas:
        z_blur = d[f"{BLUR_KEY}{sigma}"]
        if len(z_blur) != n_test:
            raise RuntimeError("blurred test set has a different size")
        out[sigma] = z_blur
    return out


def blur_name(sigma: float) -> str:
    return f"blur_s{sigma}"


def held_out_conditions(classes, artefact_classes=()):
    """(tier, condition, held-out labels) of the near and artefact conditions."""
    classes = [str(c) for c in classes]
    out = []
    if len(classes) >= NEAR_MIN_CLASSES:
        out += [("near", f"holdout_{name}", [label]) for label, name in enumerate(classes)]
    if artefact_classes:
        missing = [c for c in artefact_classes if c not in classes]
        if missing:
            raise ValueError(f"artefact classes {missing} are not classes of the dataset")
        out.append((ARTEFACT, ARTEFACT, [classes.index(c) for c in artefact_classes]))
    return out


def hold_out(split: dict, labels) -> tuple[dict, np.ndarray]:
    """Remove the classes with the given labels from a split.

    Returns the reduced split, with the remaining labels renumbered 0 .. K-1 in
    their original order, and the test embeddings of the removed classes.
    """
    n_classes = len(split["classes"])
    drop = np.unique(labels)
    if len(drop) != len(labels) or not np.isin(drop, np.arange(n_classes)).all():
        raise ValueError(f"held-out labels {list(labels)} are not distinct labels of the split")
    keep = np.setdiff1d(np.arange(n_classes), drop)
    out = {"classes": np.asarray(split["classes"])[keep]}
    for part in ("train", "test"):
        m = np.isin(split[f"y_{part}"], keep)
        out[f"Z_{part}"] = split[f"Z_{part}"][m]
        out[f"y_{part}"] = np.searchsorted(keep, split[f"y_{part}"][m])
        for key in ("paths", "groups", "sites"):
            v = split[f"{key}_{part}"]
            out[f"{key}_{part}"] = None if v is None else v[m]
    z_ood = split["Z_test"][np.isin(split["y_test"], drop)]
    n_removed = len(split["y_train"]) - len(out["y_train"])
    if not (len(out["y_train"]) and len(out["y_test"]) and len(z_ood) and n_removed):
        raise RuntimeError("empty in-distribution, out-of-distribution or held-out set")
    return out, z_ood
