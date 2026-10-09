"""
audit.py
========
The reliability audit of one embeddings file.

An embeddings file (.npz) stores one train/test split of a labelled patch set
under a frozen encoder:

    Z_train, y_train, paths_train     training embeddings, labels 0 .. K-1, patch names
    Z_test,  y_test,  paths_test      the same for the test split
    classes                           class names, in label order
    groups_train, groups_test         patient or slide id per patch (optional)
    sites_train,  sites_test          site per patch (optional)
    Z_test_blur_s<sigma>              test embeddings under Gaussian blur (optional)
    Z_far                             embeddings of a second dataset (optional)

The audit has four parts.

Detection, on the stored split: the linear probe and four post-hoc scorers
(Mahalanobis, kNN, energy, maximum softmax probability) are fitted on the
training split and scored by AUROC on the conditions of `conditions.py`.

Probe metrics, on the splits of `SPLIT_SEEDS`: the pooled embeddings are split
afresh, the probe is fitted on the training side and its accuracy, expected
calibration error, AURC, excess AURC and the excess AURC expected from a random
ranking are computed on the test side. With group ids, the probe is also tuned on
folds that ignore them, and both are reported.

Blur, on the stored split: accuracy of the probe on the blurred test split.

Gate, on the stored split: the two-threshold rule of `gate.py` at one operating
point, with its rejection rates and the accuracy of the retained predictions.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from . import __version__, gate as gating
from .conditions import (ARTEFACT, BLUR_KEY, FAR, NEAR_MIN_CLASSES, blur_name, blur_sets,
                         held_out_conditions, hold_out)
from .evaluation import probe_predict, random_ranking_excess_aurc, selective_metrics
from .probe import CV_FOLDS, CV_SCORING, PROBE_CS, fit_probe, probe_logits, selected_c
from .resampling import bootstrap_indices, pooled_interval
from .scoring import (ECE_BINS, KNN_K, SCORERS, KNNScorer, MahalanobisScorer, auroc,
                      energy_score, msp_score)
from .splits import N_FOLDS, resplit, stored_split

SCHEMA_VERSION = "2.0"
SPLIT_SEEDS = (0, 1, 2, 3, 4)       # seeds of the splits drawn for the probe metrics
N_BOOT = 300                        # bootstrap resamples per split
REQUIRED = ("Z_train", "y_train", "Z_test", "y_test", "classes")


class Detectors:
    """The probe and the two feature-space scorers, fitted on one training set."""

    def __init__(self, Z, y, groups=None):
        self.probe = fit_probe(Z, y, groups=groups)
        self.maha = MahalanobisScorer().fit(Z, y)
        self.knn = KNNScorer().fit(Z)

    def scores(self, Z) -> dict[str, np.ndarray]:
        """Novelty score of every row of `Z` under each scorer of `SCORERS`."""
        logits = probe_logits(self.probe, Z)
        out = {"maha": self.maha.score(Z), "knn": self.knn.score(Z),
               "energy": energy_score(logits), "msp": msp_score(logits)}
        if tuple(out) != SCORERS:
            raise RuntimeError("scorer set differs from scoring.SCORERS")
        return out


def load_embeddings(path: str) -> dict:
    """Read an embeddings file into a dict of arrays and check its contents."""
    with np.load(path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise ValueError(f"{path}: missing arrays {missing}")
    n_classes = len(d["classes"])
    for part in ("train", "test"):
        Z, y = d[f"Z_{part}"], d[f"y_{part}"]
        if Z.ndim != 2 or len(Z) != len(y):
            raise ValueError(f"{path}: Z_{part} and y_{part} do not describe the same patches")
        if f"paths_{part}" not in d:
            d[f"paths_{part}"] = np.array([f"{part}_{i}" for i in range(len(y))])
        for key in ("groups", "sites"):
            if f"{key}_{part}" in d and len(d[f"{key}_{part}"]) != len(y):
                raise ValueError(f"{path}: {key}_{part} has a different length")
    for key in ("groups", "sites"):
        if (f"{key}_train" in d) != (f"{key}_test" in d):
            raise ValueError(f"{path}: {key} are given for one side of the split only")
    if n_classes < 2 or not np.array_equal(np.unique(d["y_train"]), np.arange(n_classes)):
        raise ValueError(f"{path}: training labels are not 0 .. {n_classes - 1}")
    if not np.isin(d["y_test"], np.arange(n_classes)).all():
        raise ValueError(f"{path}: test labels outside 0 .. {n_classes - 1}")
    return d


# --------------------------------------------------------------------------- #
# detection, on the stored split
# --------------------------------------------------------------------------- #
def condition_rows(tier, condition, s_id, s_ood) -> list[dict]:
    """One row per scorer for a condition, from the scores of its ID and OOD sets."""
    return [dict(tier=tier, condition=condition, scorer=name,
                 auroc=auroc(s_id[name], s_ood[name]),
                 n_id=len(s_id[name]), n_ood=len(s_ood[name]))
            for name in SCORERS]


def held_out_rows(tier, condition, split, held_out) -> list[dict]:
    """Refit without the classes `held_out` and score them against the remaining classes."""
    reduced, z_ood = hold_out(split, held_out)
    det = Detectors(reduced["Z_train"], reduced["y_train"], reduced["groups_train"])
    return condition_rows(tier, condition, det.scores(reduced["Z_test"]), det.scores(z_ood))


def stored_split_rows(split, shifted) -> tuple[list[dict], dict, dict]:
    """Far and blur conditions of the stored split, the probe fitted on it, and its
    accuracy on the test split and on each blurred copy.

    `shifted` is a list of (tier, condition, embeddings).
    """
    y_test = split["y_test"]
    det = Detectors(split["Z_train"], split["y_train"], split["groups_train"])
    s_id = det.scores(split["Z_test"])
    rows = []
    for tier, condition, z_ood in shifted:
        rows += condition_rows(tier, condition, s_id, det.scores(z_ood))
    probe = dict(n_train=len(split["y_train"]), n_test=len(y_test),
                 n_features=int(split["Z_train"].shape[1]), C=selected_c(det.probe),
                 accuracy=float((det.probe.predict(split["Z_test"]) == y_test).mean()))
    blurred = {condition: float((det.probe.predict(z) == y_test).mean())
               for tier, condition, z in shifted if tier == "blur"}
    return rows, probe, blurred


# --------------------------------------------------------------------------- #
# probe metrics, on fresh splits
# --------------------------------------------------------------------------- #
def split_metrics(d, seed: int, ordinal: bool, n_boot: int):
    """One split seed: (row of per-split values, bootstrap draws, row of the probe tuned
    on ungrouped folds or None)."""
    split = resplit(d, seed)
    n_classes = len(split["classes"])
    y, groups = split["y_test"], split["groups_test"]
    k = n_classes if ordinal else None

    def evaluate(probe):
        pred, conf = probe_predict(probe, split["Z_test"], n_classes)
        correct = (pred == y).astype(float)
        row = dict(seed=seed, C=selected_c(probe), error_rate=float(1.0 - correct.mean()),
                   mean_confidence=float(conf.mean()),
                   e_aurc_random=random_ranking_excess_aurc(correct, 1.0 - conf))
        row.update(selective_metrics(y, pred, conf, k))
        return row, pred, conf

    probe = fit_probe(split["Z_train"], split["y_train"], groups=split["groups_train"])
    row, pred, conf = evaluate(probe)
    row.update(n_train=len(split["y_train"]), n_test=len(y),
               n_groups_test=None if groups is None else len(np.unique(groups)))

    draws: dict[str, list[float]] = {}
    for idx in bootstrap_indices(len(y), groups, n_boot, seed):
        for name, value in selective_metrics(y[idx], pred[idx], conf[idx], k).items():
            draws.setdefault(name, []).append(value)

    ungrouped = None
    if groups is not None:
        other = fit_probe(split["Z_train"], split["y_train"], groups=split["groups_train"],
                          grouped_cv=False, scoring=CV_SCORING)
        ungrouped = evaluate(other)[0]
    return row, {name: np.array(v) for name, v in draws.items()}, ungrouped


def summarise(rows, draws=None) -> dict:
    """metric -> mean and sample standard deviation over the splits, with the pooled
    bootstrap interval where draws are given."""
    out = {}
    skip = ("seed", "n_train", "n_test", "n_groups_test", "C")
    for name in (k for k in rows[0] if k not in skip):
        values = np.array([r[name] for r in rows], dtype=float)
        entry = {"mean": float(values.mean()),
                 "sd": float(values.std(ddof=1)) if len(values) > 1 else None}
        if draws is not None and name in draws[0]:
            entry["lo"], entry["hi"] = pooled_interval([d[name] for d in draws])
        out[name] = entry
    out["C_median"] = {"mean": float(np.median([r["C"] for r in rows]))}
    e_aurc, random = out["e_aurc"]["mean"], out["e_aurc_random"]["mean"]
    out["e_aurc_share_of_random"] = {"mean": e_aurc / random if random > 0 else None}
    return out


# --------------------------------------------------------------------------- #
# the audit
# --------------------------------------------------------------------------- #
def _gate_job(split, ood, ordinal, held_out=None):
    if held_out is not None:
        split, z_ood = hold_out(split, held_out)
        ood = {ARTEFACT: z_ood}
    return gating.evaluate(split, ood, ordinal)


def audit(d: dict, far=None, far_name: str | None = None, artefact_classes=(),
          ordinal: bool = False, seeds=SPLIT_SEEDS, n_boot: int = N_BOOT, jobs: int = -1,
          source: str | None = None, model: str | None = None):
    """Audit one embeddings dict (see `load_embeddings`).

    `far` overrides the second-dataset embeddings stored in `d`. Returns the report
    (a dict that serialises to JSON) and the fitted gates by name.
    """
    split = stored_split(d)
    classes = [str(c) for c in split["classes"]]
    groups = split["groups_train"]
    if far is None and "Z_far" in d:
        far = d["Z_far"]
        far_name = far_name or (str(d["far_name"]) if "far_name" in d else None)
    blur = blur_sets(d)
    held_out = held_out_conditions(classes, artefact_classes)

    shifted = ([(FAR, FAR, far)] if far is not None else [])
    shifted += [("blur", blur_name(s), z) for s, z in blur.items()]
    gate_ood = {condition: z for _, condition, z in shifted}

    tasks = [delayed(stored_split_rows)(split, shifted)]
    tasks += [delayed(held_out_rows)(tier, cond, split, labels) for tier, cond, labels in held_out]
    tasks += [delayed(split_metrics)(d, seed, ordinal, n_boot) for seed in seeds]
    tasks += [delayed(_gate_job)(split, gate_ood, ordinal)]
    artefact = [labels for tier, _, labels in held_out if tier == ARTEFACT]
    if artefact:
        tasks += [delayed(_gate_job)(split, None, False, artefact[0])]
    results = Parallel(n_jobs=jobs)(tasks)

    (rows, probe_stored, blurred), results = results[0], results[1:]
    for r in results[:len(held_out)]:
        rows += r
    per_split = results[len(held_out):len(held_out) + len(seeds)]
    gates = results[len(held_out) + len(seeds):]

    # detection: AUROC by tier, condition and scorer
    detection = {}
    for r in rows:
        detection.setdefault(r["tier"], {}).setdefault(r["condition"], {
            "n_id": r["n_id"], "n_ood": r["n_ood"], "auroc": {}})["auroc"][r["scorer"]] = r["auroc"]
    if "near" in detection:
        detection["near_mean"] = {s: float(np.mean([c["auroc"][s]
                                                    for c in detection["near"].values()]))
                                  for s in SCORERS}

    # probe metrics over the splits, and the tuning check
    split_rows = [row for row, _, _ in per_split]
    probe = {"stored_split": probe_stored, "per_split": split_rows,
             "summary": summarise(split_rows, [draws for _, draws, _ in per_split])}
    tuning = None
    if groups is not None:
        other = [u for _, _, u in per_split]
        quality = "qwk" if ordinal else "accuracy"
        tuning = {"grouped_folds": {k: probe["summary"][k] for k in
                                    ("C_median", "ece", "mean_confidence", quality)},
                  "ungrouped_folds": {k: summarise(other)[k] for k in
                                      ("C_median", "ece", "mean_confidence", quality)},
                  "splits_with_higher_ece_ungrouped": int(sum(
                      u["ece"] > r["ece"] for r, u in zip(split_rows, other))),
                  "n_splits": len(seeds)}

    # blur: accuracy of the probe of the stored split on the blurred test split
    blur_accuracy = None
    if blurred:
        blur_accuracy = {"unblurred": probe_stored["accuracy"],
                         "blurred": blurred,
                         "change": {c: a - probe_stored["accuracy"] for c, a in blurred.items()}}

    fitted = {"gate": gates[0][0]}
    gate = dict(gates[0][1])
    if artefact:
        fitted["gate_without_artefact"] = gates[1][0]
        gate[f"reject_{ARTEFACT}"] = dict(gates[1][1])[f"reject_{ARTEFACT}"]
        gate[f"reject_id_without_{ARTEFACT}"] = dict(gates[1][1])["reject_id"]

    not_assessed = {}
    if far is None:
        not_assessed[FAR] = "no second dataset was given"
    if not blur:
        not_assessed["blur"] = "the embeddings file has no blurred test embeddings"
    if len(classes) < NEAR_MIN_CLASSES:
        not_assessed["near"] = f"fewer than {NEAR_MIN_CLASSES} classes"
    if not artefact:
        not_assessed[ARTEFACT] = "no class was named as non-diagnostic"
    if groups is None:
        not_assessed["tuning_check"] = "the patches carry no patient or slide ids"

    report = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "model": {"name": model, "embed_dim": int(split["Z_train"].shape[1]),
                  "input_size": int(d["input_size"]) if "input_size" in d else None,
                  "normalisation": None if "normalisation" not in d else
                  {"mean": [float(v) for v in d["normalisation"][0]],
                   "std": [float(v) for v in d["normalisation"][1]]}},
        "data": {
            "source": source, "classes": classes, "ordinal": bool(ordinal),
            "n_train": len(split["y_train"]), "n_test": len(split["y_test"]),
            "groups": None if groups is None else {
                "n_train": len(np.unique(groups)),
                "n_test": len(np.unique(split["groups_test"]))},
            "sites": None if split["sites_train"] is None else
            sorted(str(s) for s in np.unique(split["sites_train"])),
            "far": None if far is None else {"name": far_name, "n": len(far)},
            "blur_sigmas": list(blur),
            "artefact_classes": list(artefact_classes),
        },
        "protocol": {
            "split": f"test side: one of {N_FOLDS} folds, stratified by class"
                     + (", whole patients or slides held out" if groups is not None else
                        ", at patch level (no patient or slide ids)"),
            "probe": {"C_grid": [float(c) for c in PROBE_CS], "cv_folds": CV_FOLDS,
                      "criterion": "log-loss",
                      "folds": "grouped by patient or slide" if groups is not None else
                      "stratified by class"},
            "split_seeds": list(seeds), "bootstrap_draws_per_split": n_boot,
            "scorers": list(SCORERS), "knn_k": KNN_K, "ece_bins": ECE_BINS,
            "gate": {"id_reject_rate_nominal": gating.ID_REJECT,
                     "retain_share_nominal": gating.RETAIN,
                     "validation_seed": gating.VALIDATION_SEED},
        },
        "probe": probe,
        "detection": detection,
        "blur_accuracy": blur_accuracy,
        "gate": gate,
        "tuning_check": tuning,
        "not_assessed": not_assessed,
    }
    return report, fitted


__all__ = ["audit", "load_embeddings", "Detectors", "SPLIT_SEEDS", "N_BOOT", "BLUR_KEY"]
