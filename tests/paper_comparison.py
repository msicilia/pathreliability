"""
paper_comparison.py
===================
Comparison of an audit report with the result tables of the method paper.

`tables` maps the name of a result file of the paper's repository to its rows, as
`csv.DictReader` returns them:

    ood_table                     AUROC by tier, condition and scorer
    probe_stored                  selected C and accuracy on the stored split
    probe_splits                  probe metrics per split seed
    probe_summary                 their means and bootstrap intervals
    probe_sensitivity_protocols   the probe tuned on ungrouped folds
    blur_accuracy                 accuracy under blur
    gate_operating_point          the gate at its operating point
"""
from __future__ import annotations

import numpy as np

TABLES = ("ood_table", "probe_stored", "probe_splits", "probe_summary",
          "probe_sensitivity_protocols", "blur_accuracy", "gate_operating_point")
SPLIT_METRICS = ("C", "accuracy", "error_rate", "mean_confidence", "ece", "aurc", "e_aurc",
                 "e_aurc_random")


class Comparison:
    """Counts the compared values and collects those that differ by more than `tol`."""

    def __init__(self, tol: float):
        self.tol, self.n, self.worst, self.failures = tol, 0, 0.0, []

    def check(self, label, ours, theirs, tol=None):
        self.n += 1
        diff = abs(float(ours) - float(theirs))
        self.worst = max(self.worst, diff)
        if not diff <= (self.tol if tol is None else tol):
            self.failures.append(f"{label}: audit {ours!r}, paper {theirs!r}")


def compare(model: str, task: str, report: dict, tables: dict, ordinal: bool,
            cmp: Comparison) -> None:
    key = f"{model}/{task}"

    def mine(name, **where):
        return [r for r in tables[name] if r["model"] == model
                and all(r[k] == str(v) for k, v in where.items())]

    # detection
    ours = {}
    for tier in ("far", "blur", "near", "artefact"):
        for condition, entry in report["detection"].get(tier, {}).items():
            for scorer, value in entry["auroc"].items():
                ours[(tier, condition, scorer)] = (value, entry["n_id"], entry["n_ood"])
    rows = mine("ood_table", task=task)
    if len(rows) != len(ours):
        cmp.failures.append(f"{key}: {len(ours)} detection values, paper has {len(rows)}")
    for r in rows:
        condition = "far" if r["tier"] == "far" else r["condition"]
        value, n_id, n_ood = ours[(r["tier"], condition, r["scorer"])]
        cmp.check(f"{key} AUROC {r['condition']} {r['scorer']}", value, r["auroc"])
        cmp.check(f"{key} n_id {r['condition']}", n_id, r["n_id"], 0)
        cmp.check(f"{key} n_ood {r['condition']}", n_ood, r["n_ood"], 0)

    # probe on the stored split
    (r,) = mine("probe_stored", task=task)
    for name in ("n_train", "n_test", "n_features", "C", "accuracy"):
        cmp.check(f"{key} stored {name}", report["probe"]["stored_split"][name], r[name])

    # probe metrics per split and their summary
    for row in report["probe"]["per_split"]:
        (r,) = mine("probe_splits", task=task, seed=row["seed"])
        for name in SPLIT_METRICS + (("qwk",) if ordinal else ()):
            cmp.check(f"{key} seed {row['seed']} {name}", row[name], r[name])
    for r in mine("probe_summary", task=task):
        entry = report["probe"]["summary"].get(r["metric"])
        if entry is None:
            continue
        cmp.check(f"{key} summary {r['metric']} mean", entry["mean"], r["mean"])
        if r["lo"]:
            cmp.check(f"{key} summary {r['metric']} lo", entry["lo"], r["lo"])
            cmp.check(f"{key} summary {r['metric']} hi", entry["hi"], r["hi"])

    # the probe tuned on ungrouped folds
    rows = mine("probe_sensitivity_protocols", task=task, folds="ungrouped",
                criterion="log-loss")
    if rows:
        tuning = report["tuning_check"]["ungrouped_folds"]
        for name in ("ece", "mean_confidence", "qwk" if ordinal else "accuracy"):
            cmp.check(f"{key} ungrouped {name}", tuning[name]["mean"],
                      np.mean([float(r[name]) for r in rows]))
        cmp.check(f"{key} ungrouped C", tuning["C_median"]["mean"],
                  np.median([float(r["selected_C"]) for r in rows]))

    # accuracy under blur
    for r in mine("blur_accuracy", task=task):
        sigma = float(r["sigma"])
        accuracy = (report["blur_accuracy"]["unblurred"] if sigma == 0.0
                    else report["blur_accuracy"]["blurred"][f"blur_s{sigma}"])
        cmp.check(f"{key} blur accuracy sigma {sigma}", accuracy, r["accuracy"])

    # gate
    for r in mine("gate_operating_point", task=task):
        cmp.check(f"{key} gate {r['quantity']}", report["gate"][r["quantity"]], r["value"])
    for r in mine("gate_operating_point", task=f"{task}_artefact"):
        if r["quantity"] == "reject_artefact":
            cmp.check(f"{key} gate {r['quantity']}", report["gate"]["reject_artefact"], r["value"])
