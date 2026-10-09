"""
make_reference.py
=================
Write pathreliability/reference.json: the ranges of the method paper against
which a report sets its values.

Every range is the minimum, median and maximum, over the pathology encoders of the
paper and the tasks of one kind, of the value of one (encoder, task) pair, read from
the result files of the paper's repository. The kinds are the categorical tasks, the
grading task with region-level labels (SICAPv2) and the grading task with
whole-slide labels (PANDA).

Run from the root of this repository:
    python tests/acceptance/make_reference.py --paper-repo PATH
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "pathreliability", "reference.json")
FILES = ("ood_table", "probe_summary", "blur_accuracy", "gate_operating_point",
         "probe_sensitivity_protocols")
GATE = ("reject_id", "reject_far", "reject_artefact", "coverage_novelty_gate",
        "coverage_both_gates")


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def span(values) -> dict | None:
    """Minimum, median and maximum of the per-pair values, and their number."""
    values = [float(v) for v in values]
    if not values:
        return None
    return {"lo": min(values), "median": float(np.median(values)), "hi": max(values),
            "n": len(values)}


def kind_reference(res, models, tasks, scorers) -> dict:
    """Ranges over `models` x `tasks` of the per-pair values."""
    pair = lambda r: r["model"] in models and r["task"] in tasks

    # detection: AUROC per tier and scorer; the near tier is averaged over its classes
    auroc = defaultdict(list)
    for r in filter(pair, res["ood_table"]):
        name = {"far": "far", "artefact": "artefact", "near": "near_mean"}.get(
            r["tier"], r["condition"])
        auroc[(name, r["scorer"], r["model"], r["task"])].append(float(r["auroc"]))
    detection = defaultdict(dict)
    for name in sorted({k[0] for k in auroc}):
        for scorer in scorers:
            detection[name][scorer] = span(np.mean(v) for k, v in auroc.items()
                                           if k[0] == name and k[1] == scorer)

    # probe: calibration error and excess AURC as a share of its random value
    means = {(r["model"], r["task"], r["metric"]): float(r["mean"])
             for r in filter(pair, res["probe_summary"])}
    pairs = sorted({k[:2] for k in means})
    probe = {"ece": span(means[m, t, "ece"] for m, t in pairs),
             # undefined for a pair without errors, whose random value is zero
             "e_aurc_share_of_random": span(means[m, t, "e_aurc"] / means[m, t, "e_aurc_random"]
                                            for m, t in pairs if means[m, t, "e_aurc_random"] > 0)}

    # blur: change of accuracy against the unblurred test split
    acc = {(r["model"], r["task"], float(r["sigma"])): float(r["accuracy"])
           for r in filter(pair, res["blur_accuracy"])}
    blur = {f"blur_s{s}": span(a - acc[m, t, 0.0] for (m, t, sigma), a in acc.items()
                               if sigma == s)
            for s in sorted({k[2] for k in acc} - {0.0})}

    # gate: rejection rates and coverage
    gate = defaultdict(list)
    for r in res["gate_operating_point"]:
        task = r["task"].removesuffix("_artefact")
        if r["model"] in models and task in tasks:
            if r["task"].endswith("_artefact") != (r["quantity"] == "reject_artefact"):
                continue
            if r["quantity"] in GATE or r["quantity"].startswith("reject_blur"):
                gate[r["quantity"]].append(float(r["value"]))

    # tuning: calibration error under grouped and ungrouped folds, per task
    tuning = {}
    for task in tasks:
        ece = defaultdict(list)
        for r in res["probe_sensitivity_protocols"]:
            if r["model"] in models and r["task"] == task and r["criterion"] == "log-loss":
                ece[(r["folds"], r["model"])].append(float(r["ece"]))
        if ece:
            tuning[task] = {f"ece_{folds}_folds": span(np.mean(v) for k, v in ece.items()
                                                       if k[0] == folds)
                            for folds in ("grouped", "ungrouped")}
    return {"detection": dict(detection), "probe": probe, "blur_accuracy_change": blur,
            "gate": {q: span(v) for q, v in sorted(gate.items())}, "tuning_check": tuning}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper-repo", required=True, help="root of the paper's repository")
    args = ap.parse_args()
    sys.path.insert(0, args.paper_repo)
    from src.lib import config
    from src.lib.names import task_name
    from src.lib.scoring import SCORERS

    res = {name: read(os.path.join(args.paper_repo, "results", f"{name}.csv")) for name in FILES}
    models = [m for m in config.ENCODERS if m not in config.BASELINES]
    kinds = {"categorical": ("categorical tasks", config.CATEGORICAL),
             "ordinal_region_labels": ("grading, region-level labels", ("sicap",)),
             "ordinal_slide_labels": ("grading, whole-slide labels", ("panda",))}
    reference = {
        "source": "A Retraining-Free Reliability Audit and Abstention Gate for Frozen "
                  "Pathology Foundation Models",
        "encoders": [config.display(m) for m in models],
        "range": "minimum, median and maximum over the encoders and the tasks of one kind, "
                 "of the value of one encoder on one task",
        "task_names": {t: task_name(t) for t in config.TASKS},
        "kinds": {key: {"label": label, "tasks": list(tasks),
                        **kind_reference(res, models, tasks, SCORERS)}
                  for key, (label, tasks) in kinds.items()},
    }
    with open(OUT, "w") as f:
        json.dump(reference, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
