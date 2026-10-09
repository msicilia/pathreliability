"""
reproduce_paper.py
==================
Check that the audit reproduces the results of the method paper when it is given
the paper's embeddings.

For every requested encoder and task the audit is run on the stored embeddings of
the paper's repository, and each value is compared with the value in the result
files of that repository:

    ood_table.csv                     AUROC by tier, condition and scorer
    probe_stored.csv                  selected C and accuracy on the stored split
    probe_splits.csv                  probe metrics per split seed
    probe_summary.csv                 their means and bootstrap intervals
    probe_sensitivity_protocols.csv   the probe tuned on ungrouped folds
    blur_accuracy.csv                 accuracy under blur
    gate_operating_point.csv          the gate at its operating point

Run from the root of this repository:
    python tests/acceptance/reproduce_paper.py --paper-repo PATH --models lunit conch
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from paper_comparison import TABLES, Comparison, compare

from pathreliability.audit import audit, load_embeddings

TOLERANCE = 1e-9


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper-repo", required=True, help="root of the paper's repository")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    sys.path.insert(0, args.paper_repo)
    from src.lib import config                       # the paper's task definitions

    emb = os.path.join(args.paper_repo, "embeddings")
    res = {name: read(os.path.join(args.paper_repo, "results", f"{name}.csv"))
           for name in TABLES}
    cmp = Comparison(TOLERANCE)
    for model in args.models:
        for task in args.tasks or config.TASKS:
            d = load_embeddings(config.npz_path(model, task, emb))
            donor = config.FAR_OOD[task]
            far = load_embeddings(config.npz_path(model, donor, emb))["Z_test"]
            report, _ = audit(d, far=far, far_name=donor, ordinal=task in config.ORDINAL,
                              artefact_classes=config.ARTEFACT_CLASSES if task == "nct" else (),
                              jobs=args.jobs)
            before = len(cmp.failures)
            compare(model, task, report, res, task in config.ORDINAL, cmp)
            print(f"{model}/{task}: {'ok' if len(cmp.failures) == before else 'DIFFERS'}",
                  flush=True)
    print(f"\n{cmp.n} values compared, largest absolute difference {cmp.worst:.2e}, "
          f"{len(cmp.failures)} beyond {TOLERANCE:g}")
    for line in cmp.failures[:40]:
        print("  " + line)
    return 1 if cmp.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
