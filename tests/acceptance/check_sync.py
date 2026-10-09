"""
check_sync.py
=============
Check that the functions this tool takes from the analysis code of the method
paper are still the same code.

For every pair below, the function or class of the paper's repository and the one
of this package are compared as syntax trees, ignoring docstrings and comments.
The fitting and evaluation of the gate is adapted, not copied (it also returns the
fitted gate), and is covered by the numerical comparisons instead.

Run from the root of this repository:
    python tests/acceptance/check_sync.py --paper-repo PATH
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (file in the paper's repository, file in this package, names)
PAIRS = (
    ("src/lib/probe.py", "probe.py",
     ("fit_probe", "selected_c", "probe_logits")),
    ("src/lib/scoring.py", "scoring.py",
     ("l2_normalise", "MahalanobisScorer", "KNNScorer", "energy_score", "confidence",
      "msp_score", "auroc", "_ranked_correct", "risk_coverage_curve", "aurc", "excess_aurc",
      "selective_accuracy", "ece", "qwk")),
    ("src/lib/resampling.py", "resampling.py", ("bootstrap_indices", "pooled_interval")),
    ("src/lib/splits.py", "splits.py",
     ("split_indices", "joint_strata", "pool", "resplit", "stored_split")),
    ("src/lib/evaluation.py", "evaluation.py", ("probe_predict", "selective_metrics")),
    ("src/analysis/probe_metrics.py", "evaluation.py", ("random_ranking_excess_aurc",)),
    ("src/lib/conditions.py", "conditions.py", ("hold_out",)),
    ("src/pipeline/ood_sweep.py", "audit.py", ("Detectors",)),
)
CONSTANTS = (
    ("src/lib/probe.py", "probe.py", ("PROBE_CS", "CV_FOLDS", "CV_SEED", "CV_SCORING", "TOL",
                                      "MAX_ITER")),
    ("src/lib/scoring.py", "scoring.py", ("SCORERS", "KNN_K", "ECE_BINS")),
    ("src/lib/splits.py", "splits.py", ("N_FOLDS",)),
    ("src/lib/config.py", "audit.py", ("SPLIT_SEEDS", "N_BOOT")),
    ("src/lib/config.py", "embed.py", ("BLUR_SIGMAS",)),
    ("src/analysis/gate_operating_point.py", "gate.py", ("VALIDATION_SEED", "ID_REJECT",
                                                         "RETAIN")),
)


def definitions(path: str) -> dict[str, str]:
    """name -> dump of each top-level function, class and assignment, without docstrings."""
    with open(path) as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        for sub in ast.walk(node):
            body = getattr(sub, "body", None)
            if (isinstance(sub, (ast.FunctionDef, ast.ClassDef)) and body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                sub.body = body[1:] or [ast.Pass()]
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.dump(node.value)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper-repo", required=True, help="root of the paper's repository")
    args = ap.parse_args()
    differing, n = [], 0
    for theirs, ours, names in PAIRS + CONSTANTS:
        a = definitions(os.path.join(args.paper_repo, theirs))
        b = definitions(os.path.join(ROOT, "pathreliability", ours))
        for name in names:
            n += 1
            if name not in a or name not in b or a[name] != b[name]:
                differing.append(f"{name}: {theirs} and pathreliability/{ours}")
    print(f"{n} definitions compared, {len(differing)} differ")
    for line in differing:
        print("  " + line)
    return 1 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
