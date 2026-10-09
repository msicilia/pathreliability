"""
make_expected.py
================
Write tests/expected_paper.json: the values that the analysis code of the method
paper computes on the synthetic embeddings of tests/synthetic.py.

The synthetic embeddings are written to a temporary directory in the layout of
the paper's repository, and its analysis functions are run on them unchanged:

    src/pipeline/ood_sweep.py              run_task
    src/analysis/probe_metrics.py          run_split, summarise
    src/analysis/probe_sensitivity.py      run_job
    src/analysis/blur_accuracy.py          run_task
    src/analysis/gate_operating_point.py   run_job

tests/test_paper_equivalence.py runs the audit on the same embeddings and compares.

Run from the root of this repository:
    python tests/acceptance/make_expected.py --paper-repo PATH
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

import synthetic

OUT = os.path.join(ROOT, "tests", "expected_paper.json")
MODEL = "synthetic"


def text(rows) -> list[dict]:
    """Rows as a CSV reader would return them: every value as text."""
    return [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper-repo", required=True, help="root of the paper's repository")
    args = ap.parse_args()
    sys.path.insert(0, args.paper_repo)
    from src.analysis import blur_accuracy, gate_operating_point, probe_metrics, probe_sensitivity
    from src.lib import config
    from src.pipeline import ood_sweep

    for task, donor in synthetic.FAR.items():       # the settings the synthetic tasks rely on
        assert config.FAR_OOD[task] == donor
    assert set(config.ARTEFACT_CLASSES) == set(synthetic.ARTEFACT["nct"])
    assert tuple(config.BLUR_SIGMAS) == synthetic.BLUR_SIGMAS

    tables = {name: [] for name in ("ood_table", "probe_stored", "probe_splits", "probe_summary",
                                    "probe_sensitivity_protocols", "blur_accuracy",
                                    "gate_operating_point")}
    with tempfile.TemporaryDirectory() as emb:
        for task in synthetic.CLASSES:
            np.savez(config.npz_path(MODEL, task, emb), **synthetic.make(task))
        for task in synthetic.FAR:
            rows, probe_row = ood_sweep.run_task(MODEL, task, emb)
            tables["ood_table"] += rows
            tables["probe_stored"].append(probe_row)
            results = [probe_metrics.run_split(MODEL, task, seed, emb, config.N_BOOT)
                       for seed in config.SPLIT_SEEDS]
            tables["probe_splits"] += [row for row, _, _ in results]
            tables["probe_summary"] += probe_metrics.summarise(results, [MODEL], [task])
            for seed in config.SPLIT_SEEDS:
                tables["probe_sensitivity_protocols"] += probe_sensitivity.run_job(
                    MODEL, task, seed, emb)[1]
            if task in config.CATEGORICAL:
                tables["blur_accuracy"] += blur_accuracy.run_task(MODEL, task, emb)
            tables["gate_operating_point"] += gate_operating_point.run_job(MODEL, task, emb)

    with open(OUT, "w") as f:
        json.dump({name: text(rows) for name, rows in tables.items()}, f, indent=0)
        f.write("\n")
    print(f"wrote {OUT}: " + ", ".join(f"{len(v)} {k}" for k, v in tables.items()))


if __name__ == "__main__":
    raise SystemExit(main())
