"""The audit computes what the analysis code of the method paper computes.

tests/expected_paper.json holds the output of the paper's analysis functions on
the synthetic embeddings of tests/synthetic.py (tests/acceptance/make_expected.py).
"""
import json
import os

import pytest

import synthetic
from paper_comparison import Comparison, compare
from pathreliability.audit import audit

TOLERANCE = 1e-6
MODEL = "synthetic"


@pytest.fixture(scope="module")
def expected():
    with open(os.path.join(os.path.dirname(__file__), "expected_paper.json")) as f:
        return json.load(f)


@pytest.mark.parametrize("task", sorted(synthetic.FAR))
def test_audit_matches_paper_code(task, expected):
    far = synthetic.make(synthetic.FAR[task])["Z_test"]
    report, _ = audit(synthetic.make(task), far=far, ordinal=task in synthetic.ORDINAL,
                      artefact_classes=synthetic.ARTEFACT.get(task, ()), jobs=1)
    cmp = Comparison(TOLERANCE)
    compare(MODEL, task, report, expected, task in synthetic.ORDINAL, cmp)
    assert cmp.n > 100
    assert not cmp.failures, "\n".join(cmp.failures[:20])
