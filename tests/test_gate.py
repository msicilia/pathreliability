"""A stored gate reproduces the decisions of the audit that fitted it."""
import numpy as np
import pytest

import synthetic
from pathreliability.audit import audit
from pathreliability.gate import ABSTAIN, PREDICT, REJECT, Gate


@pytest.fixture(scope="module")
def audited():
    d = synthetic.make("nct")
    far = synthetic.make("breakhis")["Z_test"]
    report, gates = audit(d, far=far, artefact_classes=synthetic.ARTEFACT["nct"], n_boot=20,
                          jobs=1)
    return d, far, report, gates


def test_stored_gate_matches_report(audited, tmp_path):
    d, far, report, gates = audited
    path = str(tmp_path / "gate.npz")
    gates["gate"].save(path)
    gate = Gate.load(path)

    test = gate.decide(d["Z_test"])
    assert np.mean(test["decision"] == REJECT) == pytest.approx(report["gate"]["reject_id"])
    assert np.mean(test["decision"] == PREDICT) == pytest.approx(
        report["gate"]["coverage_both_gates"])
    kept = test["decision"] == PREDICT
    assert np.mean(test["label"][kept] == d["y_test"][kept]) == pytest.approx(
        report["gate"]["accuracy_both_gates"])
    assert np.mean(gate.decide(far)["decision"] == REJECT) == pytest.approx(
        report["gate"]["reject_far"])
    assert set(test["decision"]) <= {REJECT, ABSTAIN, PREDICT}


def test_gate_without_artefact_rejects_artefact(audited):
    d, _, report, gates = audited
    gate = gates["gate_without_artefact"]
    classes = [str(c) for c in d["classes"]]
    artefact = np.isin(d["y_test"], [classes.index(c) for c in synthetic.ARTEFACT["nct"]])
    assert len(gate.classes) == len(classes) - len(synthetic.ARTEFACT["nct"])
    assert np.mean(gate.decide(d["Z_test"][artefact])["decision"] == REJECT) == pytest.approx(
        report["gate"]["reject_artefact"])


def test_two_class_gate_logits():
    d = synthetic.make("breakhis")
    _, gates = audit(d, n_boot=10, jobs=1)
    out = gates["gate"].decide(d["Z_test"])
    assert set(out["label"]) <= {0, 1}
    assert ((out["confidence"] >= 0.5) & (out["confidence"] <= 1.0)).all()
