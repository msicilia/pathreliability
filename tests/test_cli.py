"""The three commands, end to end, on generated images with a numpy encoder."""
import csv
import json
import os

import numpy as np
from PIL import Image

from pathreliability.__main__ import main

ADAPTER = '''
import numpy as np
from pathreliability.adapter import Encoder


class Colour(Encoder):
    """Mean and standard deviation of each channel, and of its horizontal differences."""
    input_size = 32
    normalisation = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    def load(self):
        pass

    def embed(self, batch):
        edges = np.abs(np.diff(batch, axis=2))
        return np.concatenate([batch.mean((1, 2)), batch.std((1, 2)), edges.mean((1, 2))],
                              axis=1).astype(np.float32)
'''
COLOURS = {"g1": (200, 120, 150), "g2": (150, 110, 180), "g3": (110, 130, 200)}


def _patches(root, colours, n=40, patients=8, seed=0):
    rng = np.random.default_rng(seed)
    for name, colour in colours.items():
        os.makedirs(os.path.join(root, name))
        for i in range(n):
            image = np.clip(np.array(colour) + rng.normal(0, 25, (40, 40, 3)), 0, 255)
            Image.fromarray(image.astype(np.uint8)).save(
                os.path.join(root, name, f"pt{i % patients:02d}_{name}_{i:03d}.png"))


def test_embed_audit_apply(tmp_path):
    data, far, out = (str(tmp_path / n) for n in ("data", "far", "out"))
    _patches(data, COLOURS)
    _patches(far, {"other": (90, 190, 110)}, n=30, seed=1)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(ADAPTER)
    spec = f"{adapter}:Colour"
    emb = str(tmp_path / "emb.npz")

    assert main(["embed", "--adapter", spec, "--data", data, "--far-data", far,
                 "--group-pattern", r"^(pt\d+)_", "--out", emb]) == 0
    with np.load(emb) as d:
        assert d["Z_train"].shape[1] == 9 and len(d["Z_far"]) == 30
        assert not set(d["groups_train"]) & set(d["groups_test"])
        assert {"Z_test_blur_s1.0", "Z_test_blur_s2.0", "Z_test_blur_s3.0"} <= set(d.files)

    assert main(["audit", "--embeddings", emb, "--ordinal", "--bootstrap", "20", "--jobs", "1",
                 "--out", out]) == 0
    with open(os.path.join(out, "report.json")) as f:
        report = json.load(f)
    assert report["schema_version"] == "2.0"
    assert report["data"]["groups"] is not None and report["tuning_check"] is not None
    assert set(report["not_assessed"]) == {"artefact"}
    assert "qwk" in report["probe"]["summary"]
    assert report["detection"]["far"]["far"]["auroc"]["maha"] > 0.9
    html = open(os.path.join(out, "report.html"), encoding="utf-8").read()
    assert "The gate" in html and "Tuning check" in html

    decisions = str(tmp_path / "decisions.csv")
    assert main(["apply", "--gate", os.path.join(out, "gate.npz"), "--adapter", spec,
                 "--data", far, "--out", decisions]) == 0
    with open(decisions, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30
    assert np.mean([r["decision"] == "reject" for r in rows]) == report["gate"]["reject_far"]


def test_audit_from_images_without_ids(tmp_path):
    data, out = str(tmp_path / "data"), str(tmp_path / "out")
    _patches(data, {k: COLOURS[k] for k in ("g1", "g2")})
    adapter = tmp_path / "adapter.py"
    adapter.write_text(ADAPTER)
    assert main(["audit", "--adapter", f"{adapter}:Colour", "--data", data, "--blur",
                 "--bootstrap", "20", "--jobs", "1", "--out", out]) == 0
    with open(os.path.join(out, "report.json")) as f:
        report = json.load(f)
    assert set(report["not_assessed"]) == {"far", "blur", "near", "artefact", "tuning_check"}
    assert report["tuning_check"] is None and report["blur_accuracy"] is None


def test_errors_are_reported(tmp_path):
    assert main(["audit", "--out", str(tmp_path / "out")]) == 2
    assert main(["audit", "--embeddings", str(tmp_path / "missing.npz"),
                 "--out", str(tmp_path / "out")]) == 2
