"""
gate.py
=======
The two-threshold abstention rule: fitting, evaluation at one operating point,
storage and application.

Fitting, on one train/test split:

1. The training split is divided into a fit part and a validation part with
   `splits.split_indices`, seed `VALIDATION_SEED`: the validation part is one
   fifth of the patches or, when patches carry group ids, all patches of one
   fifth of the groups. The division is stratified by class (and by site).
2. The Mahalanobis scorer and the linear probe are fitted on the fit part.
3. Novelty threshold tau_ood: the 95th percentile of the Mahalanobis scores of
   the validation patches (5% nominal in-distribution false-reject rate). A
   patch is rejected when its score exceeds tau_ood.
4. Confidence threshold tau_conf: the quantile of the maximum softmax probability,
   among the validation patches that pass the novelty gate, that retains 80% of
   them. A patch is retained when its confidence is at least tau_conf.

Evaluation applies both thresholds unchanged to the test split and to the
out-of-distribution sets. A stored gate holds the parameters of the scorer and of
the probe together with the two thresholds, and classifies new embeddings without
access to the data it was fitted on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import scoring
from .evaluation import probe_predict
from .probe import fit_probe
from .splits import joint_strata, split_indices

VALIDATION_SEED = 0         # seed of the fit/validation division of the training split
ID_REJECT = 0.05            # nominal in-distribution false-reject rate of the novelty gate
RETAIN = 0.80               # share of the validation patches passing the novelty gate
#                             that the confidence gate retains
REJECT, ABSTAIN, PREDICT = "reject", "abstain", "predict"


@dataclass
class Gate:
    """A fitted gate: Mahalanobis scorer, standardisation and probe weights, thresholds."""
    classes: np.ndarray         # class names, in label order
    means: np.ndarray           # class means of the scorer, K x D
    precision: np.ndarray       # shared precision matrix of the scorer, D x D
    centre: np.ndarray          # standardisation of the probe input: mean, D
    scale: np.ndarray           # standardisation of the probe input: scale, D
    coef: np.ndarray            # probe weights, K x D (1 x D for two classes)
    intercept: np.ndarray       # probe intercepts, K (1 for two classes)
    tau_ood: float
    tau_conf: float

    def novelty(self, Z) -> np.ndarray:
        """Squared Mahalanobis distance to the nearest class mean."""
        Z = np.asarray(Z, dtype=np.float64)
        dists = []
        for mean in self.means:
            d = Z - mean
            dists.append(((d @ self.precision) * d).sum(axis=1))
        return np.stack(dists, axis=1).min(1)

    def logits(self, Z) -> np.ndarray:
        """Probe logits, one column per class."""
        X = (np.asarray(Z, dtype=np.float64) - self.centre) / self.scale
        d = X @ self.coef.T + self.intercept
        return np.c_[-d[:, 0] / 2.0, d[:, 0] / 2.0] if d.shape[1] == 1 else d

    def decide(self, Z) -> dict[str, np.ndarray]:
        """Decision per embedding: reject (novelty above tau_ood), abstain (confidence
        below tau_conf) or predict, with the novelty score, the predicted label and
        its confidence."""
        novelty = self.novelty(Z)
        logits = self.logits(Z)
        conf = scoring.confidence(logits)
        decision = np.where(novelty > self.tau_ood, REJECT,
                            np.where(conf >= self.tau_conf, PREDICT, ABSTAIN))
        return {"decision": decision, "novelty": novelty, "label": logits.argmax(1),
                "confidence": conf}

    def save(self, path: str) -> None:
        np.savez(path, classes=np.asarray(self.classes, dtype=str), means=self.means,
                 precision=self.precision, centre=self.centre, scale=self.scale,
                 coef=self.coef, intercept=self.intercept,
                 tau_ood=self.tau_ood, tau_conf=self.tau_conf)

    @classmethod
    def load(cls, path: str) -> "Gate":
        with np.load(path, allow_pickle=False) as d:
            return cls(classes=d["classes"], means=d["means"], precision=d["precision"],
                       centre=d["centre"], scale=d["scale"], coef=d["coef"],
                       intercept=d["intercept"], tau_ood=float(d["tau_ood"]),
                       tau_conf=float(d["tau_conf"]))

    def thresholds(self) -> dict:
        return {"tau_ood": self.tau_ood, "tau_conf": self.tau_conf,
                "novelty_scorer": "squared Mahalanobis distance to the nearest class mean",
                "reject_when": "novelty > tau_ood",
                "abstain_when": "novelty <= tau_ood and confidence < tau_conf",
                "id_reject_rate_nominal": ID_REJECT, "retain_share_nominal": RETAIN,
                "classes": [str(c) for c in self.classes]}


def evaluate(split: dict, ood: dict, ordinal: bool) -> tuple[Gate, list[tuple[str, float]]]:
    """Fit on the fit part, set both thresholds on the validation part and apply them to
    the test split and to the sets in `ood` (name -> embeddings). Returns the fitted gate
    and its (quantity, value) pairs."""
    n_classes = len(split["classes"])
    y, groups = split["y_train"], split["groups_train"]
    fit, val = split_indices(y, groups, joint_strata(y, split["sites_train"]),
                             seed=VALIDATION_SEED)
    Z_fit, Z_val = split["Z_train"][fit], split["Z_train"][val]

    detector = scoring.MahalanobisScorer().fit(Z_fit, y[fit])
    probe = fit_probe(Z_fit, y[fit], groups=None if groups is None else groups[fit])

    # thresholds: validation part only
    novelty_val = detector.score(Z_val)
    tau_ood = float(np.quantile(novelty_val, 1.0 - ID_REJECT))
    passed_val = novelty_val <= tau_ood
    _, conf_val = probe_predict(probe, Z_val, n_classes)
    tau_conf = float(np.quantile(conf_val[passed_val], 1.0 - RETAIN))

    # test split and out-of-distribution sets: thresholds applied unchanged
    y_test = split["y_test"]
    pred, conf = probe_predict(probe, split["Z_test"], n_classes)
    correct = pred == y_test
    passed = detector.score(split["Z_test"]) <= tau_ood
    kept = passed & (conf >= tau_conf)

    out = [("n_fit", len(fit)), ("n_validation", len(val)), ("n_test", len(y_test)),
           ("tau_ood", tau_ood), ("tau_conf", tau_conf),
           ("reject_validation", 1.0 - passed_val.mean()),
           ("retain_validation", (conf_val[passed_val] >= tau_conf).mean()),
           ("reject_id", 1.0 - passed.mean())]
    out += [(f"reject_{name}", (detector.score(Z) > tau_ood).mean()) for name, Z in ood.items()]
    stages = (("ungated", np.ones(len(y_test), dtype=bool)), ("novelty_gate", passed),
              ("both_gates", kept))
    out += [(f"accuracy_{name}", correct[m].mean()) for name, m in stages]
    out += [(f"coverage_{name}", m.mean()) for name, m in stages[1:]]
    if ordinal:
        out += [(f"qwk_{name}", scoring.qwk(y_test[m], pred[m], n_classes))
                for name, m in stages]

    scaler, clf = probe[0], probe[-1]
    gate = Gate(classes=np.asarray(split["classes"]), means=detector.means_,
                precision=detector.precision_, centre=scaler.mean_, scale=scaler.scale_,
                coef=clf.coef_, intercept=clf.intercept_, tau_ood=tau_ood, tau_conf=tau_conf)
    return gate, [(q, float(v)) for q, v in out]
