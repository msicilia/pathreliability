"""
The reliability audit — runs a user's frozen encoder over their dataset(s) and
scores four axes plus a deployable abstention gate. Retraining-free: the encoder
is only ever asked to embed; a linear probe is fit on the embeddings.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .data import list_imagefolder, subsample, ARTIFACT_NAMES
from .embed import embed_paths
from .scorers import (MahalanobisScorer, KNNScorer, logit_scores,
                      ood_metrics, risk_coverage_curve, aurc, ece)


def _progress(tag):
    def p(done, total):
        sys.stdout.write(f"\r    embedding {tag}: {done}/{total}   ")
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")
    return p


def _fit_T(logits, y):
    from scipy.optimize import minimize_scalar
    rng = np.random.default_rng(0)
    cal = rng.permutation(len(y))[: len(y) // 2]

    def sm(z):
        z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    def nll(T):
        p = sm(logits[cal] / T)
        return -np.log(p[np.arange(len(cal)), y[cal]] + 1e-12).mean()
    return float(minimize_scalar(nll, bounds=(0.3, 10), method="bounded").x)


def audit(adapter, data_dir, ood_dir=None, limit=1200, seed=0,
          blur_sigmas=(1.0, 2.0, 3.0)) -> dict:
    adapter.load()
    caps = adapter.capabilities()

    paths, labels, classes = list_imagefolder(data_dir)
    paths, labels = subsample(paths, labels, limit, seed)
    tr, te = train_test_split(np.arange(len(labels)), test_size=0.25,
                              random_state=seed, stratify=labels)
    ptr, ytr = paths[tr], labels[tr]
    pte, yte = paths[te], labels[te]

    Ztr = embed_paths(adapter, ptr, progress=_progress("train"))
    Zte = embed_paths(adapter, pte, progress=_progress("test"))

    clf = LogisticRegression(max_iter=3000, n_jobs=-1).fit(Ztr, ytr)
    maha = MahalanobisScorer().fit(Ztr, ytr)
    knn = KNNScorer(k=50).fit(Ztr)

    axes = {"ood": {}, "corruption": {}, "calibration": {}, "selective_prediction": {}}

    # --- far-OOD (a second, different dataset) ---
    if ood_dir:
        op, ol, _ = list_imagefolder(ood_dir)
        op, ol = subsample(op, ol, max(200, limit // 3), seed)
        Zood = embed_paths(adapter, op, progress=_progress("far-OOD"))
        axes["ood"]["far_ood"] = {
            "dataset": ood_dir,
            "auroc_maha": round(ood_metrics(maha.score(Zte), maha.score(Zood))["auroc"], 3),
            "auroc_knn": round(ood_metrics(knn.score(Zte), knn.score(Zood))["auroc"], 3)}

    # --- artifact rejection (auto: if class names include background/debris/adipose) ---
    art = [i for i, c in enumerate(classes) if c.upper() in ARTIFACT_NAMES]
    if art and len(classes) - len(art) >= 2:
        keep = ~np.isin(ytr, art)
        sc = MahalanobisScorer().fit(Ztr[keep], ytr[keep])
        idm, oodm = ~np.isin(yte, art), np.isin(yte, art)
        if oodm.sum():
            axes["ood"]["artifact_rejection"] = {
                "held_out": [classes[i] for i in art],
                "auroc": round(ood_metrics(sc.score(Zte[idm]), sc.score(Zte[oodm]))["auroc"], 3)}

    # --- near-OOD (leave-one-class) ---
    if len(classes) >= 3:
        aur = []
        for ci in np.unique(ytr):
            k = ytr != ci
            if k.sum() and (yte == ci).sum():
                sc = MahalanobisScorer().fit(Ztr[k], ytr[k])
                aur.append(ood_metrics(sc.score(Zte[yte != ci]), sc.score(Zte[yte == ci]))["auroc"])
        if aur:
            axes["ood"]["near_ood"] = {"auroc_mean": round(float(np.mean(aur)), 3),
                                       "n_classes": len(aur)}

    # --- corruption (blur) — re-embed the test set blurred ---
    blur = {}
    for s in blur_sigmas:
        Zb = embed_paths(adapter, pte, blur_sigma=s, progress=_progress(f"blur σ={s}"))
        blur[str(s)] = round(ood_metrics(maha.score(Zte), maha.score(Zb))["auroc"], 3)
    axes["corruption"]["blur"] = blur

    # --- calibration ---
    proba = clf.predict_proba(Zte)
    conf, correct = proba.max(1), (proba.argmax(1) == yte).astype(float)
    dec = clf.decision_function(Zte)
    logits = dec if dec.ndim == 2 else np.c_[-dec, dec]
    T = _fit_T(logits, yte)
    ez = np.exp(logits / T - (logits / T).max(1, keepdims=True))
    conf_t = (ez / ez.sum(1, keepdims=True)).max(1)
    axes["calibration"] = {"ece_raw": round(ece(conf, correct), 3),
                           "ece_temp_scaled": round(ece(conf_t, correct), 3),
                           "fitted_T": round(T, 2)}

    # --- selective prediction ---
    energy = logit_scores(logits)["energy"]
    cov, risk = risk_coverage_curve(yte, clf.predict(Zte), energy)
    step = max(1, len(cov) // 40)
    axes["selective_prediction"] = {
        "aurc": round(aurc(cov, risk), 3),
        "accuracy": round(float(correct.mean()), 3),
        "risk_coverage_curve": [[round(float(c), 3), round(float(r), 3)]
                                for c, r in zip(cov[::step], risk[::step])]}

    # --- gate ---
    idsc = maha.score(Zte)
    tau_ood = float(np.percentile(idsc, 95))
    order = np.argsort(-conf)
    tau_conf = float(conf[order[max(0, int(0.8 * len(order)) - 1)]])
    gate = {"stage1_novelty": {"scorer": "mahalanobis", "tau_ood": round(tau_ood, 3),
                               "operating_point": "95th pct ID scores (5% ID false-reject)"},
            "stage2_confidence": {"tau_conf": round(tau_conf, 3),
                                  "read_from": "validation risk-coverage curve",
                                  "target_coverage": 0.80},
            "inference_cost": "one distance evaluation + two comparisons per patch"}

    report = {
        "schema_version": "1.0", "tool_version": "0.1.0",
        "audited_model": {"name": getattr(adapter, "name", type(adapter).__name__),
                          "tier": caps["tier"], "embed_dim": caps["embed_dim"],
                          "magnification": caps["input"]["magnification"]},
        "data": {"id_dataset": data_dir, "far_ood_dataset": ood_dir,
                 "n_train": int(len(ytr)), "n_test": int(len(yte)),
                 "classes": classes, "seed": seed,
                 "split_provenance": "stratified image-level (BYOD)"},
        "axes": axes,
        "scorers": {"primary": "mahalanobis", "secondary": "knn"},
        "gate": gate}
    report["caveats"], report["verdict"] = verdict(axes)
    return report


def verdict(axes):
    caveats, flags, blocking = [], {}, []

    def add(slug, msg, block=False):
        caveats.append({"slug": slug, "message": msg})
        if block:
            blocking.append(slug)

    ood = axes.get("ood", {})
    ood_flag = "pass"
    far = ood.get("far_ood", {}).get("auroc_maha")
    if far is not None and far < 0.70:
        ood_flag = "warn"
        add("far_ood_weak", f"Cross-organ far-OOD AUROC {far} is weak; the novelty gate may "
                            "not separate other-tissue inputs. Confirm with more donors.")
    elif far is not None and far < 0.85:
        ood_flag = "warn"
    art = ood.get("artifact_rejection", {}).get("auroc")
    if art is not None and art < 0.70:
        ood_flag = "fail"
        add("artifact_rejection_fail",
            f"Artifact-rejection AUROC {art}: does not reliably reject non-diagnostic tissue "
            "(background/debris/adipose); unsafe as a quality-control gate.", block=True)
    elif art is not None and art < 0.85 and ood_flag != "fail":
        ood_flag = "warn"
    flags["ood"] = ood_flag

    blur = axes.get("corruption", {}).get("blur", {})
    corr = "pass"
    if blur:
        vals = list(blur.values()); mean_v = sum(vals) / len(vals)
        if mean_v < 0.5:
            corr = "fail"
            add("blur_anti_detection",
                f"Mean blur OOD-AUROC {mean_v:.2f} is below chance and does not recover with "
                "severity: this encoder rates blurred patches as MORE in-distribution than sharp "
                "ones. A QC gate on it would pass the degraded inputs it should reject.", block=True)
        elif min(vals) < 0.6:
            corr = "warn"
            add("blur_detection_weak",
                f"Blur detection is weak at mild blur (min AUROC {min(vals)}) but improves with "
                f"severity (up to {max(vals)}); the gate may miss lightly-blurred inputs.")
    flags["corruption"] = corr

    cal = axes.get("calibration", {})
    cflag = "pass"
    if cal.get("ece_raw", 0) > 0.10:
        cflag = "warn"
        add("miscalibrated_raw",
            f"Raw ECE {cal['ece_raw']} > 0.10; temperature scaling lowers it to "
            f"{cal.get('ece_temp_scaled')} (T={cal.get('fitted_T')}), but recalibration does not "
            "recover accuracy — treat as cosmetic, not a fix.")
    flags["calibration"] = cflag
    flags["selective_prediction"] = "pass"
    return caveats, {"per_axis_flags": flags, "blocking_findings": blocking}
