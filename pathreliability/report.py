"""
report.py
=========
The outputs of an audit:

    report.json     every value of the audit, with the schema version
    report.html     the same values for a reader, each beside the range of the
                    method paper
    gate.npz        the fitted gate, read by `pathreliability apply`
    gate.json       its two thresholds and the decision rule

The ranges come from reference.json, which is generated from the result files of
the paper (tests/acceptance/make_reference.py). They are context for a value, not
thresholds: the report states no verdict.
"""
from __future__ import annotations

import html
import json
import os

from .conditions import ARTEFACT, FAR

SCORER_NAMES = {"maha": "Mahalanobis", "knn": "kNN", "energy": "energy", "msp": "max. softmax"}
PRIMARY = "maha"


def load_reference() -> dict:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.json")) as f:
        return json.load(f)


def reference_kinds(report: dict, reference: dict) -> list[dict]:
    """The reference entries that apply to the audited task."""
    kinds = reference["kinds"]
    if report["data"]["ordinal"]:
        return [kinds["ordinal_region_labels"], kinds["ordinal_slide_labels"]]
    return [kinds["categorical"]]


def write_all(report: dict, gates: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    reference = load_reference()
    paths = {"report_json": os.path.join(out_dir, "report.json"),
             "report_html": os.path.join(out_dir, "report.html"),
             "gate_json": os.path.join(out_dir, "gate.json")}
    with open(paths["report_json"], "w") as f:
        json.dump({**report, "reference": reference}, f, indent=1, default=float)
    with open(paths["report_html"], "w", encoding="utf-8") as f:
        f.write(render(report, reference))
    for name, gate in gates.items():
        paths[name] = os.path.join(out_dir, f"{name}.npz")
        gate.save(paths[name])
    with open(paths["gate_json"], "w") as f:
        json.dump(gates["gate"].thresholds(), f, indent=1)
    return paths


# --------------------------------------------------------------------------- #
# report.html
# --------------------------------------------------------------------------- #
def num(v, digits: int = 2) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def signed(v, digits: int = 3) -> str:
    return "–" if v is None else f"{round(v, digits) + 0.0:+.{digits}f}".replace("+0.000", "0.000")


def span(r, fmt=num) -> str:
    """A reference range as text: minimum to maximum, with the median."""
    if r is None:
        return "–"
    lo, hi = fmt(r["lo"]), fmt(r["hi"])
    if lo == hi:
        return lo
    return f"{lo} to {hi}<div class=dim>median {fmt(r['median'])}</div>"


def table(header, rows, numeric_from: int = 1) -> str:
    def cells(tag, row):
        return "".join(f'<{tag}{" class=n" if i >= numeric_from else ""}>{c}</{tag}>'
                       for i, c in enumerate(row))
    body = "".join(f"<tr>{cells('td', r)}</tr>" for r in rows)
    return f"<table><thead><tr>{cells('th', header)}</tr></thead><tbody>{body}</tbody></table>"


def render(report: dict, reference: dict) -> str:
    e = html.escape
    data, probe, det, gate = report["data"], report["probe"], report["detection"], report["gate"]
    summary = probe["summary"]
    kinds = reference_kinds(report, reference)
    ref_head = [f"Paper: {e(k['label'])}" for k in kinds]
    ref = lambda *keys, fmt=num: [span(_dig(k, keys), fmt) for k in kinds]
    names = reference["task_names"]

    # the classifier
    def metric(label, key, note, fmt=num):
        s = summary[key]
        value = fmt(s["mean"])
        if s.get("sd") is not None:
            value += f" ± {fmt(s['sd'])}"
        if "lo" in s:
            value += f" <span class=dim>[{fmt(s['lo'])}, {fmt(s['hi'])}]</span>"
        return [f"{label}<div class=dim>{note}</div>", value]

    rows = [metric("Accuracy", "accuracy", "share of test patches classified correctly",
                   fmt=_num3) + ["–"] * len(kinds)]
    if data["ordinal"]:
        rows.append(metric("Quadratic weighted kappa", "qwk",
                           "agreement on an ordered scale; 1 is perfect") + ["–"] * len(kinds))
    rows.append(metric("Calibration error (ECE)", "ece",
                       "gap between stated confidence and accuracy; lower is better", fmt=_num3)
                + ref("probe", "ece", fmt=_num3))
    share = summary["e_aurc_share_of_random"]["mean"]
    rows.append(["Error ranking: excess AURC as a share of its random value"
                 "<div class=dim>0% when confidence ranks every error last, 100% when it "
                 "carries no information</div>",
                 "not defined: no errors" if share is None else pct(share)]
                + ref("probe", "e_aurc_share_of_random", fmt=pct))
    classifier = table(["", f"This model, {len(probe['per_split'])} splits"] + ref_head, rows)

    # detection
    scorers = report["protocol"]["scorers"]
    rows = []

    def detection_row(label, note, auroc, ref_key):
        rows.append([f"{label}<div class=dim>{note}</div>"] + [num(auroc[s]) for s in scorers]
                    + ref("detection", ref_key, PRIMARY))

    if FAR in det:
        entry = det[FAR][FAR]
        detection_row("Second dataset", f"{entry['n_ood']} patches of {e(str(data['far']['name']))}",
                      entry["auroc"], "far")
    if "near" in det:
        detection_row("Unseen class", f"each of the {len(det['near'])} classes held out in turn; "
                      "mean", det["near_mean"], "near_mean")
    if ARTEFACT in det:
        entry = det[ARTEFACT][ARTEFACT]
        detection_row("Non-diagnostic classes", f"{e(', '.join(data['artefact_classes']))} held "
                      f"out jointly; {entry['n_ood']} patches", entry["auroc"], "artefact")
    for condition, entry in det.get("blur", {}).items():
        sigma = condition.removeprefix("blur_s")
        detection_row(f"Blur, σ = {sigma}", "the test patches, blurred", entry["auroc"], condition)
    detection = table([""] + [SCORER_NAMES[s] for s in scorers]
                      + [f"{h} ({SCORER_NAMES[PRIMARY]})" for h in ref_head], rows) if rows else ""
    per_class = ""
    if "near" in det:
        per_class = "<details><summary>Unseen class, by held-out class</summary>" + table(
            ["Held-out class", "patches"] + [SCORER_NAMES[s] for s in scorers],
            [[e(c.removeprefix("holdout_")), v["n_ood"]] + [num(v["auroc"][s]) for s in scorers]
             for c, v in det["near"].items()]) + "</details>"

    # blur and accuracy
    blur = ""
    if report["blur_accuracy"]:
        b = report["blur_accuracy"]
        blur = table(["", "Accuracy", "Change"] + ref_head,
                     [["Unblurred", num(b["unblurred"], 3), "", *[""] * len(kinds)]]
                     + [[f"σ = {c.removeprefix('blur_s')}", num(b["blurred"][c], 3),
                         signed(b["change"][c])]
                        + [span(_dig(k, ("blur_accuracy_change", c)), signed) for k in kinds]
                        for c in b["blurred"]])

    # the gate
    rows = [["Test patches of the task <div class=dim>the cost of the gate; 5% by design"
             "</div>", pct(gate["reject_id"])] + ref("gate", "reject_id", fmt=pct)]
    if "reject_far" in gate:
        rows.append(["Second dataset", pct(gate["reject_far"])]
                    + ref("gate", "reject_far", fmt=pct))
    if f"reject_{ARTEFACT}" in gate:
        rows.append(["Non-diagnostic classes <div class=dim>gate fitted without them</div>",
                     pct(gate[f"reject_{ARTEFACT}"])] + ref("gate", "reject_artefact", fmt=pct))
    for q in (q for q in gate if q.startswith("reject_blur")):
        rows.append([f"Blur, σ = {q.removeprefix('reject_blur_s')}", pct(gate[q])]
                    + ref("gate", q, fmt=pct))
    rejection = table(["Rejected by the novelty threshold", "This model"] + ref_head, rows)
    quality = "qwk" if data["ordinal"] else "accuracy"
    quality_name = "QWK" if data["ordinal"] else "Accuracy"
    retained = table(
        ["", "Patches kept", quality_name + " of the kept patches"],
        [["No gate", "100%", num(gate[f"{quality}_ungated"], 3)],
         ["Novelty threshold", pct(gate["coverage_novelty_gate"]),
          num(gate[f"{quality}_novelty_gate"], 3)],
         ["Novelty and confidence thresholds", pct(gate["coverage_both_gates"]),
          num(gate[f"{quality}_both_gates"], 3)]])

    # tuning check
    tuning = ""
    if report["tuning_check"]:
        t = report["tuning_check"]
        g, u = t["grouped_folds"], t["ungrouped_folds"]
        tuning = table(
            ["Folds used to tune the classifier", "Selected C (median)", "Calibration error",
             "Mean confidence", quality_name],
            [["Grouped by patient or slide <div class=dim>used throughout this report</div>",
              f"{g['C_median']['mean']:g}", _num3(g["ece"]["mean"]),
              num(g["mean_confidence"]["mean"]), num(g[quality]["mean"], 3)],
             ["Ignoring patient or slide", f"{u['C_median']['mean']:g}", _num3(u["ece"]["mean"]),
              num(u["mean_confidence"]["mean"]), num(u[quality]["mean"], 3)]])
        tuning += (f"<p>The calibration error was higher with ungrouped folds in "
                   f"{t['splits_with_higher_ece_ungrouped']} of {t['n_splits']} splits.</p>")
        rows = [[e(names[task]), span(v["ece_grouped_folds"], _num3),
                 span(v["ece_ungrouped_folds"], _num3)]
                for k in reference["kinds"].values() for task, v in k["tuning_check"].items()]
        tuning += ("<details><summary>The same comparison in the paper</summary>"
                   + table(["Task", "Calibration error, grouped folds",
                            "Calibration error, ungrouped folds"], rows) + "</details>")

    notes = []
    if data["groups"] is None:
        notes.append("The patches carry no patient or slide ids, so the split and the tuning of "
                     "the classifier are at patch level. If several patches come from the same "
                     "patient or slide, accuracy and calibration are likely to be optimistic.")
    labels = {FAR: "Detection of a second dataset", "blur": "Blur", "near": "Unseen class",
              ARTEFACT: "Non-diagnostic classes", "tuning_check": "Tuning check"}
    notes += [f"{labels[k]} was not assessed: {e(v)}." for k, v in report["not_assessed"].items()]
    if FAR in report["not_assessed"]:
        notes.append("Without a second dataset the gate has not been tested on tissue from "
                     "another organ or source.")

    groups = ("no patient or slide ids" if data["groups"] is None else
              f"{data['groups']['n_train']} training and {data['groups']['n_test']} test "
              "patients or slides")
    p = report["protocol"]
    m = report["model"]
    preprocessing = ""
    if m.get("normalisation"):
        preprocessing = (f" Input: {m['input_size']} pixels, mean "
                         f"{', '.join(f'{v:.4g}' for v in m['normalisation']['mean'])}, standard "
                         f"deviation {', '.join(f'{v:.4g}' for v in m['normalisation']['std'])}.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reliability report: {e(str(report['model']['name']))}</title>
<style>
 :root {{ --fg:#1f2328; --dim:#656d76; --line:#d0d7de; --bg:#ffffff; --band:#f6f8fa; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --fg:#e6edf3; --dim:#9198a1; --line:#3d444d; --bg:#0d1117; --band:#151b23; }} }}
 body {{ font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:var(--fg);
   background:var(--bg); max-width:920px; margin:32px auto; padding:0 16px; line-height:1.5; }}
 h1 {{ font-size:22px; margin:0 0 4px; }}
 h2 {{ font-size:17px; margin:34px 0 6px; padding-top:14px; border-top:1px solid var(--line); }}
 p {{ margin:6px 0 10px; }}
 .dim {{ color:var(--dim); font-size:13px; font-weight:400; }}
 .wrap {{ overflow-x:auto; }}
 table {{ border-collapse:collapse; width:100%; margin:8px 0 12px; font-size:14px; }}
 th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
   vertical-align:top; }}
 th {{ background:var(--band); font-weight:600; }}
 th.n, td.n {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
 th.n {{ white-space:normal; }}
 ul {{ margin:6px 0 10px; padding-left:20px; }}
 details {{ margin:6px 0 12px; }} summary {{ cursor:pointer; color:var(--dim); }}
 code {{ background:var(--band); padding:1px 5px; border-radius:4px; font-size:13px; }}
</style></head><body>
<h1>Reliability report: {e(str(report['model']['name']))}</h1>
<div class=dim>{e(str(data['source']))} · {len(data['classes'])} classes · {data['n_train']}
 training and {data['n_test']} test patches · {groups} · embedding dimension
 {report['model']['embed_dim']}</div>

<p>This report measures how a linear classifier on the frozen embeddings of this model behaves
beyond its accuracy. Each value is shown beside the range obtained for {len(reference['encoders'])}
pathology encoders in the method paper. The ranges give context; they are not thresholds, and
the report states no verdict.</p>
{"<ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>" if notes else ""}

<h2>The classifier</h2>
<p>Mean and standard deviation over the splits, with a 95% bootstrap interval in brackets.</p>
<div class=wrap>{classifier}</div>

<h2>Detection of inputs the classifier should not receive</h2>
<p>AUROC of each novelty score at separating the test patches from the patches in each row;
0.5 is chance and 1 is perfect separation.</p>
<div class=wrap>{detection or "<p>No detection condition was assessed.</p>"}</div>
{per_class}

{"<h2>Accuracy under blur</h2><p>Whether blur harms the classifier, which is what makes its "
 "detection matter.</p><div class=wrap>" + blur + "</div>" if blur else ""}

<h2>The gate</h2>
<p>A patch is rejected when its Mahalanobis distance exceeds the novelty threshold
(<code>tau_ood</code> = {gate['tau_ood']:.4g}); otherwise the classifier abstains when its
confidence is below the confidence threshold (<code>tau_conf</code> = {gate['tau_conf']:.3f}).
Both were set on {int(gate['n_validation'])} validation patches taken from the training split
and applied unchanged to everything below.</p>
<div class=wrap>{rejection}</div>
<div class=wrap>{retained}</div>

{"<h2>Tuning check</h2><p>The classifier's regularisation is tuned by cross-validation. Folds "
 "that share a patient or slide can select weaker regularisation and make the classifier "
 "over-confident.</p><div class=wrap>" + tuning + "</div>" if tuning else ""}

<h2>Method</h2>
<p class=dim>Linear classifier: standardised embeddings, multinomial logistic regression, C
selected among {len(p['probe']['C_grid'])} values from {p['probe']['C_grid'][0]:g} to
{p['probe']['C_grid'][-1]:g} by {p['probe']['cv_folds']}-fold cross-validation
(folds {e(p['probe']['folds'])}, {e(p['probe']['criterion'])}). Split: {e(p['split'])}.
Classifier metrics: {len(p['split_seeds'])} splits of the pooled embeddings; detection, blur
and gate: the split stored in the embeddings file. Calibration error: {p['ece_bins']}
equal-width bins. kNN: k = {p['knn_k']}. The model was not retrained.{preprocessing}
Reference ranges:
{e(reference['range'])}; from “{e(reference['source'])}”. Tool version
{e(report['tool_version'])}, report schema {e(report['schema_version'])}.</p>
</body></html>
"""


def _num3(v) -> str:
    return num(v, 3)


def _dig(node, keys):
    for k in keys:
        if node is None or k not in node:
            return None
        node = node[k]
    return node
