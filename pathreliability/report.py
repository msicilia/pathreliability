"""Write report.json (machines), gate.json (deploy), report.html (humans, verdict-first)."""
from __future__ import annotations

import json
import os

_FLAG = {"pass": ("#137333", "#e6f4ea", "PASS"),
         "warn": ("#8a6d00", "#fef7e0", "WARN"),
         "fail": ("#a50e0e", "#fce8e6", "FAIL")}


def write_all(report: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    p = {"report_json": os.path.join(out_dir, "report.json"),
         "gate_json": os.path.join(out_dir, "gate.json"),
         "report_html": os.path.join(out_dir, "report.html")}
    json.dump(report, open(p["report_json"], "w"), indent=2)
    json.dump(report["gate"], open(p["gate_json"], "w"), indent=2)
    open(p["report_html"], "w").write(_html(report))
    return p


def _pill(flag):
    fg, bg, label = _FLAG.get(flag, _FLAG["warn"])
    return (f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:12px;'
            f'font-weight:600;font-size:13px">{label}</span>')


def _html(r):
    m, axes, v = r["audited_model"], r["axes"], r["verdict"]
    flags, blocking = v["per_axis_flags"], v["blocking_findings"]
    overall = ("fail" if any(x == "fail" for x in flags.values())
               else "warn" if any(x == "warn" for x in flags.values()) else "pass")
    fg, bg, label = _FLAG[overall]
    headline = ("Do not deploy this model as a quality-control / abstention gate."
                if overall == "fail" else
                "Usable with the caveats below." if overall == "warn" else
                "No blocking reliability issues found.")

    rows = "".join(
        f'<tr><td style="padding:6px 12px">{a.replace("_"," ")}</td>'
        f'<td style="padding:6px 12px">{_pill(flags.get(a,"warn"))}</td></tr>'
        for a in ("ood", "corruption", "calibration", "selective_prediction"))
    cav = "".join(f'<li style="margin:6px 0"><code>{c["slug"]}</code> — {c["message"]}</li>'
                  for c in r["caveats"]) or "<li>None.</li>"

    ood = axes.get("ood", {}); far = ood.get("far_ood", {}); art = ood.get("artifact_rejection", {})
    near = ood.get("near_ood", {}); blur = axes.get("corruption", {}).get("blur", {})
    cal = axes.get("calibration", {}); sel = axes.get("selective_prediction", {}); g = r["gate"]

    def kv(k, val):
        return (f'<tr><td style="padding:4px 12px;color:#555">{k}</td>'
                f'<td style="padding:4px 12px"><b>{val}</b></td></tr>')

    detail = "".join([
        kv("far-OOD AUROC (Maha / kNN)", f'{far.get("auroc_maha","–")} / {far.get("auroc_knn","–")}') if far else "",
        kv("artifact-rejection AUROC", f'{art.get("auroc","–")} (held out {", ".join(art.get("held_out",[]))})') if art else "",
        kv("near-OOD AUROC (mean)", near.get("auroc_mean", "–")) if near else "",
        kv("blur OOD-AUROC (σ=1/2/3)", " / ".join(str(blur[k]) for k in sorted(blur))) if blur else "",
        kv("probe accuracy", sel.get("accuracy", "–")) if sel else "",
        kv("calibration ECE (raw → temp)", f'{cal.get("ece_raw","–")} → {cal.get("ece_temp_scaled","–")} (T={cal.get("fitted_T","–")})') if cal else "",
        kv("selective-prediction AURC", sel.get("aurc", "–")) if sel else ""])

    gate = "".join([kv("stage 1 — novelty τ_ood", g["stage1_novelty"]["tau_ood"]),
                    kv("stage 2 — confidence τ_conf", g["stage2_confidence"]["tau_conf"]),
                    kv("inference cost", g["inference_cost"])])
    d = r["data"]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Reliability report — {m['name']}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:32px auto;
   padding:0 20px;color:#202124;line-height:1.5}}
 h1{{font-size:22px;margin:0 0 2px}} h2{{font-size:15px;color:#5f6368;margin:26px 0 8px;
   text-transform:uppercase;letter-spacing:.04em}}
 table{{border-collapse:collapse;width:100%}}
 code{{background:#f1f3f4;padding:1px 5px;border-radius:4px;font-size:13px}}
 .banner{{background:{bg};border-left:6px solid {fg};padding:14px 18px;border-radius:8px;margin:10px 0 4px}}
 .banner b{{color:{fg}}}
</style></head><body>
<h1>Reliability report — {m['name']}</h1>
<div style="color:#5f6368;font-size:14px">tier {m['tier']} · embed_dim {m['embed_dim']} ·
  {len(d['classes'])} classes · n_test {d['n_test']} · schema {r['schema_version']}</div>
<div class="banner"><b>{label}: {headline}</b>
  {'<div style="margin-top:6px">Blocking: '+', '.join('<code>'+b+'</code>' for b in blocking)+'</div>' if blocking else ''}</div>
<h2>Verdict by axis</h2><table>{rows}</table>
<h2>Caveats</h2><ul style="padding-left:20px;margin:4px 0">{cav}</ul>
<h2>Metrics</h2><table>{detail}</table>
<h2>Deployable gate</h2><table>{gate}</table>
<div style="color:#5f6368;font-size:13px;margin-top:6px">Regime caveat: stage 2 is trustworthy
  only where predicted confidence tracks correctness; under weak whole-slide labels, aggregate to
  the slide rather than abstain patch-by-patch.</div>
<h2>Provenance</h2><div style="color:#5f6368;font-size:13px">Retraining-free audit on frozen
  embeddings + a linear probe. No weight access, no gradients, no pretraining data. Primary
  scorer Mahalanobis, secondary kNN. ID dataset: <code>{d['id_dataset']}</code>; split:
  {d['split_provenance']}.</div>
</body></html>"""
