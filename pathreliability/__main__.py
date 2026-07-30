"""
pathreliability — audit the reliability of your own frozen encoder.

    # no code needed — wrap a standard model:
    pathreliability audit --builtin timm:vit_base_patch16_224 --data ./my_dataset

    # or your own model via a ~10-line adapter:
    pathreliability audit --adapter my_model.py:MyEncoder --data ./my_dataset \
                          --ood-data ./other_organ

--data is a folder with one subfolder per class (ImageFolder layout). Output:
report.html (open this), report.json, gate.json.
"""
from __future__ import annotations

import argparse
import os
import sys

from .adapter import load_adapter, builtin_adapter
from .audit import audit
from .report import write_all


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pathreliability")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="run a reliability audit on your encoder")
    src = a.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="path/to/file.py:ClassName (your Encoder subclass)")
    src.add_argument("--builtin", help="timm:<name> or torchvision:<name>")
    a.add_argument("--data", required=True, help="ID dataset dir (one subfolder per class)")
    a.add_argument("--ood-data", default=None, help="a different-organ dir for far-OOD (optional)")
    a.add_argument("--limit", type=int, default=1200, help="max ID images (stratified); 0 = all")
    a.add_argument("--out", default="out", help="output dir (default ./out)")
    a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cmd == "audit":
        try:
            enc = builtin_adapter(args.builtin) if args.builtin else load_adapter(args.adapter)
        except Exception as e:
            print(f"✗ could not load model: {e}", file=sys.stderr); return 2
        name = args.builtin or args.adapter
        print(f"[pathreliability] auditing '{name}' on '{args.data}'"
              + (f" (far-OOD: {args.ood_data})" if args.ood_data else "") + " …")
        try:
            report = audit(enc, args.data, ood_dir=args.ood_data,
                           limit=args.limit, seed=args.seed)
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            print(f"✗ {e}", file=sys.stderr); return 2
        paths = write_all(report, args.out)
        flags = report["verdict"]["per_axis_flags"]; block = report["verdict"]["blocking_findings"]
        overall = ("FAIL" if any(v == "fail" for v in flags.values())
                   else "WARN" if any(v == "warn" for v in flags.values()) else "PASS")
        print(f"\n  verdict: {overall}" + (f"  (blocking: {', '.join(block)})" if block else ""))
        for k, val in flags.items():
            print(f"      {k:22} {val}")
        print(f"\n  open → {paths['report_html']}")
        print(f"  gate → {paths['gate_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
