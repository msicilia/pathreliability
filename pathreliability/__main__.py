"""
pathreliability: audit the reliability of a frozen image encoder and fit an
abstention gate for it.

    # audit a model on a labelled patch set (one subdirectory per class)
    pathreliability audit --builtin timm:resnet50 --data ./patches --out ./out
    pathreliability audit --adapter my_model.py:MyEncoder --data ./patches \\
                          --groups groups.csv --far-data ./other_organ --out ./out

    # the two stages separately
    pathreliability embed --adapter my_model.py:MyEncoder --data ./patches --out emb.npz
    pathreliability audit --embeddings emb.npz --out ./out

    # apply the fitted gate to new patches
    pathreliability apply --gate ./out/gate.npz --adapter my_model.py:MyEncoder \\
                          --data ./new_patches --out decisions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np

from . import __version__
from .adapter import builtin_adapter, load_adapter, with_preprocessing
from .audit import N_BOOT, audit, load_embeddings
from .data import list_images, relative
from .embed import BLUR_SIGMAS, SPLIT_SEED, build_embeddings, embed_paths, save_embeddings
from .gate import Gate
from .report import write_all

log = logging.getLogger("pathreliability")


def _model_args(p, required: bool):
    g = p.add_mutually_exclusive_group(required=required)
    g.add_argument("--adapter", help="path/to/file.py:ClassName, a subclass of Encoder")
    g.add_argument("--builtin", help="timm:<name> or torchvision:<name>")
    p.add_argument("--batch", type=int, default=32, help="patches per forward pass")
    p.add_argument("--mean", nargs=3, type=float, metavar=("R", "G", "B"),
                   help="normalisation mean, replacing the one the model declares")
    p.add_argument("--std", nargs=3, type=float, metavar=("R", "G", "B"),
                   help="normalisation standard deviation, replacing the one the model declares")
    p.add_argument("--input-size", type=int, help="input side in pixels, replacing the one "
                                                  "the model declares")


def _embed_args(p):
    p.add_argument("--data", help="labelled patch set: one subdirectory per class")
    ids = p.add_mutually_exclusive_group()
    ids.add_argument("--groups", help="CSV with the columns path, group and optionally site: "
                                      "the patient or slide of each patch")
    ids.add_argument("--group-pattern", help="regular expression whose first group, matched "
                                             "in the file name, is the patient or slide id")
    p.add_argument("--far-data", help="directory of patches from another organ or source")
    p.add_argument("--blur", nargs="*", type=float, default=list(BLUR_SIGMAS),
                   help="blur sigmas, in pixels of a 224 x 224 input")
    p.add_argument("--limit", type=int, default=0,
                   help="embed at most this many patches, the same number per class (0: all)")
    p.add_argument("--seed", type=int, default=SPLIT_SEED, help="seed of the stored split")


def _adapter(args):
    adapter = builtin_adapter(args.builtin) if args.builtin else load_adapter(args.adapter)
    return with_preprocessing(adapter, args.mean, args.std, args.input_size)


_tenths: dict[str, int] = {}


def _progress(name, done, total):
    """Log the embedding progress of one patch set at every tenth of it."""
    tenth = 10 * done // total
    if _tenths.get(name) != tenth:
        _tenths[name] = tenth
        log.info("embedding %s: %d/%d", name, done, total)


def _embed(args) -> dict:
    if not args.data:
        raise ValueError("--data is required to embed a patch set")
    adapter = _adapter(args)
    log.info("model %s, data %s", adapter.label(), args.data)
    return build_embeddings(adapter, args.data, groups_csv=args.groups,
                            group_pattern=args.group_pattern, far_dir=args.far_data,
                            limit=args.limit, seed=args.seed, sigmas=tuple(args.blur),
                            batch=args.batch, progress=_progress)


def cmd_embed(args) -> int:
    save_embeddings(args.out, _embed(args))
    log.info("wrote %s", args.out)
    return 0


def cmd_audit(args) -> int:
    if args.embeddings:
        if args.adapter or args.builtin or args.data:
            raise ValueError("give either --embeddings or a model with --data")
        d = load_embeddings(args.embeddings)
        source = str(d["source"]) if "source" in d else args.embeddings
    else:
        if not (args.adapter or args.builtin):
            raise ValueError("give --embeddings, or a model (--adapter or --builtin) with --data")
        arrays = _embed(args)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "embeddings.npz")
        save_embeddings(path, arrays)
        log.info("wrote %s", path)
        d, source = load_embeddings(path), args.data
    far = far_name = None
    if args.far:
        other = np.load(args.far, allow_pickle=True)
        key = next((k for k in ("Z_far", "Z_test", "Z") if k in other), None)
        if key is None:
            raise ValueError(f"{args.far}: no embeddings (Z_far, Z_test or Z)")
        far, far_name = other[key], args.far
    log.info("auditing %d training and %d test patches, %d classes",
             len(d["y_train"]), len(d["y_test"]), len(d["classes"]))
    report, gates = audit(d, far=far, far_name=far_name,
                          artefact_classes=tuple(args.artefact_classes), ordinal=args.ordinal,
                          n_boot=args.bootstrap, jobs=args.jobs, source=source,
                          model=str(d["model"]) if "model" in d else None)
    paths = write_all(report, gates, args.out)
    for key, reason in report["not_assessed"].items():
        log.info("not assessed: %s (%s)", key, reason)
    log.info("report: %s", paths["report_html"])
    log.info("gate:   %s", paths["gate"])
    return 0


def cmd_apply(args) -> int:
    gate = Gate.load(args.gate)
    if args.embeddings:
        if args.data or args.adapter or args.builtin:
            raise ValueError("give either --embeddings or a model with --data")
        f = np.load(args.embeddings, allow_pickle=True)
        Z = f if isinstance(f, np.ndarray) else f[next(k for k in ("Z", "Z_test") if k in f)]
        names = np.arange(len(Z))
    else:
        if not (args.data and (args.adapter or args.builtin)):
            raise ValueError("give --embeddings, or a model (--adapter or --builtin) with --data")
        adapter = _adapter(args)
        adapter.load()
        paths = list_images(args.data)
        Z, _ = embed_paths(adapter, paths, batch=args.batch,
                           progress=lambda done, total: _progress("patches", done, total))
        names = relative(paths, args.data)
    if Z.shape[1] != gate.means.shape[1]:
        raise ValueError(f"embeddings have {Z.shape[1]} dimensions, the gate expects "
                         f"{gate.means.shape[1]}")
    out = gate.decide(Z)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch", "decision", "novelty", "predicted_class", "confidence"])
        for name, decision, novelty, label, conf in zip(
                names, out["decision"], out["novelty"], out["label"], out["confidence"]):
            w.writerow([name, decision, f"{novelty:.6g}", gate.classes[label], f"{conf:.6f}"])
    counts = {k: int((out["decision"] == k).sum()) for k in ("reject", "abstain", "predict")}
    log.info("%d patches: %s", len(Z), json.dumps(counts))
    log.info("wrote %s", args.out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pathreliability", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--log-file", help="also write the log of the run to this file")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("embed", help="embed a patch set with a model")
    _model_args(p, required=True)
    _embed_args(p)
    p.add_argument("--out", required=True, help="embeddings file to write (.npz)")
    p.set_defaults(run=cmd_embed)

    p = sub.add_parser("audit", help="audit a model and fit its gate")
    p.add_argument("--embeddings", help="embeddings file written by 'embed'")
    _model_args(p, required=False)
    _embed_args(p)
    p.add_argument("--far", help="embeddings file of a second dataset (with --embeddings)")
    p.add_argument("--artefact-classes", nargs="*", default=[],
                   help="classes that are not diagnostic tissue, e.g. background")
    p.add_argument("--ordinal", action="store_true",
                   help="the classes are ordered grades, in the order of their names")
    p.add_argument("--bootstrap", type=int, default=N_BOOT, help="bootstrap draws per split")
    p.add_argument("--jobs", type=int, default=-1, help="parallel workers")
    p.add_argument("--out", default="out", help="output directory")
    p.set_defaults(run=cmd_audit)

    p = sub.add_parser("apply", help="apply a fitted gate to new patches")
    p.add_argument("--gate", required=True, help="gate.npz written by 'audit'")
    p.add_argument("--embeddings", help="embeddings of the new patches (.npy, or .npz with Z)")
    _model_args(p, required=False)
    p.add_argument("--data", help="directory of new patches")
    p.add_argument("--out", required=True, help="CSV file of decisions to write")
    p.set_defaults(run=cmd_apply)

    args = ap.parse_args(argv)
    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log.info("pathreliability %s: %s", __version__, args.cmd)
    try:
        return args.run(args)
    except (FileNotFoundError, NotADirectoryError, ValueError, TypeError, RuntimeError) as err:
        log.error("%s", err)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
