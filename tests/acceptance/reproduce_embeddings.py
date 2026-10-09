"""
reproduce_embeddings.py
=======================
Check that the tool reproduces the results of the method paper from the images.

The patches of one task of the paper, and of the task that serves as its second
dataset, are embedded again with the preprocessing and blur of this tool, in the
paper's split. The embeddings are compared with those stored in the paper's
repository, the audit is run on the new embeddings, and its values are compared
with the paper's result files. Differences come from the image pipeline and the
arithmetic of the device only.

Three encoders with openly available weights are supported:

    lunit          --builtin timm:hf-hub:1aurent/vit_small_patch8_224.lunit_dino
    imagenet_vit   --builtin timm:vit_base_patch16_224.augreg2_in21k_ft_in1k
    dinov2_nat     an adapter around transformers' facebook/dinov2-base (class token)

Each is given the input normalisation that the paper uses for it. For the Lunit
encoder this is the normalisation published with the model, which differs from
the one stored with its timm port.

Run from the root of this repository:
    python tests/acceptance/reproduce_embeddings.py --paper-repo PATH --model lunit \\
        --data PATH/data/nct/NCT-CRC-HE-100K-NONORM --far-data PATH/data/breakhis
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from paper_comparison import TABLES, Comparison, compare

from pathreliability.adapter import Encoder, builtin_adapter, with_preprocessing
from pathreliability.audit import audit, load_embeddings
from pathreliability.data import EXTENSIONS
from pathreliability.embed import embed_paths

BUILTIN = {"lunit": "timm:hf-hub:1aurent/vit_small_patch8_224.lunit_dino",
           "imagenet_vit": "timm:vit_base_patch16_224.augreg2_in21k_ft_in1k"}


class DinoV2(Encoder):
    """DINOv2 ViT-B/14 trained on natural images; the class token is the embedding."""
    name = "facebook/dinov2-base"

    def load(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.name).to(self.device).eval()

    def embed(self, batch):
        x = self.torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        with self.torch.no_grad():
            return self.model(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy()


def locate(names, root: str) -> np.ndarray:
    """Path under `root` of each patch name; names are unique file names."""
    index = {}
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(EXTENSIONS):
                index.setdefault(f, []).append(os.path.join(r, f))
    missing = [n for n in names if len(index.get(str(n), ())) != 1]
    if missing:
        raise RuntimeError(f"{len(missing)} patches are missing or ambiguous under {root}, "
                           f"e.g. {missing[0]}")
    return np.array([index[str(n)][0] for n in names])


def agreement(name, ours, theirs):
    cosine = (ours * theirs).sum(1) / (np.linalg.norm(ours, axis=1) * np.linalg.norm(theirs, axis=1))
    scale = np.abs(theirs).mean()
    print(f"  {name:22} {ours.shape}  mean |difference| / mean |value| = "
          f"{np.abs(ours - theirs).mean() / scale:.2e}, lowest cosine = {cosine.min():.6f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper-repo", required=True)
    ap.add_argument("--model", required=True, choices=list(BUILTIN) + ["dinov2_nat"])
    ap.add_argument("--task", default="nct")
    ap.add_argument("--data", required=True, help="directory of the patches of the task")
    ap.add_argument("--far-data", required=True, help="directory of the second dataset")
    ap.add_argument("--tolerance", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    sys.path.insert(0, args.paper_repo)
    from src.lib import config

    emb = os.path.join(args.paper_repo, "embeddings")
    paper = load_embeddings(config.npz_path(args.model, args.task, emb))
    donor = config.FAR_OOD[args.task]
    paper_far = load_embeddings(config.npz_path(args.model, donor, emb))

    from src.pipeline.export_embeddings import NORM      # the paper's normalisation per encoder
    adapter = builtin_adapter(BUILTIN[args.model]) if args.model in BUILTIN else DinoV2()
    adapter = with_preprocessing(adapter, *NORM[args.model])
    adapter.load()
    print(f"{adapter.label()}: input {adapter.input_size}, normalisation {adapter.mean_std()}")
    sigmas = tuple(float(k[len("Z_test_blur_s"):]) for k in paper if k.startswith("Z_test_blur_s"))
    z_train, _ = embed_paths(adapter, locate(paper["paths_train"], args.data), batch=args.batch)
    z_test, z_blur = embed_paths(adapter, locate(paper["paths_test"], args.data), sigmas,
                                 args.batch)
    z_far, _ = embed_paths(adapter, locate(paper_far["paths_test"], args.far_data),
                           batch=args.batch)

    print("embeddings, tool against paper:")
    agreement("training split", z_train, paper["Z_train"])
    agreement("test split", z_test, paper["Z_test"])
    for s in sigmas:
        agreement(f"test split, blur {s}", z_blur[s], paper[f"Z_test_blur_s{s}"])
    agreement("second dataset", z_far, paper_far["Z_test"])

    mine = {**paper, "Z_train": z_train, "Z_test": z_test,
            **{f"Z_test_blur_s{s}": z for s, z in z_blur.items()}}
    report, _ = audit(mine, far=z_far, far_name=donor, ordinal=args.task in config.ORDINAL,
                      artefact_classes=config.ARTEFACT_CLASSES if args.task == "nct" else (),
                      jobs=args.jobs)
    tables = {}
    for name in TABLES:
        with open(os.path.join(args.paper_repo, "results", f"{name}.csv"), newline="") as f:
            tables[name] = list(csv.DictReader(f))
    cmp = Comparison(args.tolerance)
    compare(args.model, args.task, report, tables, args.task in config.ORDINAL, cmp)
    print(f"audit, tool against paper: {cmp.n} values compared, largest absolute difference "
          f"{cmp.worst:.2e}, {len(cmp.failures)} beyond {args.tolerance:g}")
    for line in cmp.failures[:40]:
        print("  " + line)
    return 1 if cmp.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
