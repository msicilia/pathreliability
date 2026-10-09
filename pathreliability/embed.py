"""
embed.py
========
From a patch set and an encoder to an embeddings file.

The patch set is split once into a training and a test side (`splits.split_indices`,
seed `SPLIT_SEED`): one fifth of the patches is held out or, when patches carry
group ids, all patches of one fifth of the groups. The encoder embeds the training
split, the test split and the test split under Gaussian blur, once per sigma. A
second, unlabelled patch set is embedded whole when it is given. The arrays are
those read by `audit.load_embeddings`.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import __version__
from .conditions import BLUR_KEY
from .data import (gaussian_blur, groups_from_csv, groups_from_pattern, list_classes,
                   list_images, load_image, normalise, relative, subsample)
from .splits import joint_strata, split_indices

SPLIT_SEED = 42                     # seed of the split stored in the embeddings file
BLUR_SIGMAS = (1.0, 2.0, 3.0)
FAR_LIMIT = 2000                    # patches of the second dataset that are embedded
WORKERS = 8                         # image-decoding threads


def embed_paths(adapter, paths, sigmas=(), batch: int = 32, progress=None):
    """Embed `paths`; also return the embeddings of the blurred inputs per sigma."""
    mean, std = adapter.mean_std()
    size = adapter.input_size
    unblurred, blurred = [], {s: [] for s in sigmas}
    with ThreadPoolExecutor(WORKERS) as pool:
        for i in range(0, len(paths), batch):
            images = np.stack(list(pool.map(lambda p: load_image(p, size), paths[i:i + batch])))
            x = normalise(images, mean, std)
            unblurred.append(_checked(adapter.embed(x), len(x)))
            for s in sigmas:
                blurred[s].append(_checked(adapter.embed(gaussian_blur(x, s)), len(x)))
            if progress:
                progress(min(i + batch, len(paths)), len(paths))
    return np.concatenate(unblurred), {s: np.concatenate(v) for s, v in blurred.items()}


def _checked(z, n: int) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or len(z) != n:
        raise RuntimeError(f"the encoder returned an array of shape {z.shape} for {n} patches; "
                           "expected one vector per patch")
    if not np.isfinite(z).all():
        raise RuntimeError("the encoder returned non-finite embeddings")
    return z


def build_embeddings(adapter, data_dir: str, groups_csv: str | None = None,
                     group_pattern: str | None = None, far_dir: str | None = None,
                     limit: int = 0, seed: int = SPLIT_SEED, sigmas=BLUR_SIGMAS,
                     batch: int = 32, progress=None) -> dict:
    """Embed a patch set and return the arrays of its embeddings file."""
    paths, labels, classes = list_classes(data_dir)
    groups = sites = None
    if groups_csv:
        groups, sites = groups_from_csv(paths, data_dir, groups_csv)
    elif group_pattern:
        groups = groups_from_pattern(paths, group_pattern)
    keep = subsample(labels, limit, seed)
    paths, labels = paths[keep], labels[keep]
    groups = None if groups is None else groups[keep]
    sites = None if sites is None else sites[keep]

    tr, te = split_indices(labels, groups, joint_strata(labels, sites), seed)
    if len(np.unique(labels[tr])) < len(classes):
        raise ValueError("a class has no patch in the training split; with group ids, every "
                         "class needs patches from several patients or slides")

    adapter.load()
    step = (lambda name: (lambda done, total: progress(name, done, total))) if progress else \
        (lambda name: None)
    z_train, _ = embed_paths(adapter, paths[tr], batch=batch, progress=step("training split"))
    z_test, z_blur = embed_paths(adapter, paths[te], sigmas, batch, step("test split"))

    names = relative(paths, data_dir)
    out = dict(Z_train=z_train, y_train=labels[tr], paths_train=names[tr],
               Z_test=z_test, y_test=labels[te], paths_test=names[te],
               classes=np.array(classes), model=np.array(adapter.label()),
               input_size=np.array(adapter.input_size),
               normalisation=np.array(adapter.mean_std(), dtype=np.float64),
               source=np.array(data_dir),
               split_seed=np.array(seed), tool_version=np.array(__version__))
    for key, values in (("groups", groups), ("sites", sites)):
        if values is not None:
            out[f"{key}_train"], out[f"{key}_test"] = values[tr], values[te]
    for s, z in z_blur.items():
        out[f"{BLUR_KEY}{float(s)}"] = z

    if far_dir:
        far = list_images(far_dir)
        if len(far) > FAR_LIMIT:
            far = np.sort(np.random.default_rng(seed).choice(far, FAR_LIMIT, replace=False))
        out["Z_far"], _ = embed_paths(adapter, far, batch=batch, progress=step("second dataset"))
        out["paths_far"] = relative(far, far_dir)
        out["far_name"] = np.array(far_dir)
    return out


def save_embeddings(path: str, arrays: dict) -> None:
    """Write an embeddings file; it is written in full and then renamed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    partial = path + ".partial.npz"
    np.savez_compressed(partial, **arrays)
    os.replace(partial, path)
