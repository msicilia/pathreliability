"""Run a user's Encoder over image paths to produce frozen embeddings."""
from __future__ import annotations

import numpy as np

from .data import load_batch, IMAGENET_MEAN, IMAGENET_STD


def _norm(adapter):
    if adapter.normalisation == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    mean, std = adapter.normalisation
    return np.asarray(mean, np.float32), np.asarray(std, np.float32)


def embed_paths(adapter, paths, blur_sigma=None, batch=64, progress=None):
    """Embed every path; returns float32 [N, embed_dim]."""
    mean, std = _norm(adapter)
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        x = load_batch(chunk, mean=mean, std=std, blur_sigma=blur_sigma)
        out.append(np.asarray(adapter.embed(x), dtype=np.float32))
        if progress:
            progress(min(i + batch, len(paths)), len(paths))
    return np.concatenate(out) if out else np.zeros((0, adapter.embed_dim), np.float32)
