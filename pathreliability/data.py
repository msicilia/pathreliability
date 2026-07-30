"""
Dataset loading and preprocessing — an ImageFolder-style layout (one subfolder
per class), plus resize, ImageNet-style normalisation, and Gaussian blur for the
corruption axis. PIL + numpy + scipy only (no torch in the tool core).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# tissue classes commonly used as non-diagnostic "artefact" novelty
ARTIFACT_NAMES = {"BACK", "BACKGROUND", "DEB", "DEBRIS", "ADI", "ADIPOSE"}


def list_imagefolder(root: str):
    """Return (paths, labels, class_names) for a folder of per-class subfolders."""
    if not os.path.isdir(root):
        raise NotADirectoryError(f"--data path is not a directory: {root}")
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    if not classes:
        raise ValueError(f"No class subfolders found under {root} "
                         "(expected one subfolder per class).")
    cmap = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        for r, _, fs in os.walk(os.path.join(root, c)):
            for f in fs:
                if f.lower().endswith(_EXT):
                    paths.append(os.path.join(r, f))
                    labels.append(cmap[c])
    return np.array(paths), np.array(labels), classes


def subsample(paths, labels, limit, seed=0):
    """Stratified cap to at most `limit` images total."""
    if not limit or len(paths) <= limit:
        return paths, labels
    rng = np.random.default_rng(seed)
    per = max(1, limit // len(np.unique(labels)))
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep.extend(rng.choice(idx, min(per, len(idx)), replace=False))
    keep = np.array(keep)
    return paths[keep], labels[keep]


def _load_rgb_224(path):
    try:
        return np.asarray(Image.open(path).convert("RGB").resize((224, 224)),
                          dtype=np.uint8)
    except Exception:
        return np.zeros((224, 224, 3), dtype=np.uint8)


def _blur(img_u8, sigma):
    from scipy.ndimage import gaussian_filter
    out = np.empty_like(img_u8)
    for c in range(3):
        out[..., c] = gaussian_filter(img_u8[..., c], sigma=sigma)
    return out


def load_batch(paths, mean=IMAGENET_MEAN, std=IMAGENET_STD, blur_sigma=None):
    """Load paths -> normalised float32 array [N,224,224,3] the adapter consumes."""
    imgs = np.stack([_load_rgb_224(p) for p in paths])          # uint8 [N,224,224,3]
    if blur_sigma:
        imgs = np.stack([_blur(im, blur_sigma) for im in imgs])
    x = imgs.astype(np.float32) / 255.0
    return (x - mean) / std
