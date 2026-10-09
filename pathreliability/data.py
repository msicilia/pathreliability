"""
data.py
=======
Patch sets on disk and their preprocessing.

A labelled patch set is a directory with one subdirectory per class. Patient or
slide ids, when available, come from a CSV file or from a pattern applied to the
file names. Images are resized on the shorter side with bicubic interpolation and
centre-cropped to the input size of the encoder, scaled to [0, 1] and normalised;
no augmentation is applied. Blur is a Gaussian filter applied to the normalised
input; sigma is expressed in pixels of a 224 x 224 input and scaled for encoders
with a larger input.
"""
from __future__ import annotations

import csv
import os
import re
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import correlate1d

Image.MAX_IMAGE_PIXELS = None
EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
REFERENCE_SIZE = 224        # input size in whose pixels the blur sigma is expressed


def _images(root: str) -> list[str]:
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs
                  if f.lower().endswith(EXTENSIONS))


def list_classes(root: str):
    """(paths, labels, class names) of a directory with one subdirectory per class."""
    if not os.path.isdir(root):
        raise NotADirectoryError(f"not a directory: {root}")
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if len(classes) < 2:
        raise ValueError(f"{root}: expected one subdirectory per class, at least two")
    paths, labels = [], []
    for label, name in enumerate(classes):
        found = _images(os.path.join(root, name))
        if not found:
            raise ValueError(f"{root}: class {name} has no images")
        paths += found
        labels += [label] * len(found)
    return np.array(paths), np.array(labels), classes


def list_images(root: str) -> np.ndarray:
    """Paths of all images under a directory, labelled or not."""
    if not os.path.isdir(root):
        raise NotADirectoryError(f"not a directory: {root}")
    paths = _images(root)
    if not paths:
        raise ValueError(f"{root}: no images")
    return np.array(paths)


def relative(paths, root: str) -> np.ndarray:
    return np.array([os.path.relpath(p, root) for p in paths])


def groups_from_pattern(paths, pattern: str) -> np.ndarray:
    """Group id of each patch: the first group of `pattern` matched in its file name."""
    regex = re.compile(pattern)
    out = []
    for p in paths:
        m = regex.search(os.path.basename(p))
        if m is None or not m.groups():
            raise ValueError(f"group pattern {pattern!r} does not match {os.path.basename(p)!r} "
                             "or has no capturing group")
        out.append(m.group(1))
    return np.array(out)


def groups_from_csv(paths, root: str, csv_path: str):
    """(groups, sites or None) of each patch, read from a CSV file.

    The file has the columns `path` and `group`, and optionally `site`. A path is
    given relative to the dataset directory; a bare file name is accepted when it
    is unique in the dataset.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"path", "group"} <= set(rows[0]):
        raise ValueError(f"{csv_path}: expected the columns path and group")
    has_site = "site" in rows[0]
    table = {os.path.normpath(r["path"]): r for r in rows}
    names = [os.path.basename(p) for p in paths]
    unique_names = {n for n, k in Counter(names).items() if k == 1}
    groups, sites = [], []
    for p, name in zip(relative(paths, root), names):
        row = table.get(os.path.normpath(p))
        if row is None and name in unique_names:
            row = table.get(name)
        if row is None:
            raise ValueError(f"{csv_path}: no row for {p}")
        groups.append(row["group"])
        sites.append(row["site"] if has_site else None)
    return np.array(groups), (np.array(sites) if has_site else None)


def subsample(labels, limit: int, seed: int = 0) -> np.ndarray:
    """Indices of at most `limit` patches, the same number per class where available."""
    labels = np.asarray(labels)
    if not limit or len(labels) <= limit:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    per_class = max(1, limit // len(np.unique(labels)))
    keep = [rng.choice(idx, min(per_class, len(idx)), replace=False)
            for idx in (np.flatnonzero(labels == c) for c in np.unique(labels))]
    return np.sort(np.concatenate(keep))


def load_image(path: str, size: int) -> np.ndarray:
    """One image as uint8 [size, size, 3]: shorter side resized to `size`, centre crop."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if (w, h) != (size, size):
            new_w, new_h = (size, int(size * h / w)) if w <= h else (int(size * w / h), size)
            if (new_w, new_h) != (w, h):
                img = img.resize((new_w, new_h), Image.BICUBIC)
            top, left = int(round((new_h - size) / 2.0)), int(round((new_w - size) / 2.0))
            img = img.crop((left, top, left + size, top + size))
        return np.asarray(img, dtype=np.uint8)


def normalise(images: np.ndarray, mean, std) -> np.ndarray:
    """uint8 [N, S, S, 3] -> float32, scaled to [0, 1] and normalised per channel."""
    x = images.astype(np.float32) / 255.0
    return (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)


def gaussian_blur(batch: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a normalised batch [N, S, S, 3].

    `sigma` is in pixels of a 224 x 224 input and is scaled with the input size.
    The kernel spans max(3, int(6 sigma) | 1) pixels and the borders are reflected.
    """
    sigma = sigma * batch.shape[1] / REFERENCE_SIZE
    width = max(3, int(6 * sigma) | 1)
    offsets = np.linspace(-(width - 1) / 2.0, (width - 1) / 2.0, width)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = (kernel / kernel.sum()).astype(np.float32)
    out = correlate1d(batch, kernel, axis=1, mode="mirror")
    return correlate1d(out, kernel, axis=2, mode="mirror")
