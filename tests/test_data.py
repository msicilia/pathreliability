"""Preprocessing, group ids and splitting."""
import numpy as np
import pytest
from PIL import Image

from pathreliability import data
from pathreliability.splits import split_indices


def _write(path, size, value=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(str(path)))
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def test_load_image_resizes_shorter_side_and_crops(tmp_path):
    _write(tmp_path / "wide.png", (300, 200))
    _write(tmp_path / "tall.png", (150, 400))
    _write(tmp_path / "exact.png", (224, 224))
    for name in ("wide.png", "tall.png", "exact.png"):
        assert data.load_image(str(tmp_path / name), 224).shape == (224, 224, 3)
    exact = np.asarray(Image.open(tmp_path / "exact.png").convert("RGB"))
    assert np.array_equal(data.load_image(str(tmp_path / "exact.png"), 224), exact)


def test_preprocessing_matches_torchvision(tmp_path):
    torch = pytest.importorskip("torch")
    transforms = pytest.importorskip("torchvision.transforms")
    TF = transforms.functional
    mean, std = data.IMAGENET
    _write(tmp_path / "a.png", (300, 260))
    for size in (224, 448):
        reference = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size), transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        with Image.open(tmp_path / "a.png") as img:
            x_ref = reference(img.convert("RGB"))[None]
        x = data.normalise(data.load_image(str(tmp_path / "a.png"), size)[None], mean, std)
        assert np.allclose(x, x_ref.permute(0, 2, 3, 1).numpy(), atol=1e-6)
        for sigma in (1.0, 2.0, 3.0):
            scaled = sigma * size / 224
            kernel = max(3, int(6 * scaled) | 1)
            blurred = TF.gaussian_blur(x_ref, kernel_size=kernel, sigma=scaled)
            assert np.allclose(data.gaussian_blur(x, sigma),
                               blurred.permute(0, 2, 3, 1).numpy(), atol=1e-5)


def test_groups_from_pattern_and_csv(tmp_path):
    for cls in ("a", "b"):
        for patient in ("p1", "p2"):
            for k in range(2):
                _write(tmp_path / "set" / cls / f"{patient}_{cls}_{k}.png", (32, 32))
    paths, labels, classes = data.list_classes(str(tmp_path / "set"))
    assert classes == ["a", "b"] and len(paths) == 8
    by_pattern = data.groups_from_pattern(paths, r"^(p\d+)_")
    assert sorted(set(by_pattern)) == ["p1", "p2"]

    csv_path = tmp_path / "groups.csv"
    rel = data.relative(paths, str(tmp_path / "set"))
    csv_path.write_text("path,group,site\n" + "\n".join(
        f"{p},{g},lab{i % 2}" for i, (p, g) in enumerate(zip(rel, by_pattern))) + "\n")
    groups, sites = data.groups_from_csv(paths, str(tmp_path / "set"), str(csv_path))
    assert np.array_equal(groups, by_pattern) and set(sites) == {"lab0", "lab1"}
    with pytest.raises(ValueError):
        data.groups_from_pattern(paths, r"^(x\d+)_")


def test_split_keeps_groups_on_one_side():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 200)
    groups = np.array([f"{label}_{g}" for label, g in zip(y, rng.integers(0, 15, len(y)))])
    train, test = split_indices(y, groups, y, seed=3)
    assert not set(groups[train]) & set(groups[test])
    assert len(train) + len(test) == len(y)


def test_subsample_is_balanced():
    labels = np.repeat([0, 1, 2], [500, 300, 40])
    keep = data.subsample(labels, 300, seed=0)
    assert np.array_equal(np.bincount(labels[keep]), [100, 100, 40])
    assert len(data.subsample(labels, 0)) == len(labels)
