"""
adapter.py
==========
The contract through which a model is given to the tool.

A model is provided in one of two ways:

  1. an adapter, a subclass of `Encoder` in a Python file:  --adapter file.py:MyEncoder
  2. a standard model wrapped by the tool:                  --builtin timm:<name>
                                                            --builtin torchvision:<name>

The tool passes `embed` a float32 batch [N, S, S, 3], resized and normalised as
the adapter declares, and expects a float32 array [N, embed_dim] back. The model
is only asked to embed; it is not retrained and no gradients are computed.
"""
from __future__ import annotations

import importlib.util

import numpy as np

from .data import IMAGENET


class Encoder:
    """Subclass this. `load` and `embed` are required."""
    name: str | None = None
    input_size: int = 224                       # side of the square input, in pixels
    normalisation = "imagenet"                  # or (mean, std), per channel

    def load(self) -> None:
        raise NotImplementedError

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """batch: float32 [N, S, S, 3], resized and normalised -> float32 [N, embed_dim]."""
        raise NotImplementedError

    def mean_std(self):
        return IMAGENET if self.normalisation == "imagenet" else self.normalisation

    def label(self) -> str:
        return self.name or type(self).__name__


def with_preprocessing(adapter: Encoder, mean=None, std=None, input_size=None) -> Encoder:
    """Replace the normalisation or the input size that an adapter declares.

    The replacement is made after `load`, which is where a wrapped standard model
    reads the preprocessing published with its weights.
    """
    if (mean is None) != (std is None):
        raise ValueError("--mean and --std are given together")
    load = adapter.load

    def load_then_replace():
        load()
        if mean is not None:
            adapter.normalisation = (list(mean), list(std))
        if input_size:
            adapter.input_size = int(input_size)

    adapter.load = load_then_replace
    return adapter


def load_adapter(spec: str) -> Encoder:
    """Instantiate the class named by 'path/to/file.py:ClassName'."""
    if ":" not in spec:
        raise ValueError("--adapter must be 'path/to/file.py:ClassName'")
    path, cls = spec.rsplit(":", 1)
    mod_spec = importlib.util.spec_from_file_location("_user_adapter", path)
    if mod_spec is None:
        raise ValueError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    enc = getattr(mod, cls)()
    if not isinstance(enc, Encoder):
        raise TypeError(f"{cls} must subclass pathreliability.adapter.Encoder")
    return enc


class _TorchEncoder(Encoder):
    """A timm or torchvision model with its classifier removed."""

    def __init__(self, kind: str, model: str):
        self.kind, self.model_name = kind, model
        self.name = f"{kind}:{model}"

    def load(self):
        import torch
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        if self.kind == "timm":
            import timm
            self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
            cfg = timm.data.resolve_data_config({}, model=self.model)
            self.normalisation = (list(cfg["mean"]), list(cfg["std"]))
            self.input_size = int(cfg["input_size"][-1])
        elif self.kind == "torchvision":
            import torchvision.models as M
            m = getattr(M, self.model_name)(weights="DEFAULT")
            for attr in ("fc", "classifier", "heads", "head"):      # remove the classifier
                if hasattr(m, attr):
                    setattr(m, attr, torch.nn.Identity())
            self.model = m
        else:
            raise ValueError(f"unknown builtin kind '{self.kind}'")
        self.model = self.model.to(self.device).eval()

    def embed(self, batch):
        x = self.torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        with self.torch.no_grad():
            z = self.model(x)
            if z.ndim > 2:
                z = z.mean(dim=tuple(range(2, z.ndim)))
        return z.float().cpu().numpy().astype(np.float32)


def builtin_adapter(spec: str) -> Encoder:
    if ":" not in spec:
        raise ValueError("--builtin must be 'timm:<name>' or 'torchvision:<name>'")
    kind, name = spec.split(":", 1)
    return _TorchEncoder(kind, name)
