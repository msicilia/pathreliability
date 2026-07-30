"""
The model plug: a ~10-line contract a user implements to expose their own model.

Two ways to provide a model:
  1. Write an adapter (subclass Encoder) and pass  --adapter path/to/file.py:MyEncoder
  2. Use a built-in wrapper for a standard model:   --builtin timm:<name>
                                                     --builtin torchvision:<name>

The tool feeds `embed()` a normalised float32 batch [N,224,224,3] and expects a
float32 embedding array [N, embed_dim] back. No weights or gradients ever leave
your process.
"""
from __future__ import annotations

import importlib.util
import numpy as np

_IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class Encoder:
    """Subclass this. Only `load` and `embed` are required."""
    embed_dim: int = 0
    normalisation = "imagenet"          # or (mean, std)
    magnification = "20x"

    def load(self) -> None:
        raise NotImplementedError

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """batch: float32 [N,224,224,3] (already resized + normalised) -> [N,embed_dim]."""
        raise NotImplementedError

    def capabilities(self) -> dict:
        mean, std = _IMAGENET if self.normalisation == "imagenet" else self.normalisation
        return {"schema_version": "1.0", "tier": "A", "embed_dim": self.embed_dim,
                "input": {"size": [224, 224], "layout": "HWC", "dtype": "uint8",
                          "colour": "RGB", "magnification": self.magnification},
                "normalisation": {"mean": list(mean), "std": list(std)}}


# --------------------------------------------------------------------------- #
# loading a user adapter from "file.py:ClassName"
# --------------------------------------------------------------------------- #
def load_adapter(spec: str) -> Encoder:
    if ":" not in spec:
        raise ValueError("--adapter must be 'path/to/file.py:ClassName'")
    path, cls = spec.rsplit(":", 1)
    mod_spec = importlib.util.spec_from_file_location("_user_adapter", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    enc = getattr(mod, cls)()
    if not isinstance(enc, Encoder):
        raise TypeError(f"{cls} must subclass pathreliability.adapter.Encoder")
    return enc


# --------------------------------------------------------------------------- #
# built-in wrappers so a user can test without writing any code
# --------------------------------------------------------------------------- #
class _TorchEncoder(Encoder):
    def __init__(self, kind, name):
        self.kind, self.name = kind, name

    def load(self):
        import torch
        self.torch = torch
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        if self.kind == "timm":
            import timm
            self.model = timm.create_model(self.name, pretrained=True, num_classes=0)
        elif self.kind == "torchvision":
            import torchvision.models as M
            m = getattr(M, self.name)(weights="DEFAULT")
            # strip the classifier -> pooled features
            for attr in ("fc", "classifier", "heads", "head"):
                if hasattr(m, attr):
                    setattr(m, attr, torch.nn.Identity())
            self.model = m
        else:
            raise ValueError(f"unknown builtin kind '{self.kind}'")
        self.model = self.model.to(self.device).eval()
        # probe embed_dim with a dummy forward
        with torch.no_grad():
            z = self._fwd(np.zeros((1, 224, 224, 3), np.float32))
        self.embed_dim = int(z.shape[1])

    def _fwd(self, batch):
        x = self.torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        with self.torch.no_grad():
            z = self.model(x)
            if z.ndim > 2:
                z = z.mean(dim=tuple(range(2, z.ndim)))
        return z.float().cpu().numpy()

    def embed(self, batch):
        return self._fwd(batch).astype(np.float32)


def builtin_adapter(spec: str) -> Encoder:
    if ":" not in spec:
        raise ValueError("--builtin must be 'timm:<name>' or 'torchvision:<name>'")
    kind, name = spec.split(":", 1)
    return _TorchEncoder(kind, name)
