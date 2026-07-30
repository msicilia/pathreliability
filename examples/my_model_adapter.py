"""
Template: wrap YOUR trained model so PathReliability can audit it.

Copy this file, fill in `load()` and `embed()`, then run:

    pathreliability audit --adapter my_model_adapter.py:MyEncoder --data ./my_dataset

The tool hands `embed()` a normalised float32 batch of shape [N, 224, 224, 3]
and expects a float32 embedding array [N, embed_dim] back. Your weights and
gradients never leave this process.
"""
import numpy as np
from pathreliability.adapter import Encoder


class MyEncoder(Encoder):
    embed_dim = 768                 # <-- your model's pooled embedding size
    normalisation = "imagenet"      # or a tuple (mean, std) your model was trained with
    magnification = "20x"           # what your patches are; declared, not assumed

    def load(self):
        # Load your trained weights ONCE. Example (timm):
        import timm, torch
        self.torch = torch
        self.model = timm.create_model("hf-hub:me/my-encoder",
                                       pretrained=True, num_classes=0).eval()

    def embed(self, batch: np.ndarray) -> np.ndarray:
        # batch: float32 [N,224,224,3], already resized + normalised by the tool.
        x = self.torch.from_numpy(batch).permute(0, 3, 1, 2)   # -> NCHW
        with self.torch.no_grad():
            return self.model(x).cpu().numpy().astype(np.float32)
