"""
Template of an adapter: the class through which PathReliability runs your model.

Copy this file, fill in `load` and `embed`, and run:

    pathreliability audit --adapter my_model_adapter.py:MyEncoder --data ./patches --out ./out

The tool passes `embed` a float32 batch [N, S, S, 3], resized to `input_size` and
normalised as declared here, and expects a float32 array [N, embed_dim] back.
"""
import numpy as np

from pathreliability.adapter import Encoder


class MyEncoder(Encoder):
    name = "my-encoder"
    input_size = 224                # side of the square input your model expects, in pixels
    normalisation = "imagenet"      # or (mean, std) per channel, as your model was trained

    def load(self):
        # load the weights once; this example uses timm
        import timm
        import torch
        self.torch = torch
        self.model = timm.create_model("hf-hub:me/my-encoder", pretrained=True,
                                       num_classes=0).eval()

    def embed(self, batch: np.ndarray) -> np.ndarray:
        x = self.torch.from_numpy(batch).permute(0, 3, 1, 2)       # to [N, 3, S, S]
        with self.torch.no_grad():
            return self.model(x).cpu().numpy().astype(np.float32)
