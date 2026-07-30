# PathReliability

*A retraining-free reliability audit and abstention gate for frozen image encoders.*

Encoders are commonly selected on predictive accuracy, which measures how often a model is correct
but is silent on whether its outputs can be trusted when it is not. PathReliability audits a frozen
encoder along the complementary axes that govern safe deployment and returns a structured
reliability report together with a deployable abstention gate. It operates post hoc on cached
embeddings and a linear probe, requiring no retraining and no access to model weights, gradients,
or pretraining data.

The audit characterises four properties that accuracy-based evaluation does not capture:

- **Out-of-distribution (novelty) detection** — whether inputs unlike the training distribution
  are flagged rather than silently classified.
- **Corruption robustness** — whether image degradation such as blur is detected.
- **Calibration** — whether predicted confidence agrees with empirical accuracy.
- **Selective prediction** — whether abstaining on low-confidence inputs improves the accuracy of
  those retained.

---

## Applicability — check before installing

This version audits **frozen feature encoders**: a model that maps a 2-D image to a fixed-length
embedding vector (the representation used under linear probing). The tool applies to your model if:

- it returns a **pooled embedding (feature vector) per image**;
- that embedding can be obtained for a batch of images, through either a standard
  `timm` / `torchvision` identifier or a short adapter (about ten lines, see below);
- a **labelled dataset** is available in one-subfolder-per-class layout for the task the encoder
  serves.

The following are **not supported** in this version:

- classifiers exposing only class probabilities, with no accessible embedding;
- slide-level or multiple-instance (MIL) aggregators;
- segmentation and detection networks;
- any method requiring model weights, gradients, or internal activations beyond the pooled
  embedding.

---

## Usage (Docker)

The tool is distributed as a container image for reproducible, dependency-isolated execution.
Docker is the supported entry point; no local Python installation is required.

```bash
docker build -t pathreliability .        # or: docker pull ghcr.io/msicilia/pathreliability
```

The input dataset is a directory with **one subfolder per class**:

```
my_dataset/
├── class_a/  img001.png  img002.png  …
├── class_b/  …
└── class_c/  …
```

Audit a standard model with no code, mounting the dataset and an output directory:

```bash
docker run --rm \
  -v "$PWD/my_dataset:/data" -v "$PWD/out:/out" \
  pathreliability audit --builtin timm:resnet50 --data /data --out /out
```

The report is written to `out/report.html` on the host. A committed example is provided at
[`examples/sample_report.html`](examples/sample_report.html) for inspection prior to running.

---

## Audit your own model

A model is provided through a small adapter (two methods, about ten lines); copy
[`examples/my_model_adapter.py`](examples/my_model_adapter.py) as a starting point. The adapter is
the entire integration surface:

```python
from pathreliability.adapter import Encoder
import numpy as np

class MyEncoder(Encoder):
    embed_dim = 768
    normalisation = "imagenet"          # or (mean, std) your model expects
    def load(self):
        import timm, torch
        self.torch = torch
        self.model = timm.create_model("hf-hub:me/my-encoder", pretrained=True,
                                       num_classes=0).eval()
    def embed(self, batch):             # batch: float32 [N,224,224,3], normalised
        x = self.torch.from_numpy(batch).permute(0, 3, 1, 2)
        with self.torch.no_grad():
            return self.model(x).cpu().numpy().astype(np.float32)
```

Mount the adapter alongside the dataset and run:

```bash
docker run --rm \
  -v "$PWD/my_dataset:/data" -v "$PWD/out:/out" \
  -v "$PWD/my_model_adapter.py:/work/adapter.py" \
  pathreliability audit --adapter /work/adapter.py:MyEncoder \
                        --data /data --ood-data /data_ood --out /out
```

The container feeds `embed` a normalised batch and expects an embedding array back; **model
weights never leave the container.** If your model has dependencies not present in the image,
extend the `Dockerfile`; serving the model separately over HTTP behind the adapter contract is
planned for a future release.

---

## What you get

In `./out/`:

| File | Intended reader | Contents |
|---|---|---|
| **`report.html`** | human reviewer | verdict banner, per-axis PASS/WARN/FAIL, caveats, metrics, and the gate |
| `report.json` | downstream tooling / records | the same content, schema-versioned |
| `gate.json` | deployment | the two thresholds (`τ_ood`, `τ_conf`) |

**Reading the verdict.** Each axis is PASS / WARN / FAIL. A **FAIL** means a blocking finding —
e.g. `blur_anti_detection` (the model rates blurred images as *more* normal, so it would pass the
degraded inputs a quality gate should reject) or `artifact_rejection_fail` (it can't tell tissue
from background/debris). The report explains each in plain language.

**The gate** is the actionable output: a two-stage decision rule applied ahead of the model at
inference — rejecting unfamiliar inputs (stage 1), then abstaining on low-confidence ones
(stage 2) — at a cost of one distance evaluation and two comparisons per image.

---

## Options

| Flag | Meaning |
|---|---|
| `--data DIR` | ID dataset (one subfolder per class) — **required** |
| `--builtin timm:<name>` \| `torchvision:<name>` | wrap a standard model with no code |
| `--adapter file.py:Class` | your own model |
| `--ood-data DIR` | a different-organ dataset → enables the cross-organ far-OOD axis |
| `--limit N` | cap ID images (stratified) for a fast first run (default 1200; 0 = all) |
| `--out DIR` | output directory (default `./out`) |

---

## Definitions

Several terms have a specific meaning in this tool:

- **Out-of-distribution (OOD).** Defined *relative to the classes the linear probe was trained
  on* — the task's in-distribution — **not** relative to the data the encoder saw during
  pretraining. An input is OOD if it falls outside those probe classes. The tool makes no claim
  about whether the encoder encountered such inputs during pretraining.
- **Far-OOD.** Inputs from a different dataset or organ than the probe's task (cross-organ).
- **Near-OOD.** A held-out class of the same domain (leave-one-class-out); harder than far-OOD.
- **Artefact rejection.** Flagging non-diagnostic material (background, debris, adipose) as OOD
  relative to diagnostic tissue.
- **Corruption robustness.** Whether degraded inputs (Gaussian blur of increasing severity) are
  detected as OOD.
- **Calibration (ECE).** Agreement between predicted confidence and empirical accuracy, summarised
  by the Expected Calibration Error.
- **Selective prediction (AURC).** Abstaining on the least-confident inputs, summarised by the
  Area Under the Risk–Coverage curve (lower is better).
- **Abstention gate.** The deployable two-threshold rule the audit emits: a novelty threshold
  (`τ_ood`) applied to the embedding, then a confidence threshold (`τ_conf`) applied to the probe.
- **Embedding.** The fixed-length pooled feature vector the frozen encoder produces per image.
- **Linear probe.** A logistic-regression classifier fitted on the frozen embeddings; the only
  trained component of the audit.

---

Method and validation: *"Do Frozen Pathology Foundation Models Know When They Don't Know?"*
(thirteen encoders, six public datasets). This tool packages that audit for your own model.

MIT-licensed. Issues and adapters for new model families welcome.
