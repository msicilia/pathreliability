# PathReliability

A reliability audit and abstention gate for frozen image encoders.

Encoders are usually selected on accuracy, which does not show whether a model flags the inputs
it should not classify. PathReliability takes a frozen encoder and a labelled patch set, fits a
linear classifier on the embeddings, and measures how that classifier behaves beyond its
accuracy. It also fits a two-threshold gate that rejects unfamiliar patches and abstains on
uncertain ones. The encoder is only asked to embed: it is not retrained.

The tool runs the analysis of the paper *A Reliability Audit and Abstention Gate for Frozen
Pathology Foundation Models* on your model and your data. It ships no datasets. The
code and results of the paper itself are at
[pathology-fm-reliability](https://github.com/msicilia/pathology-fm-reliability).

## What it measures

- **Detection of inputs the classifier should not receive**: patches from a second dataset, an
  unseen class, classes you name as non-diagnostic, and blurred patches. Four post-hoc scorers
  are reported (Mahalanobis distance, k-nearest-neighbour distance, energy, maximum softmax
  probability).
- **Calibration**: whether the classifier's confidence matches its accuracy.
- **Error ranking**: whether low confidence identifies the errors, so that abstaining helps.
- **Accuracy under blur**: whether blur harms the classifier, which is what makes detecting it
  matter.
- **The gate at one operating point**: what it rejects, what it costs, and how accurate the
  retained predictions are.
- **A tuning check**, when patient or slide ids are given: whether tuning the classifier on
  folds that share a patient or slide makes it over-confident.

The report states measured values, each beside the range the paper obtained for eleven pathology
encoders. The ranges are context, not thresholds, and the report gives no verdict.

## What you need

- A model that maps an image patch to one embedding vector, available either as a `timm` or
  `torchvision` model or through a short adapter (see below).
- A labelled patch set: a directory with one subdirectory per class.
- Optional, and each one enables part of the audit:
  - the patient or slide of each patch (strongly recommended, see below);
  - a second patch set from another organ or source;
  - the names of the classes that are not diagnostic tissue.

Not supported: slide-level and multiple-instance models, models that expose only class
probabilities, and segmentation or detection networks.

## Quick start

With Docker (the image runs models on the CPU):

```bash
docker pull ghcr.io/msicilia/pathreliability
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/patches:/data" -v "$PWD/out:/out" \
  ghcr.io/msicilia/pathreliability audit --builtin timm:resnet50 --data /data --out /out
```

From source, in an environment that has your model's framework:

```bash
pip install .            # add ".[torch]" for the --builtin wrappers
pathreliability audit --builtin timm:resnet50 --data ./patches --out ./out
```

Open `out/report.html`. An example, for a toy encoder of the
[testkit](https://github.com/msicilia/pathreliability-testkit) on synthetic patches, is in
[`examples/sample_report.html`](examples/sample_report.html).

## Inputs

```
patches/
├── class_a/   img001.png  img002.png  ...
├── class_b/   ...
└── class_c/   ...
```

**Patient or slide ids.** Patches of the same patient or slide are alike. Without ids, the test
split and the tuning of the classifier share patients with the training data, and accuracy and
calibration are likely to be optimistic. Give the ids in one of two ways:

```bash
--groups groups.csv            # columns: path, group  (and optionally site)
--group-pattern '^(P\d+)_'     # first group of a regular expression, matched in the file name
```

With ids, whole patients or slides are held out for testing and kept together during tuning.
A `site` column (for example the laboratory) also stratifies the split by site.

**A second dataset.** `--far-data DIR` gives patches from another organ or source. Without it,
the gate is not tested on foreign tissue and the report says so.

**Non-diagnostic classes.** `--artefact-classes BACK DEB` names classes such as background or
debris. They are held out together, and a second gate is fitted without them.

**Ordered grades.** `--ordinal` treats the classes as grades in the order of their names and
adds the quadratic weighted kappa.

## Your own model

A model is given through an adapter: a class with two methods.

```python
import numpy as np
from pathreliability.adapter import Encoder

class MyEncoder(Encoder):
    input_size = 224                    # side of the square input, in pixels
    normalisation = "imagenet"          # or (mean, std) per channel

    def load(self):
        import timm, torch
        self.torch = torch
        self.model = timm.create_model("hf-hub:me/my-encoder", pretrained=True,
                                       num_classes=0).eval()

    def embed(self, batch):             # float32 [N, S, S, 3], resized and normalised
        x = self.torch.from_numpy(batch).permute(0, 3, 1, 2)
        with self.torch.no_grad():
            return self.model(x).cpu().numpy().astype(np.float32)
```

```bash
pathreliability audit --adapter my_model.py:MyEncoder --data ./patches \
    --groups groups.csv --far-data ./other_organ --out ./out
```

Images are resized on the shorter side with bicubic interpolation, centre-cropped to
`input_size`, scaled to [0, 1] and normalised. Check the normalisation: a `--builtin timm:` model
uses the values stored with its weights, which are not always those its authors published.
`--mean`, `--std` and `--input-size` replace the declared values, and the report records the
ones that were used.

In Docker, mount the adapter and, if the model needs packages that the image lacks, extend the
`Dockerfile`.

## Outputs

| File | Contents |
|---|---|
| `report.html` | the audit for a reader, each value beside the paper's range |
| `report.json` | every value, with the schema version and the reference ranges |
| `embeddings.npz` | the embeddings, so the audit can be repeated without the model |
| `gate.npz` | the fitted gate |
| `gate.json` | its two thresholds and the decision rule |
| `gate_without_artefact.npz` | the gate fitted without the non-diagnostic classes, when named |

The classifier metrics are means over five train/test splits of the embeddings, with bootstrap
intervals. Detection, blur and the gate use the one split stored in the embeddings file.

## Applying the gate

```bash
pathreliability apply --gate out/gate.npz --adapter my_model.py:MyEncoder \
    --data ./new_patches --out decisions.csv
```

Each patch receives one of three decisions:

- `reject`: its distance to the training classes exceeds the novelty threshold;
- `abstain`: it passes, but the classifier's confidence is below the confidence threshold;
- `predict`: the classifier's prediction is retained.

The thresholds are set on a validation part of the training split: the novelty threshold rejects
5% of the validation patches, and the confidence threshold retains 80% of those that pass. The
gate must be applied with the same model and preprocessing it was fitted with.

## The two stages

`audit` with a model runs two stages, which can also be run separately:

```bash
pathreliability embed --adapter my_model.py:MyEncoder --data ./patches --out emb.npz
pathreliability audit --embeddings emb.npz --out ./out
```

Embedding is the slow stage and the only one that needs the model. If you already have
embeddings, write them to an `.npz` file with the arrays listed at the top of
[`pathreliability/audit.py`](pathreliability/audit.py) and run `audit --embeddings`.

## Options

| Option | Meaning |
|---|---|
| `--data DIR` | labelled patch set, one subdirectory per class |
| `--builtin timm:<name>` or `torchvision:<name>` | a standard model, without code |
| `--adapter file.py:Class` | your own model |
| `--groups CSV`, `--group-pattern REGEX` | patient or slide id of each patch |
| `--far-data DIR` | second patch set, from another organ or source |
| `--artefact-classes A B` | classes that are not diagnostic tissue |
| `--ordinal` | the classes are ordered grades |
| `--blur 1 2 3` | blur sigmas, in pixels of a 224 x 224 input; `--blur` alone skips blur |
| `--limit N` | embed at most N patches, the same number per class (default: all) |
| `--mean`, `--std`, `--input-size` | replace the preprocessing the model declares |
| `--embeddings FILE`, `--far FILE` | audit stored embeddings |
| `--jobs N` | parallel workers (default: all cores) |
| `--log-file FILE` | also write the log of the run to a file |
| `--out DIR` | output directory (default `./out`) |

## Limits

- The audit describes one model on one patch set. It does not show how the model behaves at
  another hospital or scanner unless you supply such data as the second dataset.
- Blur is the only image degradation that is tested.
- The gate is evaluated at one operating point.
- Every class needs patches from several patients or slides, so that each can appear on both
  sides of a split.
- In the paper, an unseen grade of a grading task was detected poorly by every encoder. A gate
  of this kind should not be expected to catch unfamiliar grades.

## Validation

The functions that compute the audit are those of the paper's
[analysis code](https://github.com/msicilia/pathology-fm-reliability), and three checks in
[`tests/`](tests/) keep them so:

- `tests/test_paper_equivalence.py` runs in CI and compares the audit with the output of the
  paper's code on synthetic embeddings.
- `tests/acceptance/reproduce_paper.py` runs the audit on the paper's stored embeddings and
  compares every value with the paper's result files.
- `tests/acceptance/reproduce_embeddings.py` embeds the paper's patches again from the images
  and compares.

## Licence

MIT.
