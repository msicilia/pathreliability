# Publishing this tool to GitHub

This folder is a self-contained repository. To put it online:

## 1. Make it its own git repo

This folder currently lives inside a larger research repo. Turn it into a standalone repo:

```bash
cd pathreliability          # this folder
git init
git add .
git commit -m "PathReliability v0.1 — reliability audit + abstention gate for frozen encoders"
```

## 2. Create the GitHub repo and push

With the GitHub CLI (`gh`):

```bash
gh repo create pathreliability --public --source=. --push \
  --description "Retraining-free reliability audit and abstention gate for frozen medical-imaging encoders"
```

Or manually: create an empty repo on github.com, then

```bash
git remote add origin https://github.com/msicilia/pathreliability.git
git branch -M main
git push -u origin main
```

## 3. Build and publish the container image

Publish the image alongside the source so the tool can be run without a local Python install.
Using the GitHub Container Registry (GHCR):

```bash
docker build -t ghcr.io/msicilia/pathreliability:0.1.0 -t ghcr.io/msicilia/pathreliability:latest .
echo "$GITHUB_TOKEN" | docker login ghcr.io -u msicilia --password-stdin
docker push ghcr.io/msicilia/pathreliability:0.1.0
docker push ghcr.io/msicilia/pathreliability:latest
```

Verify the published image runs end-to-end before submission:

```bash
docker run --rm -v "$PWD/my_dataset:/data" -v "$PWD/out:/out" \
  ghcr.io/msicilia/pathreliability:latest audit --builtin timm:resnet50 --data /data --out /out
```

Record the image reference, base image, and dependency versions wherever the tool is registered,
and keep the `Dockerfile` in the repository root so the image is reproducible from source.

## 4. Recommended repo settings

- **Topics** (for discoverability): `computational-pathology`, `foundation-models`,
  `out-of-distribution`, `uncertainty-quantification`, `calibration`, `trustworthy-ai`,
  `medical-imaging`.
- **Description**: as above.
- Keep the **`out/`** directory git-ignored (already in `.gitignore`) so generated reports are
  not committed; the committed example report lives at `examples/sample_report.html`.
- Add a CITATION referencing the method paper once it is published.

## What is (and isn't) in the repo

Included: the package (`pathreliability/`), the adapter template and a sample report
(`examples/`), `pyproject.toml`, `LICENSE` (MIT), `README.md`.

Not included (intentionally): datasets, model weights, the `out/` run outputs, and any
dependency on the research repo — the tool is fully standalone.

## Companion repo

For users who want to try the tool without their own model, point them at the **testkit**
(tiny encoders with known pass/fail verdicts + a synthetic dataset). Link it from the README
once it is published.
