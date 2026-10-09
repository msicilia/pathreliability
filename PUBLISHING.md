# Releasing

The repository is https://github.com/msicilia/pathreliability.

## Before a release

Run the tests:

```bash
pip install -c constraints.txt ".[test]"
pytest -q
```

If the analysis code of the method paper has changed, run the acceptance checks against a
checkout of its repository, in an environment that has its dependencies:

```bash
python tests/acceptance/check_sync.py     --paper-repo PATH   # the copied functions are unchanged
python tests/acceptance/make_expected.py  --paper-repo PATH   # expected values of the tests
python tests/acceptance/make_reference.py --paper-repo PATH   # reference ranges of the report
python tests/acceptance/reproduce_paper.py --paper-repo PATH --models lunit conch uni
```

`reproduce_embeddings.py` repeats the comparison from the images, for encoders with open weights.

## Container image

The workflow `.github/workflows/publish.yml` runs the tests, builds the image for linux/amd64
and linux/arm64 and pushes it to `ghcr.io/msicilia/pathreliability`, on every push to `main`
(tag `latest`) and on version tags:

```bash
git tag v0.2.0 && git push origin v0.2.0      # publishes ghcr.io/msicilia/pathreliability:0.2.0
```

The image installs the versions of `constraints.txt`. Update that file when the tool is
validated with newer versions, and keep `version` in `pyproject.toml` and `__version__` in
`pathreliability/__init__.py` equal to the tag.

## What the repository holds

The package, its tests, the adapter template and a sample report. It holds no datasets and no
model weights; `out/` is ignored.
