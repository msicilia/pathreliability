# PathReliability: container image of the audit tool.
# Build:  docker build -t pathreliability .
# Run:    docker run --rm --user "$(id -u):$(id -g)" \
#             -v "$PWD/patches:/data" -v "$PWD/out:/out" \
#             pathreliability audit --builtin timm:resnet50 --data /data --out /out
# The base image is pulled through Google's mirror of Docker Hub (python:3.13-slim).
FROM mirror.gcr.io/library/python:3.13-slim

# libraries that Pillow needs to read JPEG, PNG and TIFF patches
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo libpng16-16 libtiff6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# The tool and, for running models, a CPU build of PyTorch with timm, at the versions of
# constraints.txt. amd64 takes the CPU wheels of the PyTorch index; arm64 takes those of
# PyPI, which are CPU-only. TARGETARCH is set by buildx.
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pip install --no-cache-dir -c constraints.txt \
            --extra-index-url https://download.pytorch.org/whl/cpu ".[torch]"; \
    else \
        pip install --no-cache-dir -c constraints.txt ".[torch]"; \
    fi

# an unprivileged user; model weights downloaded by --builtin are cached under /tmp
RUN useradd --create-home --uid 1000 audit
ENV HF_HOME=/tmp/cache/huggingface TORCH_HOME=/tmp/cache/torch XDG_CACHE_HOME=/tmp/cache
USER audit

WORKDIR /work
ENTRYPOINT ["pathreliability"]
CMD ["--help"]
