# PathReliability — containerised audit tool.
# Build:  docker build -t pathreliability .
# Run:    docker run --rm -v "$PWD/my_dataset:/data" -v "$PWD/out:/out" \
#             pathreliability audit --builtin timm:resnet50 --data /data --out /out
# pulled via Google's read-through mirror of Docker Hub to avoid Docker Hub
# rate-limits / auth.docker.io flakiness on CI runners (equivalent to python:3.11-slim)
FROM mirror.gcr.io/library/python:3.11-slim

# minimal system libs for Pillow / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# core tool (numpy/scikit-learn/scipy/pillow), then CPU-only torch stack for running models.
# amd64 uses PyTorch's CPU wheel index (smaller than the CUDA default); arm64 uses PyPI, which
# ships CPU-only aarch64 wheels. TARGETARCH is provided automatically by buildx.
ARG TARGETARCH
RUN pip install --no-cache-dir . \
    && if [ "$TARGETARCH" = "amd64" ]; then \
         pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
       else \
         pip install --no-cache-dir torch torchvision; \
       fi \
    && pip install --no-cache-dir timm

# audits write here unless --out is mounted elsewhere
WORKDIR /work
ENTRYPOINT ["pathreliability"]
CMD ["--help"]
