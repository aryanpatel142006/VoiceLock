# Session server for the remote GPU path (docs/DEPLOY.md). Runs `python -m hearing serve`
# on :8000 with the same /health and /ws/session contract as the laptop.
#
# Build from tracked files only (no .venv, weights or fixtures in the context):
#   git archive --format=tar HEAD | docker build -t hearing-gpu -
# CPU-only rehearsal image (no CUDA wheels):
#   git archive --format=tar HEAD | docker build -t hearing-cpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -
# Run (NVIDIA container toolkit; the GPU driver comes from the host):
#   docker run --gpus all -p 127.0.0.1:8000:8000 -v hearing-cache:/workspace \
#     -e HEARING_PIPELINES='focus_hq=<hq stage spec>' hearing-gpu
FROM python:3.12-slim

# torch wheels from this index bundle their CUDA runtime; only the host driver is needed.
# Set TORCH_INDEX="" to skip torch (fastest smoke build: passthrough stages only).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG EXTRAS=ml
# Extra pip requirements for an experiment, e.g. --build-arg EXTRA_PIP="onnxruntime-gpu clearvoice"
ARG EXTRA_PIP=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/hf-cache \
    HEARING_MODELS_DIR=/workspace/models \
    HEARING_PIPELINES=""

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Heavy, rarely changing layer first.
RUN if [ -n "$TORCH_INDEX" ]; then pip install torch torchaudio --index-url "$TORCH_INDEX"; fi

# Editable install so hearing.server finds web/console relative to the package.
COPY pyproject.toml ./
COPY hearing ./hearing
COPY web ./web
COPY eval ./eval
RUN pip install -e ".[${EXTRAS}]" \
    && if [ -n "$EXTRA_PIP" ]; then pip install $EXTRA_PIP; fi \
    && python -m hearing.device

VOLUME ["/workspace"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["python", "-m", "hearing", "serve", "--host", "0.0.0.0", "--port", "8000"]
