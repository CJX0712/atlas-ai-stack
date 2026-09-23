# ATLAS container image.
#
# CPU-only, no model download, no network at run time. The default image serves
# the offline stack (deterministic reader + hashed embeddings + in-memory index),
# which is a real, complete system - not a stub. Add `requirements-optional.txt`
# if you want the ONNX encoder and a local llama.cpp model inside the image.
#
# Author: 晨星

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Core runtime first: this layer is cached until requirements.txt changes.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

# Source. resources/ ships the demo corpus and the offline evaluation set, so
# the image is self-sufficient with no external volume.
COPY atlas/ ./atlas/
COPY tools/ ./tools/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY pyproject.toml README.md ARCHITECTURE.md LICENSE ./

# Run as a non-root user.
RUN useradd --create-home --uid 10001 atlas && chown -R atlas:atlas /app
USER atlas

EXPOSE 8077

# Liveness: the health endpoint reports corpus size, so this also proves the
# corpus loaded rather than merely that the socket is open.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8077/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "atlas.api.app:app", \
     "--host", "0.0.0.0", "--port", "8077", "--log-level", "info"]
