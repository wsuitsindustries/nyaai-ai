# === AI Service — lightweight embedding + LLM worker ===
# Can run standalone (HTTP server) or be bundled with backend.
# This Dockerfile pre-downloads models and can serve as a sidecar.

FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ONNX embedding model during build
COPY setup.py .
RUN python setup.py

# --- Production image ---
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY setup.py .

ENV PYTHONPATH=/app/src
ENV HF_HOME=/root/.cache/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from ai.embeddings import embed; import asyncio; asyncio.run(embed('healthcheck'))" || exit 1

ENV EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LABEL org.opencontainers.image.title="Nya AI Service"
LABEL org.opencontainers.image.description="Embedding + LLM orchestration layer"
LABEL org.opencontainers.image.version="0.3.0"
