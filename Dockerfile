# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifests first for layer cache
COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/uv.lock backend/uv.lock

WORKDIR /app/backend
RUN uv sync --frozen || uv sync

WORKDIR /app
# Copy only needed source (respects .dockerignore, avoids 570MB node_modules)
COPY backend ./backend
COPY scripts ./scripts
COPY data ./data
COPY README.md ./README.md

WORKDIR /app/backend
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
