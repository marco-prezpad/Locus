# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base-common

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# dev-cpu — MVP
FROM base-common AS dev-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    git make curl libatomic1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra mvp

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra mvp

ENV PATH="/app/.venv/bin:$PATH"
CMD ["sleep", "infinity"]

# base-cuda — CUDA runtime sobre Ubuntu 24.04
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS base-cuda

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git make && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.12

WORKDIR /app

# dev-gpu — B1+ (Deep Learning)
FROM base-cuda AS dev-gpu

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra all

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra all

ENV PATH="/app/.venv/bin:$PATH"
CMD ["sleep", "infinity"]

# prod — CPU only, minimum dependencies for production
FROM base-common AS prod

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
      --extra data --extra ml --extra backtest

COPY src/ ./src/
COPY configs/ ./configs/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
      --extra data --extra ml --extra backtest

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "locus"]
