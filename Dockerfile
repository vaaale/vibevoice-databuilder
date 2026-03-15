FROM nvcr.io/nvidia/pytorch:25.09-py3
#FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_LINK_MODE=copy

# System dependencies (libsox-dev for resemble-enhance audio processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsox-dev wget libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# ---------- Layer 1: dependency install (cached unless pyproject.toml changes) ----------

# Copy dependency metadata first (layer caching)
COPY pyproject.toml uv.lock README.md ./
COPY resemble-enhance/pyproject.toml resemble-enhance/README.md resemble-enhance/

COPY databuilder/ databuilder/
COPY resemble-enhance/ resemble-enhance/
COPY resemble_ai/ resemble_ai/

RUN uv sync
# ---------- Entrypoint ----------

COPY run.sh /usr/local/bin/run.sh
RUN chmod +x /usr/local/bin/run.sh

ENTRYPOINT ["/usr/local/bin/run.sh"]
