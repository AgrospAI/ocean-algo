FROM ghcr.io/osgeo/gdal:ubuntu-full-3.12.2 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

ENV PATH="/algorithm/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=python3

WORKDIR /algorithm
COPY algorithm/pyproject.toml ./
RUN uv venv && \
    uv pip install "setuptools" "numpy==2.4.2" && \
    uv sync --no-install-project --no-build-isolation


FROM ghcr.io/osgeo/gdal:ubuntu-full-3.12.2

RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /algorithm/.venv /algorithm/.venv
COPY algorithm/src/ /algorithm/src/

WORKDIR /algorithm
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/algorithm/.venv/bin:$PATH"

ENTRYPOINT ["python3", "/algorithm/src/algorithm.py"]
