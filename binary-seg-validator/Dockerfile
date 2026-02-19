# ---------------------------------------------------
# STAGE 1: Builder
# ---------------------------------------------------
FROM docker:27-dind AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /algorithm/

RUN apk add --no-cache \
    python3 \
    python3-dev \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    py3-opencv

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV PATH="/algorithm/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

ENV UV_PYTHON=python3

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=algorithm/uv.lock,target=uv.lock \
    --mount=type=bind,source=algorithm/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project


RUN cp -r /usr/lib/python3.12/site-packages/cv2 \
    /algorithm/.venv/lib/python3.12/site-packages/cv2

# ---------------------------------------------------
# STAGE 2: Runner
# ---------------------------------------------------
FROM docker:27-dind

RUN apk add --no-cache \
    python3 \
    bash \
    libstdc++ \
    libgomp \
    py3-opencv

WORKDIR /algorithm


COPY --from=builder /algorithm/.venv /algorithm/.venv

COPY algorithm/src /algorithm/src

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV PATH="/algorithm/.venv/bin:$PATH"

RUN chmod +x /algorithm/src/run.sh

ENTRYPOINT ["python3", "/algorithm/src/setup.py"]
