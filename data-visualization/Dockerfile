# Step 1 : Build the environment with dependencies
FROM python:3.12 AS builder

ENV PYTHONUNBUFFERED=1

WORKDIR /algorithm/

# Install uv
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

# Place executables in the environment at the front of the path
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#using-the-environment
ENV PATH="/algorithm/.venv/bin:$PATH"

# Compile bytecode
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1

# uv Cache
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#caching
ENV UV_LINK_MODE=copy

# Install dependencies
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=algorithm/uv.lock,target=uv.lock \
    --mount=type=bind,source=algorithm/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

FROM python:3.12-slim

WORKDIR /algorithm

# Copy the algorithm code (with the same depth)
COPY --from=builder /algorithm/.venv /algorithm/.venv
COPY algorithm/src /algorithm/src
COPY algorithm/tests /algorithm/tests

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV PATH="/algorithm/.venv/bin:$PATH"
CMD ["python", "-m", "src.algorithm"]

