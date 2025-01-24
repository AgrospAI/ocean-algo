# --- Install dependencies (with poetry) ---
FROM python:3.10-alpine AS builder

WORKDIR /algorithm

COPY algorithm/pyproject.toml algorithm/poetry.lock algorithm/LICENSE algorithm/README.md /algorithm/

# Install needed dependecies for poetry
RUN apk add build-base libffi-dev

# Install poetry and dependencies
RUN pip install poetry
RUN poetry config virtualenvs.in-project true
RUN poetry install --no-ansi

# --- Running the algorithm ---
FROM python:3.10-alpine

# Refresh the argument 
ARG WORKDIR

WORKDIR /algorithm

# Copy the algorithm code (with the same depth), and the virtual environment from the previous image
COPY algorithm/src /algorithm/src
COPY --from=builder /algorithm/.venv /algorithm/.venv

# Create a new group `algorithm` with Group ID `1000`.
RUN addgroup --gid 1000 algorithm
# Create a new user `algorithm`, sets home directory to `/algorithm`, User ID `1000`, in
# the group `algorithm`. The `-DH` option results in a system account.
RUN adduser algorithm -h /algorithm -u 1000 -G algorithm -DH
# Change the user for subsequent commands in Dockerfile to the user with ID `1000`.
USER 1000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["/algorithm/.venv/bin/python", "-u", "src/main.py"]