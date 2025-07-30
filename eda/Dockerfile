# --- Running the algorithm ---
FROM python:3.11-slim

# Dependencies
ENV LLVM_CONFIG=/usr/bin/llvm-config
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libffi-dev \
    llvm \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


# Refresh the argument
ARG WORKDIR

WORKDIR /algorithm

# Install the dependencies from the requirements.txt file
COPY algorithm/requirements.txt /algorithm/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /algorithm/requirements.txt

# Copy the algorithm code (with the same depth), and the virtual environment from the previous image
COPY algorithm/src /algorithm/src
COPY algorithm/tests /algorithm/tests
COPY entrypoint.sh /algorithm/entrypoint.sh

# Create a new group `algorithm` with Group ID `1000`.
# Create a new user `algorithm`, sets home directory to `/algorithm`, User ID `1000`, in
# the group `algorithm`. The `-DH` option results in a system account, own the directory
# RUN groupadd --gid 1000 algorithm && \
#     useradd -u 1000 -g algorithm -d /algorithm -s /bin/bash -m algorithm && \
#     chown -R algorithm:algorithm /algorithm

# # Change the user for subsequent commands in Dockerfile to the user with ID `1000`.
# USER 1000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Check if running tests 
CMD ["/bin/sh", "/algorithm/entrypoint.sh"]