#!/bin/bash

IMAGE_DIGEST=$1

set -euo pipefail

unset DOCKER_TLS_VERIFY DOCKER_CERT_PATH DOCKER_HOST

dockerd-entrypoint.sh &

echo "Waiting for Docker to be ready..."
until docker info >/dev/null 2>&1; do
    echo "Docker still not ready. Waiting..."
    sleep 1
done
echo "Docker is ready."

docker pull "$IMAGE_DIGEST"

docker run -d \
    -v /workspace:/workspace \
    -v /predictions/runs/segment:/predictions/runs/segment \
    --name model \
    "$IMAGE_DIGEST" \
    tail -f /dev/null

exec python3 /algorithm/src/algorithm.py