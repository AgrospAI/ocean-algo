#!/bin/bash
set -euo pipefail

unset DOCKER_TLS_VERIFY DOCKER_CERT_PATH DOCKER_HOST

dockerd-entrypoint.sh > /tmp/dockerd.log 2>&1 &
dockerd_pid=$!

echo "Waiting for Docker to be ready..."
until docker info >/dev/null 2>&1; do
    echo "Docker still not ready. Waiting..."
    sleep 1
done
echo "Docker is ready."

docker pull aleixsegura29/harvest_predictor:latest

docker run -d \
    -v /workspace:/workspace \
    -v /predictions/runs/segment:/predictions/runs/segment \
    --name agrospai_algo_validation \
    registry.agrospai.udl.cat/library/agrospai_apple_inference \
    tail -f /dev/null

docker ps -a

echo "Running main.py..."
exec python3 src/main.py

echo "Cleaning up..."
docker stop agrospai_algo_validation || true
docker rm agrospai_algo_validation || true
kill $dockerd_pid || true