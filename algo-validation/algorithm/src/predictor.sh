#!/bin/bash
set -euo pipefail

while true; do
  ls -l /workspace
	for img in $(find /workspace -type f -name "*.png"); do
    [ -f "$img" ] || continue
    yolo predict model=yolov8s-seg.pt source="$img" save=True project=/workspace/runs/segment
    rm "$img"
  done
	sleep 2
done
