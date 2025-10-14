#!/bin/sh
set -e

if [ -n "$TEST" ]; then
    pytest -v
elif [ -n "$DEV" ]; then
    python3 -u src/main.py
else
    python3
fi
