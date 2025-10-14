#!/bin/bash
export PYTHONPATH=/algorithm/src:$PYTHONPATH
[[ -z "${TEST}" ]] && { [[ -z "${DEV}" ]] && python3 || python3 -u src/main.py; } || pytest -v
