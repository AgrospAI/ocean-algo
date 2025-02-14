if [[ -z "${TEST}" ]]; then
    cd src
    /algorithm/.venv/bin/python -u main.py
else
    /algorithm/.venv/bin/python -m pytest tests/
fi