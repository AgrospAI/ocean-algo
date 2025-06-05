FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    build-essential \
    python3-dev \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /algorithm

COPY algorithm/requirements.txt /algorithm/requirements.txt
RUN pip install --no-cache-dir -r /algorithm/requirements.txt

RUN pip install ultralytics --no-deps

COPY algorithm/src /algorithm/src
COPY algorithm/tests /algorithm/tests
COPY entrypoint.sh /algorithm/entrypoint.sh

RUN chmod +x /algorithm/src/predictor.sh

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python3", "src/main.py"]
