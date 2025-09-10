FROM python:3.12-slim

WORKDIR /algorithm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py /algorithm/inference.py
COPY deeplabv3_apples.pth /algorithm/deeplabv3_apples.pth

ENV PYTHONUNBUFFERED=1
