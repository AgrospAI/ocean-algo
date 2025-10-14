FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /algorithm

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"

COPY src /algorithm/src
COPY deeplabv3_apples.pth /algorithm/deeplabv3_apples.pth

CMD ["python", "src/main.py"]
