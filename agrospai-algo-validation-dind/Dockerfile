FROM docker:27-dind

RUN apk add --no-cache \
    python3 py3-pip bash \
    build-base cmake ninja pkgconfig musl-dev libffi-dev linux-headers \
    jpeg-dev libpng-dev libwebp-dev tiff-dev ffmpeg-dev openblas-dev glib-dev \
    py3-opencv opencv-dev

WORKDIR /algorithm

COPY algorithm/requirements.txt /algorithm/requirements.txt

# Need to create a venv because docker:27-dind uses Alpine Linux which uses apk as a package manager

RUN python3 -m venv --system-site-packages /opt/venv \
    && source /opt/venv/bin/activate \
    && pip install --no-cache-dir -r /algorithm/requirements.txt 

ENV PATH="/opt/venv/bin:${PATH}"

COPY algorithm/src /algorithm/src
COPY algorithm/tests /algorithm/tests
COPY entrypoint.sh /algorithm/entrypoint.sh

RUN chmod +x /algorithm/src/run.sh

CMD ["bash", "/algorithm/src/run.sh"]
