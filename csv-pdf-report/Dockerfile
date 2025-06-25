FROM python:3.11-alpine

# Install dependencies for PDF generation and plotting
RUN apk add --no-cache \
    libgomp \
    libstdc++ \
    build-base \
    # Additional dependencies for matplotlib and reportlab
    freetype \
    freetype-dev \
    libpng \
    libpng-dev \
    jpeg \
    jpeg-dev \
    zlib \
    zlib-dev \
    # Font dependencies for PDF generation
    fontconfig \
    ttf-dejavu

WORKDIR /algorithm

# Copy the algorithm code
COPY algorithm/src /algorithm/src
COPY algorithm/tests /algorithm/tests
COPY algorithm/requirements.txt /algorithm/requirements.txt
COPY entrypoint.sh /algorithm/entrypoint.sh

# Install the dependencies
RUN pip install --no-cache-dir -r /algorithm/requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Check if running dev & tests
CMD ["sh", "/algorithm/entrypoint.sh"]