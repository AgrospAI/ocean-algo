FROM python:3.12-slim

WORKDIR /algorithm

# Install the dependencies from the requirements.txt file
COPY algorithm/requirements.txt /algorithm/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /algorithm/requirements.txt

# Copy the algorithm code (with the same depth), and the virtual environment from the previous image
COPY algorithm/src /algorithm/src
COPY algorithm/tests /algorithm/tests
COPY entrypoint.sh /algorithm/entrypoint.sh

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Check if running tests
CMD ["/bin/sh", "/algorithm/entrypoint.sh"]
