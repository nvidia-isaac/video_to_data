FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install flask celery[redis] redis opencv-python-headless Jinja2

# Copy common utilities
COPY modules/common /app/modules/common

# Copy orchestrator code
COPY modules/orchestrator /app/modules/orchestrator

ENV PYTHONPATH=/app

CMD ["python", "modules/orchestrator/api.py"]

