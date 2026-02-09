FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install MoGE from source/pip
RUN pip install git+https://github.com/microsoft/MoGE.git#egg=moge

RUN pip install flask requests Pillow opencv-python celery[redis] redis

# Copy common utilities
COPY modules/common /workspace/modules/common

# Copy service code
COPY modules/moge /workspace/modules/moge

ENV PYTHONPATH=/workspace