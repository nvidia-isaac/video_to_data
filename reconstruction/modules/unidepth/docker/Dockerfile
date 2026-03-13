FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install UniDepth
RUN pip install git+https://github.com/lpiccinelli-eth/UniDepth.git

RUN pip install requests Pillow opencv-python

# Copy common utilities
COPY modules/common /workspace/modules/common

# Copy service code
COPY modules/unidepth /workspace/modules/unidepth

ENV PYTHONPATH=/workspace
