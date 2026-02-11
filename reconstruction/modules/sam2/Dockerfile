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

# Install SAM2
RUN pip install git+https://github.com/facebookresearch/sam2.git

# Install decord for low-mem predictor
RUN pip install decord

RUN pip install huggingface-hub requests Pillow opencv-python

# Copy common utilities
COPY modules/common /workspace/modules/common

# Copy service code
COPY modules/sam2 /workspace/modules/sam2

ENV PYTHONPATH=/workspace
