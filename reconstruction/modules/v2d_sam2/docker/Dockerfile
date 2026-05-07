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

COPY v2d_common /workspace/v2d_common
COPY v2d_mv /workspace/v2d_mv
COPY v2d_sam2 /workspace/v2d_sam2

RUN pip install -e "/workspace/v2d_common[io]" -e /workspace/v2d_mv -e /workspace/v2d_sam2/lib
