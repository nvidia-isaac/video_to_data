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

COPY v2d_datatypes /workspace/v2d_datatypes
COPY v2d_moge/lib /workspace/v2d_moge/lib

RUN pip install -e /workspace/v2d_datatypes -e /workspace/v2d_moge/lib
