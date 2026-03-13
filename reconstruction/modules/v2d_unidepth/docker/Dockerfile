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

COPY v2d_datatypes/lib /workspace/v2d_datatypes/lib
COPY v2d_unidepth/lib /workspace/v2d_unidepth/lib

RUN pip install -e /workspace/v2d_datatypes/lib/ -e /workspace/v2d_unidepth/lib/
