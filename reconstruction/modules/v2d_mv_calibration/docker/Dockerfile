FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install ceres-solver for pyceres
RUN pip install pyceres

COPY . /workspace

RUN pip install -e /workspace/v2d_common -e /workspace/v2d_mv -e /workspace/v2d_mv_calibration/lib
