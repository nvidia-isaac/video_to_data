FROM nvcr.io/nvidia/tensorrt:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    cmake \
    build-essential \
    libopencv-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libabsl-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install && \
    git clone https://github.com/nvidia-isaac/pyCuSFM /pycusfm && \
    cd /pycusfm && \
    bash setup.bash cuda12 && \
    pip install -e .

COPY v2d_cusfm/__init__.py /workspace/v2d_cusfm/__init__.py
COPY v2d_cusfm/lib /workspace/v2d_cusfm/lib
COPY v2d_cusfm/configs/backpack /pycusfm/pycusfm/configs/backpack
RUN pip install -e /workspace/v2d_cusfm/lib

ENV PYTHONPATH=/workspace:$PYTHONPATH

WORKDIR /workspace
