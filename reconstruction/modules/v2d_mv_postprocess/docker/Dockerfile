FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    libosmesa6-dev \
    libegl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

COPY v2d_common /workspace/v2d_common
RUN pip install --no-cache-dir -e /workspace/v2d_common

COPY v2d_mv /workspace/v2d_mv
RUN pip install --no-cache-dir -e "/workspace/v2d_mv[io,vis,math]"

COPY v2d_mv_postprocess/lib /workspace/v2d_mv_postprocess/lib
RUN pip install --no-cache-dir -e /workspace/v2d_mv_postprocess/lib

ENV PYOPENGL_PLATFORM=egl
