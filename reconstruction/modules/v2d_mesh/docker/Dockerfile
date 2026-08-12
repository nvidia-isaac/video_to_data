FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps: osmesa for pyrender offscreen rendering, libgl for trimesh
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    libegl1-mesa \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install pytest -e /workspace/v2d_common -e /workspace/v2d_mesh/lib

ENV PYOPENGL_PLATFORM=egl
