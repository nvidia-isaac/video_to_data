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

RUN pip install pyrender pyglet trimesh wis3d opencv-python imageio[pyav] av tqdm matplotlib

COPY . /workspace

RUN pip install -e /workspace/v2d_common \
                -e "/workspace/v2d_mv[io,vis,math]" \
                -e /workspace/v2d_mv_postprocess/lib

ENV PYOPENGL_PLATFORM=egl
