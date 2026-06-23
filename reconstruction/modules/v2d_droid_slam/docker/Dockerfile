FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    ninja-build \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    opencv-python \
    Pillow \
    pyyaml \
    tqdm \
    matplotlib \
    einops \
    evo \
    gdown \
    open3d

RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

RUN git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git /opt/droid-slam

RUN cd /opt/droid-slam/thirdparty/lietorch && \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install --no-build-isolation -e .

RUN cd /opt/droid-slam && \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install --no-build-isolation -e .

ENV PYTHONPATH=/opt/droid-slam:/opt/droid-slam/droid_slam:${PYTHONPATH}

COPY . /workspace

RUN pip install -e /workspace/v2d_common -e /workspace/v2d_droid_slam/lib
