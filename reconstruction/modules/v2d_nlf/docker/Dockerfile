FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-6 \
    libegl1-mesa \
    libgles2-mesa \
    libgbm1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install git+https://github.com/isarandi/smplfitter.git

RUN pip3 install pyrender trimesh h5py opencv-python torchvision open3d \
    transformations ruamel.yaml kornia einops joblib gdown

RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"

RUN git clone https://github.com/NVlabs/nvdiffrast /tmp/nvdiffrast && \
    cd /tmp/nvdiffrast && \
    pip3 install setuptools wheel ninja && \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip3 install . --no-build-isolation && \
    rm -rf /tmp/nvdiffrast

RUN pip3 install warp-lang

COPY . /workspace

RUN pip install -e "/workspace/v2d_mv[io,math]" \
    -e /workspace/v2d_common \
    -e /workspace/v2d_mesh/lib \
    -e /workspace/v2d_foundation_pose/lib \
    -e /workspace/v2d_nlf/lib
