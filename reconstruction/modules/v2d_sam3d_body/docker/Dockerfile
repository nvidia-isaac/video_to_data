FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    zip \
    libosmesa6-dev \
    libegl1-mesa-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Heavy Python dependencies
RUN pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill \
    pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
    networkx==3.2.1 roma joblib cython jsonlines loguru optree fvcore pycocotools \
    tensorboard huggingface_hub webdataset rtree

# Pytorch3D (requires CUDA build; no GPU visible during docker build so we
# must specify architectures explicitly)
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"

COPY . /workspace

RUN pip install -e /workspace/v2d_common \
                -e "/workspace/v2d_mv[all]" \
                -e /workspace/v2d_sam3d_body/lib

# sam_3d_body and tools are accessed via PYTHONPATH (legacy package structure)
ENV PYTHONPATH=/workspace/v2d_sam3d_body/lib
ENV PYOPENGL_PLATFORM=egl
ENV TORCH_HOME=/tmp/torch_cache
