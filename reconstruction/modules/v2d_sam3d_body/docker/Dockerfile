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
    && rm -rf /var/lib/apt/lists/*

# Heavy Python dependencies
RUN pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill \
    pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
    networkx==3.2.1 roma joblib cython jsonlines loguru optree fvcore pycocotools \
    tensorboard huggingface_hub webdataset

# Pytorch3D (requires CUDA build)
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"

# Detectron2 (requires --no-build-isolation)
RUN pip install 'detectron2[all] @ git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation

# MoGe
RUN pip install git+https://github.com/microsoft/MoGe.git

COPY . /workspace

RUN pip install -e /workspace/v2d_common \
                -e /workspace/v2d_mv \
                -e /workspace/v2d_sam3d_body/lib

# sam_3d_body and tools are accessed via PYTHONPATH (legacy package structure)
ENV PYTHONPATH=/workspace/v2d_sam3d_body/lib
ENV PYOPENGL_PLATFORM=egl
ENV TORCH_HOME=/tmp/torch_cache
