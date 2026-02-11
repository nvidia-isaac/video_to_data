FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install gsplat
RUN pip3 install gsplat

# Install kaolin
RUN pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html

# Install Pytorch3D
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install MoGE
RUN pip3 install git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b#egg=moge

# Install other requirements
RUN pip3 install requests tqdm pillow opencv-python omegaconf hydra-core seaborn utils3d trimesh

# Install sam-3d-objects
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/third_party/sam-3d-objects && \
    cd /workspace/third_party/sam-3d-objects && \
    pip3 install -e . --no-deps

RUN pip install loguru timm open3d astor easydict
RUN pip install cumm-cu124 spconv-cu124
RUN pip install lightning
RUN pip install xatlas pyvista pymeshfix igraph imageio
RUN git clone https://github.com/NVlabs/nvdiffrast /workspace/third_party/nvdiffrast && \
    cd /workspace/third_party/nvdiffrast && \
    pip install setuptools wheel ninja && \ 
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install . --no-build-isolation


# Copy common utilities
COPY modules/common /workspace/modules/common

# Copy service code
COPY modules/sam3d /workspace/modules/sam3d
ENV HYDRA_FULL_ERROR=1
ENV CONDA_PREFIX=/opt/conda
ENV CUDA_HOME=/usr/local/cuda
ENV LIDRA_SKIP_INIT=1

ENV PYTHONPATH=/workspace
