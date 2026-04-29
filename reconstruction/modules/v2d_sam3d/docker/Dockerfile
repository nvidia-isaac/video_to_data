FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
# libegl1 + libgles2: required by open3d's Filament/EGL OffscreenRenderer
#   used in v2d.sam3d.lib.render_textured_video. Without these, headless
#   GPU rendering fails with "Could not find library libEGL.so.1" + segfault.
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Heavy deps that need special install procedures
RUN pip3 install gsplat
RUN pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip3 install git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b#egg=moge

RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /tmp/sam-3d-objects && \
    pip3 install /tmp/sam-3d-objects --no-deps && \
    rm -rf /tmp/sam-3d-objects

# networkx is required for trimesh to honor GLB scene-graph node transforms.
# Without it, scene.geometry.items() returns raw mesh vertices missing the
# per-node transform — render_debug_image projections land at the wrong place.
RUN pip install loguru timm open3d astor easydict networkx
RUN pip install cumm-cu124 spconv-cu124
RUN pip install lightning
RUN pip install xatlas pyvista pymeshfix igraph imageio
RUN git clone https://github.com/NVlabs/nvdiffrast /tmp/nvdiffrast && \
    pip install setuptools wheel ninja && \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install /tmp/nvdiffrast --no-build-isolation && \
    rm -rf /tmp/nvdiffrast

# Required by sam-3d-objects on A100/H100/H200: inference_pipeline.set_attention_backend()
# unconditionally sets ATTN_BACKEND=flash_attn on those GPUs and the sparse module then
# does `import flash_attn` at import time. Other GPUs (e.g. RTX 6000 Ada) keep ATTN=sdpa
# and never touch this package.
RUN pip install packaging && \
    MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation

COPY . /workspace

RUN pip install -e /workspace/v2d_common -e /workspace/v2d_sam3d/lib

# SAM3D weights are NOT baked at build time — see workflow/README.md §3a in
# the osmo/hoi_reconstruction repo for the docker cp + commit flow that
# layers them in after this base image is built (avoids putting 15GB into
# the build context and dealing with build-context symlink limitations).

ENV HYDRA_FULL_ERROR=1
ENV CONDA_PREFIX=/opt/conda
ENV CUDA_HOME=/usr/local/cuda
ENV LIDRA_SKIP_INIT=1
