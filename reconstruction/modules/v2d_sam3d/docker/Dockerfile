FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=12.0
ENV CUMM_CUDA_ARCH_LIST=12.0
WORKDIR /app

# The Blackwell workstation GPUs used by this project are compute capability
# 12.0. Keep the source-built CUDA extensions native to that architecture;
# this also prevents headless Docker builds from trying to auto-detect a GPU.
ARG MAX_BUILD_JOBS=4
ARG PYTORCH3D_COMMIT=33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba
ARG GSPLAT_COMMIT=2323de5905d5e90e035f792fe65bad0fedd413e7
ARG CUMM_COMMIT=d971d02a0b29b7e6da46186564f37bde77b99dcc
ARG SPCONV_COMMIT=263d6b47425ef843c82f997b12d8b714013d216c
ARG NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae
ARG MOGE_COMMIT=a8c37341bc0325ca99b9d57981cc3bb2bd3e255b
ARG SAM3D_COMMIT=f91db411c50efee93d8db7aeb323885650f6f722

# libegl1 + libgles2 support pyrender's headless EGL path. The NVIDIA EGL
# implementation itself is injected by nvidia-container-toolkit at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    python3-dev \
    python3-pip \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "setuptools<81" \
    wheel \
    ninja \
    packaging \
    "pccm==0.4.16" \
    "ccimport==0.4.4" \
    "pybind11==2.13.6" \
    "fire==0.7.1"

# NVIDIA publishes a Kaolin wheel specifically for torch 2.7.1 + CUDA 12.8.
RUN python -m pip install --no-cache-dir \
    "kaolin==0.18.0" \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu128.html

# PyTorch3D 0.7.9 is the newest tagged release. Build it against this image's
# torch/CUDA rather than mixing in a wheel built for an older ABI.
RUN FORCE_CUDA=1 MAX_JOBS=${MAX_BUILD_JOBS} \
    python -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_COMMIT}"

# SAM3D currently depends on the gsplat 1.5.3 API at this exact upstream
# commit. Installing from source bakes sm_120 kernels into the image.
RUN MAX_JOBS=${MAX_BUILD_JOBS} \
    python -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_COMMIT}"

# spconv 2.3.8 predates CUDA 12.8 wheels. Current cumm adds explicit sm_120
# support, so build both upstream sources ahead-of-time for CUDA 12.8. spconv's
# stale distribution metadata caps cumm below 0.8; --no-deps is intentional
# here because the source-compatible cumm 0.8.2 wheel is installed first.
RUN git clone --filter=blob:none https://github.com/FindDefinition/cumm.git /tmp/cumm && \
    git -C /tmp/cumm checkout ${CUMM_COMMIT} && \
    cd /tmp/cumm && \
    CUMM_CUDA_VERSION=12.8 CUMM_DISABLE_JIT=1 MAX_JOBS=${MAX_BUILD_JOBS} \
        python setup.py bdist_wheel && \
    python -m pip install --no-cache-dir --no-deps /tmp/cumm/dist/*.whl && \
    git clone --filter=blob:none https://github.com/traveller59/spconv.git /tmp/spconv && \
    git -C /tmp/spconv checkout ${SPCONV_COMMIT} && \
    cd /tmp/spconv && \
    CUMM_CUDA_VERSION=12.8 SPCONV_DISABLE_JIT=1 MAX_JOBS=${MAX_BUILD_JOBS} \
        python setup.py bdist_wheel && \
    python -m pip install --no-cache-dir --no-deps /tmp/spconv/dist/*.whl && \
    rm -rf /tmp/cumm /tmp/spconv

# nvdiffrast 0.4.0 includes the Blackwell rasterizer fixes and an ahead-of-time
# PyTorch extension build. Pin the tagged source and compile only sm_120.
RUN MAX_JOBS=${MAX_BUILD_JOBS} \
    python -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git@${NVDIFFRAST_COMMIT}"

RUN python -m pip install --no-cache-dir \
    "git+https://github.com/microsoft/MoGe.git@${MOGE_COMMIT}#egg=moge" \
    loguru \
    timm \
    open3d \
    astor \
    easydict \
    networkx \
    lightning \
    xatlas \
    pyvista \
    pymeshfix \
    igraph \
    imageio \
    "pyrender==0.1.45" \
    "pyglet==2.1.15"

# Blackwell is not in SAM3D's A100/H100/H200 FlashAttention allow-list; its
# inference pipeline uses torch SDPA here, so no flash-attn extension is needed.
RUN python -m pip install --no-cache-dir --no-deps \
    "git+https://github.com/facebookresearch/sam-3d-objects.git@${SAM3D_COMMIT}"

ENV PYOPENGL_PLATFORM=egl
RUN python -c "import networkx, pyrender; print('sam3d_egl_imports=pass')"

# Keep the build context narrow: these are the only local packages needed by
# this image. World-readable sources also work under non-root runtime users.
COPY v2d_common /workspace/v2d_common
COPY v2d_sam3d /workspace/v2d_sam3d
RUN chmod -R a+rX /workspace && \
    python -m pip install --no-cache-dir \
        -e /workspace/v2d_common \
        -e /workspace/v2d_sam3d/lib

# SAM3D weights are mounted at runtime; they are deliberately not baked in.
ENV HYDRA_FULL_ERROR=1
ENV CONDA_PREFIX=/opt/conda
ENV LIDRA_SKIP_INIT=1
