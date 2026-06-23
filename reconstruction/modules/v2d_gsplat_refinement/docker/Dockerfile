FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    libglm-dev \
    && rm -rf /var/lib/apt/lists/*

# Build native CUDA extensions for the same arches as the rest of the repo.
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
ENV FORCE_CUDA=1

RUN pip install --no-cache-dir \
    "numpy<2" \
    Pillow opencv-python tqdm \
    trimesh \
    imageio imageio-ffmpeg \
    scipy \
    roma

# manotorch (and chumpy, its transitive runtime dep) aren't on PyPI for 3.11.
RUN pip install --no-cache-dir \
    git+https://github.com/lixiny/manotorch.git \
    git+https://github.com/mattloper/chumpy.git

# gsplat builds CUDA kernels at install time; pinning a known-good tag keeps
# Dockerfile reproducibility independent of upstream HEAD.
RUN pip install --no-cache-dir --no-binary gsplat gsplat==1.4.0

COPY . /workspace
RUN pip install -e /workspace/v2d_common \
    -e /workspace/v2d_gsplat_refinement/lib

# Sanity-check imports at build time.
RUN python -c "import gsplat; from gsplat.rendering import rasterization; \
               from manotorch.manolayer import ManoLayer; \
               from v2d.gsplat_refinement.lib import refine; \
               print('gsplat_refinement imports OK')"
