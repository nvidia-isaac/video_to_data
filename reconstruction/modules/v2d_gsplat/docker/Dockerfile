FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Redirect the JIT extension cache to a fixed, world-readable location.
# This must be set before gsplat is installed so the build step and all
# subsequent container runs (as any user) share the same compiled .so.
ENV TORCH_EXTENSIONS_DIR=/opt/torch_extensions
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# gsplat: pip-installable Gaussian splatting rasterizer
RUN pip install gsplat

# Pre-compile gsplat's CUDA JIT extension at image-build time so containers
# start instantly.  nvcc (from the -devel base) does the compilation; no GPU
# device is needed.  The resulting .so lands in TORCH_EXTENSIONS_DIR and is
# reused by every container run (any uid) without recompilation.
RUN mkdir -p /opt/torch_extensions && \
    MAX_JOBS=8 python3 -c "from gsplat.cuda._backend import _C; print('gsplat CUDA extension pre-compiled')" && \
    chmod -R a+rX /opt/torch_extensions

# SMPL body model library
RUN pip install smplx

# Mesh I/O and processing
RUN pip install trimesh

# SSIM loss
RUN pip install pytorch-msssim

# Image / video I/O
RUN pip install Pillow opencv-python scipy tqdm h5py chumpy && \
    python3 -c "\
p='/opt/conda/lib/python3.11/site-packages/chumpy/__init__.py'; \
t=open(p).read(); \
t=t.replace('from numpy import bool, int, float, complex, object, unicode, str, nan, inf', \
            'from numpy import nan, inf; unicode=str'); \
open(p,'w').write(t)" && \
    grep -rl 'inspect\.getargspec' /opt/conda/lib/python3.11/site-packages/chumpy/ | \
    xargs sed -i 's/inspect\.getargspec/inspect.getfullargspec/g'

COPY . /workspace

RUN pip install -e /workspace/v2d_common \
    -e /workspace/v2d_gsplat/lib
