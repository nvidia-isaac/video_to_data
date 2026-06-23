FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=egl
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg \
    libgl1 libegl1 libglib2.0-0 libxrender1 libsm6 libxext6 libglfw3 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Vendor HaMeR at a pinned commit. We only need its hand-reconstruction code
# (models/, datasets/, utils/) — the demo-side detection chain (vitdet,
# vitpose) is intentionally NOT installed.
ARG HAMER_REPO=https://github.com/geopavlakos/hamer.git
ARG HAMER_COMMIT=3a01849f4148352e9260b69bf28b65d1671a4905
RUN git clone --recursive ${HAMER_REPO} /opt/hamer && \
    cd /opt/hamer && git checkout ${HAMER_COMMIT} && \
    git submodule update --init --recursive

RUN pip install --no-cache-dir \
    "numpy<2" \
    Pillow opencv-python tqdm \
    pyrender trimesh \
    smplx \
    yacs hydra-core omegaconf \
    pytorch-lightning \
    timm einops \
    webdataset braceexpand dill av joblib loguru \
    scikit-image

# manotorch and chumpy aren't on PyPI for Python 3.11 — pull from git.
RUN pip install --no-cache-dir \
    git+https://github.com/lixiny/manotorch.git \
    git+https://github.com/mattloper/chumpy.git

# Install HaMeR (skip its detector deps — we drive crops from SAM2 masks).
RUN pip install --no-deps -e /opt/hamer

ENV PYTHONPATH=/opt/hamer

COPY . /workspace
RUN pip install -e /workspace/v2d_common \
    -e /workspace/v2d_hamer/lib

# Sanity-check imports at build time so a broken install fails the build.
RUN python -c "from hamer.models import load_hamer; from hamer.datasets.utils import generate_image_patch_cv2; print('hamer imports OK')"
