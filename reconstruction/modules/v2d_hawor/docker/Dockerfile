FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV HAWOR_ROOT=/opt/hawor
ENV PYOPENGL_PLATFORM=egl
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg build-essential ninja-build \
    libgl1 libegl1 libglib2.0-0 libxrender1 libsm6 libxext6 libglfw3 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ARG HAWOR_REPO=https://github.com/ThunderVVV/HaWoR.git
ARG HAWOR_COMMIT=main
RUN git clone --recursive ${HAWOR_REPO} /opt/hawor && \
    cd /opt/hawor && git checkout ${HAWOR_COMMIT} && \
    git submodule update --init --recursive

RUN pip install --no-cache-dir \
    "numpy<2" Pillow opencv-python tqdm scipy joblib natsort gdown \
    einops timm yacs omegaconf hydra-core smplx pyrender trimesh \
    pytorch-lightning==2.2.4 lightning-utilities torchmetrics==1.4.0 \
    imageio imageio-ffmpeg av

RUN if [ -f /opt/hawor/requirements.txt ]; then pip install --no-cache-dir -r /opt/hawor/requirements.txt || true; fi
RUN cd /opt/hawor/thirdparty/DROID-SLAM && python setup.py install
RUN pip install --no-cache-dir "matplotlib<3.9"
RUN chmod -R a+w /opt/hawor

ENV PYTHONPATH=/opt/hawor:/opt/hawor/thirdparty/DROID-SLAM:/opt/hawor/thirdparty/DROID-SLAM/droid_slam

COPY . /workspace
RUN pip install -e /workspace/v2d_hawor/lib

RUN python -c "import demo; print('hawor import OK')"
