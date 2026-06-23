FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=egl
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg \
    libgl1 libegl1 libglib2.0-0 libxrender1 libsm6 libxext6 libglfw3 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy<2" \
    Pillow opencv-python tqdm \
    pyrender trimesh \
    smplx \
    yacs hydra-core omegaconf einops timm \
    huggingface_hub transformers safetensors \
    ultralytics dill

# manotorch + chumpy (same as hamer; needed by the MANO layer used in render).
RUN pip install --no-cache-dir \
    git+https://github.com/lixiny/manotorch.git \
    git+https://github.com/mattloper/chumpy.git

# WiLoR mini pipeline — provides WiLorHandPose3dEstimationPipeline (detector + reconstructor).
RUN pip install --no-cache-dir git+https://github.com/warmshao/WiLoR-mini.git

COPY . /workspace
RUN pip install -e /workspace/v2d_common \
    -e /workspace/v2d_wilor/lib

# Sanity-check at build time so a broken install fails the build, not the run.
RUN python -c "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; print('wilor_mini import OK')"
