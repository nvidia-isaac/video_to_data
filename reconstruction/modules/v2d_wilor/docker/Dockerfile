# PyTorch 2.5 / CUDA 12.4 wheels do not contain sm_120 kernels for the
# RTX PRO 6000 Blackwell used by this investigation.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
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
    ultralytics==8.1.34 scikit-image roma dill

# manotorch + chumpy (same as hamer; needed by the MANO layer used in render).
RUN pip install --no-cache-dir \
    git+https://github.com/lixiny/manotorch.git@a2a70c591f91551078b7bb2af9b5d9f275b626e0 \
    git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da

# WiLoR mini pipeline — provides WiLorHandPose3dEstimationPipeline (detector + reconstructor).
# --no-deps keeps the base image's torch build; the deps are declared above.
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/warmshao/WiLoR-mini.git@ebec42f94c389070cdd7dda6fd1bf0b4a659c960

COPY . /workspace
RUN pip install -e /workspace/v2d_common \
    -e /workspace/v2d_wilor/lib

# Editable installs import from /workspace at runtime. Docker COPY preserves
# restrictive source modes, so make the tree readable by the host UID used by
# the container wrappers.
RUN chmod -R a+rX /workspace

# Sanity-check at build time so a broken install fails the build, not the run.
RUN python -c "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; print('wilor_mini import OK')"
