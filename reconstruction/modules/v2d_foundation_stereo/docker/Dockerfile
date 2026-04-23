FROM nvcr.io/nvidia/tensorrt:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python-headless numpy Pillow "cuda-python==12.6.*"

COPY . /workspace

RUN pip install -e /workspace/v2d_common -e /workspace/v2d_mv -e /workspace/v2d_foundation_stereo/lib

# Bake model weights into image (ONNX required; engine is GPU-arch-specific cache)
COPY weights/foundationstereo /data/models/foundationstereo
