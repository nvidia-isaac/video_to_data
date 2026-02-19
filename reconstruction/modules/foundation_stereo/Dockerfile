FROM nvcr.io/nvidia/tensorrt:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python-headless numpy Pillow "cuda-python==12.6.*"

COPY modules/common /workspace/modules/common
COPY modules/foundation_stereo /workspace/modules/foundation_stereo

ENV PYTHONPATH=/workspace
