FROM nvcr.io/nvidia/pytorch:25.06-py3
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    zip \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install -e /workspace/v2d_common -e /workspace/v2d_moge/lib

# The NGC PyTorch base ships a torch build compiled against NumPy 1.x; the
# moge/lib install above pulls in NumPy 2.x transitively, which breaks
# torch<->numpy interop at runtime ("RuntimeError: Numpy is not available").
# Pin back down, matching the same fix already used in
# robotic_grounding/workflow/Dockerfile for the same NGC/Isaac Lab base.
RUN pip install "numpy<2.0.0"
