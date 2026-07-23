# CUDA 12.4 / PyTorch 2.5 wheels stop at sm_90 and fail with "no kernel
# image" on the RTX PRO 6000 Blackwell (sm_120). CUDA 12.8 PyTorch wheels
# include Blackwell support while retaining compatibility with Ada/Hopper.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    zip \
    && rm -rf /var/lib/apt/lists/*

COPY v2d_common /workspace/v2d_common
COPY v2d_moge /workspace/v2d_moge

RUN pip install -e /workspace/v2d_common -e /workspace/v2d_moge/lib

# Host runners execute as the calling user; editable installs must remain
# readable even when the checkout has group-only directory permissions.
RUN chmod -R a+rX /workspace
