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
COPY v2d_mv /workspace/v2d_mv
COPY v2d_sam2 /workspace/v2d_sam2

RUN pip install -e "/workspace/v2d_common[io]" -e /workspace/v2d_mv -e /workspace/v2d_sam2/lib

# The host runners deliberately execute containers as the calling user so
# generated artifacts are not left owned by root.  Docker COPY preserves the
# repository's group-only directory modes, while editable installs import from
# /workspace at runtime; make those sources readable by the unprivileged user.
RUN chmod -R a+rX /workspace
