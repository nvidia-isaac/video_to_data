# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

LABEL org.opencontainers.image.title="v2d E2FGVI inpainting"
LABEL org.opencontainers.image.source="https://github.com/MarionLepert/phantom-E2FGVI.git"
LABEL org.opencontainers.image.revision="5b45ffe400288006facb350e00d319bfc6c5cbd3"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/e2fgvi

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg git libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# This SHA is the requested Phantom/E2FGVI release commit. Verify it in the
# image layer so a moved branch or incomplete fetch cannot silently change code.
RUN git init /opt/e2fgvi \
    && git -C /opt/e2fgvi remote add origin https://github.com/MarionLepert/phantom-E2FGVI.git \
    && git -C /opt/e2fgvi fetch --depth=1 origin 5b45ffe400288006facb350e00d319bfc6c5cbd3 \
    && git -C /opt/e2fgvi checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/e2fgvi rev-parse HEAD)" = "5b45ffe400288006facb350e00d319bfc6c5cbd3"

# Torchvision supplies the modulated deform-convolution operator used by the
# inference-only MMCV compatibility package in this stage.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torchvision==0.22.1

COPY lib /workspace/e2fgvi/lib
RUN python -m pip install --no-cache-dir /workspace/e2fgvi/lib

WORKDIR /workspace
ENTRYPOINT []
