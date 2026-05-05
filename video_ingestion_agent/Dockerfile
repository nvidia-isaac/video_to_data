# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile for Video Ingestion Agent
#
# Based on the public NVIDIA CUDA + cuDNN devel image. We install Python,
# uv, and the pipeline (which pulls torch + vLLM from PyPI) on top. This
# mirrors what nvidia-cosmos/cosmos-reason2 does upstream and avoids the
# pre-baked-package ABI mismatches that the heavyweight NGC PyTorch image
# would force us to work around.
#
# Does NOT bundle model weights -- vLLM downloads them at runtime via HF_TOKEN.
#
# Build:
#   docker build -t video_ingestion_agent .
#
# Override the base via --build-arg BASE_IMAGE=<image> if a different
# CUDA / cuDNN combination is needed.

ARG CUDA_VERSION=12.8.1
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# System deps:
#   - python3.12 + venv + dev headers: the Ubuntu 24.04 default Python
#     (python3-dev is needed for Triton's runtime JIT, which compiles
#     small C extensions against Python.h when vLLM warms up its kernels)
#   - build-essential: gcc/g++/make for the same JIT path
#   - ffmpeg: video decoding for the ingestion pipeline
#   - opencv runtime libs: libxcb / libgl / libglib / libsm / libxext / libxrender
#   - ca-certificates / curl / git: HF download + diagnostics
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-dev python-is-python3 \
        build-essential \
        ffmpeg \
        ca-certificates curl git \
        libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Triton needs ptxas to JIT vLLM's CUDA kernels (the cudnn-devel base
# ships ptxas at this path).
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# uv for fast, lock-aware installs. --break-system-packages is only used
# here to install uv itself into Ubuntu 24.04's PEP 668 system Python;
# the project itself goes into a project venv via `uv sync` below.
RUN pip install --no-cache-dir --break-system-packages uv

WORKDIR /workspace/video_ingestion_agent

# Copy spec + lockfile + source for layer caching.
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

# Sync from the lockfile so the container reproduces the exact versions
# in the host .venv (vllm 0.15.1 / torch 2.9.1 / transformers 4.57.6, ...).
# --frozen aborts if uv.lock is stale relative to pyproject.toml.
# The [server] extra pins vLLM to >=0.15.1,<0.20 (0.20+ has a Qwen3-VL
# deepstack regression); torch is pulled with bundled CUDA libs matching
# the system CUDA above.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra server --extra benchmark --extra webapp

# Make the project venv the default Python on PATH so subsequent
# `python` / `pip` invocations resolve to the locked environment.
ENV VIRTUAL_ENV="/workspace/video_ingestion_agent/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Copy remaining project files (scripts, configs, data, etc.).
# .venv/ is in .dockerignore, so this won't clobber the synced env.
COPY . /workspace/video_ingestion_agent/
