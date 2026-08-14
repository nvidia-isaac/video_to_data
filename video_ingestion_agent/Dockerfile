# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
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

# Dockerfile for Video Ingestion Agent — unified image for all backends.
#
# One image, one venv, one lockfile: the same environment serves Qwen3-VL,
# Cosmos3-Nano (Reasoner tower via the vllm-cosmos3 plugin), and the API
# backends. The former Dockerfile.cosmos variant is folded in here; unlike
# it, this image installs strictly from uv.lock (--frozen), so builds are
# reproducible and match the host dev venv byte-for-byte.
#
# Stack (locked): Python 3.13 / torch 2.11+cu130 / vllm 0.21.x /
# vllm-cosmos3 (rev-pinned in pyproject.toml) / transformers 5.x.
#
# Based on the public NVIDIA CUDA + cuDNN devel image. We install uv and
# the pipeline on top; the devel base also provides the CUDA toolchain
# that flashinfer's ninja JIT needs at engine init.
#
# Does NOT bundle model weights -- vLLM downloads them at runtime via HF_TOKEN.
#
# Build:
#   docker build -t video_ingestion_agent .
#
# Override the base via --build-arg BASE_IMAGE=<image> if a different
# CUDA / cuDNN combination is needed.

ARG CUDA_VERSION=13.0.1
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# System deps:
#   - python3 + venv + dev headers: bootstrap Python for uv itself (the
#     project venv uses the uv-managed 3.13 from .python-version;
#     python3-dev is needed for Triton's runtime JIT, which compiles
#     small C extensions against Python.h when vLLM warms up its kernels)
#   - build-essential: gcc/g++/make for the same JIT path
#   - ninja-build: flashinfer JIT-compiles kernels at vLLM engine init and
#     shells out to `ninja`; without it EngineCore dies with
#     FileNotFoundError: 'ninja' during model load
#   - ffmpeg: video decoding for the ingestion pipeline
#   - opencv runtime libs: libxcb / libgl / libglib / libsm / libxext / libxrender
#   - ca-certificates / curl / git: HF download + git deps (vllm-cosmos3)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-dev python-is-python3 \
        build-essential ninja-build \
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
COPY pyproject.toml uv.lock README.md .python-version ./
COPY src/ src/

# Sync from the lockfile so the container reproduces the exact versions
# in the host .venv (vllm 0.21.x / torch 2.11+cu130 / transformers 5.x /
# vllm-cosmos3, ...). --frozen aborts if uv.lock is stale relative to
# pyproject.toml. torch resolves from the cu130 index declared in
# pyproject's [tool.uv] section, matching the CUDA 13 base above.
# GIT_LFS_SKIP_SMUDGE guards the vllm-cosmos3 git fetch (cosmos-framework
# has an LFS object missing upstream; irrelevant to the python package).
RUN --mount=type=cache,target=/root/.cache/uv \
    GIT_LFS_SKIP_SMUDGE=1 \
    uv sync --frozen --extra local --extra server --extra benchmark --extra webapp

# Make the project venv the default Python on PATH so subsequent
# `python` / `pip` invocations resolve to the locked environment.
ENV VIRTUAL_ENV="/workspace/video_ingestion_agent/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Copy remaining project files (scripts, configs, data, etc.).
# .venv/ is in .dockerignore, so this won't clobber the synced env.
COPY . /workspace/video_ingestion_agent/
