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
# Based on the cosmos_reason_2 image which has a known-good CUDA 12.8 + PyTorch
# stack. We install vLLM and the pipeline dependencies on top.
# Does NOT bundle model weights -- vLLM downloads them at runtime via HF_TOKEN.
#
# Build:
#   docker build -t video_ingestion_agent .
#
# The osmo workflow uses this as the base image.

ARG BASE_IMAGE=nvcr.io/nvstaging/isaac-amr/cosmos_reason_2
FROM ${BASE_IMAGE}

# Set working directory
WORKDIR /workspace/video_ingestion_agent

# Copy dependency spec first for better layer caching
COPY pyproject.toml README.md ./
COPY src/ src/

# Install vLLM + pipeline deps into the existing venv
# The base image already has PyTorch + CUDA, so vLLM builds quickly
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install vllm && \
    uv pip install -e ".[benchmark,webapp]"

# Copy remaining project files (scripts, configs, etc.)
COPY . /workspace/video_ingestion_agent/
