# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM nvcr.io/nvidia/isaac-sim:6.0.1

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps \
    coacd==1.0.11 fast-simplification==0.1.13 trimesh==4.12.2
USER 1234:1234

COPY v2d_hoi_object_reconstruction/mesh_to_usd /workspace/mesh_to_usd
WORKDIR /workspace/mesh_to_usd/runtime
