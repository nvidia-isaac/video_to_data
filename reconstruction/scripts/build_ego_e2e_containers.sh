#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build only the Docker images required by the ego e2e pipeline
# (modules/v2d_pipelines/run_v2d_ego_e2e.py).
#
# Use this when you don't need every container. For everything, run
# ./scripts/build_containers.sh instead.
#
# Run from reconstruction/ or repo root. Requires Docker and (optionally)
# NVIDIA Container Toolkit.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Modules under the v2d.<name>.docker namespace
MODULES=(anycalib moge grounding_dino sam2 sam3d foundation_pose hamer)

for module in "${MODULES[@]}"; do
  echo "Building v2d_${module}..."
  python -m "v2d.${module}.docker.build"
done

# Modules with their own build entry points
echo "Building v2d_ego_hand_reconstruction..."
python modules/v2d_ego_hand_reconstruction/docker/build.py
echo "Building v2d_hand_alignment..."
python modules/v2d_hand_alignment/docker/build.py

echo "All ego-e2e containers built successfully."
