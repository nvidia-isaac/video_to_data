#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build only Docker images required by run_ego_reconstruction.py.
#
# Run from reconstruction/ or repo root. Requires Docker and NVIDIA Container Toolkit.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODULES=(anycalib moge grounding_dino sam2 sam3d foundation_pose mediapipe hamer wilor geocalib droid_slam gsplat_refinement)

for module in "${MODULES[@]}"; do
  echo "Building v2d_${module}..."
  python -m "v2d.${module}.docker.build"
done

# Modules with their own build entry points.
echo "Building v2d_ego_hand_reconstruction..."
python modules/v2d_ego_hand_reconstruction/docker/build.py
echo "Building v2d_hand_alignment..."
python modules/v2d_hand_alignment/docker/build.py

echo "All ego reconstruction containers built successfully."
