#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build all v2d Docker images.
# Run from reconstruction/ or repo root. Requires Docker and (optionally) NVIDIA Container Toolkit.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODULES=(unidepth moge anycalib sam2 sam3d grounding_dino mediapipe hamer wilor droid_slam foundation_stereo foundation_pose gsplat_refinement)

for module in "${MODULES[@]}"; do
  echo "Building v2d_${module}..."
  python -m "v2d.${module}.docker.build"
done

# These modules use a different package namespace
echo "Building v2d_cusfm..."
python modules/v2d_cusfm/docker/build.py
echo "Building v2d_bundlesdf..."
python modules/v2d_bundlesdf/docker/build.py
echo "Building v2d_hoi_object_reconstruction..."
python modules/v2d_hoi_object_reconstruction/docker/build.py
echo "Building v2d_ego_hand_reconstruction..."
python modules/v2d_ego_hand_reconstruction/docker/build.py
echo "Building v2d_hand_alignment..."
python modules/v2d_hand_alignment/docker/build.py

echo "All containers built successfully."
