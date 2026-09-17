#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install only the host-side packages required by run_ego_reconstruction.py.
#
# Run from reconstruction/ or repo root.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
# Install into the same interpreter environment that runs run_e2e.sh.
PYTHON_BIN="${PYTHON:-python3}"

echo "Installing host packages for ego reconstruction..."
# Ubuntu's SciPy 1.8 binary is incompatible when dependency resolution selects NumPy 2.
"$PYTHON_BIN" -m pip install Pillow "scipy>=1.13,<2" \
  -e modules/v2d_common \
  -e modules/v2d_docker \
  -e modules/v2d_depth \
  -e modules/v2d_anycalib/docker \
  -e modules/v2d_geocalib/docker \
  -e modules/v2d_droid_slam/docker \
  -e modules/v2d_ego_hand_reconstruction/docker \
  -e modules/v2d_foundation_pose/docker \
  -e modules/v2d_grounding_dino/docker \
  -e modules/v2d_gsplat_refinement/docker \
  -e modules/v2d_hamer/docker \
  -e modules/v2d_hand_alignment/docker \
  -e modules/v2d_hawor/docker \
  -e modules/v2d_mediapipe/docker \
  -e modules/v2d_moge/docker \
  -e modules/v2d_sam2/docker \
  -e modules/v2d_sam3d/docker \
  -e modules/v2d_wilor/docker \
  -e modules/v2d_pipelines

echo "Done. Next: build Docker images with ./scripts/build_ego_reconstruction_packages.sh"
