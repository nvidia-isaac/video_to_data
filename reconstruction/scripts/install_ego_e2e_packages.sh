#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install only the host-side packages required by the ego e2e pipeline
# (modules/v2d_pipelines/run_v2d_ego_e2e.py).
#
# Use this when you don't need the full repo install. For everything, run
# ./scripts/install_packages.sh instead.
#
# Run from reconstruction/ or repo root.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Installing host packages for the ego e2e pipeline..."
pip install -e modules/v2d_common \
  -e modules/v2d_docker \
  -e modules/v2d_depth \
  -e modules/v2d_anycalib/docker \
  -e modules/v2d_ego_hand_reconstruction/docker \
  -e modules/v2d_foundation_pose/docker \
  -e modules/v2d_grounding_dino/docker \
  -e modules/v2d_hamer/docker \
  -e modules/v2d_hand_alignment/docker \
  -e modules/v2d_moge/docker \
  -e modules/v2d_sam2/docker \
  -e modules/v2d_sam3d/docker \
  -e modules/v2d_pipelines

echo "Done. Next: build the matching containers with ./scripts/build_ego_e2e_containers.sh"
