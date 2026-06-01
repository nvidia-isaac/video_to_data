#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install all lightweight packages: docker orchestration + v2d_pipelines.
# Run from reconstruction/ or repo root.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Installing v2d docker packages and v2d_pipelines..."
pip install -e modules/v2d_common \
  -e modules/v2d_docker \
  -e modules/v2d_mv \
  -e modules/v2d_depth \
  -e modules/v2d_viz \
  -e modules/v2d_anycalib/docker \
  -e modules/v2d_bundlesdf/docker \
  -e modules/v2d_cusfm/docker \
  -e modules/v2d_depth_anything/docker \
  -e modules/v2d_detectron2/docker \
  -e modules/v2d_droid_slam/docker \
  -e modules/v2d_ego_hand_reconstruction/docker \
  -e modules/v2d_foundation_pose/docker \
  -e modules/v2d_foundation_stereo/docker \
  -e modules/v2d_grounding_dino/docker \
  -e modules/v2d_gsplat_refinement/docker \
  -e modules/v2d_hamer/docker \
  -e modules/v2d_hand_alignment/docker \
  -e modules/v2d_hoi_object_reconstruction/docker \
  -e modules/v2d_mediapipe/docker \
  -e modules/v2d_mesh/docker \
  -e modules/v2d_moge/docker \
  -e modules/v2d_mv_calibration/docker \
  -e modules/v2d_mv_postprocess/docker \
  -e modules/v2d_mv_preprocess/docker \
  -e modules/v2d_rosbag/docker \
  -e modules/v2d_sam2/docker \
  -e modules/v2d_sam3d/docker \
  -e modules/v2d_sam3d_body/docker \
  -e modules/v2d_unidepth/docker \
  -e modules/v2d_wilor/docker \
  -e modules/v2d_pipelines

echo "Done. Run 'python -m v2d.pipelines.run_example_pipeline' or build containers with ./build_containers.sh"
