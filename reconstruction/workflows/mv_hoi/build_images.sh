#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build all Docker images needed by the MV HOI pipelines (calibration + reconstruction).
#
# Usage:
#   ./build_images.sh          # build all
#   ./build_images.sh sam2     # build only v2d_sam2
#
# Run from anywhere — paths are resolved relative to this script.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULES_DIR="$SCRIPT_DIR/../../modules"

IMAGES=(
    v2d_rosbag
    v2d_mv_calibration
    v2d_mv_preprocess
    v2d_foundation_stereo
    v2d_grounding_dino
    v2d_sam2
    v2d_foundation_pose
    v2d_detectron2
    v2d_sam3d_body
    v2d_mv_postprocess
)

if [ $# -gt 0 ]; then
    IMAGES=()
    for arg in "$@"; do
        if [[ "$arg" != v2d_* ]]; then
            arg="v2d_${arg}"
        fi
        IMAGES+=("$arg")
    done
fi

for module in "${IMAGES[@]}"; do
    echo "=== Building ${module} ==="
    python "$MODULES_DIR/${module}/docker/build.py"
    echo ""
done

echo "Done: built ${#IMAGES[@]} image(s)."
