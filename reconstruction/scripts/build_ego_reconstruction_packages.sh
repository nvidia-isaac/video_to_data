#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build only Docker images required by run_ego_reconstruction.py.
#
# Run from reconstruction/ or repo root. Requires Docker and NVIDIA Container Toolkit.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
# Build modules with the same interpreter environment that runs run_e2e.sh.
PYTHON_BIN="${PYTHON:-python3}"

# The E2E HaMeR route omits unrelated HAWOR and DynHaMR images to limit disk use.
# Keep `all` as the backward-compatible default for standalone callers.
MODE="all"
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--mode" ]]; then
    echo "Usage: $0 [--mode all|hamer]" >&2
    exit 2
  fi
  MODE="$2"
fi

case "$MODE" in
  all)
    MODULES=(anycalib moge grounding_dino sam2 sam3d foundation_pose hamer wilor hawor geocalib droid_slam gsplat_refinement)
    ;;
  hamer)
    MODULES=(anycalib moge grounding_dino sam2 sam3d foundation_pose hamer wilor geocalib droid_slam gsplat_refinement)
    ;;
  *)
    echo "Unknown mode: $MODE (expected all or hamer)" >&2
    exit 2
    ;;
esac

for module in "${MODULES[@]}"; do
  echo "Building v2d_${module}..."
  "$PYTHON_BIN" -m "v2d.${module}.docker.build"
done

# The legacy ViPE/DynHaMR pipeline and its alignment adapter are not used by
# the HaMeR path. Keep them in the standalone full-build mode.
if [[ "$MODE" == "all" ]]; then
  echo "Building v2d_ego_hand_reconstruction..."
  "$PYTHON_BIN" modules/v2d_ego_hand_reconstruction/docker/build.py
  echo "Building v2d_hand_alignment..."
  "$PYTHON_BIN" modules/v2d_hand_alignment/docker/build.py
fi

echo "Ego reconstruction containers built successfully (mode: $MODE)."
