#!/bin/bash
# Install all lightweight packages: docker orchestration wrappers.
# Run from reconstruction/ or repo root.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Installing v2d docker packages..."
pip install -e modules/v2d_depth \
  -e modules/v2d_sam2/docker \
  -e modules/v2d_sam3d/docker \
  -e modules/v2d_unidepth/docker \
  -e modules/v2d_moge/docker \
  -e modules/v2d_depth_anything/docker \
  -e modules/v2d_nlf/docker \
  -e modules/v2d_foundation_pose/docker \
  -e modules/v2d_foundation_stereo/docker \
  -e modules/v2d_grounding_dino/docker \
  -e modules/v2d_cusfm/docker \
  -e modules/v2d_bundlesdf/docker \
  -e modules/v2d_hoi_object_reconstruction/docker

echo "Done. Build containers with ./build_containers.sh"
