#!/bin/bash
# Install all lightweight packages: docker orchestration + v2d_pipelines.
# Run from reconstruction/ or repo root.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Installing v2d docker packages and v2d_pipelines..."
pip install -e modules/v2d_sam2/docker \
  -e modules/v2d_sam3d/docker \
  -e modules/v2d_unidepth/docker \
  -e modules/v2d_moge/docker \
  -e modules/v2d_nlf/docker \
  -e modules/v2d_foundation_pose/docker \
  -e modules/v2d_foundation_stereo/docker \
  -e modules/v2d_grounding_dino/docker \
  -e modules/v2d_cusfm/docker \
  -e modules/v2d_bundlesdf/docker \
  -e modules/v2d_hoi_object_reconstruction/docker \
  -e modules/v2d_ego_hand_reconstruction/docker \
  -e modules/v2d_pipelines

echo "Done. Run 'python -m v2d.pipelines.run_example_pipeline' or build containers with ./build_containers.sh"
