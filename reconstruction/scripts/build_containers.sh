#!/bin/bash
# Build all v2d Docker images.
# Run from reconstruction/ or repo root. Requires Docker and (optionally) NVIDIA Container Toolkit.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODULES=(unidepth moge sam2 sam3d grounding_dino foundation_stereo foundation_pose nlf)

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

echo "All containers built successfully."
