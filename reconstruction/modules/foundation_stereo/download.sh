#!/bin/bash
# Download Foundation Stereo ONNX model from NVIDIA NGC.
# The TensorRT engine is compiled at runtime (GPU-architecture-specific).
#
# Usage: bash download.sh [MODEL_DIR]
# Default MODEL_DIR: /data/foundation_stereo/models

set -e

MODEL_DIR="${1:-${MODEL_DIR:-/data/foundation_stereo/models}}"
ONNX_FILENAME="deployable_foundationstereo_small_576x960_v2.0.onnx"
ONNX_URL="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/foundationstereo/deployable_v2.0/files?redirect=true&path=${ONNX_FILENAME}"

mkdir -p "$MODEL_DIR"
DEST="$MODEL_DIR/$ONNX_FILENAME"

if [ -f "$DEST" ]; then
    echo "ONNX already exists at $DEST, skipping download."
    exit 0
fi

echo "Downloading Foundation Stereo ONNX to $DEST ..."
curl -L "$ONNX_URL" -o "$DEST"
echo "Download complete: $DEST"
