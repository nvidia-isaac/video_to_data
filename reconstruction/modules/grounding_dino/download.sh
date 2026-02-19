#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RECONSTRUCTION_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
MODEL_DIR="${1:-${MODEL_DIR:-"$RECONSTRUCTION_DIR/data/grounding_dino/models"}}"
MODEL_FILE="$MODEL_DIR/groundingdino_swint_ogc.pth"

if [ -f "$MODEL_FILE" ]; then
    echo "Checkpoint already exists at $MODEL_FILE, skipping download."
    exit 0
fi

mkdir -p "$MODEL_DIR"
echo "Downloading GroundingDINO SwinT-OGC checkpoint to $MODEL_FILE..."
wget -q --show-progress \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -O "$MODEL_FILE"
echo "GroundingDINO checkpoint downloaded."
