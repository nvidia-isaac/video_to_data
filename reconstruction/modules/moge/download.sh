#!/bin/bash
set -e
# Get the script directory (modules directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RECONSTRUCTION_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
DATA_DIR=${DATA_DIR:-"$RECONSTRUCTION_DIR/data"}
CHECKPOINT_DIR=${1:-"$DATA_DIR/moge/checkpoints/moge-2-vitl-normal"}
mkdir -p "$CHECKPOINT_DIR"
echo "Downloading MoGE v2 checkpoint to $CHECKPOINT_DIR..."
hf download Ruicheng/moge-2-vitl-normal --local-dir "$CHECKPOINT_DIR"
echo "MoGE v2 checkpoint downloaded."
