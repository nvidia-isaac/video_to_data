#!/bin/bash
set -e
# Get the script directory (modules directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RECONSTRUCTION_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
DATA_DIR=${DATA_DIR:-"$RECONSTRUCTION_DIR/data"}
CHECKPOINT_DIR=${1:-"$DATA_DIR/unidepth/checkpoints/unidepth-v2-vitl14"}
mkdir -p "$CHECKPOINT_DIR"
echo "Downloading UniDepth v2 checkpoint to $CHECKPOINT_DIR..."
hf download lpiccinelli/unidepth-v2-vitl14 --local-dir "$CHECKPOINT_DIR"
echo "UniDepth v2 checkpoint downloaded."
