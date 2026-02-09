#!/bin/bash
set -e
# Get the script directory (modules directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CHECKPOINT_DIR=${1:-"$SCRIPT_DIR/data/checkpoints/unidepth-v2-vitl14"}
mkdir -p "$CHECKPOINT_DIR"
echo "Downloading UniDepth v2 checkpoint to $CHECKPOINT_DIR..."
hf download lpiccinelli/unidepth-v2-vitl14 --local-dir "$CHECKPOINT_DIR"
echo "UniDepth v2 checkpoint downloaded."
