#!/bin/bash
set -e
# Get the script directory (modules directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CHECKPOINT_DIR=${1:-"$SCRIPT_DIR/data/checkpoints/sam2.1-hiera-large"}
mkdir -p "$CHECKPOINT_DIR"
echo "Downloading SAM 2.1 checkpoint to $CHECKPOINT_DIR..."
hf download facebook/sam2.1-hiera-large --local-dir "$CHECKPOINT_DIR"
echo "SAM 2.1 checkpoint downloaded."
