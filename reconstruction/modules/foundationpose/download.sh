#!/bin/bash
set -e
# Get the script directory (modules directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RECONSTRUCTION_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
DATA_DIR=${DATA_DIR:-"$RECONSTRUCTION_DIR/data"}
WEIGHTS_DIR=${1:-"$DATA_DIR/foundationpose/checkpoints/weights"}
mkdir -p "$WEIGHTS_DIR"
echo "Note: FoundationPose weights need to be downloaded manually."
echo "Please download weights to: $WEIGHTS_DIR"
echo "See FoundationPose documentation for download instructions."
