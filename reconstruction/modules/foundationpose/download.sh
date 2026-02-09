#!/bin/bash
set -e
# Get the script directory (modules directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WEIGHTS_DIR=${1:-"$SCRIPT_DIR/data/checkpoints/weights"}
mkdir -p "$WEIGHTS_DIR"
echo "Note: FoundationPose weights need to be downloaded manually."
echo "Please download weights to: $WEIGHTS_DIR"
echo "See FoundationPose documentation for download instructions." 