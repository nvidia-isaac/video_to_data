#!/bin/bash
set -e

# Get the script directory (modules directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CHECKPOINTS_DIR=${1:-"$SCRIPT_DIR/data/checkpoints/sam3d"}

mkdir -p "$CHECKPOINTS_DIR/hf-download"

echo "Note: SAM 3D Objects checkpoints are gated on Hugging Face."
echo "Please request access at: https://huggingface.co/facebook/sam-3d-objects"
echo "After access is granted, login using: hf login"
echo ""

if ! command -v hf &> /dev/null
then
    echo "hf could not be found. "
    exit 1
fi

echo "Downloading SAM 3D checkpoints to $CHECKPOINTS_DIR/hf-download..."
hf download facebook/sam-3d-objects --local-dir "$CHECKPOINTS_DIR/hf-download"

echo "Downloading MoGE v1 checkpoint (required for SAM3D)..."
# We download it to the HF cache directory mapped in docker-compose
export HF_HOME="$CHECKPOINTS_DIR/hf_home"
mkdir -p "$HF_HOME"
hf download Ruicheng/moge-vitl

echo "Downloading DINOv2 checkpoint..."
mkdir -p "$CHECKPOINTS_DIR/torch_home/checkpoints"
if [ ! -f "$CHECKPOINTS_DIR/torch_home/checkpoints/dinov2_vitl14_reg4_pretrain.pth" ]; then
    # Use curl as a fallback if wget is missing, or just use curl
    curl -L https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth -o "$CHECKPOINTS_DIR/torch_home/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
fi

echo "SAM 3D checkpoints and dependencies downloaded."
