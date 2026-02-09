#!/bin/bash

# Directory for weights
WEIGHTS_DIR="modules/nlf/data/weights"
mkdir -p $WEIGHTS_DIR

# Download NLF TorchScript weights
# Note: Using the URL from the original run_nlf.py context if available, 
# otherwise pointing to where it can be found.
# The original code used: 'models/nlf_l_multi_0.3.2.torchscript'
# Assuming a public URL or providing instructions.

WEIGHTS_FILE="nlf_l_multi_0.3.2.torchscript"
URL="https://github.com/isarandi/nlf/releases/download/v0.3.2/$WEIGHTS_FILE"

if [ ! -f "$WEIGHTS_DIR/$WEIGHTS_FILE" ]; then
    echo "Downloading NLF weights..."
    wget -O "$WEIGHTS_DIR/$WEIGHTS_FILE" "$URL"
else
    echo "NLF weights already exist."
fi

echo "--------------------------------------------------"
echo "SMPL/SMPLH models must be downloaded manually due to licensing."
echo "Please place them in: modules/nlf/data/smpl_models/"
echo "Expected structure:"
echo "modules/nlf/data/smpl_models/"
echo "├── smpl/"
echo "│   ├── SMPL_FEMALE.pkl"
echo "│   ├── SMPL_MALE.pkl"
echo "│   └── SMPL_NEUTRAL.pkl"
echo "└── smplh/"
echo "    ├── female/"
echo "    │   └── model.npz"
echo "    └── male/"
echo "        └── model.npz"
echo "--------------------------------------------------"

