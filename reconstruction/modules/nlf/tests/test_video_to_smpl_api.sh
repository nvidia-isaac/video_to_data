#!/bin/bash

# Simple test script for NLF video_to_smpl API
# Usage: ./test_video_to_smpl_api.sh <video_path> <masks_zip_path>

VIDEO_PATH=$1
MASKS_ZIP=$2
GENDER=${3:-"neutral"}
MODEL_TYPE=${4:-"smplh"}

if [ -z "$VIDEO_PATH" ] || [ -z "$MASKS_ZIP" ]; then
    echo "Usage: $0 <video_path> <masks_zip_path> [gender] [model_type]"
    exit 1
fi

# Example intrinsics (HD)
INTRINSICS='{"fx": 1000, "fy": 1000, "cx": 960, "cy": 540, "width": 1920, "height": 1080}'

echo "Submitting NLF task..."
curl -X POST http://localhost:8005/process/video_to_smpl \
  -F "video=@$VIDEO_PATH" \
  -F "masks=@$MASKS_ZIP" \
  -F "gender=$GENDER" \
  -F "model_type=$MODEL_TYPE" \
  -F "intrinsics=$INTRINSICS" \
  --output nlf_results.zip

echo "Results saved to nlf_results.zip"


