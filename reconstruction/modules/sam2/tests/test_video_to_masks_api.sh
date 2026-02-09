#!/bin/bash
# Test script for SAM2 video_to_masks API endpoint
# Usage: ./test_video_to_masks_api.sh

API_URL="http://localhost:8001/process/video_to_masks"
TEST_DATA_DIR="$(dirname "$0")/test_data"

echo "Testing SAM2 video_to_masks API..."
echo "API URL: $API_URL"
echo "Test data directory: $TEST_DATA_DIR"

if [ ! -f "$TEST_DATA_DIR/test_video.mp4" ]; then
    echo "Error: test_video.mp4 not found in $TEST_DATA_DIR"
    exit 1
fi

if [ ! -f "$TEST_DATA_DIR/test_prompts.json" ]; then
    echo "Error: test_prompts.json not found in $TEST_DATA_DIR"
    exit 1
fi

curl -X POST "$API_URL" \
    -F "video=@$TEST_DATA_DIR/test_video.mp4" \
    -F "prompts=@$TEST_DATA_DIR/test_prompts.json" \
    --output sam2_video_to_masks_results.zip \
    --progress-bar

if [ $? -eq 0 ]; then
    echo ""
    echo "Success! Results saved to sam2_video_to_masks_results.zip"
else
    echo ""
    echo "Error: API call failed"
    exit 1
fi

