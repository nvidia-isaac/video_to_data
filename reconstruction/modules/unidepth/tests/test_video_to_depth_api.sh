#!/bin/bash
# Test script for UniDepth video_to_depth API endpoint
# Usage: ./test_video_to_depth_api.sh

API_URL="http://localhost:8003/process/video_to_depth"
TEST_DATA_DIR="$(dirname "$0")/test_data"

echo "Testing UniDepth video_to_depth API..."
echo "API URL: $API_URL"
echo "Test data directory: $TEST_DATA_DIR"

if [ ! -f "$TEST_DATA_DIR/test_video.mp4" ]; then
    echo "Error: test_video.mp4 not found in $TEST_DATA_DIR"
    exit 1
fi

curl -X POST "$API_URL" \
    -F "video=@$TEST_DATA_DIR/test_video.mp4" \
    -F "batch_size=8" \
    --output unidepth_video_to_depth_results.zip \
    --progress-bar

if [ $? -eq 0 ]; then
    echo ""
    echo "Success! Results saved to unidepth_video_to_depth_results.zip"
else
    echo ""
    echo "Error: API call failed"
    exit 1
fi

