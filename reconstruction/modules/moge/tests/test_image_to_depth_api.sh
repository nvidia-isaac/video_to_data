#!/bin/bash
# Test script for MoGe image_to_depth API endpoint
# Usage: ./test_image_to_depth_api.sh

API_URL="http://localhost:8002/process/image_to_depth"
TEST_DATA_DIR="$(dirname "$0")/test_data"

echo "Testing MoGe image_to_depth API..."
echo "API URL: $API_URL"
echo "Test data directory: $TEST_DATA_DIR"

# Check if we have a test image (we might need to extract one from video or use a placeholder)
# For now, we'll check if there's a test image file
if [ ! -f "$TEST_DATA_DIR/test_image.jpg" ] && [ ! -f "$TEST_DATA_DIR/test_image.png" ]; then
    echo "Warning: No test image found. Creating a placeholder test..."
    echo "Note: You may need to provide a test image file (test_image.jpg or test_image.png) in $TEST_DATA_DIR"
    echo "Skipping test..."
    exit 0
fi

# Use the first available image
TEST_IMAGE=""
if [ -f "$TEST_DATA_DIR/test_image.jpg" ]; then
    TEST_IMAGE="$TEST_DATA_DIR/test_image.jpg"
elif [ -f "$TEST_DATA_DIR/test_image.png" ]; then
    TEST_IMAGE="$TEST_DATA_DIR/test_image.png"
fi

curl -X POST "$API_URL" \
    -F "image=@$TEST_IMAGE" \
    --output moge_image_to_depth_results.zip \
    --progress-bar

if [ $? -eq 0 ]; then
    echo ""
    echo "Success! Results saved to moge_image_to_depth_results.zip"
else
    echo ""
    echo "Error: API call failed"
    exit 1
fi

