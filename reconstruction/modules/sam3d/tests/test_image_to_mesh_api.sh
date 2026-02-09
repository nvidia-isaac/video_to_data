#!/bin/bash
# Test script for SAM3D image_to_mesh API endpoint
# Usage: ./test_image_to_mesh_api.sh

API_URL="http://localhost:8004/process/image_to_mesh"
TEST_DATA_DIR="$(dirname "$0")/test_data"

echo "Testing SAM3D image_to_mesh API..."
echo "API URL: $API_URL"
echo "Test data directory: $TEST_DATA_DIR"

if [ ! -f "$TEST_DATA_DIR/test_image.jpg" ] && [ ! -f "$TEST_DATA_DIR/test_image_2.jpg" ]; then
    echo "Error: test image not found in $TEST_DATA_DIR"
    exit 1
fi

if [ ! -f "$TEST_DATA_DIR/test_mask.png" ] && [ ! -f "$TEST_DATA_DIR/test_mask_2.png" ]; then
    echo "Error: test mask not found in $TEST_DATA_DIR"
    exit 1
fi

# Use the first available image and mask
TEST_IMAGE=""
TEST_MASK=""

if [ -f "$TEST_DATA_DIR/test_image_2.jpg" ]; then
    TEST_IMAGE="$TEST_DATA_DIR/test_image_2.jpg"
elif [ -f "$TEST_DATA_DIR/test_image.jpg" ]; then
    TEST_IMAGE="$TEST_DATA_DIR/test_image.jpg"
fi

if [ -f "$TEST_DATA_DIR/test_mask_2.png" ]; then
    TEST_MASK="$TEST_DATA_DIR/test_mask_2.png"
elif [ -f "$TEST_DATA_DIR/test_mask.png" ]; then
    TEST_MASK="$TEST_DATA_DIR/test_mask.png"
fi

curl -X POST "$API_URL" \
    -F "image=@$TEST_IMAGE" \
    -F "mask=@$TEST_MASK" \
    --output sam3d_image_to_mesh_results.zip \
    --progress-bar

if [ $? -eq 0 ]; then
    echo ""
    echo "Success! Results saved to sam3d_image_to_mesh_results.zip"
else
    echo ""
    echo "Error: API call failed"
    exit 1
fi

