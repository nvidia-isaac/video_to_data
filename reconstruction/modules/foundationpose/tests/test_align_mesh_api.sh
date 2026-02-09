#!/bin/bash
# Test script for FoundationPose align_mesh API endpoint
# Usage: ./test_align_mesh_api.sh

API_URL="http://localhost:8005/process/align_mesh"
TEST_DATA_DIR="$(dirname "$0")/test_data"

echo "Testing FoundationPose align_mesh API..."
echo "API URL: $API_URL"
echo "Test data directory: $TEST_DATA_DIR"

echo "Note: This endpoint requires mesh, depth, mask, intrinsics, and transform files."
echo "Skipping test - requires test data setup..."
exit 0

# Example when test data is available:
# curl -X POST "$API_URL" \
#     -F "mesh=@$TEST_DATA_DIR/test_mesh.glb" \
#     -F "depth=@$TEST_DATA_DIR/test_depth.png" \
#     -F "mask=@$TEST_DATA_DIR/test_mask.png" \
#     -F "intrinsics=@$TEST_DATA_DIR/test_intrinsics.json" \
#     -F "transform=@$TEST_DATA_DIR/test_transform.json" \
#     --output foundationpose_align_mesh_results.zip \
#     --progress-bar

