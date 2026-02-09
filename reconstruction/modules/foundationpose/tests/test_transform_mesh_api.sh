#!/bin/bash
# Test script for FoundationPose transform_mesh API endpoint
# Usage: ./test_transform_mesh_api.sh

API_URL="http://localhost:8005/process/transform_mesh"
TEST_DATA_DIR="$(dirname "$0")/test_data"

echo "Testing FoundationPose transform_mesh API..."
echo "API URL: $API_URL"
echo "Test data directory: $TEST_DATA_DIR"

echo "Note: This endpoint requires input_mesh and transform files."
echo "Skipping test - requires test data setup..."
exit 0

# Example when test data is available:
# curl -X POST "$API_URL" \
#     -F "input_mesh=@$TEST_DATA_DIR/test_mesh.glb" \
#     -F "transform=@$TEST_DATA_DIR/test_transform.json" \
#     --output foundationpose_transform_mesh_results.zip \
#     --progress-bar

