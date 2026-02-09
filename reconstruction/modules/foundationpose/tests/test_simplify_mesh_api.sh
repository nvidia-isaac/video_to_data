#!/bin/bash
# Test script for FoundationPose simplify_mesh API endpoint
# Usage: ./test_simplify_mesh_api.sh

API_URL="http://localhost:8005/process/simplify_mesh"
TEST_DATA_DIR="$(dirname "$0")/test_data"

echo "Testing FoundationPose simplify_mesh API..."
echo "API URL: $API_URL"
echo "Test data directory: $TEST_DATA_DIR"

echo "Note: This endpoint requires input_mesh file."
echo "Skipping test - requires test data setup..."
exit 0

# Example when test data is available:
# curl -X POST "$API_URL" \
#     -F "input_mesh=@$TEST_DATA_DIR/test_mesh.glb" \
#     -F "faces=5000" \
#     -F "factor=0.5" \
#     --output foundationpose_simplify_mesh_results.zip \
#     --progress-bar

