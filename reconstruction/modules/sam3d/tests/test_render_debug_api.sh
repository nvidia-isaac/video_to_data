#!/bin/bash
# Test script for SAM3D render_debug API endpoint
# Usage: ./test_render_debug_api.sh

API_URL="http://localhost:8004/process/render_debug"
TEST_DATA_DIR="$(dirname "$0")/test_data"

echo "Testing SAM3D render_debug API..."
echo "API URL: $API_URL"
echo "Test data directory: $TEST_DATA_DIR"

echo "Note: This test requires pre-generated mesh, transform, and intrinsics files."
echo "You may need to run image_to_mesh first to generate these files."
echo ""
echo "Skipping test - requires mesh output from image_to_mesh..."
exit 0

# Uncomment and update paths when you have the required files:
# curl -X POST "$API_URL" \
#     -F "image=@$TEST_DATA_DIR/test_image.jpg" \
#     -F "mesh=@$TEST_DATA_DIR/output/mesh.glb" \
#     -F "transform=@$TEST_DATA_DIR/output/transform.json" \
#     -F "intrinsics=@$TEST_DATA_DIR/output/intrinsics.json" \
#     --output sam3d_render_debug_results.zip \
#     --progress-bar

