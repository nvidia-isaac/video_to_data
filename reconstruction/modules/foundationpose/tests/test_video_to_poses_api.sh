#!/bin/bash
# Test script for FoundationPose video_to_poses API endpoint
# Usage: ./test_video_to_poses_api.sh

API_URL="http://localhost:8005/process/video_to_poses"
TEST_DATA_DIR="$(dirname "$0")/test_data"

echo "Testing FoundationPose video_to_poses API..."
echo "API URL: $API_URL"
echo "Test data directory: $TEST_DATA_DIR"

echo "Note: This endpoint requires video, mesh, and optionally depth/masks/intrinsics."
echo "Skipping test - requires test data setup..."
exit 0

# Example when test data is available:
# curl -X POST "$API_URL" \
#     -F "video=@$TEST_DATA_DIR/test_video.mp4" \
#     -F "mesh=@$TEST_DATA_DIR/test_mesh.glb" \
#     -F "camera_intrinsics=@$TEST_DATA_DIR/test_intrinsics.json" \
#     --output foundationpose_video_to_poses_results.zip \
#     --progress-bar

