#!/bin/bash
# Create output directories for all test services.
# Run from reconstruction/ before first test run: ./scripts/init_test_output_dirs.sh

set -e
DATA="${1:-./data}"
TESTS="$DATA/tests"

mkdir -p "$TESTS/sam2-video-to-masks-test/output"
mkdir -p "$TESTS/sam2-annotate-test/output"
mkdir -p "$TESTS/moge-video-to-depth-test/output"
mkdir -p "$TESTS/moge-image-to-depth-test/output"
mkdir -p "$TESTS/unidepth-video-to-depth-test/output"
mkdir -p "$TESTS/unidepth-image-to-depth-test/output"
mkdir -p "$TESTS/sam3d-image-to-mesh-test/output"
mkdir -p "$TESTS/sam3d-render-debug-test/output"
mkdir -p "$TESTS/foundationpose-video-to-poses-test/output"
mkdir -p "$TESTS/foundationpose-align-mesh-test/output"
mkdir -p "$TESTS/foundationpose-estimate-scale-test/output"
mkdir -p "$TESTS/foundationpose-transform-mesh-test/output"
mkdir -p "$TESTS/foundationpose-simplify-mesh-test/output"
mkdir -p "$TESTS/foundationpose-render-overlay-test/output"
mkdir -p "$TESTS/nlf-video-to-smpl-test/output"
mkdir -p "$TESTS/nlf-render-smpl-overlay-test/output"
mkdir -p "$TESTS/nlf-align-to-depth-test/output"
mkdir -p "$TESTS/nlf-align-depth-to-smpl-test/output"
mkdir -p "$TESTS/nlf-render-smpl-depth-test/output"
mkdir -p "$TESTS/grounding-dino-image-to-object-bboxes-test/output"
mkdir -p "$TESTS/grounding-dino-image-list-to-object-bboxes-test/output"
mkdir -p "$TESTS/grounding-dino-video-to-object-bboxes-test/output"
mkdir -p "$TESTS/foundation-stereo-image-list-to-depth-test/output"
mkdir -p "$TESTS/foundation-stereo-image-to-depth-test/output"

echo "Created test output directories under $TESTS"
