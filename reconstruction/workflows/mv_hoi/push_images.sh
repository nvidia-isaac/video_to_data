#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
# Tag and push all MV HOI Docker images to the registry.
#
# Usage:
#   ./push_images.sh                             # auto-bump patch from latest DB version
#   ./push_images.sh -m "fix OOM"                # auto-bump with message
#   ./push_images.sh 1.2.0                       # explicit version
#   ./push_images.sh 1.2.0 -m "initial release"  # explicit version with message
#
# The version must be a valid semver (X.Y.Z) greater than the latest in the DB.
# After all pushes succeed, the version is recorded in the DB.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="nvcr.io/nvstaging/isaac-amr"

IMAGES=(
    v2d_rosbag
    v2d_mv_calibration
    v2d_mv_preprocess
    v2d_foundation_stereo
    v2d_grounding_dino
    v2d_sam2
    v2d_foundation_pose
    v2d_detectron2
    v2d_sam3d_body
    v2d_mv_postprocess
)

# If $1 looks like semver, consume it as the explicit version.
EXPLICIT_VERSION=""
if [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    EXPLICIT_VERSION="$1"
    shift
fi

MESSAGE=""
while getopts "m:" opt; do
    case $opt in
        m) MESSAGE="$OPTARG" ;;
        *) echo "Usage: $0 [version] [-m \"message\"]" >&2; exit 1 ;;
    esac
done

# Resolve version: explicit arg, or auto-bump patch from latest in DB.
export SCRIPT_DIR
VERSION=$(EXPLICIT_VERSION="$EXPLICIT_VERSION" python3 - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["SCRIPT_DIR"])
from db import init_db, get_latest_version, parse_semver, validate_semver_gt
init_db()
explicit = os.environ.get("EXPLICIT_VERSION", "")
latest = get_latest_version()
if explicit:
    parse_semver(explicit)
    validate_semver_gt(explicit, latest)
    print(explicit)
else:
    if latest is None:
        print("0.1.0")
    else:
        major, minor, patch = parse_semver(latest)
        print(f"{major}.{minor}.{patch + 1}")
PYEOF
)

LATEST=$(python3 - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["SCRIPT_DIR"])
from db import init_db, get_latest_version
init_db()
print(get_latest_version() or "")
PYEOF
)
if [ -n "$LATEST" ]; then
    echo "Latest version: ${LATEST}"
fi
echo "New version:    ${VERSION} ✓"

echo ""
echo "=== Pushing as ${VERSION} ==="
echo "    Message: ${MESSAGE}"
echo ""

for LOCAL in "${IMAGES[@]}"; do
    REMOTE="${REGISTRY}/mv_hoi_${LOCAL#v2d_}"
    echo "--- ${LOCAL} → ${REMOTE} ---"
    docker tag "${LOCAL}" "${REMOTE}:latest"
    docker tag "${LOCAL}" "${REMOTE}:${VERSION}"
    docker push "${REMOTE}:latest"
    docker push "${REMOTE}:${VERSION}"
    echo ""
done

# All pushes succeeded — record the version
python3 - "$VERSION" "$MESSAGE" <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["SCRIPT_DIR"])
from db import init_db, insert_version
init_db()
insert_version(sys.argv[1], sys.argv[2])
print(f"Recorded pipeline version {sys.argv[1]}")
PYEOF

echo "=== Done: pushed all images as :latest and :${VERSION} ==="
