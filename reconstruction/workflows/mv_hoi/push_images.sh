#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Tag and push all MV HOI Docker images to the registry.
#
# Usage:
#   ./push_images.sh                             # auto-bump patch from latest registry version
#   ./push_images.sh -m "fix OOM"                # auto-bump with message
#   ./push_images.sh 1.2.0                       # explicit version
#   ./push_images.sh 1.2.0 -m "initial release"  # explicit version with message
#
# The version must be a valid semver (X.Y.Z) greater than the latest in the configured registry.
# After all pushes succeed, the version is cached in the local DB.

set -euo pipefail

: "${V2D_IMAGE_REGISTRY:?Set V2D_IMAGE_REGISTRY to your registry/namespace before publishing}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A leading non-option is the explicit version; Python validates its semver syntax.
EXPLICIT_VERSION=""
if [[ -n "${1:-}" && "${1}" != -* ]]; then
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
shift $((OPTIND - 1))
if [ "$#" -ne 0 ]; then
    echo "Usage: $0 [version] [-m \"message\"]" >&2
    exit 1
fi

VERSION_ARGS=()
if [ -n "$EXPLICIT_VERSION" ]; then
    VERSION_ARGS=(--version "$EXPLICIT_VERSION")
fi
VERSION=$(python3 "$SCRIPT_DIR/orchestration/registry_versions.py" resolve-push "${VERSION_ARGS[@]}")
echo "New version:    ${VERSION} ✓"

echo ""
echo "=== Pushing as ${VERSION} ==="
echo "    Message: ${MESSAGE}"
echo ""

MANAGED_IMAGES=$(python3 "$SCRIPT_DIR/orchestration/registry_versions.py" managed-images)
while read -r LOCAL REMOTE; do
    echo "--- ${LOCAL} → ${REMOTE} ---"
    docker tag "${LOCAL}" "${REMOTE}:latest"
    docker tag "${LOCAL}" "${REMOTE}:${VERSION}"
    docker push "${REMOTE}:latest"
    docker push "${REMOTE}:${VERSION}"
    echo ""
done <<< "$MANAGED_IMAGES"

# All pushes succeeded — cache the remote version locally.
export SCRIPT_DIR
python3 - "$VERSION" "$MESSAGE" <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["SCRIPT_DIR"])
from orchestration.db import ensure_version_cached, init_db
init_db()
ensure_version_cached(sys.argv[1], sys.argv[2])
print(f"Cached pipeline version {sys.argv[1]}")
PYEOF

echo "=== Done: pushed all images as :latest and :${VERSION} ==="
