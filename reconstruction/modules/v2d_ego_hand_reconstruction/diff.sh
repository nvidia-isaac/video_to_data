#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
# Show local modifications to vendored content vs upstream IsaacTeleop.
# Usage: ./diff.sh            (summary of changed files)
#        ./diff.sh --full     (full unified diff)
#        ./diff.sh --patch    (git-format patch, apply with: git apply <patch>)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor"
REPO="https://github.com/NVIDIA/IsaacTeleop.git"
BRANCH="main"
DIR="src/postprocessing/egocentric_hand_reconstruction"

if [ ! -d "$VENDOR_DIR" ]; then
    echo "vendor/ not found. Run ./sync.sh first." >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "Fetching upstream ($BRANCH)..." >&2
git clone --filter=blob:none --sparse --branch "$BRANCH" --depth 1 "$REPO" "$tmp/repo" 2>/dev/null
cd "$tmp/repo" && git sparse-checkout set "$DIR" 2>/dev/null && cd - > /dev/null

upstream="$tmp/repo/$DIR"

case "${1:-}" in
    --patch)
        # Produce a git-format patch with paths relative to the IsaacTeleop repo root.
        # Apply in an IsaacTeleop clone with: git apply <patch>
        cp -r "$VENDOR_DIR/." "$tmp/repo/$DIR/"
        cd "$tmp/repo"
        git diff || true
        ;;
    --full)
        diff -ru "$upstream/" "$VENDOR_DIR/" || true
        ;;
    *)
        diff -rq "$upstream/" "$VENDOR_DIR/" && echo "(no differences)" || true
        ;;
esac
