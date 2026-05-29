#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="https://github.com/NVIDIA/IsaacTeleop.git"
BRANCH="main"
DIR="src/postprocessing/egocentric_hand_reconstruction"

VENDOR_DIR="$SCRIPT_DIR/vendor"

echo "Syncing $DIR from $REPO ($BRANCH) into vendor/..."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git clone --filter=blob:none --sparse --branch "$BRANCH" --depth 1 "$REPO" "$tmp"
cd "$tmp"
git sparse-checkout set "$DIR"
cd - > /dev/null

rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR"
cp -r "$tmp/$DIR/." "$VENDOR_DIR/"

echo "Done. Files synced to $VENDOR_DIR/"
