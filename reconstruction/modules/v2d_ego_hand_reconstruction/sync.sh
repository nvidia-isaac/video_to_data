#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="https://github.com/NVIDIA/IsaacTeleop.git"
BRANCH="ego4robo/0.0.1"
DIR="src/postprocessing/egocentric_hand_reconstruction"

echo "Syncing $DIR from $REPO ($BRANCH)..."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git clone --filter=blob:none --sparse --branch "$BRANCH" --depth 1 "$REPO" "$tmp"
cd "$tmp"
git sparse-checkout set "$DIR"
cd - > /dev/null

cp -r "$tmp/$DIR/." "$SCRIPT_DIR/"

echo "Done. Files synced to $SCRIPT_DIR/"
