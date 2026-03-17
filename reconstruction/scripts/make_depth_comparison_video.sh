#!/usr/bin/env bash
# Create a side-by-side comparison video of two depth folders.
# Usage: ./scripts/make_depth_comparison_video.sh [left_dir] [right_dir] [output] [fps]

set -euo pipefail

LEFT_DIR="${1:-data/outputs/yellow_spray_updated/depth}"
RIGHT_DIR="${2:-data/outputs/yellow_spray_updated/depth_aligned}"
OUTPUT="${3:-data/outputs/yellow_spray_updated/depth_comparison.mp4}"
FPS="${4:-30}"

mkdir -p "$(dirname "$OUTPUT")"

ffmpeg -y \
  -framerate "$FPS" -i "$LEFT_DIR/%06d.png" \
  -framerate "$FPS" -i "$RIGHT_DIR/%06d.png" \
  -filter_complex "
    [0:v]drawtext=text='depth':fontsize=28:fontcolor=white:x=10:y=10[left];
    [1:v]drawtext=text='depth_aligned':fontsize=28:fontcolor=white:x=10:y=10[right];
    [left][right]hstack[out]
  " \
  -map "[out]" \
  -c:v libx264 -pix_fmt yuv420p \
  "$OUTPUT"

echo "Saved: $OUTPUT"
