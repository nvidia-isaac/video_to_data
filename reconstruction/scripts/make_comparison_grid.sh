#!/usr/bin/env bash
# Stack comparison.mp4s for 5 objects into a labelled grid video (1920 wide).
# Rows (top to bottom): dust_brush, airplane, electric_drill_toy, wooden_spatula, yellow_spray
# Columns: DA3 raw | DA3 EKF Smoothed | MoGe raw | MoGe EKF Smoothed
set -e
cd "$(dirname "$0")/.."

DUST_BRUSH="data/objects/dust_brush/sessions/Session_20260310_135759_f50/outputs/comparison.mp4"
AIRPLANE="data/objects/airplane/sessions/Session_20260310_130642_f50/outputs/comparison.mp4"
DRILL="data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/comparison.mp4"
SPATULA="data/objects/wooden_spatula/sessions/Session_20260310_134639_f50/outputs/comparison.mp4"
YELLOW="data/objects/yellow_spray/sessions/Session_20260310_141158_f50/outputs/comparison.mp4"
OUTPUT="data/comparison_grid.mp4"

FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTSIZE=26
LABEL_H=40   # height of label bar added above the stacked rows

# Column centres at 1920px wide (4 equal columns of 480px each)
C1=240
C2=720
C3=1200
C4=1680

ffmpeg -y \
    -i "$DUST_BRUSH" \
    -i "$AIRPLANE" \
    -i "$DRILL" \
    -i "$SPATULA" \
    -i "$YELLOW" \
    -filter_complex "
        [0]scale=1920:-2[r0];
        [1]scale=1920:-2[r1];
        [2]scale=1920:-2[r2];
        [3]scale=1920:-2[r3];
        [4]scale=1920:-2[r4];
        [r0][r1][r2][r3][r4]vstack=inputs=5,
        pad=iw:ih+${LABEL_H}:0:${LABEL_H}:black,
        drawtext=fontfile='${FONT}':text='Depth Anything 3':fontcolor=white:fontsize=${FONTSIZE}:x=${C1}-text_w/2:y=(${LABEL_H}-text_h)/2:box=0,
        drawtext=fontfile='${FONT}':text='Depth Anything 3 (EKF Smoothed)':fontcolor=white:fontsize=${FONTSIZE}:x=${C2}-text_w/2:y=(${LABEL_H}-text_h)/2:box=0,
        drawtext=fontfile='${FONT}':text='MoGe':fontcolor=white:fontsize=${FONTSIZE}:x=${C3}-text_w/2:y=(${LABEL_H}-text_h)/2:box=0,
        drawtext=fontfile='${FONT}':text='MoGe (EKF Smoothed)':fontcolor=white:fontsize=${FONTSIZE}:x=${C4}-text_w/2:y=(${LABEL_H}-text_h)/2:box=0
        [out]
    " \
    -map "[out]" \
    -c:v libx264 -crf 18 -preset fast \
    "$OUTPUT"

echo "Done: $OUTPUT"
