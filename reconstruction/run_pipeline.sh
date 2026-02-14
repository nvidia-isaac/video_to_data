#!/bin/bash

# run_pipeline.sh
# This script executes the video-to-policy reconstruction pipeline using Docker containers.
# It operates on a job directory provided as an argument.

set -e

DIRECTORY=$1
REFERENCE_FRAME=$2
if [ -z "$REFERENCE_FRAME" ]; then
    REFERENCE_FRAME=0
fi
FORMATTED_FRAME=$(printf "%06d" "$REFERENCE_FRAME")
echo "$FORMATTED_FRAME"
# echo "Step 0: Extracting frame $REFERENCE_FRAME from video for SAM3D..."
docker compose run ffmpeg \
    -i "$DIRECTORY/video.mp4" \
    -frames:v 1 \
    -vf "select=eq(n\,$REFERENCE_FRAME)"\
    "$DIRECTORY/frame_${REFERENCE_FRAME}.jpg" \
    -y

docker compose run sam2-video-to-masks \
    --video_path "$DIRECTORY/video.mp4" \
    --prompts_path "$DIRECTORY/prompts.json" \
    --masks_dir "$DIRECTORY/masks"

docker compose run unidepth-video-to-depth \
    --video_path "$DIRECTORY/video.mp4" \
    --depth_folder "$DIRECTORY/depth" \
    --intrinsics_folder "$DIRECTORY/intrinsics"

docker compose run nlf-video-to-smpl \
    --video_path "$DIRECTORY/video.mp4" \
    --masks_dir "$DIRECTORY/masks/0" \
    --intrinsics_path "$DIRECTORY/intrinsics/${FORMATTED_FRAME}.json" \
    --gender "neutral" \
    --output_path "$DIRECTORY/smpl_results.h5"

docker compose run nlf-render-smpl-depth \
    --smpl_params_path "$DIRECTORY/smpl_results.h5" \
    --intrinsics_path "$DIRECTORY/intrinsics/${FORMATTED_FRAME}.json" \
    --output_depth_folder "$DIRECTORY/smpl_depth" \
    --output_mask_folder "$DIRECTORY/smpl_depth_masks"

docker compose run nlf-align-depth-to-smpl \
    --depth_folder "$DIRECTORY/depth" \
    --smpl_depth_folder "$DIRECTORY/smpl_depth" \
    --output_depth_folder "$DIRECTORY/smpl_depth_aligned" \
    --masks_folder "$DIRECTORY/masks/0" \
    --smpl_masks_folder "$DIRECTORY/smpl_depth_masks"


docker compose run sam3d-image-to-mesh \
    --image_path "$DIRECTORY/frame_${REFERENCE_FRAME}.jpg" \
    --mask_path "$DIRECTORY/masks/1/${FORMATTED_FRAME}.png" \
    --mesh_path "$DIRECTORY/mesh.glb" \
    --transform_path "$DIRECTORY/transform.json" \
    --intrinsics_path "$DIRECTORY/intrinsics.json" \
    --with_layout_postprocess

docker compose run foundationpose-simplify-mesh \
    --input-mesh "$DIRECTORY/mesh.glb" \
    --output-mesh "$DIRECTORY/mesh_simplified.glb" \
    --factor 0.15

docker compose run foundationpose-align-mesh \
    --mesh "$DIRECTORY/mesh_simplified.glb" \
    --depth "$DIRECTORY/smpl_depth_aligned/${FORMATTED_FRAME}.png" \
    --mask "$DIRECTORY/masks/1/${FORMATTED_FRAME}.png" \
    --intrinsics "$DIRECTORY/intrinsics/${FORMATTED_FRAME}.json" \
    --transform "$DIRECTORY/transform.json" \
    --output-transform "$DIRECTORY/transform_aligned.json"

docker compose run foundationpose-estimate-scale \
    --mesh "$DIRECTORY/mesh_simplified.glb" \
    --rgb "$DIRECTORY/frame_${REFERENCE_FRAME}.jpg" \
    --depth "$DIRECTORY/smpl_depth_aligned/${FORMATTED_FRAME}.png" \
    --mask "$DIRECTORY/masks/1/${FORMATTED_FRAME}.png" \
    --intrinsics "$DIRECTORY/intrinsics/${FORMATTED_FRAME}.json" \
    --transform "$DIRECTORY/transform_aligned.json" \
    --output-transform "$DIRECTORY/transform_aligned_fp.json" \
    --num-levels 3 \
    --num-samples-per-level 10 \
    --level-size 3.0

docker compose run foundationpose-transform-mesh \
    --input-mesh "$DIRECTORY/mesh_simplified.glb" \
    --output-mesh "$DIRECTORY/mesh_input.glb" \
    --transform "$DIRECTORY/transform_aligned.json"

docker compose run foundationpose-video-to-poses \
    --video_path "$DIRECTORY/video.mp4" \
    --depth_folder "$DIRECTORY/smpl_depth_aligned" \
    --masks_folder "$DIRECTORY/masks/1" \
    --camera_intrinsics_path "$DIRECTORY/intrinsics/${FORMATTED_FRAME}.json" \
    --mesh_path "$DIRECTORY/mesh_input.glb" \
    --poses_dir "$DIRECTORY/poses" \
    --reference_frame "$REFERENCE_FRAME" \
    --debug_dir "$DIRECTORY/foundationpose_debug"

docker compose run foundationpose-render-overlay \
    --video_path "$DIRECTORY/video.mp4" \
    --poses_dir "$DIRECTORY/poses" \
    --mesh_path "$DIRECTORY/mesh_input.glb" \
    --camera_intrinsics_path "$DIRECTORY/intrinsics/${FORMATTED_FRAME}.json" \
    --output_dir "$DIRECTORY/render_foundationpose"



docker compose run --rm -p 8080:8080 viz --dir "$DIRECTORY"

# # echo "Pipeline completed successfully! Results are in $OUTPUT_DIR"
