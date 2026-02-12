#!/bin/bash

# run_pipeline.sh
# This script executes the video-to-policy reconstruction pipeline using Docker containers.
# It operates on a job directory provided as an argument.

set -e

DIRECTORY=$1

echo "Step 0: Extracting frame 0 from video for SAM3D..."
docker compose run ffmpeg \
    -i "$DIRECTORY/video.mp4" -frames:v 1 "$DIRECTORY/frame_0.jpg" -y

# echo "Step 1 & 2: Running SAM2 to compute masks..."
docker compose run sam2-video-to-masks \
    --video_path "$DIRECTORY/video.mp4" \
    --prompts_path "$DIRECTORY/prompts.json" \
    --masks_dir "$DIRECTORY/masks"

# echo "Step 3: Running MoGE to compute depth..."
docker compose run moge-video-to-depth \
    --video_path "$DIRECTORY/video.mp4" \
    --depth_folder "$DIRECTORY/depth" \
    --intrinsics_folder "$DIRECTORY/intrinsics"


# echo "Step 4: Running SAM3D to get mesh..."
docker compose run sam3d-image-to-mesh \
    --image_path "$DIRECTORY/frame_0.jpg" \
    --mask_path "$DIRECTORY/masks/1/000000.png" \
    --mesh_path "$DIRECTORY/mesh.glb" \
    --transform_path "$DIRECTORY/transform.json" \
    --intrinsics_path "$DIRECTORY/intrinsics.json"

# # echo "Step 5: Simplifying mesh..."
docker compose run foundationpose-simplify-mesh \
    --input-mesh "$DIRECTORY/mesh.glb" \
    --output-mesh "$DIRECTORY/mesh_simplified.glb" \
    --factor 0.1

# # echo "Step 6: Aligning mesh scale..."
docker compose run foundationpose-align-mesh \
    --mesh "$DIRECTORY/mesh_simplified.glb" \
    --depth "$DIRECTORY/depth/000000.png" \
    --mask "$DIRECTORY/masks/1/000000.png" \
    --intrinsics "$DIRECTORY/intrinsics/000000.json" \
    --transform "$DIRECTORY/transform.json" \
    --output-transform "$DIRECTORY/transform_aligned.json"

# # echo "Step 7: Transforming mesh..."
docker compose run foundationpose-transform-mesh \
    --input-mesh "$DIRECTORY/mesh_simplified.glb" \
    --output-mesh "$DIRECTORY/mesh_input.glb" \
    --transform "$DIRECTORY/transform_aligned.json"

# echo "Step 8: Running FoundationPose to track poses..."
docker compose run foundationpose-video-to-poses \
    --video_path "$DIRECTORY/video.mp4" \
    --depth_folder "$DIRECTORY/depth" \
    --masks_folder "$DIRECTORY/masks/1" \
    --camera_intrinsics_path "$DIRECTORY/intrinsics/000000.json" \
    --mesh_path "$DIRECTORY/mesh_input.glb" \
    --poses_dir "$DIRECTORY/poses" \
    --reference_frame 0

# echo "Step 9: Rendering FoundationPose overlay..."
docker compose run foundationpose-render-overlay \
    --video_path "$DIRECTORY/video.mp4" \
    --poses_dir "$DIRECTORY/poses" \
    --mesh_path "$DIRECTORY/mesh_input.glb" \
    --camera_intrinsics_path "$DIRECTORY/intrinsics/000000.json" \
    --output_dir "$DIRECTORY/render_foundationpose"

# echo "Step 10: Running NLF to compute SMPL parameters..."
# # Assuming object_id 0 is the person in SAM2 masks
docker compose run nlf-video-to-smpl \
    --video_path "$DIRECTORY/video.mp4" \
    --masks_dir "$DIRECTORY/masks/0" \
    --intrinsics_path "$DIRECTORY/intrinsics/000000.json" \
    --gender "neutral" \
    --output_path "$DIRECTORY/smpl_results.h5"

# echo "Step 11: Rendering NLF SMPL overlay..."
docker compose run nlf-render-smpl-overlay \
    --video_path "$DIRECTORY/video.mp4" \
    --smpl_params_path "$DIRECTORY/smpl_results.h5" \
    --intrinsics_path "$DIRECTORY/intrinsics/000000.json" \
    --output_dir "$DIRECTORY/render_nlf"

# echo "Pipeline completed successfully! Results are in $OUTPUT_DIR"
