#!/bin/bash

# docker compose --profile exec run moge-video-to-depth \
#   --video_path /data/test_video.mp4 \
#   --depth_folder /data/test_depth \
#   --intrinsics_folder /data/test_intrinsics

# docker compose --profile exec run sam2-video-to-masks \
#   --video_path /data/test_video.mp4 \
#   --prompts_path /data/test_prompts.json \
#   --masks_dir /data/test_masks

# # extract first image using ffmpeg
# ffmpeg -i /data/test_video.mp4 -ss 00:00:00 -vframes 1 /data/test_image.jpg

# docker compose --profile exec run sam3d-image-to-mesh \
#   --image_path /data/test_image.jpg \
#   --mask_path /data/test_masks/0/000000.png \
#   --mesh_path /data/test_mesh.glb \
#   --transform_path /data/test_transform.json \
#   --intrinsics_path /data/test_intrinsics.json


# docker compose --profile exec run foundationpose-simplify-mesh \
#   --input-mesh /data/test_mesh.glb \
#   --output-mesh /data/test_simplified_mesh.glb \
#   --factor 0.1


# docker compose --profile exec run foundationpose-align-mesh \
#   --mesh /data/test_simplified_mesh.glb \
#   --depth /data/test_depth/000000.png \
#   --mask /data/test_masks/0/000000.png \
#   --intrinsics /data/test_intrinsics.json \
#   --transform /data/test_transform.json \
#   --output-transform /data/test_aligned_transform.json


# docker compose --profile exec run foundationpose-transform-mesh \
#   --input-mesh /data/test_simplified_mesh.glb \
#   --output-mesh /data/test_transformed_mesh.glb \
#   --transform /data/test_aligned_transform.json


# docker compose --profile exec run foundationpose-video-to-poses \
#   --video_path /data/test_video.mp4 \
#   --depth_folder /data/test_depth \
#   --masks_folder /data/test_masks/0 \
#   --camera_intrinsics_path /data/test_intrinsics.json \
#   --mesh_path /data/test_transformed_mesh.glb \
#   --poses_dir /data/test_poses \
#   --reference_frame 0 \
#   --debug_dir /data/test_foundationpose_debug

# convert image folder to mp4 using ffmpeg
# ffmpeg -framerate 30 -i data/test_foundationpose_debug/%06d.png -c:v libx264 -pix_fmt yuv420p data/test_video_debug.mp4