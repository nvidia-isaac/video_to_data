import json
import os
from typing import Literal

from v2d.pipelines.extract_images import extract_images
from v2d.pipelines.frames_to_video import frames_to_video
from v2d.pipelines.stitch_videos import stitch_videos
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.mesh.docker.run_mesh_simplify import run_mesh_simplify
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_correct_depth_scale import run_correct_depth_scale
from v2d.depth.lib.align_depth_sequence import align_depth_sequence
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.pipelines.extract_images import extract_images
from v2d.pipelines.frames_to_video import frames_to_video
from v2d.pipelines.stitch_videos import stitch_videos
from v2d.depth_anything.docker.run_video_to_depth import run_video_to_depth as run_video_to_depth_depth_anything
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_video_to_depth_moge


def _dino_detections_to_sam2_prompts(
    detections_path: str,
    frame_index: int,
    object_id: int,
) -> Sam2Prompts:
    with open(detections_path) as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError(f"Grounding DINO found no detections in {detections_path}")
    top = detections[0]
    box = BoundingBox.from_dict(top["box"])
    return Sam2Prompts(prompts=[
        Sam2Prompt(frame_index=frame_index, object_id=object_id, box=box)
    ])

name = "airplane"
prompt = "a toy airplane"
session = "Session_20260310_130642"
reference_frame = 40
reference_frame_name = f"{reference_frame:06d}"

mesh_path = f"data/objects/{name}/mesh/textured_mesh.obj"
video_path = f"data/objects/{name}/sessions/{session}/{session}_color.mp4"

output_dir = f"data/objects/{name}/sessions/{session}/outputs"

# 1 - Extract frames
images_dir = os.path.join(output_dir, "frames")
if False:
    extract_images(video_path, images_dir)


# 2 - Run DINO detection
dino_detections_path = os.path.join(output_dir, "dino_detections.json")
if False:
    run_image_to_object_bboxes(
        image_path=f"{images_dir}/{reference_frame_name}.png",
        output_path=dino_detections_path,
        prompt=prompt,
        model_dir="data/weights/grounding_dino",
    )

# 3 - Run SAM2 segmentation
if False:
    sam2_prompts_path = os.path.join(output_dir, "sam2_prompts.json")
    prompts = _dino_detections_to_sam2_prompts(dino_detections_path, reference_frame, 1)
    with open(sam2_prompts_path, "w") as f:
        json.dump(prompts.to_dict(), f, indent=2)
    run_video_to_masks(
        video_path=video_path,
        prompts_path=sam2_prompts_path,
        masks_dir=os.path.join(output_dir, "masks"),
        weights_dir="data/weights/sam2",
    )

vipe_intrinsics_path = os.path.join(output_dir, "intrinsics_vipe.json")
manual_intrinsics_path = os.path.join(output_dir, "intrinsics_manual.json")

# 4 - Run Depth estimation (depth anything)
depth_anything_dir = os.path.join(output_dir, "depth_anything")
depth_dir_depth_anything = os.path.join(depth_anything_dir, "depth")
intrinsics_dir_depth_anything = os.path.join(depth_anything_dir, "intrinsics")
intrinsics_depth_anything_stable_path = os.path.join(depth_anything_dir, "intrinsics_stable.json")
    
moge_intrinsics_stable_path = os.path.join(output_dir, "moge", "intrinsics_stable.json")
if False:
    run_video_to_depth_depth_anything(
        video_path=video_path,
        depth_folder=depth_dir_depth_anything,
        intrinsics_folder=intrinsics_dir_depth_anything,
        weights_path="data/weights/depth_anything_metric",
        dev=True,
        input_intrinsics_path=moge_intrinsics_stable_path,
    )

    stabilize_intrinsics(
        intrinsics_folder=intrinsics_dir_depth_anything,
        output_path=intrinsics_depth_anything_stable_path,
    )

depth_anything_vipe_dir = os.path.join(output_dir, "depth_anything_vipe")
depth_anything_vipe_depth_dir = os.path.join(depth_anything_vipe_dir, "depth")
depth_anything_vipe_intrinsics_dir = os.path.join(depth_anything_vipe_dir, "intrinsics")
depth_anything_vipe_intrinsics_stable_path = os.path.join(depth_anything_vipe_dir, "intrinsics_stable.json")
if False:
    run_video_to_depth_depth_anything(
        video_path=video_path,
        depth_folder=depth_anything_vipe_depth_dir,
        intrinsics_folder=depth_anything_vipe_intrinsics_dir,
        weights_path="data/weights/depth_anything_metric",
        dev=True,
        input_intrinsics_path=vipe_intrinsics_path,
    )

    stabilize_intrinsics(
        intrinsics_folder=depth_anything_vipe_intrinsics_dir,
        output_path=depth_anything_vipe_intrinsics_stable_path,
    )

# 5 - Run Depth estimation (moge)
moge_dir = os.path.join(output_dir, "moge")
moge_depth_dir = os.path.join(moge_dir, "depth")
moge_intrinsics_dir = os.path.join(moge_dir, "intrinsics")
moge_intrinsics_stable_path = os.path.join(moge_dir, "intrinsics_stable.json")
if True:
    run_video_to_depth_moge(
        video_path=video_path,
        depth_folder=moge_depth_dir,
        intrinsics_folder=moge_intrinsics_dir,
        weights_path="data/weights/moge",
        dev=True
    )

    stabilize_intrinsics(
        intrinsics_folder=moge_intrinsics_dir,
        output_path=moge_intrinsics_stable_path,
    )

# Run foundation pose tracking
foundation_pose_dir = os.path.join(output_dir, "foundation_pose")

poses_dir = os.path.join(foundation_pose_dir, "poses_moge")
fp_depth_dir = moge_depth_dir
fp_intrinsics_path = moge_intrinsics_stable_path

if True:
    run_video_to_poses(
        video_path=video_path,
        depth_folder=fp_depth_dir,
        masks_folder=os.path.join(output_dir, "masks", "1"),
        camera_intrinsics_path=fp_intrinsics_path,
        mesh_path=mesh_path,
        poses_dir=poses_dir,
        weights_dir="data/weights/foundation_pose",
        reference_frame=reference_frame
    )


# Generate video
renders_dir = os.path.join(foundation_pose_dir, "renders_moge")
if True:
    run_render_poses(
        mesh_path=mesh_path,
        poses_dir=poses_dir,
        frames_dir=os.path.join(output_dir, "frames"),
        intrinsics_path=fp_intrinsics_path,
        output_dir=renders_dir
    )

# Make video
output_video_path = os.path.join(output_dir, "renders_moge.mp4")
if True:
    frames_to_video(renders_dir, output_video_path)


# stitch_videos(
#     [
#         os.path.join(output_dir, "renders_moge.mp4"),
#         os.path.join(output_dir, "renders_moge_manual.mp4"),
#         os.path.join(output_dir, "renders_da.mp4"),
#         os.path.join(output_dir, "renders_da_vipe.mp4"),
#     ],
#     os.path.join(output_dir, "renders_all.mp4"),
# )