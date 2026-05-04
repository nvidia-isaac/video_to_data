"""
End-to-end ego hand + object reconstruction pipeline from a single video.

Given an MP4 and a text prompt describing the held object, this script runs:
  1.  Frame extraction
  2.  MoGe monocular depth + intrinsics
  3.  Grounding DINO object detection (reference frame only)
  4.  SAM2 object mask tracking (full video)
  5.  SAM3D textured mesh generation (reference frame + depth)
  6.  FoundationPose scale estimation → scaled mesh
  7.  FoundationPose 6-DOF tracking
  8.  EKF smoothing of object poses
  9.  Ego hand reconstruction (ViPE + Dyn-HaMR)
 10.  Hand/object depth alignment (align_world_results)

Output directory layout:
  <output_dir>/
  ├── frames/                    # Extracted video frames
  ├── depth/                     # MoGe depth PNGs
  ├── intrinsics/                # Per-frame intrinsics JSONs
  ├── intrinsics_stable.json     # Temporally stabilised intrinsics
  ├── dino_detections.json       # Grounding DINO bboxes (frame 0)
  ├── sam2_prompts.json          # SAM2 prompt file
  ├── masks/1/                   # Per-frame object masks
  ├── mesh/
  │   ├── textured_mesh.obj      # SAM3D textured mesh
  │   ├── mesh_transform.json    # SAM3D scale/pose transform
  │   └── mesh_intrinsics.json   # SAM3D-estimated intrinsics
  ├── mesh_scaled.obj            # Depth-aligned, scale-corrected mesh
  ├── scale.json                 # Estimated mesh scale factor
  ├── poses/                     # Raw FoundationPose per-frame JSONs
  ├── poses_smoothed/            # EKF-smoothed per-frame pose JSONs
  ├── hand_reconstruction/       # Dyn-HaMR + ViPE outputs
  │   ├── MANO_RIGHT.pkl         # (auto-copied by run_reconstruction)
  │   ├── BMC/                   # (auto-copied by run_reconstruction)
  │   └── logs/                  # Dyn-HaMR logs (world_results.npz here)
  └── world_results_aligned.npz  # Final depth-aligned hand + object poses

Usage:
    python modules/v2d_pipelines/run_v2d_ego_e2e.py \\
        --video_path data/my_video.mp4 \\
        --prompt "blue cup" \\
        --output_dir data/outputs/my_video \\
        --moge_weights             data/weights/moge \\
        --grounding_dino_weights   data/weights/grounding_dino \\
        --sam2_weights             data/weights/sam2 \\
        --sam3d_weights            data/weights/sam3d \\
        --foundation_pose_weights  data/weights/foundation_pose \\
        --hand_reconstruction_weights data/weights/hand

Run from reconstruction/.
"""

import argparse
import glob
import json
import os

from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.utils import extract_images
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.hand_alignment.docker.run_align_world_results import run_align_world_results
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d_ego_hand_reconstruction.docker.run_reconstruction import run_reconstruction


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end ego hand + object reconstruction from a single video."
    )
    p.add_argument("--video_path",   required=True,
                   help="Input MP4 video.")
    p.add_argument("--prompt",       required=True,
                   help="Grounding DINO text prompt for the held object (e.g. 'blue cup').")
    p.add_argument("--output_dir",   required=True,
                   help="Root output directory.")

    # Weight paths (all default to data/weights/<model> relative to cwd)
    p.add_argument("--moge_weights",            default="data/weights/moge",
                   help="MoGe model weights directory. (default: data/weights/moge)")
    p.add_argument("--grounding_dino_weights",  default="data/weights/grounding_dino",
                   help="Grounding DINO weights directory. (default: data/weights/grounding_dino)")
    p.add_argument("--sam2_weights",            default="data/weights/sam2",
                   help="SAM2 weights directory. (default: data/weights/sam2)")
    p.add_argument("--sam3d_weights",           default="data/weights/sam3d",
                   help="SAM3D weights directory. (default: data/weights/sam3d)")
    p.add_argument("--foundation_pose_weights", default="data/weights/foundation_pose",
                   help="FoundationPose weights directory. (default: data/weights/foundation_pose)")
    p.add_argument("--hand_reconstruction_weights", default="data/weights/hand",
                   help="Ego hand reconstruction weights directory containing MANO_RIGHT.pkl "
                        "and BMC/. (default: data/weights/hand)")
    p.add_argument("--mano_weights", default=None,
                   help="MANO model directory for hand alignment. "
                        "(default: --hand_reconstruction_weights)")

    # Optional tuning
    p.add_argument("--reference_frame", type=int, default=0,
                   help="Frame used for DINO detection, SAM3D, and FP registration (default: 0).")
    p.add_argument("--smooth_sigma",    type=float, default=5.0,
                   help="Gaussian sigma (frames) for hand translation smoothing (default: 5.0).")
    p.add_argument("--dev", action="store_true",
                   help="Mount local module source into containers (live-edit mode).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


def _step(label: str, done: bool) -> bool:
    if done:
        print(f"  [skip] {label}")
        return True
    print(f"  [run ] {label}")
    return False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_v2d_ego_e2e(
    video_path: str,
    prompt: str,
    output_dir: str,
    moge_weights: str,
    grounding_dino_weights: str,
    sam2_weights: str,
    sam3d_weights: str,
    foundation_pose_weights: str,
    hand_reconstruction_weights: str,
    mano_weights: str | None = None,
    reference_frame: int = 0,
    smooth_sigma: float = 5.0,
    dev: bool = False,
) -> None:
    video_path  = os.path.abspath(video_path)
    output_dir  = os.path.abspath(output_dir)
    mano_weights = os.path.abspath(mano_weights or hand_reconstruction_weights)
    os.makedirs(output_dir, exist_ok=True)

    OBJECT_ID = 1

    # -- Output paths --------------------------------------------------------
    frames_dir          = f"{output_dir}/frames"
    depth_dir           = f"{output_dir}/depth"
    intrinsics_dir      = f"{output_dir}/intrinsics"
    intrinsics_stable   = f"{output_dir}/intrinsics_stable.json"
    dino_detections     = f"{output_dir}/dino_detections.json"
    sam2_prompts        = f"{output_dir}/sam2_prompts.json"
    masks_dir           = f"{output_dir}/masks"
    mesh_dir            = f"{output_dir}/mesh"
    mesh_path           = f"{mesh_dir}/textured_mesh.obj"
    mesh_transform      = f"{mesh_dir}/mesh_transform.json"
    mesh_intrinsics     = f"{mesh_dir}/mesh_intrinsics.json"
    mesh_scaled         = f"{output_dir}/mesh_scaled.obj"
    scale_path          = f"{output_dir}/scale.json"
    poses_dir           = f"{output_dir}/poses"
    poses_smooth_dir    = f"{output_dir}/poses_smoothed"
    hand_recon_dir      = f"{output_dir}/hand_reconstruction"
    world_aligned       = f"{output_dir}/world_results_aligned.npz"

    ref_rgb    = f"{frames_dir}/{reference_frame:06d}.png"
    ref_depth  = f"{depth_dir}/{reference_frame:06d}.png"
    ref_mask   = f"{masks_dir}/{OBJECT_ID}/{reference_frame:06d}.png"

    print(f"\n{'='*60}")
    print(f"  video : {os.path.basename(video_path)}")
    print(f"  prompt: {prompt!r}")
    print(f"  output: {output_dir}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Step 1: Extract frames
    # -----------------------------------------------------------------------
    if not _step("Extract frames", _has_files(frames_dir)):
        extract_images(video_path, frames_dir)

    # -----------------------------------------------------------------------
    # Step 2: MoGe depth
    # -----------------------------------------------------------------------
    if not _step("MoGe depth + intrinsics", _has_files(depth_dir)):
        run_moge_depth(
            video_path        = video_path,
            depth_folder      = depth_dir,
            intrinsics_folder = intrinsics_dir,
            weights_path      = moge_weights,
            dev               = dev,
        )

    if not _step("Stabilise intrinsics", os.path.exists(intrinsics_stable)):
        stabilize_intrinsics(intrinsics_dir, intrinsics_stable)

    # -----------------------------------------------------------------------
    # Step 3: Grounding DINO → SAM2 prompts
    # -----------------------------------------------------------------------
    if not _step("Grounding DINO (frame 0)", os.path.exists(dino_detections)):
        run_image_to_object_bboxes(
            image_path  = ref_rgb,
            output_path = dino_detections,
            prompt      = prompt,
            model_dir   = grounding_dino_weights,
            dev         = dev,
        )

    if not os.path.exists(sam2_prompts):
        with open(dino_detections) as f:
            detections = json.load(f)
        if not detections:
            raise RuntimeError(
                f"Grounding DINO found no objects — check prompt: {prompt!r}"
            )
        box = BoundingBox.from_dict(detections[0]["box"])
        prompts = Sam2Prompts(
            prompts=[Sam2Prompt(frame_index=reference_frame, object_id=OBJECT_ID, box=box)]
        )
        with open(sam2_prompts, "w") as f:
            json.dump(prompts.to_dict(), f, indent=2)

    # -----------------------------------------------------------------------
    # Step 4: SAM2 object mask tracking
    # -----------------------------------------------------------------------
    if not _step("SAM2 mask tracking", _has_files(f"{masks_dir}/{OBJECT_ID}")):
        run_video_to_masks(
            video_path  = video_path,
            prompts_path= sam2_prompts,
            masks_dir   = masks_dir,
            weights_dir = sam2_weights,
            dev         = dev,
        )

    # -----------------------------------------------------------------------
    # Step 5: SAM3D textured mesh generation
    # -----------------------------------------------------------------------
    os.makedirs(mesh_dir, exist_ok=True)
    if not _step("SAM3D mesh generation", os.path.exists(mesh_path)):
        run_image_to_mesh(
            image_path            = ref_rgb,
            mask_path             = ref_mask,
            mesh_path             = mesh_path,
            transform_path        = mesh_transform,
            intrinsics_path       = mesh_intrinsics,
            weights_dir           = sam3d_weights,
            with_texture_baking   = True,
            with_mesh_postprocess = True,
            depth_path            = ref_depth,
            depth_intrinsics_path = intrinsics_stable,
            depth_mask_path       = ref_mask,
            dev                   = dev,
        )

    # -----------------------------------------------------------------------
    # Step 6: FoundationPose scale estimation
    # -----------------------------------------------------------------------
    if not _step("Scale estimation", os.path.exists(mesh_scaled)):
        run_estimate_mesh_scale(
            mesh_path             = mesh_path,
            rgb_path              = ref_rgb,
            depth_path            = ref_depth,
            mask_path             = ref_mask,
            intrinsics_path       = intrinsics_stable,
            weights_dir           = foundation_pose_weights,
            scale_path            = scale_path,
            rescaled_mesh_path    = mesh_scaled,
            lo                    = 0.5,
            hi                    = 2.0,
            n_samples             = 9,
            n_levels              = 4,
            iou_weight            = 1.0,
            depth_weight          = 1.0,
            registration_iterations = 5,
            dev                   = dev,
        )

    # -----------------------------------------------------------------------
    # Step 7: FoundationPose 6-DOF tracking
    # -----------------------------------------------------------------------
    if not _step("FoundationPose tracking", _has_files(poses_dir)):
        run_video_to_poses(
            video_path            = video_path,
            depth_folder          = depth_dir,
            masks_folder          = f"{masks_dir}/{OBJECT_ID}",
            camera_intrinsics_path= intrinsics_stable,
            mesh_path             = mesh_scaled,
            poses_dir             = poses_dir,
            weights_dir           = foundation_pose_weights,
            reference_frame       = reference_frame,
            mask_depth            = True,
            dev                   = dev,
        )

    # -----------------------------------------------------------------------
    # Step 8: EKF smoothing
    # -----------------------------------------------------------------------
    if not _step("EKF pose smoothing", _has_files(poses_smooth_dir)):
        run_ekf_smoothing(
            poses_dir             = poses_dir,
            mesh_path             = mesh_scaled,
            intrinsics_path       = intrinsics_stable,
            weights_dir           = foundation_pose_weights,
            output_dir            = poses_smooth_dir,
            masks_folder          = f"{masks_dir}/{OBJECT_ID}",
            process_noise_xy      = 0.01,
            process_noise_z       = 0.01,
            process_noise_r       = 0.02,
            measurement_noise_xy  = 0.01,
            measurement_noise_z   = 0.04,
            measurement_noise_r   = 0.02,
            dev                   = dev,
        )

    # -----------------------------------------------------------------------
    # Step 9: Ego hand reconstruction (ViPE + Dyn-HaMR)
    # -----------------------------------------------------------------------
    _world_results = glob.glob(
        f"{hand_recon_dir}/logs/**/*_world_results.npz", recursive=True
    )
    if not _step("Ego hand reconstruction (ViPE + Dyn-HaMR)",
                 len(_world_results) == 1):
        run_reconstruction(
            video_input = video_path,
            output_dir  = hand_recon_dir,
            weights_dir = hand_reconstruction_weights,
        )
        _world_results = glob.glob(
            f"{hand_recon_dir}/logs/**/*_world_results.npz", recursive=True
        )
        if len(_world_results) != 1:
            raise RuntimeError(
                f"Expected 1 world_results.npz after reconstruction, "
                f"found: {_world_results}"
            )

    world_results_npz = _world_results[0]
    print(f"  world_results: {world_results_npz}")

    # -----------------------------------------------------------------------
    # Step 10: Hand/object depth alignment
    # -----------------------------------------------------------------------
    if not _step("Hand/object depth alignment", os.path.exists(world_aligned)):
        run_align_world_results(
            input_hand_data  = world_results_npz,
            depth_dir        = depth_dir,
            depth_intrinsics = intrinsics_stable,
            mano_model_dir   = mano_weights,
            output_hand_data = world_aligned,
            object_masks_dir = f"{masks_dir}/{OBJECT_ID}",
            object_poses_dir = poses_smooth_dir,
            smooth_sigma     = smooth_sigma,
            dev              = dev,
        )

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Aligned hand + object: {world_aligned}")
    print(f"  Scaled mesh:           {mesh_scaled}")
    print(f"  Smoothed poses:        {poses_smooth_dir}/")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_v2d_ego_e2e(
        video_path                  = args.video_path,
        prompt                      = args.prompt,
        output_dir                  = args.output_dir,
        moge_weights                = args.moge_weights,
        grounding_dino_weights      = args.grounding_dino_weights,
        sam2_weights                = args.sam2_weights,
        sam3d_weights               = args.sam3d_weights,
        foundation_pose_weights     = args.foundation_pose_weights,
        hand_reconstruction_weights = args.hand_reconstruction_weights,
        mano_weights                = args.mano_weights,
        reference_frame             = args.reference_frame,
        smooth_sigma                = args.smooth_sigma,
        dev                         = args.dev,
    )
