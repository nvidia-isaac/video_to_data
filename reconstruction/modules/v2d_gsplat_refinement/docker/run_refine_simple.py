# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from v2d.docker.container import run_in_container
from v2d.gsplat_refinement.docker._config import IMAGE_NAME, MODULES_DIR


def run_refine_simple(
    frames_dir: str,
    depth_dir: str | None,
    intrinsics_path: str,
    object_mesh_path: str,
    object_poses_dir: str,
    object_mask_dir: str,
    refined_object_poses_dir: str,
    overlay_path: str,
    left_hand_pose_dir: str | None = None,
    left_hand_mask_dir: str | None = None,
    right_hand_pose_dir: str | None = None,
    right_hand_mask_dir: str | None = None,
    refined_left_hand_pose_dir: str | None = None,
    refined_right_hand_pose_dir: str | None = None,
    refined_camera_poses_dir: str | None = None,
    mano_assets_root: str | None = None,
    background_pose_init_dir: str | None = None,
    n_epochs: int = 20,
    batch_size: int = 32,
    train_resolution_scale: float = 0.5,
    lr_gaussians: float = 3e-3,
    lr_object_pose: float = 3e-3,
    lr_object_scale: float = 0.0,
    lr_hand_pose: float = 3e-3,
    lr_hand_articulation: float = 0.0,
    lr_hand_shape: float = 0.0,
    lr_hand_scale: float = 0.0,
    lr_camera_pose: float = 3e-3,
    lr_schedule: str = "constant",
    lr_cosine_min_factor: float = 0.0,
    w_smooth_object_rot: float = 0.1,
    w_smooth_object_trans: float = 0.1,
    w_smooth_hand_rot: float = 0.1,
    w_smooth_hand_articulation: float = 0.1,
    w_smooth_hand_trans: float = 0.1,
    w_smooth_hand_object_relative_rot: float = 0.0,
    w_smooth_hand_object_relative_trans: float = 0.0,
    w_smooth_camera_rot: float = 0.1,
    w_smooth_camera_trans: float = 0.1,
    w_mask: float = 1.0,
    w_relative_depth: float = 0.0,
    w_perceptual: float = 0.0,
    vgg_weights_path: str | None = None,
    perceptual_resize: int = 224,
    w_hand_object_penetration: float = 0.0,
    hand_object_penetration_margin: float = 0.003,
    object_sdf_resolution: int = 96,
    hand_object_penetration_max_verts: int = 0,
    mask_background: bool = False,
    bg_ref_frame: int | None = None,
    bg_init_stride: int = 10,
    bg_voxel_size: float = 0.005,
    bg_max_points: int = 50000,
    valid_mask_threshold: float = 0.04,
    valid_mask_erode_iters: int = 2,
    face_normal_thin_factor_obj: float = 0.25,
    face_normal_thin_factor_hand: float = 0.25,
    init_opacity_obj: float = 0.9,
    init_opacity_hand: float = 0.9,
    init_opacity_bg: float = 0.9,
    init_gaussian_scale_factor: float = 1.0,
    render_every: int = 25,
    progress_dir: str | None = None,
    debug_frame_idx: int | None = None,
    fps: float = 30.0,
    seed: int = 0,
    dev: bool = False,
) -> None:
    if not mask_background and depth_dir is None:
        raise ValueError("depth_dir is required unless --mask-background is set")
    if float(w_relative_depth) > 0.0 and depth_dir is None:
        raise ValueError("depth_dir is required when w_relative_depth is nonzero")
    if vgg_weights_path is not None:
        os.makedirs(
            os.path.dirname(os.path.abspath(vgg_weights_path)) or ".",
            exist_ok=True,
        )

    inputs: dict[str, str] = {
        "frames_dir":       frames_dir,
        "intrinsics_path":  intrinsics_path,
        "object_mesh_path": object_mesh_path,
        "object_poses_dir": object_poses_dir,
        "object_mask_dir":  object_mask_dir,
    }
    if depth_dir is not None:
        inputs["depth_dir"] = depth_dir
    for k, v in {
        "left_hand_pose_dir":      left_hand_pose_dir,
        "left_hand_mask_dir":      left_hand_mask_dir,
        "right_hand_pose_dir":     right_hand_pose_dir,
        "right_hand_mask_dir":     right_hand_mask_dir,
        "mano_assets_root":        mano_assets_root,
        "background_pose_init_dir": background_pose_init_dir,
        "vgg_weights_path":        vgg_weights_path,
    }.items():
        if v is not None:
            inputs[k] = v

    outputs: dict[str, str] = {
        "refined_object_poses_dir": refined_object_poses_dir,
        "overlay_path":             overlay_path,
    }
    for k, v in {
        "refined_left_hand_pose_dir":  refined_left_hand_pose_dir,
        "refined_right_hand_pose_dir": refined_right_hand_pose_dir,
        "refined_camera_poses_dir":    refined_camera_poses_dir,
        "progress_dir":                 progress_dir,
    }.items():
        if v is not None:
            outputs[k] = v

    extra_args: dict[str, object] = {
        "n_epochs":              n_epochs,
        "batch_size":            batch_size,
        "train_resolution_scale": train_resolution_scale,
        "lr_gaussians":          lr_gaussians,
        "lr_object_pose":        lr_object_pose,
        "lr_object_scale":       lr_object_scale,
        "lr_hand_pose":          lr_hand_pose,
        "lr_hand_articulation":  lr_hand_articulation,
        "lr_hand_shape":         lr_hand_shape,
        "lr_hand_scale":         lr_hand_scale,
        "lr_camera_pose":        lr_camera_pose,
        "lr_schedule":           lr_schedule,
        "lr_cosine_min_factor":  lr_cosine_min_factor,
        "w_smooth_object_rot":   w_smooth_object_rot,
        "w_smooth_object_trans": w_smooth_object_trans,
        "w_smooth_hand_rot":     w_smooth_hand_rot,
        "w_smooth_hand_articulation": w_smooth_hand_articulation,
        "w_smooth_hand_trans":   w_smooth_hand_trans,
        "w_smooth_hand_object_relative_rot": w_smooth_hand_object_relative_rot,
        "w_smooth_hand_object_relative_trans": w_smooth_hand_object_relative_trans,
        "w_smooth_camera_rot":   w_smooth_camera_rot,
        "w_smooth_camera_trans": w_smooth_camera_trans,
        "w_mask":                w_mask,
        "w_relative_depth":      w_relative_depth,
        "w_perceptual":          w_perceptual,
        "perceptual_resize":     perceptual_resize,
        "w_hand_object_penetration": w_hand_object_penetration,
        "hand_object_penetration_margin": hand_object_penetration_margin,
        "object_sdf_resolution": object_sdf_resolution,
        "hand_object_penetration_max_verts": hand_object_penetration_max_verts,
        "mask-background":       mask_background,
        "bg_ref_frame":          bg_ref_frame,
        "bg_init_stride":        bg_init_stride,
        "bg_voxel_size":         bg_voxel_size,
        "bg_max_points":         bg_max_points,
        "valid_mask_threshold":  valid_mask_threshold,
        "valid_mask_erode_iters": valid_mask_erode_iters,
        "face_normal_thin_factor_obj":  face_normal_thin_factor_obj,
        "face_normal_thin_factor_hand": face_normal_thin_factor_hand,
        "init_opacity_obj":      init_opacity_obj,
        "init_opacity_hand":     init_opacity_hand,
        "init_opacity_bg":       init_opacity_bg,
        "init_gaussian_scale_factor": init_gaussian_scale_factor,
        "render_every":          render_every,
        "debug_frame_idx":       debug_frame_idx,
        "fps":                   fps,
        "seed":                  seed,
    }

    run_in_container(
        image       = IMAGE_NAME,
        module      = "v2d.gsplat_refinement.lib.refine_simple",
        inputs      = inputs,
        outputs     = outputs,
        extra_args  = extra_args,
        dev         = dev,
        modules_dir = MODULES_DIR,
        gpus        = True,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Minimal 2DGS hand/object/camera refinement")
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--depth_dir", default=None,
                   help="Depth frames for background initialization. Required unless --mask-background is set.")
    p.add_argument("--intrinsics_path", required=True)
    p.add_argument("--object_mesh_path", required=True)
    p.add_argument("--object_poses_dir", required=True)
    p.add_argument("--object_mask_dir", required=True)
    p.add_argument("--refined_object_poses_dir", required=True)
    p.add_argument("--overlay_path", required=True)
    p.add_argument("--left_hand_pose_dir", default=None)
    p.add_argument("--left_hand_mask_dir", default=None)
    p.add_argument("--right_hand_pose_dir", default=None)
    p.add_argument("--right_hand_mask_dir", default=None)
    p.add_argument("--refined_left_hand_pose_dir", default=None)
    p.add_argument("--refined_right_hand_pose_dir", default=None)
    p.add_argument("--refined_camera_poses_dir", default=None)
    p.add_argument("--mano_assets_root", default=None)
    p.add_argument("--background_pose_init_dir", default=None)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--train_resolution_scale", type=float, default=0.5)
    p.add_argument("--lr_gaussians", type=float, default=3e-4)
    p.add_argument("--lr_object_pose", type=float, default=3e-4)
    p.add_argument("--lr_object_scale", type=float, default=0.0,
                   help="LR for global object scale. 0 disables.")
    p.add_argument("--lr_hand_pose", type=float, default=3e-4,
                   help="LR for hand root global_orient and cam_t. 0 disables root pose refinement.")
    p.add_argument("--lr_hand_articulation", type=float, default=0.0,
                   help="LR for MANO finger articulation hand_pose. 0 disables.")
    p.add_argument("--lr_hand_shape", type=float, default=0.0,
                   help="LR for shared MANO betas/shape per hand track. 0 disables.")
    p.add_argument("--lr_hand_scale", type=float, default=0.0,
                   help="LR for shared per-track hand_scale. 0 disables.")
    p.add_argument("--lr_camera_pose", type=float, default=3e-4)
    p.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant",
                   help="Learning-rate schedule for all optimizer groups.")
    p.add_argument("--lr_cosine_min_factor", type=float, default=0.0,
                   help="Final LR multiplier for --lr_schedule cosine. 0 decays to zero.")
    p.add_argument("--w_smooth_object_rot", type=float, default=0.1)
    p.add_argument("--w_smooth_object_trans", type=float, default=0.1)
    p.add_argument("--w_smooth_hand_rot", type=float, default=0.1)
    p.add_argument("--w_smooth_hand_articulation", type=float, default=0.1,
                   help="Temporal rotation smoothness for MANO finger hand_pose.")
    p.add_argument("--w_smooth_hand_trans", type=float, default=0.1)
    p.add_argument("--w_smooth_hand_object_relative_rot", type=float, default=0.0,
                   help="Temporal smoothness for hand root rotation relative to object pose. 0 disables.")
    p.add_argument("--w_smooth_hand_object_relative_trans", type=float, default=0.0,
                   help="Temporal smoothness for hand root translation in the object frame. "
                        "0 disables.")
    p.add_argument("--w_smooth_camera_rot", type=float, default=0.1)
    p.add_argument("--w_smooth_camera_trans", type=float, default=0.1)
    p.add_argument("--w_mask", type=float, default=1.0,
                   help="Weight for L1 segmentation mask loss. 0 disables.")
    p.add_argument("--w_relative_depth", type=float, default=0.0,
                   help="Weight for log-depth-gradient loss against MoGe depth. 0 disables.")
    p.add_argument("--w_perceptual", type=float, default=0.0,
                   help="Weight for VGG16 perceptual feature L1 loss. 0 disables.")
    p.add_argument("--vgg_weights_path", default=None,
                   help="Optional local VGG16 state_dict path. If missing, torchvision ImageNet weights are downloaded there.")
    p.add_argument("--perceptual_resize", type=int, default=224,
                   help="Square resize for VGG perceptual loss. <=0 uses training resolution.")
    p.add_argument("--w_hand_object_penetration", type=float, default=0.0,
                   help="Weight for 3D hand/object SDF penetration loss. 0 disables.")
    p.add_argument("--hand_object_penetration_margin", type=float, default=0.003,
                   help="Signed-distance margin in meters/object units for hand/object separation.")
    p.add_argument("--object_sdf_resolution", type=int, default=96,
                   help="Resolution per axis for the object SDF grid used by hand/object penetration.")
    p.add_argument("--hand_object_penetration_max_verts", type=int, default=0,
                   help="Max MANO verts per hand for penetration loss. 0 uses all vertices.")
    p.add_argument("--mask-background", "--mask_background", dest="mask_background", action="store_true",
                   help="Black out target background and omit background Gaussians/camera refinement.")
    p.add_argument("--bg_ref_frame", type=int, default=None)
    p.add_argument("--bg_init_stride", type=int, default=10)
    p.add_argument("--bg_voxel_size", type=float, default=0.005)
    p.add_argument("--bg_max_points", type=int, default=50000)
    p.add_argument("--valid_mask_threshold", type=float, default=0.04)
    p.add_argument("--valid_mask_erode_iters", type=int, default=2)
    p.add_argument("--face_normal_thin_factor_obj", type=float, default=0.25)
    p.add_argument("--face_normal_thin_factor_hand", type=float, default=0.25)
    p.add_argument("--init_opacity_obj", type=float, default=0.9)
    p.add_argument("--init_opacity_hand", type=float, default=0.9)
    p.add_argument("--init_opacity_bg", type=float, default=0.9)
    p.add_argument("--init_gaussian_scale_factor", type=float, default=1.0,
                   help="Multiplier for all initial Gaussian scales/radii.")
    p.add_argument("--render_every", type=int, default=25)
    p.add_argument("--progress_dir", default=None)
    p.add_argument("--debug_frame_idx", type=int, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dev", action="store_true")
    args = p.parse_args()
    run_refine_simple(**vars(args))
