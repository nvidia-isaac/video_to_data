# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified egocentric reconstruction entrypoint.

This script keeps the old pipeline scripts untouched and maps a smaller public
CLI onto the existing runners:

- ``--hand_tracking dynhamr`` runs the legacy ViPE + DynHaMR path with MoGe as
  the only downstream depth source.
- ``--hand_tracking hamer`` runs the WiLoR/SAM2/HaMeR path, with optional
  DROID-SLAM, GeoCalib gravity alignment, and refine_simple.py gsplat
  refinement.

Run from ``reconstruction/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Callable
from pathlib import Path


GSPLAT_DEFAULTS = {
    "epochs": 10,
    "batch_size": 4,
    "train_resolution_scale": 0.5,
    "lr_gaussians": 1e-3,
    "lr_object_pose": 1e-3,
    "lr_object_scale": 1e-3,
    "lr_hand_pose": 1e-3,
    "lr_hand_articulation": 1e-3,
    "lr_hand_shape": 1e-3,
    "lr_hand_scale": 1e-3,
    "lr_camera_pose": 1e-3,
    "init_opacity_obj": 0.5,
    "init_opacity_hand": 0.5,
    "init_opacity_bg": 0.5,
    "init_gaussian_scale_factor": 1.0,
    "w_perceptual": 1.0,
    "w_hand_object_penetration": 0.0,
    "hand_object_penetration_margin": 0.003,
    "object_sdf_resolution": 96,
    "hand_object_penetration_max_verts": 0,
    "perceptual_resize": 224,
    "lr_schedule": "cosine",
    "lr_cosine_min_factor": 0.1,
    "w_smooth_object_rot": 1000.0,
    "w_smooth_object_trans": 1000.0,
    "w_smooth_hand_rot": 1000.0,
    "w_smooth_hand_articulation": 1000.0,
    "w_smooth_hand_trans": 1000.0,
    "w_smooth_camera_rot": 1000.0,
    "w_smooth_camera_trans": 1000.0,
    "w_mask": 1.0,
    "w_relative_depth": 1.0,
    "render_every": 25,
    "w_smooth_hand_object_relative_rot": 1000.0,
    "w_smooth_hand_object_relative_trans": 1000.0,
}



def _import_run_ego_wilor() -> Callable[..., None]:
    try:
        from v2d.pipelines.run_ego_wilor import run_ego_wilor
    except ModuleNotFoundError as exc:  # Direct execution before editable install.
        if exc.name not in {
            "v2d",
            "v2d.pipelines",
            "v2d.pipelines.run_ego_wilor",
        }:
            raise
        from run_ego_wilor import run_ego_wilor
    return run_ego_wilor


def _import_run_v2d_ego_e2e() -> Callable[..., None]:
    try:
        from v2d.pipelines.run_v2d_ego_e2e import run_v2d_ego_e2e
    except ModuleNotFoundError as exc:  # Direct execution before editable install.
        if exc.name not in {
            "v2d",
            "v2d.pipelines",
            "v2d.pipelines.run_v2d_ego_e2e",
        }:
            raise
        from run_v2d_ego_e2e import run_v2d_ego_e2e
    return run_v2d_ego_e2e


def _import_result_bundle_tools() -> tuple[Callable[..., str], Callable[..., str], Callable[..., bool]]:
    from v2d.common.result_bundle import (
        gravity_align_result_bundle,
        result_bundle_has_gravity_alignment,
        write_result_bundle,
    )
    return write_result_bundle, gravity_align_result_bundle, result_bundle_has_gravity_alignment


def _import_droid_slam() -> Callable[..., None]:
    from v2d.droid_slam.docker.run_video_to_slam import run_video_to_slam
    return run_video_to_slam


def _import_geocalib() -> Callable[..., None]:
    from v2d.geocalib.docker.run_video_to_calibration import run_video_to_calibration
    return run_video_to_calibration


def _import_threejs_exporter() -> Callable[..., dict]:
    try:
        from v2d.pipelines.export_result_threejs_scene import export_scene
    except ModuleNotFoundError as exc:  # Direct execution before editable install.
        if exc.name not in {
            "v2d",
            "v2d.pipelines",
            "v2d.pipelines.export_result_threejs_scene",
        }:
            raise
        from export_result_threejs_scene import export_scene
    return export_scene


def _has_files(directory: Path) -> bool:
    return directory.is_dir() and any(directory.iterdir())


def _step(label: str, done: bool) -> bool:
    if done:
        print(f"  [skip] {label}")
        return True
    print(f"  [run ] {label}")
    return False


def _result_bundle_done(result_dir: Path) -> bool:
    return (result_dir / "result.npz").exists() and (result_dir / "mesh.obj").exists()


def _postprocess_video_path(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    if args.undistort:
        return output_dir / "anycalib" / "undistorted.mp4"
    return Path(args.video).resolve()


def _copy_result_bundle(src: Path, dst: Path) -> None:
    if not _result_bundle_done(src):
        raise FileNotFoundError(f"Missing source result bundle: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _run_droid_slam_postprocess(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    slam_poses_dir = output_dir / "slam_poses"
    slam_trajectory = output_dir / "slam_trajectory.txt"
    if not _step("DROID-SLAM trajectory", _has_files(slam_poses_dir)):
        run_video_to_slam = _import_droid_slam()
        run_video_to_slam(
            video_path=str(_postprocess_video_path(args)),
            poses_folder=str(slam_poses_dir),
            weights_path=args.droid_slam_weights,
            input_intrinsics_path=str(output_dir / "intrinsics_stable.json"),
            align_to_depth_folder=str(output_dir / "depth"),
            trajectory_path=str(slam_trajectory),
            dev=args.dev,
        )
    return slam_poses_dir


def _run_geocalib_postprocess(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    geocalib_dir = output_dir / "geocalib"
    intrinsics_path = geocalib_dir / "intrinsics.json"
    distortion_path = geocalib_dir / "distortion.json"
    gravity_path = geocalib_dir / "gravity.json"
    calibration_path = geocalib_dir / "calibration.json"
    done = all(p.exists() for p in (intrinsics_path, distortion_path, gravity_path, calibration_path))
    if not _step("GeoCalib intrinsics + gravity", done):
        geocalib_dir.mkdir(parents=True, exist_ok=True)
        run_video_to_calibration = _import_geocalib()
        run_video_to_calibration(
            video_path=str(_postprocess_video_path(args)),
            intrinsics_path=str(intrinsics_path),
            distortion_path=str(distortion_path),
            gravity_path=str(gravity_path),
            calibration_path=str(calibration_path),
            weights_path=args.geocalib_weights_path,
            weights=args.geocalib_weights,
            camera_model=args.geocalib_camera_model,
            num_samples=args.geocalib_num_samples,
            dev=args.dev,
        )
    return calibration_path


def _write_dynhamr_slam_result(args: argparse.Namespace, camera_to_world_dir: Path) -> Path:
    output_dir = Path(args.output_dir).resolve()
    result_dir = output_dir / "result_slam"
    if not _step("Package result_slam/", _result_bundle_done(result_dir)):
        write_result_bundle, _, _ = _import_result_bundle_tools()
        write_result_bundle(
            result_dir=str(result_dir),
            frames_dir=str(output_dir / "frames"),
            intrinsics_path=str(output_dir / "intrinsics_stable.json"),
            mesh_path=str(output_dir / "mesh_scaled_moge.obj"),
            object_poses_dir=str(output_dir / "poses_smoothed_moge"),
            object_scale=1.0,
            camera_to_world_dir=str(camera_to_world_dir),
            left_hand_dir=str(output_dir / "hamer_aligned_from_dynhamr_moge" / "2"),
            right_hand_dir=str(output_dir / "hamer_aligned_from_dynhamr_moge" / "3"),
            source_manifest={
                "pipeline": "ego_reconstruction",
                "base_pipeline": "ego_e2e",
                "depth_source": "moge",
                "camera_pose_source": "droid_slam",
                "object_id": 1,
                "left_hand_id": 2,
                "right_hand_id": 3,
            },
        )
    return result_dir


def _copy_stage_result(args: argparse.Namespace, src_result_dir: Path, suffix: str) -> Path:
    output_dir = Path(args.output_dir).resolve()
    dst = output_dir / suffix
    if not _step(f"Copy {suffix}/", _result_bundle_done(dst)):
        _copy_result_bundle(src_result_dir, dst)
    return dst


def _gravity_align_stage_result(args: argparse.Namespace, src_result_dir: Path, suffix: str) -> Path:
    output_dir = Path(args.output_dir).resolve()
    dst = output_dir / suffix
    _, gravity_align_result_bundle, result_bundle_has_gravity_alignment = _import_result_bundle_tools()
    if not result_bundle_has_gravity_alignment(str(dst), target=args.gravity_align_target):
        _copy_result_bundle(src_result_dir, dst)
    if not _step(f"Gravity-align {suffix}/", result_bundle_has_gravity_alignment(str(dst), target=args.gravity_align_target)):
        calibration_path = _run_geocalib_postprocess(args)
        gravity_align_result_bundle(
            result_dir=str(dst),
            geocalib_calibration_path=str(calibration_path),
            target=args.gravity_align_target,
        )
    return dst


def _finalize_result_bundle(args: argparse.Namespace, base_result_dir: Path, *, dynhamr: bool) -> Path:
    final_result_dir = base_result_dir
    if args.run_droid_slam:
        if dynhamr:
            slam_poses_dir = _run_droid_slam_postprocess(args)
            final_result_dir = _write_dynhamr_slam_result(args, slam_poses_dir)
        else:
            # The HaMeR runner already packages result/ with SLAM poses when
            # --run_droid_slam is set. Keep a named copy for comparison.
            final_result_dir = _copy_stage_result(args, base_result_dir, "result_slam")

    if args.run_gravity_alignment:
        suffix = "result_slam_gravity_aligned" if args.run_droid_slam else "result_gravity_aligned"
        final_result_dir = _gravity_align_stage_result(args, final_result_dir, suffix)

    return final_result_dir


def _default_threejs_mano_assets_root(args: argparse.Namespace) -> Path:
    if args.threejs_mano_assets_root is not None:
        return Path(args.threejs_mano_assets_root)
    if args.hand_tracking == "hamer":
        return Path(args.hamer_weights) / "_DATA" / "data"
    return Path(args.mano_weights or args.hand_reconstruction_weights)


def _threejs_scene_done(output_dir: Path) -> bool:
    required = ("index.html", "scene_data.js", "three.module.js", "OrbitControls.js")
    return all((output_dir / name).exists() and (output_dir / name).stat().st_size > 0 for name in required)


def _export_threejs_result(args: argparse.Namespace, result_dir: Path) -> None:
    if not args.export_threejs_result:
        return
    output_dir = Path(args.threejs_output_dir) if args.threejs_output_dir else result_dir / "threejs_scene"
    if not _step("Export Three.js result scene", _threejs_scene_done(output_dir)):
        export_scene = _import_threejs_exporter()
        data = export_scene(
            result_dir=result_dir,
            output_dir=output_dir,
            frames_dir=Path(args.output_dir).resolve() / "frames",
            depth_dir=Path(args.output_dir).resolve() / "depth",
            intrinsics_dir=Path(args.output_dir).resolve() / "intrinsics",
            frames=args.threejs_frames,
            num_frames=args.threejs_num_frames,
            depth_stride=max(1, args.threejs_depth_stride),
            max_depth_points=max(1, args.threejs_max_depth_points),
            max_depth=args.threejs_max_depth,
            max_object_faces=args.threejs_max_object_faces,
            mano_assets_root=_default_threejs_mano_assets_root(args),
        )
        print(f"Wrote Three.js scene: {output_dir / 'index.html'}")
        print(f"  frames: {data['meta']['frame_indices']}")
        print(f"  hand meshes: {data['meta']['hand_mesh_status']}")


def _add_weight_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--moge_weights", default="data/weights/moge")
    p.add_argument("--grounding_dino_weights", default="data/weights/grounding_dino")
    p.add_argument("--sam2_weights", default="data/weights/sam2")
    p.add_argument("--sam3d_weights", default="data/weights/sam3d")
    p.add_argument("--foundation_pose_weights", default="data/weights/foundation_pose")
    p.add_argument("--anycalib_weights", default="data/weights/anycalib")
    p.add_argument("--wilor_weights", default="data/weights/wilor")
    p.add_argument("--hamer_weights", default="data/weights/hamer")
    p.add_argument("--droid_slam_weights", default="data/weights/droid_slam")
    p.add_argument("--geocalib_weights_path", default="data/weights/geocalib")
    p.add_argument("--hand_reconstruction_weights", default="data/weights/hand")
    p.add_argument("--mano_weights", default=None)


def _add_gsplat_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("gsplat refinement")
    g.add_argument("--gsplat_refine_epochs", type=int, default=GSPLAT_DEFAULTS["epochs"])
    g.add_argument("--gsplat_refine_batch_size", type=int, default=GSPLAT_DEFAULTS["batch_size"])
    g.add_argument(
        "--gsplat_refine_train_resolution_scale",
        type=float,
        default=GSPLAT_DEFAULTS["train_resolution_scale"],
    )
    g.add_argument("--gsplat_refine_lr_gaussians", type=float, default=GSPLAT_DEFAULTS["lr_gaussians"])
    g.add_argument("--gsplat_refine_lr_object_pose", type=float, default=GSPLAT_DEFAULTS["lr_object_pose"])
    g.add_argument("--gsplat_refine_lr_object_scale", type=float, default=GSPLAT_DEFAULTS["lr_object_scale"])
    g.add_argument("--gsplat_refine_lr_hand_pose", type=float, default=GSPLAT_DEFAULTS["lr_hand_pose"])
    g.add_argument(
        "--gsplat_refine_lr_hand_articulation",
        type=float,
        default=GSPLAT_DEFAULTS["lr_hand_articulation"],
    )
    g.add_argument("--gsplat_refine_lr_hand_shape", type=float, default=GSPLAT_DEFAULTS["lr_hand_shape"])
    g.add_argument("--gsplat_refine_lr_hand_scale", type=float, default=GSPLAT_DEFAULTS["lr_hand_scale"])
    g.add_argument("--gsplat_refine_lr_camera_pose", type=float, default=GSPLAT_DEFAULTS["lr_camera_pose"])
    g.add_argument("--gsplat_refine_init_opacity_obj", type=float, default=GSPLAT_DEFAULTS["init_opacity_obj"])
    g.add_argument("--gsplat_refine_init_opacity_hand", type=float, default=GSPLAT_DEFAULTS["init_opacity_hand"])
    g.add_argument("--gsplat_refine_init_opacity_bg", type=float, default=GSPLAT_DEFAULTS["init_opacity_bg"])
    g.add_argument(
        "--gsplat_refine_init_gaussian_scale_factor",
        type=float,
        default=GSPLAT_DEFAULTS["init_gaussian_scale_factor"],
    )
    g.add_argument("--gsplat_refine_w_perceptual", type=float, default=GSPLAT_DEFAULTS["w_perceptual"])
    g.add_argument(
        "--gsplat_refine_w_hand_object_penetration",
        type=float,
        default=GSPLAT_DEFAULTS["w_hand_object_penetration"],
    )
    g.add_argument(
        "--gsplat_refine_hand_object_penetration_margin",
        type=float,
        default=GSPLAT_DEFAULTS["hand_object_penetration_margin"],
    )
    g.add_argument(
        "--gsplat_refine_object_sdf_resolution",
        type=int,
        default=GSPLAT_DEFAULTS["object_sdf_resolution"],
    )
    g.add_argument(
        "--gsplat_refine_hand_object_penetration_max_verts",
        type=int,
        default=GSPLAT_DEFAULTS["hand_object_penetration_max_verts"],
    )
    g.add_argument(
        "--gsplat_refine_perceptual_resize",
        type=int,
        default=GSPLAT_DEFAULTS["perceptual_resize"],
    )
    g.add_argument(
        "--gsplat_refine_lr_schedule",
        choices=("constant", "cosine"),
        default=GSPLAT_DEFAULTS["lr_schedule"],
    )
    g.add_argument(
        "--gsplat_refine_lr_cosine_min_factor",
        type=float,
        default=GSPLAT_DEFAULTS["lr_cosine_min_factor"],
    )
    g.add_argument(
        "--gsplat_refine_w_smooth_object_rot",
        type=float,
        default=GSPLAT_DEFAULTS["w_smooth_object_rot"],
    )
    g.add_argument(
        "--gsplat_refine_w_smooth_object_trans",
        type=float,
        default=GSPLAT_DEFAULTS["w_smooth_object_trans"],
    )
    g.add_argument(
        "--gsplat_refine_w_smooth_hand_rot",
        type=float,
        default=GSPLAT_DEFAULTS["w_smooth_hand_rot"],
    )
    g.add_argument(
        "--gsplat_refine_w_smooth_hand_articulation",
        type=float,
        default=GSPLAT_DEFAULTS["w_smooth_hand_articulation"],
    )
    g.add_argument(
        "--gsplat_refine_w_smooth_hand_trans",
        type=float,
        default=GSPLAT_DEFAULTS["w_smooth_hand_trans"],
    )
    g.add_argument(
        "--gsplat_refine_w_smooth_camera_rot",
        type=float,
        default=GSPLAT_DEFAULTS["w_smooth_camera_rot"],
    )
    g.add_argument(
        "--gsplat_refine_w_smooth_camera_trans",
        type=float,
        default=GSPLAT_DEFAULTS["w_smooth_camera_trans"],
    )
    g.add_argument("--gsplat_refine_w_mask", type=float, default=GSPLAT_DEFAULTS["w_mask"])
    g.add_argument(
        "--gsplat_refine_w_relative_depth",
        type=float,
        default=GSPLAT_DEFAULTS["w_relative_depth"],
    )
    g.add_argument("--gsplat_refine_render_every", type=int, default=GSPLAT_DEFAULTS["render_every"])
    g.add_argument(
        "--gsplat_refine_w_smooth_hand_object_relative_rot",
        type=float,
        default=GSPLAT_DEFAULTS["w_smooth_hand_object_relative_rot"],
    )
    g.add_argument(
        "--gsplat_refine_w_smooth_hand_object_relative_trans",
        type=float,
        default=GSPLAT_DEFAULTS["w_smooth_hand_object_relative_trans"],
    )
    g.add_argument("--gsplat_refine_mask_background", action="store_true")
    g.add_argument("--gsplat_refine_debug_frame", type=int, default=None)
    g.add_argument("--gsplat_refine_vgg_weights_path", default=None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", "--video_path", dest="video", required=True, help="Input MP4 video.")
    p.add_argument("--object_prompt", "--prompt", dest="object_prompt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--object_mesh", "--object_mesh_path", dest="object_mesh", default=None)
    p.add_argument(
        "--skip_object_scale_estimation",
        action="store_true",
        help="For --object_mesh, trust the provided mesh scale and track it directly.",
    )
    p.add_argument("--hand_tracking", choices=("dynhamr", "hamer"), required=True)
    p.add_argument("--reference_frame", type=int, default=0)
    p.add_argument("--undistort", action="store_true")
    p.add_argument("--run_droid_slam", action="store_true")
    p.add_argument("--run_gravity_alignment", action="store_true")
    p.add_argument("--run_gsplat_refinement", action="store_true")
    p.add_argument("--export_threejs_result", action="store_true")
    p.add_argument("--depth_source", choices=("moge", "vipe"), default=None, help=argparse.SUPPRESS)
    p.add_argument("--reregister_iou_thresh", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dev", action="store_true")

    p.add_argument("--sam2_bbox_pad", type=float, default=0.05)
    p.set_defaults(sam2_square_wilor_bboxes=False)
    p.add_argument("--sam2_square_wilor_bboxes", dest="sam2_square_wilor_bboxes", action="store_true")
    p.add_argument("--no_sam2_square_wilor_bboxes", dest="sam2_square_wilor_bboxes", action="store_false")
    p.add_argument("--sam2_hand_prompt_source", choices=("box", "wilor_mask"), default="wilor_mask")
    p.add_argument("--hamer_bbox_expansion", type=float, default=1.7)
    p.add_argument("--hamer_mask_min_pixels", type=int, default=None)
    p.add_argument("--min_iou", type=float, default=0.1)
    p.add_argument("--mask_min_pixels", type=int, default=256)
    p.add_argument("--interp_betas", choices=("fixed", "interp"), default="fixed")
    p.add_argument("--interp_max_gap_frames", type=int, default=15)

    p.add_argument("--geocalib_weights", choices=("pinhole", "distorted"), default="pinhole")
    p.add_argument(
        "--geocalib_camera_model",
        choices=("pinhole", "simple_radial", "radial", "simple_divisional"),
        default="pinhole",
    )
    p.add_argument("--geocalib_num_samples", type=int, default=8)
    p.add_argument("--use_geocalib_intrinsics", action="store_true")
    p.add_argument(
        "--gravity_align_target",
        choices=("z_up", "z_down", "y_up", "y_down", "x_up", "x_down"),
        default="z_up",
    )

    p.add_argument("--bg_init_stride", type=int, default=10)
    p.add_argument("--bg_voxel_size", type=float, default=0.005)
    p.add_argument("--bg_max_points", type=int, default=100000)

    p.add_argument("--threejs_output_dir", default=None)
    p.add_argument("--threejs_frames", default=None)
    p.add_argument("--threejs_num_frames", type=int, default=8)
    p.add_argument("--threejs_depth_stride", type=int, default=24)
    p.add_argument("--threejs_max_depth_points", type=int, default=2500)
    p.add_argument("--threejs_max_depth", type=float, default=8.0)
    p.add_argument("--threejs_max_object_faces", type=int, default=20000)
    p.add_argument("--threejs_mano_assets_root", default=None)

    _add_weight_args(p)
    _add_gsplat_args(p)
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.depth_source is not None and args.depth_source != "moge":
        print("WARNING: --depth_source is deprecated and ignored; this pipeline always uses MoGe.")
    elif args.depth_source is not None:
        print("WARNING: --depth_source is deprecated and ignored; MoGe is always used.")

    if args.object_mesh is not None:
        args.object_mesh = os.path.abspath(args.object_mesh)
        if not os.path.isfile(args.object_mesh):
            raise FileNotFoundError(f"Object mesh not found: {args.object_mesh}")
        if not args.skip_object_scale_estimation:
            print(
                "WARNING: --object_mesh uses the provided mesh scale directly. "
                "Pass --skip_object_scale_estimation to make this explicit."
            )
    elif args.skip_object_scale_estimation:
        print("WARNING: --skip_object_scale_estimation has no effect without --object_mesh.")

    if args.hand_tracking == "dynhamr":
        unsupported = []
        if args.object_mesh is not None:
            unsupported.append("--object_mesh")
        if args.run_gsplat_refinement:
            unsupported.append("--run_gsplat_refinement")
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(
                f"{joined} are only supported with --hand_tracking hamer in this first consolidated entrypoint."
            )


def _run_dynhamr(args: argparse.Namespace) -> Path:
    run_v2d_ego_e2e = _import_run_v2d_ego_e2e()
    run_v2d_ego_e2e(
        video_path=args.video,
        prompt=args.object_prompt,
        output_dir=args.output_dir,
        moge_weights=args.moge_weights,
        grounding_dino_weights=args.grounding_dino_weights,
        sam2_weights=args.sam2_weights,
        sam3d_weights=args.sam3d_weights,
        foundation_pose_weights=args.foundation_pose_weights,
        hand_reconstruction_weights=args.hand_reconstruction_weights,
        mano_weights=args.mano_weights,
        anycalib_weights=args.anycalib_weights,
        undistort=args.undistort,
        depth_source="moge",
        reference_frame=args.reference_frame,
        reregister_iou_thresh=args.reregister_iou_thresh,
        seed=args.seed,
        dev=args.dev,
    )
    return _finalize_result_bundle(
        args,
        Path(args.output_dir).resolve() / "result",
        dynhamr=True,
    )


def _run_hamer(args: argparse.Namespace) -> Path:
    from v2d.pipelines.mano_assets import prepare_hamer_mano_assets

    source_root = Path(args.mano_weights or args.hand_reconstruction_weights)
    source_mano = source_root / "models" / "MANO_RIGHT.pkl"
    staged_mano = prepare_hamer_mano_assets(args.hamer_weights, source_mano)
    print(f"  HaMeR MANO asset: {staged_mano}")
    run_ego_wilor = _import_run_ego_wilor()
    run_ego_wilor(
        video_path=args.video,
        output_dir=args.output_dir,
        sam2_weights=args.sam2_weights,
        wilor_weights=args.wilor_weights,
        reference_frame=args.reference_frame,
        sam2_bbox_pad=args.sam2_bbox_pad,
        sam2_square_wilor_bboxes=args.sam2_square_wilor_bboxes,
        sam2_hand_prompt_source=args.sam2_hand_prompt_source,
        undistort=args.undistort,
        anycalib_weights=args.anycalib_weights,
        run_geocalib=False,
        geocalib_weights_path=args.geocalib_weights_path,
        geocalib_weights=args.geocalib_weights,
        geocalib_camera_model=args.geocalib_camera_model,
        geocalib_num_samples=args.geocalib_num_samples,
        use_geocalib_intrinsics=args.use_geocalib_intrinsics,
        gravity_align=False,
        gravity_align_target=args.gravity_align_target,
        moge_weights=args.moge_weights,
        min_iou=args.min_iou,
        mask_min_pixels=args.mask_min_pixels,
        interp_betas=args.interp_betas,
        interp_max_gap_frames=args.interp_max_gap_frames,
        object_prompt=args.object_prompt,
        object_mesh_path=args.object_mesh,
        grounding_dino_weights=args.grounding_dino_weights,
        sam3d_weights=args.sam3d_weights,
        foundation_pose_weights=args.foundation_pose_weights,
        reregister_iou_thresh=args.reregister_iou_thresh,
        run_slam=args.run_droid_slam,
        droid_slam_weights=args.droid_slam_weights,
        bg_init_stride=args.bg_init_stride,
        bg_voxel_size=args.bg_voxel_size,
        bg_max_points=args.bg_max_points,
        hand_pose_source="hamer",
        run_hamer_pass=True,
        hamer_weights=args.hamer_weights,
        hamer_bbox_expansion=args.hamer_bbox_expansion,
        hamer_mask_min_pixels=args.hamer_mask_min_pixels,
        run_refinement=False,
        run_refinement_simple=args.run_gsplat_refinement,
        simple_refinement_epochs=args.gsplat_refine_epochs,
        simple_refinement_batch_size=args.gsplat_refine_batch_size,
        simple_refinement_train_resolution_scale=args.gsplat_refine_train_resolution_scale,
        simple_refinement_lr_gaussians=args.gsplat_refine_lr_gaussians,
        simple_refinement_lr_object_pose=args.gsplat_refine_lr_object_pose,
        simple_refinement_lr_object_scale=args.gsplat_refine_lr_object_scale,
        simple_refinement_lr_hand_pose=args.gsplat_refine_lr_hand_pose,
        simple_refinement_lr_hand_articulation=args.gsplat_refine_lr_hand_articulation,
        simple_refinement_lr_hand_shape=args.gsplat_refine_lr_hand_shape,
        simple_refinement_lr_hand_scale=args.gsplat_refine_lr_hand_scale,
        simple_refinement_lr_camera_pose=args.gsplat_refine_lr_camera_pose,
        simple_refinement_lr_schedule=args.gsplat_refine_lr_schedule,
        simple_refinement_lr_cosine_min_factor=args.gsplat_refine_lr_cosine_min_factor,
        simple_refinement_w_smooth_object_rot=args.gsplat_refine_w_smooth_object_rot,
        simple_refinement_w_smooth_object_trans=args.gsplat_refine_w_smooth_object_trans,
        simple_refinement_w_smooth_hand_rot=args.gsplat_refine_w_smooth_hand_rot,
        simple_refinement_w_smooth_hand_articulation=args.gsplat_refine_w_smooth_hand_articulation,
        simple_refinement_w_smooth_hand_trans=args.gsplat_refine_w_smooth_hand_trans,
        simple_refinement_w_smooth_hand_object_relative_rot=args.gsplat_refine_w_smooth_hand_object_relative_rot,
        simple_refinement_w_smooth_hand_object_relative_trans=args.gsplat_refine_w_smooth_hand_object_relative_trans,
        simple_refinement_w_smooth_camera_rot=args.gsplat_refine_w_smooth_camera_rot,
        simple_refinement_w_smooth_camera_trans=args.gsplat_refine_w_smooth_camera_trans,
        simple_refinement_w_mask=args.gsplat_refine_w_mask,
        simple_refinement_w_relative_depth=args.gsplat_refine_w_relative_depth,
        simple_refinement_w_perceptual=args.gsplat_refine_w_perceptual,
        simple_refinement_vgg_weights_path=args.gsplat_refine_vgg_weights_path,
        simple_refinement_perceptual_resize=args.gsplat_refine_perceptual_resize,
        simple_refinement_w_hand_object_penetration=args.gsplat_refine_w_hand_object_penetration,
        simple_refinement_hand_object_penetration_margin=args.gsplat_refine_hand_object_penetration_margin,
        simple_refinement_object_sdf_resolution=args.gsplat_refine_object_sdf_resolution,
        simple_refinement_hand_object_penetration_max_verts=args.gsplat_refine_hand_object_penetration_max_verts,
        simple_refinement_mask_background=args.gsplat_refine_mask_background,
        simple_refinement_render_every=args.gsplat_refine_render_every,
        simple_refinement_debug_frame=args.gsplat_refine_debug_frame,
        init_opacity_obj=args.gsplat_refine_init_opacity_obj,
        init_opacity_hand=args.gsplat_refine_init_opacity_hand,
        init_opacity_bg=args.gsplat_refine_init_opacity_bg,
        init_gaussian_scale_factor=args.gsplat_refine_init_gaussian_scale_factor,
        dev=args.dev,
    )
    return _finalize_result_bundle(
        args,
        Path(args.output_dir).resolve() / "result",
        dynhamr=False,
    )


def run_from_args(args: argparse.Namespace) -> None:
    _validate_args(args)
    if args.hand_tracking == "dynhamr":
        final_result_dir = _run_dynhamr(args)
    elif args.hand_tracking == "hamer":
        final_result_dir = _run_hamer(args)
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported hand tracking mode: {args.hand_tracking}")
    _export_threejs_result(args, final_result_dir)
    print(f"  final_result_bundle: {final_result_dir}/")


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
