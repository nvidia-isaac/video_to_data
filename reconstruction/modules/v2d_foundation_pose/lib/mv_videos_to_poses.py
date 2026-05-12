"""Multi-view 6-DoF object tracking with FoundationPose."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import trimesh

from v2d.common.datatypes import DepthImage, Mask
from v2d.mesh.lib.mesh import Mesh
from v2d.common.video import FrameSource, get_video_writer
from v2d.mv.math.numpy_fn import pose_two_euro_filter
from v2d.mv.rig import RigConfig

from .fp_utils import draw_posed_3d_box, draw_xyz_axis
from .multiview_tracker import MultiViewTracker
from .symmetry import load_symmetry_group


@dataclass
class RecoveryConfig:
    enabled: bool = True
    refine_iter: int = 5
    min_views: int = 1
    min_valid_depth_pixels: int = 25
    visible_ratio_cutoff: float = 0.3
    attempt_stride: int = 1
    under_supported_enabled: bool = True
    mask_pose_iou_cutoff: float = 0.05
    mask_explained_ratio_cutoff: float = 0.10


@dataclass
class RepairConfig:
    enabled: bool = True
    max_step_translation_m: float = 0.15
    max_step_rotation_deg: float = 45.0
    visible_ratio_tolerance: float = 0.10
    recovery_trigger_enabled: bool = True
    recovery_trigger_max_window: int = 30
    recovery_trigger_anchor_stable_frames: int = 5
    snap_trigger_enabled: bool = True
    snap_trigger_max_span: int = 150
    snap_trigger_anchor_stable_frames: int = 8
    snap_trigger_outlier_window: int = 15
    snap_trigger_rotation_mad_scale: float = 5.0
    snap_trigger_translation_mad_scale: float = 5.0
    snap_trigger_min_rotation_deg: float = 10.0
    snap_trigger_max_translation_m: float = 0.08
    snap_trigger_max_burst_frames: int = 3
    recovery_min_views: int = 1
    track_refine_iter: int = 2


@dataclass
class TrackingFrameContext:
    frame_sources: list[FrameSource]
    depth_sources: list[FrameSource]
    mask_sources: list[FrameSource]
    cam_names: list[str]
    Ks: list[np.ndarray]
    Ts: list[np.ndarray]
    scale_target_size: tuple[int, int] | None


def mv_videos_to_poses(
    cam_names: list[str],
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    rgb_paths: list[Path],
    depth_dirs: list[Path],
    mask_dirs: list[Path],
    mesh_path: Path,
    weights_dir: str,
    pose_path: Path,
    symmetry_path: Path | None = None,
    scale: float = 0.5,
    depth_direction_trust: float = 0.5,
    visible_ratio_cutoff_high: float = 0.3,
    visible_ratio_cutoff_low: float = 0.01,
    precision_high: float = 1.0,
    precision_low: float = 0.01,
    est_refine_iter: int = 5,
    track_refine_iter: int = 2,
    recovery_config: RecoveryConfig | None = None,
    repair_config: RepairConfig | None = None,
    smooth_across_recovery: bool = False,
    debug: int = 0,
):
    """Run multi-view FoundationPose tracking.

    Args:
        cam_names: per-camera names (used for debug output naming).
        cam_intrinsics: list of (3,3) K matrices, one per camera.
        cam_extrinsics: list of (4,4) cam-to-world transforms, one per camera.
        rgb_paths: per-camera paths to RGB frames (image dir, .h5, or video file).
        depth_dirs: per-camera depth directories (inverse-depth PNGs via DepthImage).
        mask_dirs: per-camera object mask directories (first PNG used for registration).
        mesh_path: path to the object mesh file.
        weights_dir: path to FoundationPose weights.
        pose_path: output path for filtered poses .npy file.
        symmetry_path: optional path to a BOP-style symmetry annotation JSON
            (typically `<mesh_dir>/output_symmetry.json`). When provided and
            the file exists, per-view registrations at frame 0 are
            canonicalized into a common equivalence-class representative
            before averaging. If None or missing, behavior matches the
            no-symmetry case.
        scale: resolution scale factor (e.g. 0.5 for half resolution). Scales
            intrinsics and resizes images/depths/masks accordingly.
        depth_direction_trust: weight for depth axis in anisotropic translation averaging.
        visible_ratio_cutoff_high: visibility ratio at which a camera gets full precision.
        visible_ratio_cutoff_low: visibility ratio below which a camera is excluded.
            Set equal to cutoff_high for hard cutoff with uniform weighting.
        precision_high: precision weight for cameras at or above cutoff_high.
        precision_low: precision weight for cameras at cutoff_low.
        est_refine_iter: refinement iterations for registration.
        track_refine_iter: refinement iterations for tracking.
        recovery_config: online recovery settings for MultiViewTracker.
        repair_config: offline post-forward repair settings.
        smooth_across_recovery: smooth valid poses across invalid/lost spans when True.
        debug: 0=off, 1=overlay videos after processing, 2=also per-frame images every 30 frames.
    """
    pose_path = Path(pose_path)
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_config = recovery_config or RecoveryConfig()
    repair_config = repair_config or RepairConfig(
        recovery_min_views=recovery_config.min_views,
        track_refine_iter=track_refine_iter,
    )
    repair_config.recovery_min_views = recovery_config.min_views
    repair_config.track_refine_iter = track_refine_iter

    num_cameras = len(cam_intrinsics)
    Ks = cam_intrinsics
    Ts = cam_extrinsics

    mesh = Mesh.load(str(mesh_path))
    tm = mesh.to_trimesh()
    _, obb_extents = trimesh.bounds.oriented_bounds(tm)
    print(f"Mesh: {len(tm.vertices)} verts, OBB extents={obb_extents}, min={obb_extents.min():.4f}")

    symmetry_group = None
    if symmetry_path is not None and Path(symmetry_path).exists():
        symmetry_group = load_symmetry_group(symmetry_path)
        print(f"Loaded symmetry group: {len(symmetry_group)} elements from {symmetry_path}")
    else:
        print(f"No symmetry annotation; canonicalization disabled (symmetry_path={symmetry_path})")

    register_debug_path = str(pose_path.parent / "register_debug.json") if debug >= 1 else None
    tracker = MultiViewTracker(
        mesh, weights_dir, num_cameras,
        depth_direction_trust=depth_direction_trust,
        visible_ratio_cutoff_high=visible_ratio_cutoff_high,
        visible_ratio_cutoff_low=visible_ratio_cutoff_low,
        precision_high=precision_high,
        precision_low=precision_low,
        symmetry_group=symmetry_group,
        recovery_enabled=recovery_config.enabled,
        recovery_refine_iter=recovery_config.refine_iter,
        recovery_min_views=recovery_config.min_views,
        recovery_min_valid_depth_pixels=recovery_config.min_valid_depth_pixels,
        recovery_visible_ratio_cutoff=recovery_config.visible_ratio_cutoff,
        recovery_attempt_stride=recovery_config.attempt_stride,
        under_supported_recovery_enabled=recovery_config.under_supported_enabled,
        mask_pose_iou_cutoff=recovery_config.mask_pose_iou_cutoff,
        mask_explained_ratio_cutoff=recovery_config.mask_explained_ratio_cutoff,
        register_debug_path=register_debug_path,
    )

    frame_sources = [FrameSource.from_path(p) for p in rgb_paths]
    mask_sources = [FrameSource.from_path(d) for d in mask_dirs]
    depth_sources = [FrameSource.from_path(d) for d in depth_dirs]

    num_frames = frame_sources[0].n_frames
    for j, (fs, ds, ms) in enumerate(zip(frame_sources, depth_sources, mask_sources)):
        if fs.n_frames != num_frames or ds.n_frames != num_frames or ms.n_frames != num_frames:
            raise ValueError(
                f"camera {cam_names[j]}: frame count mismatch "
                f"(rgb={fs.n_frames}, depth={ds.n_frames}, mask={ms.n_frames}, expected={num_frames})"
            )
    frame_iterators = [fs.iter_frames() for fs in frame_sources]

    all_poses = []
    select_mask = []
    output_poses = []
    pose_valid_mask = []
    tracking_status = []
    visible_ratios_history = []

    if debug >= 2:
        debug_image_dirs = []
        for cam_name in cam_names:
            d = pose_path.parent / f"{cam_name}_fp_poses"
            d.mkdir(parents=True, exist_ok=True)
            debug_image_dirs.append(d)

    Ks_orig = list(Ks)
    if scale != 1.0:
        W, H = frame_sources[0].image_size
        target_size = (int(W * scale), int(H * scale))
        sx, sy = target_size[0] / W, target_size[1] / H
        S = np.diag([sx, sy, 1.0])
        Ks = [S @ K for K in Ks]
        print(f"Scaling inputs by {scale}: ({W}, {H}) -> {target_size}")
    else:
        target_size = None

    front_cam_idx = _front_camera_index(cam_names)
    print(
        "Logging front-camera masked depth diagnostics "
        f"from {cam_names[front_cam_idx]} to tracking_status.json"
    )

    print(f"Starting multi-view tracking for {num_frames} frames across {num_cameras} cameras")
    for i in tqdm(range(num_frames), desc="Tracking"):
        rgbs = [next(it) for it in frame_iterators]
        depths = [DepthImage.from_array(depth_sources[j][i]).depth for j in range(num_cameras)]
        masks = [mask_sources[j][i] > 128 for j in range(num_cameras)]

        for j in range(num_cameras):
            if target_size is not None:
                rgbs[j] = cv2.resize(rgbs[j], target_size, interpolation=cv2.INTER_AREA)
            if depths[j].shape[:2] != rgbs[j].shape[:2]:
                depths[j] = cv2.resize(
                    depths[j], (rgbs[j].shape[1], rgbs[j].shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            if masks[j].shape[:2] != rgbs[j].shape[:2]:
                masks[j] = cv2.resize(
                    masks[j].astype(np.uint8), (rgbs[j].shape[1], rgbs[j].shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

        result = tracker.track(
            rgbs, depths, masks, Ks, Ts,
            frame_index=i,
            register_iteration=est_refine_iter,
            track_iteration=track_refine_iter,
        )
        avg_pose = result.avg_pose
        world_poses = result.world_poses
        visible_ratios = result.visible_ratios
        select_idx = result.select_idx
        status = _status_with_camera_names(result.status, cam_names, frame_index=i)
        front_depth_stats = _masked_depth_stats(depths[front_cam_idx], masks[front_cam_idx])
        status["front_camera_name"] = cam_names[front_cam_idx]
        status["front_masked_depth_mean_m"] = front_depth_stats["mean_m"]
        status["front_masked_depth_valid_pixels"] = front_depth_stats["valid_pixels"]
        pose_valid = bool(status.get("pose_valid", bool(select_idx.any())))

        all_poses.append(world_poses)
        frame_mask = np.zeros(num_cameras, dtype=bool)
        frame_mask[select_idx] = True
        select_mask.append(frame_mask)
        output_poses.append(avg_pose.reshape(4, 4))
        pose_valid_mask.append(pose_valid)
        tracking_status.append(status)
        visible_ratios_history.append(np.asarray(visible_ratios, dtype=float))

        if debug >= 2 and i % 30 == 0:
            for j in range(num_cameras):
                vis = rgbs[j].copy()
                if pose_valid:
                    incam_pose = np.linalg.inv(Ts[j]) @ avg_pose
                    center_pose = incam_pose @ np.linalg.inv(tracker.to_origin)
                    vis = draw_posed_3d_box(
                        Ks[j], img=vis, ob_in_cam=center_pose,
                        bbox=tracker.bbox, linewidth=max(1, round(2 * scale)),
                    )
                    vis = draw_xyz_axis(
                        vis, ob_in_cam=center_pose, scale=0.1, K=Ks[j],
                        thickness=max(1, round(3 * scale)), transparency=0, is_input_rgb=True,
                    )
                iio.imwrite(debug_image_dirs[j] / f"{i:06d}.png", vis)

    output_poses = np.array(output_poses)
    pose_valid_mask = np.array(pose_valid_mask, dtype=bool)
    select_mask = np.array(select_mask, dtype=bool)
    visible_ratios_history = np.array(visible_ratios_history, dtype=float)
    parent = pose_path.parent

    np.save(parent / "poses_forward.npy", output_poses)
    np.save(parent / "pose_valid_mask_forward.npy", pose_valid_mask)
    with open(parent / "tracking_status_forward.json", "w") as f:
        json.dump(tracking_status, f, indent=2)

    repair_status = []
    if repair_config.enabled:
        print("Running backward repair")
        frame_context = TrackingFrameContext(
            frame_sources=frame_sources,
            depth_sources=depth_sources,
            mask_sources=mask_sources,
            cam_names=cam_names,
            Ks=Ks,
            Ts=Ts,
            scale_target_size=target_size,
        )
        (
            output_poses,
            pose_valid_mask,
            tracking_status,
            select_mask,
            visible_ratios_history,
            repair_status,
            repair_candidate_poses,
        ) = _run_backward_repair(
            tracker=tracker,
            poses=output_poses,
            pose_valid_mask=pose_valid_mask,
            tracking_status=tracking_status,
            select_mask=select_mask,
            visible_ratios_history=visible_ratios_history,
            frame_context=frame_context,
            repair_config=repair_config,
        )
        np.save(parent / "poses_repaired.npy", output_poses)
        np.save(parent / "poses_repair_candidate.npy", repair_candidate_poses)
        with open(parent / "repair_status.json", "w") as f:
            json.dump(repair_status, f, indent=2)

    print(f"Applying Two Euro filter to {pose_valid_mask.sum()} valid poses")
    filtered_poses = _filter_valid_pose_segments(
        output_poses,
        pose_valid_mask,
        smooth_across_recovery=smooth_across_recovery,
    )
    np.save(pose_path, filtered_poses)
    np.save(parent / "pose_valid_mask.npy", pose_valid_mask)
    with open(parent / "tracking_status.json", "w") as f:
        json.dump(tracking_status, f, indent=2)
    print(f"Saved poses to {pose_path}")
    print(f"Saved pose validity mask to {parent / 'pose_valid_mask.npy'}")

    if debug >= 1:
        np.save(parent / "all_poses_forward.npy", np.array(all_poses))
        np.save(parent / "select_mask.npy", select_mask)
        np.save(parent / "visible_ratios.npy", visible_ratios_history)

        _render_tiled_debug_video(
            filtered_poses, cam_names, frame_sources, Ks_orig, Ts, tracker,
            select_mask, pose_valid_mask, pose_path, num_frames, scale=scale,
        )

    # if debug > 1:
    #     _render_debug_videos(
    #         filtered_poses, cam_names, frame_sources, Ks_orig, Ts, tracker,
    #         pose_path, num_frames,
    #     )


def _render_debug_videos(
    poses: np.ndarray,
    cam_names: list[str],
    frame_sources: list[FrameSource],
    Ks: list[np.ndarray],
    Ts: list[np.ndarray],
    tracker: MultiViewTracker,
    pose_path: Path,
    num_frames: int,
    pose_valid_mask: np.ndarray | None = None,
):
    """Render per-camera overlay videos of the smoothed 3D bbox trajectory."""
    for j, (cam_name, fs) in enumerate(zip(cam_names, frame_sources)):
        video_path = pose_path.parent / f"{cam_name}_fp_poses.mp4"
        writer = get_video_writer(video_path, fps=30, crf=23)
        frame_iter = fs.iter_frames()
        for i in tqdm(range(num_frames), desc=f"Debug video [{cam_name}]"):
            rgb = next(frame_iter)
            vis = rgb
            if pose_valid_mask is None or pose_valid_mask[i]:
                pose = poses[i]
                incam_pose = np.linalg.inv(Ts[j]) @ pose
                center_pose = incam_pose @ np.linalg.inv(tracker.to_origin)
                vis = draw_posed_3d_box(
                    Ks[j], img=vis, ob_in_cam=center_pose, bbox=tracker.bbox,
                )
                vis = draw_xyz_axis(
                    vis, ob_in_cam=center_pose, scale=0.1, K=Ks[j],
                    thickness=3, transparency=0, is_input_rgb=True,
                )
            writer.write_frame(vis)
        writer.close()
        print(f"Debug video saved: {video_path}")


_VIEW_ORDER = {"front": 0, "back": 1, "left": 2, "right": 3}


def _view_sort_key(name: str) -> int:
    lower = name.lower()
    for key, order in _VIEW_ORDER.items():
        if key in lower:
            return order
    return len(_VIEW_ORDER)


def _front_camera_index(cam_names: list[str]) -> int:
    for i, name in enumerate(cam_names):
        if "front" in name.lower():
            return i
    return min(range(len(cam_names)), key=lambda i: _view_sort_key(cam_names[i]))


def _masked_depth_stats(depth: np.ndarray, mask: np.ndarray) -> dict:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth >= 0.001)
    valid_depths = np.asarray(depth, dtype=np.float64)[valid]
    if len(valid_depths) == 0:
        return {"mean_m": None, "valid_pixels": 0}
    return {
        "mean_m": float(valid_depths.mean()),
        "valid_pixels": int(len(valid_depths)),
    }


def _render_tiled_debug_video(
    poses: np.ndarray,
    cam_names: list[str],
    frame_sources: list[FrameSource],
    Ks: list[np.ndarray],
    Ts: list[np.ndarray],
    tracker: MultiViewTracker,
    select_mask: np.ndarray,
    pose_valid_mask: np.ndarray,
    pose_path: Path,
    num_frames: int,
    scale: float,
    border_width: int = 10,
):
    """Render a 2x2 tiled video with green borders on best-view cameras.

    Views are arranged in Z-order: front (top-left), back (top-right),
    left (bottom-left), right (bottom-right).

    Args:
        select_mask: (num_frames, num_cameras) boolean array indicating which
            cameras were selected as best for each frame.
    """
    sorted_cam_indices = sorted(range(len(cam_names)), key=lambda i: _view_sort_key(cam_names[i]))
    frame_iters = [frame_sources[j].iter_frames() for j in sorted_cam_indices]

    box_lw = max(1, round(2 * scale))
    axis_thickness = max(1, round(3 * scale))
    border_width = max(1, round(border_width * scale))

    video_path = pose_path.parent / "mv_tiled_fp_poses.mp4"
    writer = None

    for i in tqdm(range(num_frames), desc="Tiled debug video"):
        tiles = []
        pose_valid = bool(pose_valid_mask[i])
        for slot, orig_j in enumerate(sorted_cam_indices):
            rgb = next(frame_iters[slot])
            W_orig, H_orig = rgb.shape[1], rgb.shape[0]
            tile_w = int(W_orig * scale)
            tile_h = int(H_orig * scale)
            vis = cv2.resize(rgb, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            K_tile = Ks[orig_j].copy()
            K_tile[0] *= tile_w / W_orig
            K_tile[1] *= tile_h / H_orig
            if pose_valid:
                pose = poses[i]
                incam_pose = np.linalg.inv(Ts[orig_j]) @ pose
                center_pose = incam_pose @ np.linalg.inv(tracker.to_origin)
                vis = draw_posed_3d_box(
                    K_tile, img=vis, ob_in_cam=center_pose, bbox=tracker.bbox,
                    linewidth=box_lw,
                )
                vis = draw_xyz_axis(
                    vis, ob_in_cam=center_pose, scale=0.1, K=K_tile,
                    thickness=axis_thickness, transparency=0, is_input_rgb=True,
                )
            if not pose_valid:
                border_color = (160, 160, 160)
            elif select_mask[i, orig_j]:
                border_color = (0, 255, 0)
            else:
                border_color = (255, 0, 0)
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), border_color, border_width)
            tiles.append(vis)

        top = np.concatenate([tiles[0], tiles[1]], axis=1)
        bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
        tiled = np.concatenate([top, bottom], axis=0)
        frame_text = f"Frame {i}"
        (tw, th), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(
            tiled, frame_text, (tiled.shape[1] - tw - 10, th + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )

        if writer is None:
            writer = get_video_writer(video_path, fps=30, crf=23)
        writer.write_frame(tiled)

    if writer is not None:
        writer.close()
    print(f"Tiled debug video saved: {video_path}")


def _run_backward_repair(
    tracker: MultiViewTracker,
    poses: np.ndarray,
    pose_valid_mask: np.ndarray,
    tracking_status: list[dict],
    select_mask: np.ndarray,
    visible_ratios_history: np.ndarray,
    frame_context: TrackingFrameContext,
    repair_config: RepairConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Repair recovery windows and tracked pose snaps by tracking backward."""
    repair_records: list[dict] = []
    candidate_poses = np.full_like(poses, np.nan, dtype=float)
    forward_poses = poses.copy()
    forward_status = [dict(status) for status in tracking_status]
    if len(poses) == 0:
        return (
            poses, pose_valid_mask, tracking_status, select_mask,
            visible_ratios_history, repair_records, candidate_poses,
        )

    repair_anchors = _detect_repair_anchors(
        statuses=tracking_status,
        poses=poses,
        num_cameras=len(frame_context.cam_names),
        repair_config=repair_config,
    )
    if not repair_anchors:
        return (
            poses, pose_valid_mask, tracking_status, select_mask,
            visible_ratios_history, repair_records, candidate_poses,
        )

    old_recovery_enabled = tracker.recovery_enabled
    old_under_supported_recovery_enabled = tracker.under_supported_recovery_enabled
    sequential_cache: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    winning_poses: list[np.ndarray | None] = [None] * len(poses)
    winning_statuses: list[dict | None] = [None] * len(poses)
    winning_select: list[np.ndarray | None] = [None] * len(poses)
    winning_visible: list[np.ndarray | None] = [None] * len(poses)
    winning_anchor: list[int | None] = [None] * len(poses)

    try:
        tracker.recovery_enabled = False
        tracker.under_supported_recovery_enabled = False
        for anchor in repair_anchors:
            start_floor = int(anchor["start_floor"])
            trigger_frame = int(anchor["trigger_frame"])
            anchor_idx = anchor["anchor_frame"]
            record = {
                "trigger_type": anchor["trigger_type"],
                "trigger_frame": int(trigger_frame),
                "start_floor": int(start_floor),
                "start_floor_status": forward_status[start_floor].get("status"),
                "anchor_min_views": anchor.get("anchor_min_views"),
                "anchor_frame": None if anchor_idx is None else int(anchor_idx),
                "accepted_frames": [],
                "kept_existing_frames": [],
                "rejected_frames": [],
            }
            for key in [
                "snap_start_frame",
                "snap_end_frame",
                "snap_step_frames",
                "snap_rotation_deg",
                "snap_translation_m",
                "snap_max_step_rotation_deg",
                "snap_max_step_translation_m",
            ]:
                if key in anchor:
                    record[key] = anchor[key]
            if anchor_idx is None:
                record["failure_reason"] = "no_stable_future_anchor"
                repair_records.append(record)
                continue
            if int(anchor_idx) <= start_floor:
                record["failure_reason"] = "anchor_not_after_start_floor"
                repair_records.append(record)
                continue

            tracker.seed_pose(forward_poses[anchor_idx], frame_context.Ts)
            next_future_pose = forward_poses[anchor_idx].copy()

            for frame_idx in range(anchor_idx - 1, start_floor - 1, -1):
                try:
                    rgbs, depths, masks = _load_tracking_inputs_at(
                        frame_sources=frame_context.frame_sources,
                        depth_sources=frame_context.depth_sources,
                        mask_sources=frame_context.mask_sources,
                        frame_index=frame_idx,
                        scale_target_size=frame_context.scale_target_size,
                        sequential_cache=sequential_cache,
                    )
                except Exception as exc:  # pragma: no cover - defensive diagnostics path
                    record["rejected_frames"].append({
                        "frame": int(frame_idx),
                        "reason": f"frame_load_failed: {exc}",
                    })
                    record["stopped_after_rejection"] = True
                    break

                result = tracker.track(
                    rgbs,
                    depths,
                    masks,
                    frame_context.Ks,
                    frame_context.Ts,
                    frame_index=frame_idx,
                    register_iteration=1,
                    track_iteration=repair_config.track_refine_iter,
                )
                accepted, reject_reason, delta = _accept_backward_result(
                    result=result,
                    forward_status=forward_status[frame_idx],
                    next_future_pose=next_future_pose,
                    max_step_translation_m=repair_config.max_step_translation_m,
                    max_step_rotation_deg=repair_config.max_step_rotation_deg,
                    visible_ratio_tolerance=repair_config.visible_ratio_tolerance,
                )
                if not accepted:
                    record["rejected_frames"].append({
                        "frame": int(frame_idx),
                        "reason": reject_reason,
                        **delta,
                    })
                    record["stopped_after_rejection"] = True
                    break

                candidate_pose = result.avg_pose.reshape(4, 4)
                original_status = dict(forward_status[frame_idx])
                repaired_status = _status_with_camera_names(
                    result.status, frame_context.cam_names, frame_index=frame_idx,
                )
                repaired_status = {**original_status, **repaired_status}
                repaired_status.update({
                    "status": "repair_replaced",
                    "pose_valid": True,
                    "repair_replaced": True,
                    "repair_original_status": original_status.get("status"),
                    "repair_anchor_frame": int(anchor_idx),
                    "repair_trigger_type": anchor["trigger_type"],
                    "repair_pose_delta_translation_m": delta["translation_m"],
                    "repair_pose_delta_rotation_deg": delta["rotation_deg"],
                    "repair_forward_visible_ratios": original_status.get("visible_ratios", []),
                    "repair_forward_select_idx": original_status.get("select_idx", []),
                    "repair_forward_selected_view_count": delta["forward_selected_view_count"],
                    "repair_candidate_selected_view_count": delta["candidate_selected_view_count"],
                    "repair_forward_visible_ratio_sum": delta["forward_visible_ratio_sum"],
                    "repair_candidate_visible_ratio_sum": delta["candidate_visible_ratio_sum"],
                    "repair_candidate_visible_ratios": np.asarray(result.visible_ratios, dtype=float).tolist(),
                    "repair_candidate_select_idx": np.asarray(result.select_idx, dtype=bool).tolist(),
                })
                for key in ["snap_start_frame", "snap_end_frame", "snap_rotation_deg", "snap_translation_m"]:
                    if key in anchor:
                        repaired_status[f"repair_{key}"] = anchor[key]

                if winning_poses[frame_idx] is None:
                    winning_poses[frame_idx] = candidate_pose.copy()
                    winning_statuses[frame_idx] = _json_safe(repaired_status)
                    winning_select[frame_idx] = np.asarray(result.select_idx, dtype=bool)
                    winning_visible[frame_idx] = np.asarray(result.visible_ratios, dtype=float)
                    winning_anchor[frame_idx] = int(anchor_idx)
                    record["accepted_frames"].append(int(frame_idx))
                else:
                    record["kept_existing_frames"].append({
                        "frame": int(frame_idx),
                        "existing_anchor_frame": int(winning_anchor[frame_idx]),
                    })
                next_future_pose = candidate_pose.copy()

            record["accepted_frames"] = sorted(record["accepted_frames"])
            repair_records.append(_json_safe(record))
    finally:
        tracker.recovery_enabled = old_recovery_enabled
        tracker.under_supported_recovery_enabled = old_under_supported_recovery_enabled

    for frame_idx, candidate_pose in enumerate(winning_poses):
        if candidate_pose is None:
            continue
        poses[frame_idx] = candidate_pose
        pose_valid_mask[frame_idx] = True
        tracking_status[frame_idx] = winning_statuses[frame_idx]
        select_mask[frame_idx] = winning_select[frame_idx]
        visible_ratios_history[frame_idx] = winning_visible[frame_idx]
        candidate_poses[frame_idx] = candidate_pose

    return (
        poses, pose_valid_mask, tracking_status, select_mask,
        visible_ratios_history, repair_records, candidate_poses,
    )


def _detect_repair_anchors(
    statuses: list[dict],
    poses: np.ndarray,
    num_cameras: int,
    repair_config: RepairConfig,
) -> list[dict]:
    anchors: list[dict] = []
    if repair_config.recovery_trigger_enabled:
        for start_idx in _backward_repair_start_indices(statuses):
            anchors.append({
                "trigger_type": "recovery_trigger",
                "trigger_frame": int(start_idx),
                "start_floor": int(start_idx),
                "anchor_frame": _find_backward_repair_anchor(
                    statuses,
                    poses,
                    start_idx=start_idx,
                    max_window=repair_config.recovery_trigger_max_window,
                    min_future_stable_frames=repair_config.recovery_trigger_anchor_stable_frames,
                    recovery_min_views=repair_config.recovery_min_views,
                    max_step_translation_m=repair_config.max_step_translation_m,
                    max_step_rotation_deg=repair_config.max_step_rotation_deg,
                ),
                "anchor_min_views": int(repair_config.recovery_min_views),
            })

    if repair_config.snap_trigger_enabled:
        snap_bursts = _detect_rotation_snap_bursts(
            poses=poses,
            statuses=statuses,
            outlier_window=repair_config.snap_trigger_outlier_window,
            rotation_mad_scale=repair_config.snap_trigger_rotation_mad_scale,
            translation_mad_scale=repair_config.snap_trigger_translation_mad_scale,
            min_rotation_deg=repair_config.snap_trigger_min_rotation_deg,
            max_translation_m=repair_config.snap_trigger_max_translation_m,
            max_burst_frames=repair_config.snap_trigger_max_burst_frames,
        )
        for burst in snap_bursts:
            snap_start = int(burst["snap_start_frame"])
            snap_end = int(burst["snap_end_frame"])
            start_idx = _snap_repair_start_index(
                statuses, snap_start, max_span=repair_config.snap_trigger_max_span,
            )
            anchor_idx, anchor_min_views = _find_snap_repair_anchor(
                statuses=statuses,
                poses=poses,
                search_after_idx=snap_end,
                max_window=repair_config.snap_trigger_max_span,
                stable_frames=repair_config.snap_trigger_anchor_stable_frames,
                num_cameras=num_cameras,
                recovery_min_views=repair_config.recovery_min_views,
                max_step_translation_m=repair_config.max_step_translation_m,
                max_step_rotation_deg=repair_config.max_step_rotation_deg,
            )
            anchors.append({
                **burst,
                "trigger_type": "snap_trigger",
                "trigger_frame": snap_end,
                "start_floor": int(start_idx),
                "anchor_frame": anchor_idx,
                "anchor_min_views": anchor_min_views,
            })

    return sorted(
        anchors,
        key=lambda s: (
            s["anchor_frame"] is not None,
            int(s["anchor_frame"]) if s["anchor_frame"] is not None else -1,
            int(s["trigger_frame"]),
        ),
        reverse=True,
    )


def _detect_rotation_snap_bursts(
    poses: np.ndarray,
    statuses: list[dict],
    outlier_window: int,
    rotation_mad_scale: float,
    translation_mad_scale: float,
    min_rotation_deg: float,
    max_translation_m: float,
    max_burst_frames: int,
) -> list[dict]:
    if len(poses) < 2:
        return []
    translations, rotations = _pose_step_deltas(poses)
    snap_steps = []
    radius = max(1, int(outlier_window))
    for i in range(1, len(poses)):
        if not (
            _is_tracked_snap_status(statuses[i - 1])
            and _is_tracked_snap_status(statuses[i])
        ):
            continue
        rot_med, rot_mad = _local_median_mad(rotations, i, radius)
        trans_med, trans_mad = _local_median_mad(translations, i, radius)
        rotation_threshold = rot_med + rotation_mad_scale * max(rot_mad, 1.0)
        translation_threshold = trans_med + translation_mad_scale * max(trans_mad, 0.005)
        if rotations[i] < min_rotation_deg:
            continue
        if rotations[i] <= rotation_threshold:
            continue
        if translations[i] > max_translation_m:
            continue
        if translations[i] > translation_threshold:
            continue
        snap_steps.append(i)

    bursts = []
    for group in _group_consecutive_indices(snap_steps, max_group_size=max(1, int(max_burst_frames))):
        start_frame = int(group[0] - 1)
        end_frame = int(group[-1])
        trans_m, rot_deg = _pose_delta(poses[start_frame], poses[end_frame])
        bursts.append({
            "snap_start_frame": start_frame,
            "snap_end_frame": end_frame,
            "snap_step_frames": [int(i) for i in group],
            "snap_rotation_deg": float(rot_deg),
            "snap_translation_m": float(trans_m),
            "snap_max_step_rotation_deg": float(np.max(rotations[group])),
            "snap_max_step_translation_m": float(np.max(translations[group])),
        })
    return bursts


def _pose_step_deltas(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translations = np.full(len(poses), np.nan, dtype=float)
    rotations = np.full(len(poses), np.nan, dtype=float)
    for i in range(1, len(poses)):
        translations[i], rotations[i] = _pose_delta(poses[i - 1], poses[i])
    return translations, rotations


def _local_median_mad(values: np.ndarray, center_idx: int, radius: int) -> tuple[float, float]:
    lo = max(1, center_idx - radius)
    hi = min(len(values), center_idx + radius + 1)
    sample = np.asarray(values[lo:hi], dtype=float)
    offsets = np.arange(lo, hi)
    sample = sample[(offsets != center_idx) & np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 0.0
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    return med, mad


def _group_consecutive_indices(indices: list[int], max_group_size: int) -> list[list[int]]:
    if not indices:
        return []
    groups = []
    current = [int(indices[0])]
    for idx in indices[1:]:
        idx = int(idx)
        if idx == current[-1] + 1 and len(current) < max_group_size:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    return groups


def _is_tracked_snap_status(status: dict) -> bool:
    return status.get("status") == "tracked" and bool(status.get("pose_valid", False))


def _snap_repair_start_index(statuses: list[dict], snap_start_frame: int, max_span: int) -> int:
    lo = max(0, int(snap_start_frame) - max(1, int(max_span)))
    for i in range(int(snap_start_frame), lo - 1, -1):
        if _is_snap_span_origin_status(statuses[i]):
            return i
    return lo


def _is_snap_span_origin_status(status: dict) -> bool:
    return _is_backward_repair_start(status) or bool(status.get("under_supported", False))


def _find_snap_repair_anchor(
    statuses: list[dict],
    poses: np.ndarray,
    search_after_idx: int,
    max_window: int,
    stable_frames: int,
    num_cameras: int,
    recovery_min_views: int,
    max_step_translation_m: float,
    max_step_rotation_deg: float,
) -> tuple[int | None, int | None]:
    preferred_views = max(1, int(num_cameras))
    anchor = _find_backward_repair_anchor_after(
        statuses,
        poses,
        search_after_idx=search_after_idx,
        max_window=max_window,
        min_future_stable_frames=stable_frames,
        min_views=preferred_views,
        max_step_translation_m=max_step_translation_m,
        max_step_rotation_deg=max_step_rotation_deg,
    )
    if anchor is not None:
        return anchor, preferred_views

    fallback_views = max(int(recovery_min_views), preferred_views - 1)
    if fallback_views >= preferred_views:
        return None, None
    anchor = _find_backward_repair_anchor_after(
        statuses,
        poses,
        search_after_idx=search_after_idx,
        max_window=max_window,
        min_future_stable_frames=stable_frames,
        min_views=fallback_views,
        max_step_translation_m=max_step_translation_m,
        max_step_rotation_deg=max_step_rotation_deg,
    )
    if anchor is not None:
        return anchor, fallback_views
    return None, None


def _load_tracking_inputs_at(
    frame_sources: list[FrameSource],
    depth_sources: list[FrameSource],
    mask_sources: list[FrameSource],
    frame_index: int,
    scale_target_size: tuple[int, int] | None,
    sequential_cache: dict[tuple[str, int], dict[int, np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    num_cameras = len(frame_sources)
    rgbs = [
        _read_frame_source_at(frame_sources[j], frame_index, ("rgb", j), sequential_cache)
        for j in range(num_cameras)
    ]
    depths = [
        DepthImage.from_array(
            _read_frame_source_at(depth_sources[j], frame_index, ("depth", j), sequential_cache)
        ).depth
        for j in range(num_cameras)
    ]
    masks = [
        _read_frame_source_at(mask_sources[j], frame_index, ("mask", j), sequential_cache) > 128
        for j in range(num_cameras)
    ]

    for j in range(num_cameras):
        if scale_target_size is not None:
            rgbs[j] = cv2.resize(rgbs[j], scale_target_size, interpolation=cv2.INTER_AREA)
        if depths[j].shape[:2] != rgbs[j].shape[:2]:
            depths[j] = cv2.resize(
                depths[j], (rgbs[j].shape[1], rgbs[j].shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        if masks[j].shape[:2] != rgbs[j].shape[:2]:
            masks[j] = cv2.resize(
                masks[j].astype(np.uint8), (rgbs[j].shape[1], rgbs[j].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
    return rgbs, depths, masks


def _read_frame_source_at(
    source: FrameSource,
    frame_index: int,
    cache_key: tuple[str, int],
    sequential_cache: dict[tuple[str, int], dict[int, np.ndarray]],
) -> np.ndarray:
    try:
        return source[frame_index]
    except (RuntimeError, NotImplementedError):
        cache = sequential_cache.setdefault(cache_key, {})
        if frame_index in cache:
            return cache[frame_index]
        for i, frame in enumerate(source.iter_frames()):
            if i == frame_index:
                cache[i] = frame
                return frame
        raise IndexError(f"Frame index {frame_index} out of range for {source.path}")


def _backward_repair_start_indices(statuses: list[dict]) -> list[int]:
    return [i for i, status in enumerate(statuses) if _is_backward_repair_start(status)]


def _is_backward_repair_start(status: dict) -> bool:
    if status.get("status") in {"held", "recovered", "partially_recovered"}:
        return True
    if bool(status.get("recovery_attempted", False)):
        return True
    if not bool(status.get("pose_valid", False)):
        return True
    return bool(status.get("under_supported_initialization", False))


def _find_backward_repair_anchor(
    statuses: list[dict],
    poses: np.ndarray,
    start_idx: int,
    max_window: int,
    min_future_stable_frames: int,
    recovery_min_views: int,
    max_step_translation_m: float,
    max_step_rotation_deg: float,
) -> int | None:
    return _find_backward_repair_anchor_after(
        statuses,
        poses,
        search_after_idx=start_idx,
        max_window=max_window,
        min_future_stable_frames=min_future_stable_frames,
        min_views=recovery_min_views,
        max_step_translation_m=max_step_translation_m,
        max_step_rotation_deg=max_step_rotation_deg,
    )


def _find_backward_repair_anchor_after(
    statuses: list[dict],
    poses: np.ndarray,
    search_after_idx: int,
    max_window: int,
    min_future_stable_frames: int,
    min_views: int,
    max_step_translation_m: float,
    max_step_rotation_deg: float,
) -> int | None:
    if min_future_stable_frames <= 0:
        return None
    end = min(len(statuses), search_after_idx + max(1, int(max_window)) + 1)
    last_run_start = end - int(min_future_stable_frames)
    for run_start in range(search_after_idx + 1, last_run_start + 1):
        run_end = run_start + int(min_future_stable_frames)
        if not all(
            _is_stable_backward_anchor_status(statuses[i], min_views)
            for i in range(run_start, run_end)
        ):
            continue
        deltas_ok = True
        for i in range(run_start + 1, run_end):
            trans_m, rot_deg = _pose_delta(poses[i - 1], poses[i])
            if trans_m > max_step_translation_m or rot_deg > max_step_rotation_deg:
                deltas_ok = False
                break
        if deltas_ok:
            return run_end - 1
    return None


def _is_stable_backward_anchor_status(status: dict, recovery_min_views: int) -> bool:
    if status.get("status") != "tracked":
        return False
    if not bool(status.get("pose_valid", False)):
        return False
    if bool(status.get("under_supported", False)):
        return False
    if _selected_view_count(status) < recovery_min_views:
        return False
    return _visible_ratio_sum(status) > 0.0


def _accept_backward_result(
    result,
    forward_status: dict,
    next_future_pose: np.ndarray,
    max_step_translation_m: float,
    max_step_rotation_deg: float,
    visible_ratio_tolerance: float = 0.0,
) -> tuple[bool, str | None, dict]:
    status = result.status
    trans_m, rot_deg = _pose_delta(result.avg_pose, next_future_pose)
    candidate_count = int(np.count_nonzero(np.asarray(result.select_idx, dtype=bool)))
    forward_count = _selected_view_count(forward_status)
    candidate_visible_sum = float(np.nansum(np.asarray(result.visible_ratios, dtype=float)))
    forward_visible_sum = _visible_ratio_sum(forward_status)
    delta = {
        "translation_m": float(trans_m),
        "rotation_deg": float(rot_deg),
        "candidate_selected_view_count": int(candidate_count),
        "forward_selected_view_count": int(forward_count),
        "candidate_visible_ratio_sum": float(candidate_visible_sum),
        "forward_visible_ratio_sum": float(forward_visible_sum),
    }
    if not bool(status.get("pose_valid", False)):
        return False, "repair_pose_invalid", delta
    if candidate_visible_sum + float(visible_ratio_tolerance) + 1e-6 < forward_visible_sum:
        return False, "lower_visible_ratio_sum", delta
    if trans_m > max_step_translation_m:
        return False, "translation_jump", delta
    if rot_deg > max_step_rotation_deg:
        return False, "rotation_jump", delta
    return True, None, delta


def _selected_view_count(status: dict) -> int:
    select_idx = np.asarray(status.get("select_idx", []), dtype=bool)
    return int(np.count_nonzero(select_idx))


def _visible_ratio_sum(status: dict) -> float:
    visible_ratios = np.asarray(status.get("visible_ratios", []), dtype=float)
    if visible_ratios.size == 0:
        return 0.0
    return float(np.nansum(visible_ratios))


def _pose_delta(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[float, float]:
    pose_a = np.asarray(pose_a, dtype=float).reshape(4, 4)
    pose_b = np.asarray(pose_b, dtype=float).reshape(4, 4)
    trans_m = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
    rel_R = pose_a[:3, :3].T @ pose_b[:3, :3]
    cos_angle = (np.trace(rel_R) - 1.0) / 2.0
    rot_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    return trans_m, rot_deg


def _filter_valid_pose_segments(
    poses: np.ndarray,
    pose_valid_mask: np.ndarray,
    smooth_across_recovery: bool,
) -> np.ndarray:
    """Filter valid pose estimates while leaving invalid placeholders unchanged."""
    filtered = poses.copy()
    valid_indices = np.where(pose_valid_mask)[0]
    if len(valid_indices) == 0:
        return filtered

    if smooth_across_recovery:
        filtered[valid_indices] = pose_two_euro_filter(poses[valid_indices])
        return filtered

    split_points = np.where(np.diff(valid_indices) > 1)[0] + 1
    for segment in np.split(valid_indices, split_points):
        if len(segment) == 0:
            continue
        filtered[segment] = pose_two_euro_filter(poses[segment])
    return filtered


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _status_with_camera_names(status: dict, cam_names: list[str], frame_index: int) -> dict:
    record = dict(status)
    record["frame_index"] = int(frame_index)
    select_idx = np.asarray(record.get("select_idx", []), dtype=bool)
    if len(select_idx) == len(cam_names):
        record["selected_cameras"] = [
            cam_names[j] for j, selected in enumerate(select_idx) if selected
        ]

    candidate_idx = np.asarray(record.get("recovery_candidate_idx", []), dtype=bool)
    if len(candidate_idx) == len(cam_names):
        record["recovery_candidate_cameras"] = [
            cam_names[j] for j, selected in enumerate(candidate_idx) if selected
        ]

    recovery_select_idx = np.asarray(record.get("recovery_select_idx", []), dtype=bool)
    if len(recovery_select_idx) == len(cam_names):
        record["recovery_selected_cameras"] = [
            cam_names[j] for j, selected in enumerate(recovery_select_idx) if selected
        ]

    under_supported_candidate_idx = np.asarray(
        record.get("under_supported_recovery_candidate_idx", []), dtype=bool,
    )
    if len(under_supported_candidate_idx) == len(cam_names):
        record["under_supported_recovery_candidate_cameras"] = [
            cam_names[j] for j, selected in enumerate(under_supported_candidate_idx)
            if selected
        ]

    under_supported_select_idx = np.asarray(
        record.get("under_supported_recovery_select_idx", []), dtype=bool,
    )
    if len(under_supported_select_idx) == len(cam_names):
        record["under_supported_recovery_selected_cameras"] = [
            cam_names[j] for j, selected in enumerate(under_supported_select_idx)
            if selected
        ]
    return _json_safe(record)


def _recovery_config_from_cfg(cfg) -> RecoveryConfig:
    recovery = cfg.recovery
    under_supported = recovery.get("under_supported", {})
    return RecoveryConfig(
        enabled=recovery.get("enabled", True),
        refine_iter=recovery.get("refine_iter", 5),
        min_views=recovery.get("min_views", 1),
        min_valid_depth_pixels=recovery.get("min_valid_depth_pixels", 25),
        visible_ratio_cutoff=recovery.get("visible_ratio_cutoff", 0.3),
        attempt_stride=recovery.get("attempt_stride", 1),
        under_supported_enabled=under_supported.get("enabled", True),
        mask_pose_iou_cutoff=under_supported.get("mask_pose_iou_cutoff", 0.05),
        mask_explained_ratio_cutoff=under_supported.get("mask_explained_ratio_cutoff", 0.10),
    )


def _repair_config_from_cfg(
    cfg,
    recovery_min_views: int,
    track_refine_iter: int,
) -> RepairConfig:
    repair = cfg.repair
    arbitration = repair.get("arbitration", {})
    recovery_trigger = repair.get("recovery_trigger", {})
    snap_trigger = repair.get("snap_trigger", {})
    return RepairConfig(
        enabled=repair.get("enabled", True),
        max_step_translation_m=arbitration.get("max_step_translation_m", 0.15),
        max_step_rotation_deg=arbitration.get("max_step_rotation_deg", 45.0),
        visible_ratio_tolerance=arbitration.get("visible_ratio_tolerance", 0.10),
        recovery_trigger_enabled=recovery_trigger.get("enabled", True),
        recovery_trigger_max_window=recovery_trigger.get("max_window", 30),
        recovery_trigger_anchor_stable_frames=recovery_trigger.get("anchor_stable_frames", 5),
        snap_trigger_enabled=snap_trigger.get("enabled", True),
        snap_trigger_max_span=snap_trigger.get("max_span", 150),
        snap_trigger_anchor_stable_frames=snap_trigger.get("anchor_stable_frames", 8),
        snap_trigger_outlier_window=snap_trigger.get("outlier_window", 15),
        snap_trigger_rotation_mad_scale=snap_trigger.get("rotation_mad_scale", 5.0),
        snap_trigger_translation_mad_scale=snap_trigger.get("translation_mad_scale", 5.0),
        snap_trigger_min_rotation_deg=snap_trigger.get("min_rotation_deg", 10.0),
        snap_trigger_max_translation_m=snap_trigger.get("max_translation_m", 0.08),
        snap_trigger_max_burst_frames=snap_trigger.get("max_burst_frames", 3),
        recovery_min_views=recovery_min_views,
        track_refine_iter=track_refine_iter,
    )


def mv_videos_to_poses_from_config(cfg):
    """Resolve config fields and call mv_videos_to_poses."""
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)
    scale = cfg.get("scale", 1.0)

    cam_names: list[str] = []
    cam_intrinsics: list[np.ndarray] = []
    cam_extrinsics: list[np.ndarray] = []
    rgb_paths: list[Path] = []
    depth_dirs: list[Path] = []
    mask_dirs: list[Path] = []

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        cam_names.append(cam.name)
        cam_intrinsics.append(cam.param.K)
        cam_extrinsics.append(cam.param.T)

        rgb_paths.append(Path(cfg.rgb_path_template.format(cam_name=cam.name)))
        depth_dirs.append(Path(cfg.depth_path_template.format(cam_name=cam.name)))
        mask_dirs.append(Path(cfg.mask_path_template.format(cam_name=cam.name)))

    mesh_path = Path(cfg.mesh_path)
    symmetry_path = cfg.get("symmetry_path", None)
    if symmetry_path is None:
        auto = mesh_path.parent / "output_symmetry.json"
        if auto.exists():
            symmetry_path = auto
    else:
        symmetry_path = Path(symmetry_path)

    track_refine_iter = cfg.get("track_refine_iter", 2)
    recovery_config = _recovery_config_from_cfg(cfg)
    repair_config = _repair_config_from_cfg(
        cfg,
        recovery_min_views=recovery_config.min_views,
        track_refine_iter=track_refine_iter,
    )

    mv_videos_to_poses(
        cam_names=cam_names,
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        rgb_paths=rgb_paths,
        depth_dirs=depth_dirs,
        mask_dirs=mask_dirs,
        mesh_path=mesh_path,
        weights_dir=cfg.weights_dir,
        pose_path=Path(cfg.pose_path),
        symmetry_path=symmetry_path,
        scale=scale,
        depth_direction_trust=cfg.get("depth_direction_trust", 0.5),
        visible_ratio_cutoff_high=cfg.get("visible_ratio_cutoff_high", 0.3),
        visible_ratio_cutoff_low=cfg.get("visible_ratio_cutoff_low", 0.3),
        precision_high=cfg.get("precision_high", 1.0),
        precision_low=cfg.get("precision_low", 0.01),
        est_refine_iter=cfg.get("est_refine_iter", 5),
        track_refine_iter=track_refine_iter,
        recovery_config=recovery_config,
        repair_config=repair_config,
        smooth_across_recovery=cfg.get("smooth_across_recovery", False),
        debug=cfg.get("debug", 0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-view 6-DoF object tracking with FoundationPose")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--rgb_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--symmetry_path", type=str, default=None,
                        help="Optional BOP-style symmetry JSON (defaults to <mesh_dir>/output_symmetry.json)")
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scale", type=float, default=None,
                        help="Scale factor for input resolution (e.g. 0.5 for half)")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--debug", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_videos_to_poses.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))

    overrides = {}
    for key in [
        "camera_params_path", "rgb_dir", "depth_dir",
        "mask_dir", "mesh_path", "symmetry_path", "weights_dir", "output_dir", "scale", "debug",
    ]:
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    mv_videos_to_poses_from_config(cfg)
