"""Multi-view 6-DoF object tracking with FoundationPose."""

import argparse
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import trimesh

from v2d.common.datatypes import DepthImage, Mask
from v2d.mesh.lib.mesh import Mesh
from v2d.mv.io.video import FrameSource, get_video_writer
from v2d.mv.math.numpy_fn import pose_two_euro_filter
from v2d.mv.rig import RigConfig

from .fp_utils import draw_posed_3d_box, draw_xyz_axis
from .multiview_tracker import MultiViewTracker


def mv_videos_to_poses(
    cam_names: list[str],
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    frame_sources: list[FrameSource],
    depth_dirs: list[Path],
    mask_dirs: list[Path],
    mesh_path: Path,
    weights_dir: str,
    pose_path: Path,
    scale: float = 0.5,
    depth_direction_trust: float = 0.5,
    visible_ratio_cutoff_high: float = 0.3,
    visible_ratio_cutoff_low: float = 0.01,
    precision_high: float = 1.0,
    precision_low: float = 0.01,
    est_refine_iter: int = 5,
    track_refine_iter: int = 2,
    debug: int = 0,
):
    """Run multi-view FoundationPose tracking.

    Args:
        cam_names: per-camera names (used for debug output naming).
        cam_intrinsics: list of (3,3) K matrices, one per camera.
        cam_extrinsics: list of (4,4) cam-to-world transforms, one per camera.
        frame_sources: per-camera RGB FrameSource (image dir or video).
        depth_dirs: per-camera depth directories (inverse-depth PNGs via DepthImage).
        mask_dirs: per-camera object mask directories (first PNG used for registration).
        mesh_path: path to the object mesh file.
        weights_dir: path to FoundationPose weights.
        pose_path: output path for filtered poses .npy file.
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
        debug: 0=off, 1=overlay videos after processing, 2=also per-frame images every 30 frames.
    """
    pose_path = Path(pose_path)
    pose_path.parent.mkdir(parents=True, exist_ok=True)

    num_cameras = len(cam_intrinsics)
    Ks = cam_intrinsics
    Ts = cam_extrinsics

    mesh = Mesh.load(str(mesh_path))
    tm = mesh.to_trimesh()
    _, obb_extents = trimesh.bounds.oriented_bounds(tm)
    print(f"Mesh: {len(tm.vertices)} verts, OBB extents={obb_extents}, min={obb_extents.min():.4f}")
    tracker = MultiViewTracker(
        mesh, weights_dir, num_cameras,
        depth_direction_trust=depth_direction_trust,
        visible_ratio_cutoff_high=visible_ratio_cutoff_high,
        visible_ratio_cutoff_low=visible_ratio_cutoff_low,
        precision_high=precision_high,
        precision_low=precision_low,
    )

    mask_file_lists = []
    for d in mask_dirs:
        files = sorted(Path(d).glob("*.png"))
        if not files:
            raise FileNotFoundError(f"No PNG masks found in {d}")
        mask_file_lists.append(files)

    depth_file_lists = []
    for d in depth_dirs:
        files = sorted(Path(d).glob("*.png"))
        if not files:
            raise FileNotFoundError(f"No depth PNG files in {d}")
        depth_file_lists.append(files)

    num_frames = frame_sources[0].n_frames
    for j, (fs, dl, ml) in enumerate(zip(frame_sources, depth_file_lists, mask_file_lists)):
        if fs.n_frames != num_frames or len(dl) != num_frames or len(ml) != num_frames:
            raise ValueError(
                f"camera {cam_names[j]}: frame count mismatch "
                f"(rgb={fs.n_frames}, depth={len(dl)}, mask={len(ml)}, expected={num_frames})"
            )
    frame_iterators = [fs.iter_frames() for fs in frame_sources]

    all_poses = []
    select_mask = []
    output_poses = []

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

    print(f"Starting multi-view tracking for {num_frames} frames across {num_cameras} cameras")
    for i in tqdm(range(num_frames), desc="Tracking"):
        rgbs = [next(it) for it in frame_iterators]
        depths = [DepthImage.load(str(depth_file_lists[j][i])).depth for j in range(num_cameras)]
        masks = [Mask.load(str(mask_file_lists[j][i])).mask > 0.5 for j in range(num_cameras)]

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

        if i == 0:
            avg_pose, world_poses, visible_ratios, select_idx = tracker.register(
                rgbs, depths, masks, Ks, Ts, iteration=est_refine_iter,
            )
        else:
            avg_pose, world_poses, visible_ratios, select_idx = tracker.track(
                rgbs, depths, masks, Ks, Ts, iteration=track_refine_iter,
            )

        all_poses.append(world_poses)
        frame_mask = np.zeros(num_cameras, dtype=bool)
        frame_mask[select_idx] = True
        select_mask.append(frame_mask)
        output_poses.append(avg_pose.reshape(4, 4))

        if debug >= 2 and i % 30 == 0:
            for j in range(num_cameras):
                incam_pose = np.linalg.inv(Ts[j]) @ avg_pose
                center_pose = incam_pose @ np.linalg.inv(tracker.to_origin)
                vis = draw_posed_3d_box(
                    Ks[j], img=rgbs[j].copy(), ob_in_cam=center_pose,
                    bbox=tracker.bbox, linewidth=max(1, round(2 * scale)),
                )
                vis = draw_xyz_axis(
                    vis, ob_in_cam=center_pose, scale=0.1, K=Ks[j],
                    thickness=max(1, round(3 * scale)), transparency=0, is_input_rgb=True,
                )
                iio.imwrite(debug_image_dirs[j] / f"{i:06d}.png", vis)

    output_poses = np.array(output_poses)

    print(f"Applying Two Euro filter to {output_poses.shape[0]} poses")
    filtered_poses = pose_two_euro_filter(output_poses)
    np.save(pose_path, filtered_poses)

    if debug >= 1:
        parent = pose_path.parent
        np.save(parent / "poses_raw.npy", output_poses)
        np.save(parent / "all_poses_raw.npy", np.array(all_poses))
        np.save(parent / "select_mask.npy", np.array(select_mask))

        _render_tiled_debug_video(
            filtered_poses, cam_names, frame_sources, Ks_orig, Ts, tracker,
            np.array(select_mask), pose_path, num_frames, scale=scale,
        )

    # if debug > 1:
    #     _render_debug_videos(
    #         filtered_poses, cam_names, frame_sources, Ks_orig, Ts, tracker,
    #         pose_path, num_frames,
    #     )

    print(f"Saved poses to {pose_path}")


def _render_debug_videos(
    poses: np.ndarray,
    cam_names: list[str],
    frame_sources: list[FrameSource],
    Ks: list[np.ndarray],
    Ts: list[np.ndarray],
    tracker: MultiViewTracker,
    pose_path: Path,
    num_frames: int,
):
    """Render per-camera overlay videos of the smoothed 3D bbox trajectory."""
    for j, (cam_name, fs) in enumerate(zip(cam_names, frame_sources)):
        video_path = pose_path.parent / f"{cam_name}_fp_poses.mp4"
        writer = get_video_writer(video_path, fps=30, crf=23)
        frame_iter = fs.iter_frames()
        for i in tqdm(range(num_frames), desc=f"Debug video [{cam_name}]"):
            rgb = next(frame_iter)
            pose = poses[i]
            incam_pose = np.linalg.inv(Ts[j]) @ pose
            center_pose = incam_pose @ np.linalg.inv(tracker.to_origin)
            vis = draw_posed_3d_box(
                Ks[j], img=rgb, ob_in_cam=center_pose, bbox=tracker.bbox,
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


def _render_tiled_debug_video(
    poses: np.ndarray,
    cam_names: list[str],
    frame_sources: list[FrameSource],
    Ks: list[np.ndarray],
    Ts: list[np.ndarray],
    tracker: MultiViewTracker,
    select_mask: np.ndarray,
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
        for slot, orig_j in enumerate(sorted_cam_indices):
            rgb = next(frame_iters[slot])
            W_orig, H_orig = rgb.shape[1], rgb.shape[0]
            tile_w = int(W_orig * scale)
            tile_h = int(H_orig * scale)
            vis = cv2.resize(rgb, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            K_tile = Ks[orig_j].copy()
            K_tile[0] *= tile_w / W_orig
            K_tile[1] *= tile_h / H_orig
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
            if select_mask[i, orig_j]:
                border_color = (0, 255, 0)
            else:
                border_color = (255, 0, 0)
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), border_color, border_width)
            tiles.append(vis)

        top = np.concatenate([tiles[0], tiles[1]], axis=1)
        bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
        tiled = np.concatenate([top, bottom], axis=0)

        if writer is None:
            writer = get_video_writer(video_path, fps=30, crf=23)
        writer.write_frame(tiled)

    if writer is not None:
        writer.close()
    print(f"Tiled debug video saved: {video_path}")


def mv_videos_to_poses_from_config(cfg):
    """Resolve config fields and call mv_videos_to_poses."""
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)
    scale = cfg.get("scale", 1.0)

    cam_names: list[str] = []
    cam_intrinsics: list[np.ndarray] = []
    cam_extrinsics: list[np.ndarray] = []
    frame_sources: list[FrameSource] = []
    depth_dirs: list[Path] = []
    mask_dirs: list[Path] = []

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        cam_names.append(cam.name)
        cam_intrinsics.append(cam.param.K)
        cam_extrinsics.append(cam.param.T)

        if cfg.image_dir is not None:
            frame_sources.append(
                FrameSource(image_dir=Path(cfg.image_path_template.format(cam_name=cam.name)))
            )
        elif cfg.video_dir is not None:
            frame_sources.append(
                FrameSource(video_path=Path(cfg.video_path_template.format(cam_name=cam.name)))
            )
        else:
            raise ValueError("At least one of image_dir or video_dir is required")

        depth_dirs.append(Path(cfg.depth_path_template.format(cam_name=cam.name)))
        mask_dirs.append(Path(cfg.mask_path_template.format(cam_name=cam.name)))

    mv_videos_to_poses(
        cam_names=cam_names,
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        frame_sources=frame_sources,
        depth_dirs=depth_dirs,
        mask_dirs=mask_dirs,
        mesh_path=Path(cfg.mesh_path),
        weights_dir=cfg.weights_dir,
        pose_path=Path(cfg.pose_path),
        scale=scale,
        depth_direction_trust=cfg.get("depth_direction_trust", 0.5),
        visible_ratio_cutoff_high=cfg.get("visible_ratio_cutoff_high", 0.3),
        visible_ratio_cutoff_low=cfg.get("visible_ratio_cutoff_low", 0.3),
        precision_high=cfg.get("precision_high", 1.0),
        precision_low=cfg.get("precision_low", 0.01),
        est_refine_iter=cfg.get("est_refine_iter", 5),
        track_refine_iter=cfg.get("track_refine_iter", 2),
        debug=cfg.get("debug", 0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-view 6-DoF object tracking with FoundationPose")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scale", type=float, default=None,
                        help="Scale factor for input resolution (e.g. 0.5 for half)")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--debug", type=int, default=None)
    args = parser.parse_args()

    default_config = Path(__file__).parent / "mv_videos_to_poses.yaml"
    config_path = args.config_path or str(default_config)
    cfg = OmegaConf.load(config_path)

    overrides = {}
    for key in [
        "camera_params_path", "image_dir", "video_dir", "depth_dir",
        "mask_dir", "mesh_path", "weights_dir", "output_dir", "scale", "debug",
    ]:
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    mv_videos_to_poses_from_config(cfg)
