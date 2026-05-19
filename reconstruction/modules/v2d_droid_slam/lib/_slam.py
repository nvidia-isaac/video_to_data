"""
Internal DROID-SLAM driver shared by video_to_slam and image_list_to_slam.

Loads the upstream Droid runner, drives tracking over a frame iterator,
optionally scale-aligns the result to a reference metric-depth source,
and writes outputs in the v2d common formats.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Iterator

import cv2
import numpy as np
import torch

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Transform3d


@dataclass
class Frame:
    frame_idx: int                              # source frame index (zero-padded filename stem)
    image_bgr: np.ndarray                       # (H, W, 3) uint8
    intrinsics: CameraIntrinsics                # intrinsics at the source resolution
    prior_depth: np.ndarray | None = None       # (H, W) float32 metres — fed into BA as sensor prior
    align_to_depth: np.ndarray | None = None    # (H, W) float32 metres — used post-hoc for scale alignment


def _resize_for_droid(
    image_bgr: np.ndarray,
    fxfycxcy: np.ndarray,
    image_size: tuple[int, int],
    prior_depth: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Resize a frame to DROID's network input and rescale intrinsics to match.

    Returns image as (1, 3, H, W) uint8 BGR (upstream motion_filter adds its
    own [None]+[2,1,0] reorder to get RGB internally), scaled intrinsics, and
    optional depth resized to the network input resolution. Invalid (non-positive
    or non-finite) depth pixels are zeroed; depth_video.append() interprets
    zero as "no prior at this pixel".
    """
    target_h, target_w = image_size
    src_h, src_w = image_bgr.shape[:2]
    image_bgr = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(image_bgr).permute(2, 0, 1)[None]  # (1, 3, H, W) uint8

    sx = target_w / src_w
    sy = target_h / src_h
    fx, fy, cx, cy = fxfycxcy
    scaled = torch.tensor([fx * sx, fy * sy, cx * sx, cy * sy], dtype=torch.float32)

    depth_t: torch.Tensor | None = None
    if prior_depth is not None:
        d = prior_depth.astype(np.float32)
        invalid = ~np.isfinite(d) | (d <= 0)
        d[invalid] = 0.0
        if d.shape != (target_h, target_w):
            d = cv2.resize(d, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        depth_t = torch.from_numpy(d)

    return image, scaled, depth_t


def _load_droid_class():
    """Import the upstream Droid class lazily so module import stays cheap."""
    from droid import Droid  # provided by the DROID-SLAM repo on PYTHONPATH
    return Droid


def _get_keyframe_disps(droid, upsample: bool) -> np.ndarray:
    """Return per-keyframe disparity. Upsampled to source resolution if available."""
    video = droid.video
    if upsample and hasattr(video, "disps_up"):
        return video.disps_up.detach().cpu().numpy()
    return video.disps.detach().cpu().numpy()


def _solve_global_scale(droid, frames_by_idx: dict[int, Frame], upsample: bool) -> float:
    """Compute a robust scale factor that maps DROID depth → reference depth.

    Sampled at every keyframe where reference depth is available. Per-keyframe
    median ratios are aggregated into a final global median.
    """
    video = droid.video
    n_kf = int(video.counter.value)
    disps = _get_keyframe_disps(droid, upsample)
    tstamps = video.tstamp.detach().cpu().numpy().astype(np.int64)

    per_kf_scales: list[float] = []
    for kf in range(n_kf):
        frame_idx = int(tstamps[kf])
        if frame_idx not in frames_by_idx:
            continue
        ref = frames_by_idx[frame_idx].align_to_depth
        if ref is None:
            continue
        disp = disps[kf]
        # DROID disparity is in network-resolution pixels; resize to source.
        ref_h, ref_w = ref.shape[:2]
        if disp.shape != (ref_h, ref_w):
            disp = cv2.resize(disp, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
        valid = (disp > 1e-3) & (ref > 1e-3) & np.isfinite(ref)
        if valid.sum() < 64:
            continue
        droid_depth = 1.0 / disp[valid]
        ratios = ref[valid] / droid_depth
        per_kf_scales.append(float(np.median(ratios)))

    if not per_kf_scales:
        return 1.0
    return float(np.median(per_kf_scales))


def _write_pointcloud_ply(droid, scale: float, path: str,
                          min_views: int = 2,
                          filter_thresh: float = 0.005) -> None:
    """Dump the post-BA keyframe point cloud as a binary-little-endian PLY.

    Uses upstream's depth_filter for multi-view consistency: a pixel is
    kept only if its depth is consistent with at least ``min_views`` other
    keyframes. This is what produces clean (non-fuzzy) clouds.
    """
    import torch
    import droid_backends
    from lietorch import SE3

    video = droid.video
    n_kf = int(video.counter.value)
    if n_kf == 0:
        print("[droid-slam] no keyframes; skipping PLY")
        return

    # depth_filter inspects the FULL buffer using dirty_index to know which
    # slots are valid, so we hand it the unsliced tensors as upstream does.
    dirty_index = torch.arange(n_kf, device=video.poses.device)
    thresh = filter_thresh * torch.ones(n_kf, device=video.poses.device)
    count = droid_backends.depth_filter(
        video.poses, video.disps, video.intrinsics[0], dirty_index, thresh
    )  # (N, h, w) per-pixel consistent-view count

    poses = video.poses[:n_kf].detach()
    disps = video.disps[:n_kf].detach()
    images = video.images[:n_kf].detach()
    intrinsics = video.intrinsics[0].detach()

    cam_to_world = SE3(poses).inv()
    points_world = droid_backends.iproj(cam_to_world.data, disps, intrinsics)
    points_world = (points_world * scale).cpu().numpy()       # (N, h, w, 3)

    colors_rgb = images[:, [2, 1, 0], 3::8, 3::8]
    colors_rgb = colors_rgb.permute(0, 2, 3, 1).cpu().numpy()  # (N, h, w, 3) RGB

    disp_np = disps.cpu().numpy()
    disp_mean = disp_np.reshape(n_kf, -1).mean(axis=1).reshape(n_kf, 1, 1)
    consistency_mask = count.cpu().numpy() >= min_views
    mask = consistency_mask & (disp_np > 0.5 * disp_mean) & (disp_np > 1e-3)

    points_arr = points_world[mask].astype(np.float32)
    colors_arr = colors_rgb[mask].astype(np.uint8)
    if points_arr.size == 0:
        print("[droid-slam] no points passed consistency filter; skipping PLY")
        return

    t_arr = cam_to_world.data[:, :3].cpu().numpy()
    n_total = int(mask.size)
    print(f"[droid-slam] PLY: {n_kf} keyframes, {points_arr.shape[0]} / {n_total} points "
          f"kept (min_views={min_views}); "
          f"cam x∈[{t_arr[:,0].min():.3f},{t_arr[:,0].max():.3f}] "
          f"y∈[{t_arr[:,1].min():.3f},{t_arr[:,1].max():.3f}] "
          f"z∈[{t_arr[:,2].min():.3f},{t_arr[:,2].max():.3f}]")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {points_arr.shape[0]}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        interleaved = np.empty(points_arr.shape[0],
                               dtype=[("xyz", np.float32, 3), ("rgb", np.uint8, 3)])
        interleaved["xyz"] = points_arr
        interleaved["rgb"] = colors_arr
        interleaved.tofile(f)


def _se3_quat_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert DROID's [tx,ty,tz,qx,qy,qz,qw] world-to-camera into a 4×4 matrix."""
    from scipy.spatial.transform import Rotation
    tx, ty, tz, qx, qy, qz, qw = pose.tolist()
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


def run_droid_slam(
    frame_iter: Callable[[], Iterator[Frame]],
    weights_path: str,
    poses_folder: str,
    depth_folder: str | None = None,
    pointcloud_path: str | None = None,
    trajectory_path: str | None = None,
    image_size: tuple[int, int] = (384, 512),
    buffer_size: int = 512,
    beta: float = 0.3,
    filter_thresh: float = 2.4,
    warmup: int = 8,
    keyframe_thresh: float = 4.0,
    frontend_thresh: float = 16.0,
    frontend_window: int = 25,
    frontend_radius: int = 2,
    frontend_nms: int = 1,
    backend_thresh: float = 22.0,
    backend_radius: int = 2,
    backend_nms: int = 3,
    upsample: bool = False,
    pointcloud_min_views: int = 2,
) -> None:
    """Run DROID-SLAM over a frame iterator and write outputs.

    The frame iterator is invoked twice: once for tracking, then again for
    scale alignment so we can correlate keyframes with their reference depth
    without holding all frames in memory.
    """
    os.makedirs(poses_folder, exist_ok=True)
    if depth_folder is not None:
        os.makedirs(depth_folder, exist_ok=True)

    weights_file = os.path.join(weights_path, "droid.pth") \
        if os.path.isdir(weights_path) else weights_path
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"droid.pth not found at {weights_file}")

    args = SimpleNamespace(
        weights=weights_file,
        image_size=list(image_size),
        buffer=buffer_size,
        beta=beta,
        filter_thresh=filter_thresh,
        warmup=warmup,
        keyframe_thresh=keyframe_thresh,
        frontend_thresh=frontend_thresh,
        frontend_window=frontend_window,
        frontend_radius=frontend_radius,
        frontend_nms=frontend_nms,
        backend_thresh=backend_thresh,
        backend_radius=backend_radius,
        backend_nms=backend_nms,
        upsample=upsample,
        disable_vis=True,
        stereo=False,
    )

    Droid = _load_droid_class()
    droid: object | None = None

    frames_by_idx: dict[int, Frame] = {}
    ordered_indices: list[int] = []

    for frame in frame_iter():
        if droid is None:
            droid = Droid(args)
        fxfycxcy = np.array([frame.intrinsics.fx, frame.intrinsics.fy,
                             frame.intrinsics.cx, frame.intrinsics.cy], dtype=np.float32)
        image_t, intr_t, depth_t = _resize_for_droid(
            frame.image_bgr, fxfycxcy, image_size, prior_depth=frame.prior_depth,
        )
        droid.track(frame.frame_idx, image_t, depth=depth_t, intrinsics=intr_t)
        frames_by_idx[frame.frame_idx] = frame
        ordered_indices.append(frame.frame_idx)

    if droid is None:
        raise RuntimeError("frame_iter produced no frames")

    # Reconstruct the stream for terminate() — used internally by DROID
    # for the final global BA pass.
    def replay():
        for f in frame_iter():
            fxfycxcy = np.array([f.intrinsics.fx, f.intrinsics.fy,
                                 f.intrinsics.cx, f.intrinsics.cy], dtype=np.float32)
            image_t, intr_t, _ = _resize_for_droid(
                f.image_bgr, fxfycxcy, image_size, prior_depth=None,
            )
            yield (f.frame_idx, image_t, intr_t)

    traj_est = droid.terminate(replay())  # (N_frames, 7) camera-to-world [tx,ty,tz,qx,qy,qz,qw]

    scale = _solve_global_scale(droid, frames_by_idx, upsample)
    if scale != 1.0:
        print(f"[droid-slam] applying global scale {scale:.6f} from reference depth")

    # Write per-frame poses.
    from scipy.spatial.transform import Rotation
    for row_i, frame_idx in enumerate(ordered_indices):
        tx, ty, tz, qx, qy, qz, qw = traj_est[row_i].tolist()
        M = np.eye(4)
        M[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        M[:3, 3] = [tx * scale, ty * scale, tz * scale]
        Transform3d.from_matrix(M).save(
            os.path.join(poses_folder, f"{frame_idx:06d}.json")
        )

    # Optional TUM trajectory.
    if trajectory_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(trajectory_path)) or ".", exist_ok=True)
        with open(trajectory_path, "w") as f:
            for row_i, frame_idx in enumerate(ordered_indices):
                tx, ty, tz, qx, qy, qz, qw = traj_est[row_i].tolist()
                f.write(f"{frame_idx} {tx*scale} {ty*scale} {tz*scale} "
                        f"{qx} {qy} {qz} {qw}\n")

    # Optional per-keyframe depth output (scaled, upsampled to source resolution).
    if depth_folder is not None:
        video = droid.video
        n_kf = int(video.counter.value)
        disps = _get_keyframe_disps(droid, upsample)
        tstamps = video.tstamp.detach().cpu().numpy().astype(np.int64)

        for kf in range(n_kf):
            disp = disps[kf]
            frame_idx = int(tstamps[kf])
            frame = frames_by_idx.get(frame_idx)
            if frame is None:
                continue
            tgt_h, tgt_w = frame.intrinsics.height, frame.intrinsics.width
            if disp.shape != (tgt_h, tgt_w):
                disp = cv2.resize(disp, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
            # Dense depth output: every pixel gets a value. Tiny-disparity pixels
            # (effectively "no information") map to large depth and therefore
            # near-black in the inverse-depth PNG encoding — distinguishable
            # from depth≈0, which would have been white and visually misleading.
            safe_disp = np.maximum(disp, 1e-6)
            depth = (1.0 / safe_disp) * scale
            DepthImage(depth=depth.astype(np.float32)).to_pil_image().save(
                os.path.join(depth_folder, f"{frame_idx:06d}.png")
            )

    if pointcloud_path is not None:
        _write_pointcloud_ply(droid, scale, pointcloud_path,
                              min_views=pointcloud_min_views)
