# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run HaWoR and export canonical v2d hand-track folders.

The native HaWoR workspace is preserved separately. The canonical track export
uses HaWoR camera-space chunks so downstream v2d tools can consume the same
``bbox`` + ``mano`` + ``camera.cam_t`` schema as HaMeR/WiLoR.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from dataclasses import dataclass



_HAWOR_ROOT = os.environ.get("HAWOR_ROOT", "/opt/hawor")
_EXPORT_VERSION = 2
_LEFT_CONVERSION = "mirror_conjugate_axis_angle_yz"
_MOTION_CACHE_VERSION = 1


def _link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        if os.path.exists(dst):
            return
        os.unlink(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _first_existing(candidates: list[str], label: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    formatted = "\n    ".join(candidates)
    raise FileNotFoundError(f"Missing HaWoR {label}. Tried:\n    {formatted}")


def _prepare_mano_links(weights_dir: str) -> None:
    shared_weights = os.path.dirname(os.path.abspath(weights_dir))

    mano_sources: dict[str, str] = {}
    for filename in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl"):
        src = _first_existing([
            os.path.join(weights_dir, "_DATA", "data", "mano", filename),
            os.path.join(weights_dir, "mano", filename),
            os.path.join(weights_dir, filename),
            os.path.join(shared_weights, "hand", filename),
            os.path.join(shared_weights, "hand", "models", filename),
            os.path.join(shared_weights, "hamer", "_DATA", "data", "mano", filename),
            os.path.join(shared_weights, "hamer", "_DATA", "data", "models", filename),
            os.path.join(shared_weights, "wilor", "pretrained_models", filename),
            os.path.join(shared_weights, "wilor", "pretrained_models", "models", filename),
        ], filename)
        mano_sources[filename] = src
        _link_or_copy(src, os.path.join(_HAWOR_ROOT, "_DATA", "data", "mano", filename))

    # HaWoR's post-processing helpers use a separate hard-coded root for the
    # left-hand MANO model, distinct from the checkpoint config's right-hand
    # ``_DATA/data/mano`` path. Populate both upstream locations.
    _link_or_copy(
        mano_sources["MANO_LEFT.pkl"],
        os.path.join(_HAWOR_ROOT, "_DATA", "data_left", "mano_left", "MANO_LEFT.pkl"),
    )
    _link_or_copy(
        mano_sources["MANO_RIGHT.pkl"],
        os.path.join(_HAWOR_ROOT, "_DATA", "data_right", "mano_right", "MANO_RIGHT.pkl"),
    )

    mean_params = _first_existing([
        os.path.join(weights_dir, "_DATA", "data", "mano_mean_params.npz"),
        os.path.join(weights_dir, "mano_mean_params.npz"),
        os.path.join(shared_weights, "hamer", "_DATA", "data", "mano_mean_params.npz"),
        os.path.join(shared_weights, "wilor", "pretrained_models", "mano_mean_params.npz"),
    ], "mano_mean_params.npz")
    for dst in (
        os.path.join(_HAWOR_ROOT, "_DATA", "data", "mano_mean_params.npz"),
        os.path.join(_HAWOR_ROOT, "_DATA", "data_left", "mano_mean_params.npz"),
        os.path.join(_HAWOR_ROOT, "_DATA", "data_right", "mano_mean_params.npz"),
    ):
        _link_or_copy(mean_params, dst)


def _prepare_weight_links(weights_dir: str) -> None:
    links = [
        (os.path.join(weights_dir, "external", "detector.pt"), os.path.join(_HAWOR_ROOT, "weights", "external", "detector.pt")),
        (os.path.join(weights_dir, "external", "droid.pth"), os.path.join(_HAWOR_ROOT, "weights", "external", "droid.pth")),
        (os.path.join(weights_dir, "external", "metric_depth_vit_large_800k.pth"), os.path.join(_HAWOR_ROOT, "thirdparty", "Metric3D", "weights", "metric_depth_vit_large_800k.pth")),
        (os.path.join(weights_dir, "hawor", "model_config.yaml"), os.path.join(_HAWOR_ROOT, "weights", "hawor", "model_config.yaml")),
        (os.path.join(weights_dir, "hawor", "checkpoints", "hawor.ckpt"), os.path.join(_HAWOR_ROOT, "weights", "hawor", "checkpoints", "hawor.ckpt")),
        (os.path.join(weights_dir, "hawor", "checkpoints", "infiller.pt"), os.path.join(_HAWOR_ROOT, "weights", "hawor", "checkpoints", "infiller.pt")),
    ]
    for src, dst in links:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing HaWoR weight: {src}")
        _link_or_copy(src, dst)
    _prepare_mano_links(weights_dir)


def _rotmat_to_axis_angle(R):
    import numpy as np
    R = np.asarray(R, dtype=np.float64)
    trace = np.einsum("...ii->...", R)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    vec = np.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], axis=-1)
    sin_theta = np.sin(theta)
    scale = theta / (2.0 * sin_theta + 1e-12)
    out = vec * scale[..., None]
    out = np.where((theta < 1e-7)[..., None], np.zeros_like(out), out)
    return out.astype(np.float64)


def _mirror_conjugate_axis_angle(axis_angle):
    """Convert left-hand mirrored rotations into canonical right-MANO rotations.

    v2d's hand-track schema stores all MANO pose parameters in right-hand MANO
    convention. Downstream renderers mirror vertices for ``is_right == false``.
    HaWoR's left-hand post-processing already flips left poses into a left-hand
    convention, so undo that mirror at the rotation level before export:
    R_canonical = M @ R_left @ M, M = diag(-1, 1, 1). In axis-angle form that
    negates the y/z components.
    """
    import numpy as np
    aa = np.asarray(axis_angle, dtype=np.float64).copy()
    aa[..., 1] *= -1.0
    aa[..., 2] *= -1.0
    return aa


def _as_time_array(value):
    import numpy as np
    arr = np.asarray(value)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _image_size(seq_folder: str) -> tuple[int, int]:
    import cv2
    files = sorted(glob.glob(os.path.join(seq_folder, "extracted_images", "*.jpg")))
    if not files:
        files = sorted(glob.glob(os.path.join(seq_folder, "extracted_images", "*.png")))
    if not files:
        raise FileNotFoundError(f"No extracted HaWoR images under {seq_folder}")
    img = cv2.imread(files[0])
    if img is None:
        raise RuntimeError(f"Could not read extracted image: {files[0]}")
    h, w = img.shape[:2]
    return int(w), int(h)


def _bbox_from_det_box(box) -> dict[str, float]:
    import numpy as np
    vals = np.asarray(box, dtype=np.float64).reshape(-1)[:4]
    x0, y0, x1, y1 = vals.tolist()
    # HaWoR detector boxes are expected to be xyxy. If a source unexpectedly
    # gives xywh, this check keeps the exported bbox valid without adding a
    # second accepted hand-track schema.
    if x1 <= x0 or y1 <= y0:
        x1 = x0 + max(0.0, float(vals[2]))
        y1 = y0 + max(0.0, float(vals[3]))
    return {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}


def _interpolate_boxes(boxes):
    import numpy as np
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    valid = np.any(boxes != 0, axis=1)
    if valid.all() or not valid.any():
        return boxes
    idx = np.arange(len(boxes))
    valid_idx = idx[valid]
    for col in range(4):
        boxes[:, col] = np.interp(idx, valid_idx, boxes[valid_idx, col])
    return boxes


def _bbox_lookup(seq_folder: str, start_idx: int, end_idx: int) -> dict[bool, dict[int, dict[str, float]]]:
    import numpy as np
    path = os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}", "model_tracks.npy")
    if not os.path.exists(path):
        return {False: {}, True: {}}
    tracks = np.load(path, allow_pickle=True).item()
    out: dict[bool, dict[int, dict[str, float]]] = {False: {}, True: {}}
    for track in tracks.values():
        valid = [entry for entry in track if entry.get("det", False)]
        if not valid:
            continue
        handed = [float(np.asarray(entry["det_handedness"]).reshape(-1)[0]) for entry in valid]
        is_right = bool(np.mean(handed) > 0.5)
        entries = sorted(track, key=lambda entry: int(entry["frame"]))
        frames = np.asarray([int(entry["frame"]) for entry in entries], dtype=np.int64)
        boxes = np.concatenate([
            np.asarray(entry["det_box"], dtype=np.float64).reshape(-1)[:4].reshape(1, 4)
            for entry in entries
        ], axis=0)
        non_zero = np.where(np.any(boxes != 0, axis=1))[0]
        if len(non_zero) == 0:
            continue
        lo, hi = int(non_zero[0]), int(non_zero[-1])
        boxes[lo:hi + 1] = _interpolate_boxes(boxes[lo:hi + 1])
        for frame_idx, box in zip(frames[lo:hi + 1], boxes[lo:hi + 1]):
            out[is_right][int(frame_idx)] = _bbox_from_det_box(box)
    return out


def _frame_chunks(seq_folder: str, start_idx: int, end_idx: int):
    import joblib
    path = os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}", "frame_chunks_all.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"HaWoR frame chunk file missing: {path}")
    return joblib.load(path)


def _close_focal(a: float | None, b: float | None, tol: float = 1e-3) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def _read_float_file(path: str) -> float | None:
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def _motion_cache_meta_path(seq_folder: str, start_idx: int, end_idx: int) -> str:
    return os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}", "v2d_hawor_motion_cache.json")


def _motion_cache_current(
    seq_folder: str,
    start_idx: int,
    end_idx: int,
    requested_focal: float | None,
) -> bool:
    if requested_focal is None:
        return True

    meta_path = _motion_cache_meta_path(seq_folder, start_idx, end_idx)
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if (
            int(meta.get("cache_version", -1)) == _MOTION_CACHE_VERSION
            and _close_focal(meta.get("requested_focal_length"), requested_focal)
        ):
            return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

    # Legacy cache: est_focal.txt existed before the metadata file. Treat it as
    # current only when it already matches the requested focal.
    est_focal = _read_float_file(os.path.join(seq_folder, "est_focal.txt"))
    return _close_focal(est_focal, requested_focal)


def _clear_hawor_motion_cache(seq_folder: str, start_idx: int, end_idx: int) -> None:
    tracks_dir = os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}")
    for cache_path in (
        os.path.join(seq_folder, "cam_space"),
        os.path.join(seq_folder, "SLAM"),
    ):
        if os.path.isdir(cache_path):
            shutil.rmtree(cache_path)
    for cache_path in (
        os.path.join(seq_folder, "est_focal.txt"),
        os.path.join(seq_folder, "world_space_res.pth"),
        os.path.join(tracks_dir, "frame_chunks_all.npy"),
        os.path.join(tracks_dir, "model_masks.npy"),
        _motion_cache_meta_path(seq_folder, start_idx, end_idx),
    ):
        if os.path.exists(cache_path):
            os.remove(cache_path)


def _write_motion_cache_meta(
    seq_folder: str,
    start_idx: int,
    end_idx: int,
    requested_focal: float | None,
    est_focal: float,
) -> None:
    if requested_focal is None:
        return
    meta_path = _motion_cache_meta_path(seq_folder, start_idx, end_idx)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "cache_version": _MOTION_CACHE_VERSION,
                "requested_focal_length": float(requested_focal),
                "estimated_focal_length": float(est_focal),
            },
            f,
            indent=2,
        )


def _write_export_metadata(
    hand_tracks_dir: str,
    focal_length: float,
    left_id: int,
    right_id: int,
    start_idx: int,
    end_idx: int,
) -> None:
    with open(os.path.join(hand_tracks_dir, "_metadata.json"), "w") as f:
        json.dump(
            {
                "export_version": _EXPORT_VERSION,
                "format": "v2d_hand_track_v1",
                "source": "hawor",
                "left_conversion": _LEFT_CONVERSION,
                "focal_length": float(focal_length),
                "left_id": int(left_id),
                "right_id": int(right_id),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
            },
            f,
            indent=2,
        )


def _export_camera_space_tracks(
    seq_folder: str,
    hand_tracks_dir: str,
    focal_length: float,
    left_id: int,
    right_id: int,
    start_idx: int,
    end_idx: int,
    frame_chunks_all=None,
) -> None:
    import numpy as np
    W, H = _image_size(seq_folder)
    bboxes = _bbox_lookup(seq_folder, start_idx, end_idx)
    chunks = frame_chunks_all if frame_chunks_all is not None else _frame_chunks(seq_folder, start_idx, end_idx)
    os.makedirs(hand_tracks_dir, exist_ok=True)
    for track_id in set((left_id, right_id)):
        out_dir = os.path.join(hand_tracks_dir, str(track_id))
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

    for hand_idx, (is_right, track_id) in enumerate(((False, left_id), (True, right_id))):
        out_dir = os.path.join(hand_tracks_dir, str(track_id))
        os.makedirs(out_dir, exist_ok=True)
        chunk_dir = os.path.join(seq_folder, "cam_space", str(hand_idx))
        if not os.path.isdir(chunk_dir):
            continue
        for chunk_i, path in enumerate(sorted(glob.glob(os.path.join(chunk_dir, "*.json")))):
            with open(path) as f:
                data = json.load(f)
            frames = [int(f) for f in np.asarray(chunks[hand_idx][chunk_i]).reshape(-1)]
            root_rot = _as_time_array(data["init_root_orient"])
            hand_rot = _as_time_array(data["init_hand_pose"])
            trans = _as_time_array(data["init_trans"])
            betas = _as_time_array(data["init_betas"])
            root_aa = _rotmat_to_axis_angle(root_rot).reshape(len(frames), 3)
            hand_aa = _rotmat_to_axis_angle(hand_rot).reshape(len(frames), -1, 3)
            if not is_right:
                root_aa = _mirror_conjugate_axis_angle(root_aa)
                hand_aa = _mirror_conjugate_axis_angle(hand_aa)
            if betas.ndim == 1:
                betas = np.tile(betas[None, :], (len(frames), 1))
            elif betas.shape[0] != len(frames):
                betas = np.tile(betas.reshape(1, -1), (len(frames), 1))

            for i, frame_idx in enumerate(frames):
                bbox = bboxes[is_right].get(frame_idx)
                if bbox is None:
                    # Canonical v2d hand tracks require bbox. Skip frames where
                    # HaWoR only has an infilled/motion-only state.
                    continue
                rec = {
                    "track_id": int(track_id),
                    "frame_idx": int(frame_idx),
                    "is_right": bool(is_right),
                    "image_size": [int(W), int(H)],
                    "bbox": bbox,
                    "mano": {
                        "betas": np.asarray(betas[i], dtype=np.float64).reshape(-1)[:10].tolist(),
                        "global_orient": root_aa[i].tolist(),
                        "hand_pose": hand_aa[i].reshape(-1)[:45].tolist(),
                    },
                    "camera": {
                        "cam_t": np.asarray(trans[i], dtype=np.float64).reshape(3).tolist(),
                        "focal_length": float(focal_length),
                    },
                }
                with open(os.path.join(out_dir, f"{frame_idx:06d}.json"), "w") as f:
                    json.dump(rec, f, indent=2)
    _write_export_metadata(hand_tracks_dir, focal_length, left_id, right_id, start_idx, end_idx)



def _patch_renderer_mask_fallback() -> None:
    """Avoid HaWoR's PyTorch3D rasterizer for per-frame hand masks.

    The upstream video path renders MANO meshes only to create model_masks.npy
    for masked DROID-SLAM. Some PyTorch3D builds in the HaWoR environment are
    CPU-only and fail when the rasterizer is called on CUDA tensors. For v2d we
    only need a conservative moving-hand mask there, so project the MANO
    vertices with the renderer intrinsics and fill a padded 2D bbox instead.
    """
    try:
        import cv2
        import numpy as np
        import torch
        from lib.vis.renderer import Renderer
    except Exception as exc:
        print(f"Warning: could not install HaWoR render fallback: {exc}")
        return

    if getattr(Renderer.render_multiple, "_v2d_projected_mask", False):
        return

    def _render_multiple_projected_mask(self, verts_list, faces, colors_list, cameras, lights):
        height = int(self.height)
        width = int(self.width)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        mask_u8 = np.zeros((height, width), dtype=np.uint8)

        if isinstance(verts_list, torch.Tensor):
            iterable = list(torch.unbind(verts_list, dim=0))
        else:
            iterable = list(verts_list)

        K = self.K.detach().float().cpu().numpy()
        if K.ndim == 3:
            K = K[0]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        for verts in iterable:
            verts_np = verts.detach().float().cpu().reshape(-1, 3).numpy()
            if verts_np.size == 0:
                continue
            z = verts_np[:, 2]
            valid = np.isfinite(z) & (z > 1e-4)
            if not np.any(valid):
                continue
            u = fx * verts_np[valid, 0] / z[valid] + cx
            v = fy * verts_np[valid, 1] / z[valid] + cy
            inside = np.isfinite(u) & np.isfinite(v)
            if not np.any(inside):
                continue
            u = u[inside]
            v = v[inside]
            x0 = int(np.floor(np.clip(u.min(), 0, width - 1)))
            x1 = int(np.ceil(np.clip(u.max(), 0, width - 1)))
            y0 = int(np.floor(np.clip(v.min(), 0, height - 1)))
            y1 = int(np.ceil(np.clip(v.max(), 0, height - 1)))
            if x1 <= x0 or y1 <= y0:
                continue
            pad = max(8, int(round(0.05 * max(x1 - x0, y1 - y0))))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(width - 1, x1 + pad)
            y1 = min(height - 1, y1 + pad)
            cv2.rectangle(mask_u8, (x0, y0), (x1, y1), 1, thickness=-1)

        return image, mask_u8.astype(bool)

    _render_multiple_projected_mask._v2d_projected_mask = True
    Renderer.render_multiple = _render_multiple_projected_mask

@dataclass
class _Args:
    video_path: str
    img_focal: float | None
    checkpoint: str
    infiller_weight: str
    input_type: str = "file"
    vis_mode: str = "none"


def run_hawor(
    video: str,
    hand_tracks_dir: str,
    weights: str,
    native_dir: str | None = None,
    focal_length: float = -1.0,
    left_id: int = 2,
    right_id: int = 3,
    max_num: int = 1000,
) -> None:
    import sys

    if _HAWOR_ROOT not in sys.path:
        sys.path.insert(0, _HAWOR_ROOT)
    from demo import detect_track_video, hawor_infiller, hawor_motion_estimation, hawor_slam
    _patch_renderer_mask_fallback()

    video = os.path.abspath(video)
    hand_tracks_dir = os.path.abspath(hand_tracks_dir)
    native_dir = os.path.abspath(native_dir or os.path.join(os.path.dirname(hand_tracks_dir), "hawor_native"))
    os.makedirs(native_dir, exist_ok=True)
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        os.path.join(native_dir, "ultralytics_config"),
    )
    os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
    _prepare_weight_links(os.path.abspath(weights))

    work_video = os.path.join(native_dir, os.path.basename(video))
    if not os.path.exists(work_video):
        try:
            os.symlink(video, work_video)
        except OSError:
            shutil.copy2(video, work_video)

    prev_cwd = os.getcwd()
    try:
        os.chdir(_HAWOR_ROOT)
        args = _Args(
            video_path=work_video,
            img_focal=None if float(focal_length) <= 0 else float(focal_length),
            checkpoint=os.path.join(_HAWOR_ROOT, "weights", "hawor", "checkpoints", "hawor.ckpt"),
            infiller_weight=os.path.join(_HAWOR_ROOT, "weights", "hawor", "checkpoints", "infiller.pt"),
        )
        start_idx, end_idx, seq_folder, _imgfiles = detect_track_video(args)
        requested_focal = None if float(focal_length) <= 0 else float(focal_length)
        if not _motion_cache_current(seq_folder, start_idx, end_idx, requested_focal):
            print(
                "HaWoR focal/config changed; clearing native motion cache "
                f"for {start_idx}_{end_idx}"
            )
            _clear_hawor_motion_cache(seq_folder, start_idx, end_idx)
        frame_chunks_all, est_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx)
        hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
        _write_motion_cache_meta(seq_folder, start_idx, end_idx, requested_focal, float(est_focal))
    finally:
        os.chdir(prev_cwd)

    _export_camera_space_tracks(
        seq_folder, hand_tracks_dir, float(est_focal), left_id, right_id, start_idx, end_idx, frame_chunks_all
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--hand_tracks_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--native_dir", default=None)
    parser.add_argument("--focal_length", type=float, default=-1.0)
    parser.add_argument("--left_id", type=int, default=2)
    parser.add_argument("--right_id", type=int, default=3)
    parser.add_argument("--max_num", type=int, default=1000)
    args = parser.parse_args()
    run_hawor(
        video=args.video,
        hand_tracks_dir=args.hand_tracks_dir,
        weights=args.weights,
        native_dir=args.native_dir,
        focal_length=args.focal_length,
        left_id=args.left_id,
        right_id=args.right_id,
        max_num=args.max_num,
    )


if __name__ == "__main__":
    main()
