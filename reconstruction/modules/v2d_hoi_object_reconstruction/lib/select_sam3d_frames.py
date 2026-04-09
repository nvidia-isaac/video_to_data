"""Select representative frames for SAM3D reconstruction.

Uses CuSFM camera trajectory to pick one frame per azimuthal angle bin,
preferring frames with the largest object mask area within each bin.

Falls back to top-N by mask area if SfM data is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# SfM pose loading
# ─────────────────────────────────────────────────────────────────────────────

def _aa_to_matrix(aa: dict) -> np.ndarray:
    axis = np.array([aa["x"], aa["y"], aa["z"]])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec((axis / norm) * np.deg2rad(aa["angle_degrees"])).as_matrix()


def _load_sfm_keyframes(
    sfm_keyframes_path: Path,
    frames_meta_path: Path,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load CuSFM left-camera keyframe seq_indices and world positions.

    Returns (seq_indices, positions) or None if data is missing.
    seq_indices[i] is the sequential frame index matching left/*.jpg filenames.
    """
    if not sfm_keyframes_path.exists() or not frames_meta_path.exists():
        return None

    with open(frames_meta_path) as f:
        meta = json.load(f)
    cam_params = meta["camera_params_id_to_camera_params"]

    left_sids: dict[int, int] = {}
    right_sids: set[int] = set()
    for kf in meta["keyframes_metadata"]:
        cam_id = kf["camera_params_id"]
        sid = int(kf["synced_sample_id"])
        sensor = cam_params[cam_id]["sensor_meta_data"]["sensor_name"]
        if "front_stereo_camera_left" in sensor:
            left_sids[sid] = int(kf["timestamp_microseconds"])
        elif "front_stereo_camera_right" in sensor:
            right_sids.add(sid)
    common_sids = sorted(set(left_sids) & right_sids)
    ts_to_seq_idx = {left_sids[sid]: i for i, sid in enumerate(common_sids)}

    with open(sfm_keyframes_path) as f:
        sfm = json.load(f)

    frames: list[tuple[int, np.ndarray]] = []
    for kf in sfm["keyframes_metadata"]:
        if "front_stereo_camera_left" not in kf.get("image_name", ""):
            continue
        ts_us = int(kf["timestamp_microseconds"])
        seq_idx = ts_to_seq_idx.get(ts_us)
        if seq_idx is None:
            continue
        aa = kf["camera_to_world"]["axis_angle"]
        t = kf["camera_to_world"]["translation"]
        R = _aa_to_matrix(aa)
        # Camera position in world = R @ [0,0,0] + t = t (since c2w)
        pos = np.array([t["x"], t["y"], t["z"]])
        frames.append((seq_idx, pos))

    if not frames:
        return None

    frames.sort(key=lambda x: x[0])
    seq_indices = np.array([f[0] for f in frames])
    positions = np.array([f[1] for f in frames])
    return seq_indices, positions


# ─────────────────────────────────────────────────────────────────────────────
# Azimuthal angle computation
# ─────────────────────────────────────────────────────────────────────────────

def _cumulative_azimuth(positions: np.ndarray) -> np.ndarray:
    """Fit a plane via PCA, project positions onto it, return cumulative azimuth."""
    centroid = positions.mean(axis=0)
    _, _, Vt = np.linalg.svd(positions - centroid, full_matrices=False)
    basis_u, basis_v = Vt[0], Vt[1]
    pts_c = (positions - centroid) @ np.stack([basis_u, basis_v], axis=1)
    pts_c -= pts_c.mean(axis=0)
    angles_raw = np.arctan2(pts_c[:, 1], pts_c[:, 0])
    angles_unwrap = np.unwrap(angles_raw)
    angles_unwrap -= angles_unwrap[0]
    return np.rad2deg(angles_unwrap)


# ─────────────────────────────────────────────────────────────────────────────
# Mask area helper
# ─────────────────────────────────────────────────────────────────────────────

def _mask_area(mask_path: Path) -> int:
    arr = np.array(Image.open(mask_path).convert("L"))
    return int(np.sum(arr > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def select_frames_by_angle_bins(
    job_dir: Path,
    bin_deg: float = 60.0,
) -> list[str]:
    """Select one frame per azimuthal bin using the CuSFM camera trajectory.

    Covers both Stage-1 and Stage-2 of the scan.  Within each bin the frame
    with the largest mask area (best object visibility) is chosen.

    The transition region around stage1_end_frame (from
    stage1_detect_debug/result.json) is excluded to avoid the manual-flip
    frames where masks are unreliable.

    Returns a list of zero-padded frame ID strings e.g. ['000842', '001203'].
    Returns [] if SfM data is unavailable (caller should fall back).
    """
    sfm_kf = job_dir / "sfm" / "keyframes" / "frames_meta.json"
    frames_meta = job_dir / "frames_meta.json"
    masks_dir = job_dir / "masks" / "0"

    sfm_data = _load_sfm_keyframes(sfm_kf, frames_meta)
    if sfm_data is None:
        print("  [select_frames] SfM data not found, will fall back to mask-area selection")
        return []

    seq_indices, positions = sfm_data
    angles_deg = _cumulative_azimuth(positions)

    # Exclude transition frames (the manual flip) using stage1_detect result
    detect_result = job_dir / "stage1_detect_debug" / "result.json"
    if detect_result.exists():
        with open(detect_result) as f:
            det = json.load(f)
        stage1_end = det.get("stage1_end_frame")
        if stage1_end is not None:
            # Exclude a window of ±30 frames around stage1_end as a conservative buffer
            transition_lo = max(0, stage1_end - 30)
            transition_hi = stage1_end + 60
            keep = ~((seq_indices >= transition_lo) & (seq_indices <= transition_hi))
            seq_indices = seq_indices[keep]
            angles_deg = angles_deg[keep]

    if len(seq_indices) == 0:
        return []

    angle_min = angles_deg.min()
    angle_max = angles_deg.max()
    n_bins = max(1, int(np.ceil((angle_max - angle_min) / bin_deg)))
    bin_edges = np.linspace(angle_min, angle_max, n_bins + 1)

    selected: list[str] = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = np.where((angles_deg >= lo) & (angles_deg < hi))[0]
        if len(in_bin) == 0:
            continue
        best_idx: Optional[int] = None
        best_area = -1
        for i in in_bin:
            seq = seq_indices[i]
            mask_path = masks_dir / f"{seq:06d}.png"
            if not mask_path.exists():
                continue
            area = _mask_area(mask_path)
            if area > best_area:
                best_area = area
                best_idx = int(seq)
        if best_idx is not None:
            selected.append(f"{best_idx:06d}")

    return selected


def select_frames_fallback(job_dir: Path, n: int = 6) -> list[str]:
    """Fallback: return top-n frame IDs by mask area."""
    masks_dir = job_dir / "masks" / "0"
    if not masks_dir.is_dir():
        return []
    scored = []
    for p in sorted(masks_dir.iterdir()):
        if p.suffix.lower() != ".png":
            continue
        scored.append((_mask_area(p), p.stem))
    scored.sort(reverse=True)
    return [stem for _, stem in scored[:n]]
