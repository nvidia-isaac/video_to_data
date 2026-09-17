# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select representative frames for SAM3D reconstruction.

Two-stage captures use cumulative orbit-angle bins. Stationary-object captures
use camera viewing-direction diversity and therefore do not require a planar or
ordered orbit. Both methods prefer frames with a large visible object mask.
"""

from __future__ import annotations

import argparse
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
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load CuSFM left-camera indices, positions, and viewing directions.

    Returns ``(seq_indices, positions, view_directions)`` or ``None`` if data
    is missing. Viewing directions are the camera +Z axes in world space.
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

    frames: list[tuple[int, np.ndarray, np.ndarray]] = []
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
        # CuSFM camera-to-world uses the OpenCV camera convention (+Z forward).
        view_direction = R[:, 2]
        frames.append((seq_idx, pos, view_direction))

    if not frames:
        return None

    frames.sort(key=lambda x: x[0])
    seq_indices = np.array([f[0] for f in frames])
    positions = np.array([f[1] for f in frames])
    view_directions = np.array([f[2] for f in frames])
    return seq_indices, positions, view_directions


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

    seq_indices, positions, _ = sfm_data
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


def _select_pose_diverse_candidates(
    candidates: list[tuple[int, np.ndarray, int]],
    count: int,
) -> list[str]:
    """Greedily select mask-visible frames with diverse viewing directions.

    The largest-mask candidate seeds the selection. Each later candidate
    maximizes its minimum angular distance from the directions already chosen;
    mask area and then frame order break ties deterministically.
    """
    normalized: list[tuple[int, np.ndarray, int]] = []
    for seq_idx, direction, area in candidates:
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12 and area > 0:
            normalized.append((seq_idx, direction / norm, area))
    if not normalized:
        return []

    remaining = sorted(normalized, key=lambda item: item[0])
    first = max(remaining, key=lambda item: (item[2], -item[0]))
    selected = [first]
    remaining.remove(first)

    while remaining and len(selected) < count:
        def score(candidate: tuple[int, np.ndarray, int]) -> tuple[float, int, int]:
            min_angle = min(
                float(np.arccos(np.clip(np.dot(candidate[1], chosen[1]), -1.0, 1.0)))
                for chosen in selected
            )
            return min_angle, candidate[2], -candidate[0]

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)

    return [f"{seq_idx:06d}" for seq_idx, _, _ in selected]


def select_frames_by_view_diversity(
    job_dir: Path,
    count: int = 6,
) -> list[str]:
    """Select pose-diverse views for a stationary object and arbitrary scan path."""
    sfm_kf = job_dir / "sfm" / "keyframes" / "frames_meta.json"
    frames_meta = job_dir / "frames_meta.json"
    masks_dir = job_dir / "masks" / "0"

    sfm_data = _load_sfm_keyframes(sfm_kf, frames_meta)
    if sfm_data is None:
        print("  [select_frames] SfM data not found, will fall back to mask-area selection")
        return []

    seq_indices, _, view_directions = sfm_data
    candidates: list[tuple[int, np.ndarray, int]] = []
    for seq_idx, view_direction in zip(seq_indices, view_directions):
        mask_path = masks_dir / f"{int(seq_idx):06d}.png"
        if not mask_path.exists():
            continue
        area = _mask_area(mask_path)
        if area > 0:
            candidates.append((int(seq_idx), view_direction, area))
    return _select_pose_diverse_candidates(candidates, count)


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


def select_frames(
    job_dir: Path,
    *,
    capture_mode: str = "two_stage",
    bin_deg: float = 60.0,
    stationary_count: int = 6,
    fallback_count: int = 6,
) -> list[str]:
    """Select frames for the requested capture contract."""
    selected, _ = select_frames_with_report(
        job_dir,
        capture_mode=capture_mode,
        bin_deg=bin_deg,
        stationary_count=stationary_count,
        fallback_count=fallback_count,
    )
    return selected


def select_frames_with_report(
    job_dir: Path,
    *,
    capture_mode: str = "two_stage",
    bin_deg: float = 60.0,
    stationary_count: int = 6,
    fallback_count: int = 6,
) -> tuple[list[str], dict]:
    """Select frames and return provenance describing the selection policy."""
    if capture_mode == "two_stage":
        selected = select_frames_by_angle_bins(job_dir, bin_deg=bin_deg)
        method = "cumulative_orbit_angle_bins"
        motion_assumption = "stationary_then_reoriented_then_stationary"
    elif capture_mode == "stationary":
        selected = select_frames_by_view_diversity(job_dir, count=stationary_count)
        method = "camera_view_direction_farthest_point"
        motion_assumption = "object_stationary_throughout"
    else:
        raise ValueError(f"Unsupported capture_mode: {capture_mode}")

    if not selected:
        print("[select_frames] SfM selection empty; using mask-area fallback")
        selected = select_frames_fallback(job_dir, n=fallback_count)
        method = "mask_area_fallback"

    report = {
        "capture_mode": capture_mode,
        "object_motion_assumption": motion_assumption,
        "object_motion_validation": "capture_procedure_contract",
        "selection_method": method,
        "selected_frames": selected,
    }
    if capture_mode == "two_stage":
        report["angle_bin_degrees"] = bin_deg
    else:
        report["requested_view_count"] = stationary_count
    return selected, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--capture_mode",
        choices=["two_stage", "stationary"],
        default="two_stage",
    )
    parser.add_argument("--bin_deg", type=float, default=60.0)
    parser.add_argument("--stationary_count", type=int, default=6)
    parser.add_argument("--fallback_count", type=int, default=6)
    args = parser.parse_args()
    if args.bin_deg <= 0:
        parser.error("--bin_deg must be greater than 0")
    if args.fallback_count < 1:
        parser.error("--fallback_count must be at least 1")
    if args.stationary_count < 1:
        parser.error("--stationary_count must be at least 1")

    selected, report = select_frames_with_report(
        args.job_dir,
        capture_mode=args.capture_mode,
        bin_deg=args.bin_deg,
        stationary_count=args.stationary_count,
        fallback_count=args.fallback_count,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(selected, indent=2) + "\n")
    report_path = args.output_path.parent / "selection_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[select_frames] selected {len(selected)} frames: {selected}")
    print(f"[select_frames] provenance: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
