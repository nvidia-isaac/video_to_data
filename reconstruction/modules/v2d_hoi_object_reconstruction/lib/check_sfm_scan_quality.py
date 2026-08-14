#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate CuSFM poses before using them for automatic two-loop stage splitting."""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def _axis_angle_to_rotation(axis_angle):
    axis = np.array([
        axis_angle["x"],
        axis_angle["y"],
        axis_angle["z"],
    ], dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return Rotation.identity()
    return Rotation.from_rotvec((axis / norm) * math.radians(axis_angle["angle_degrees"]))


def _load_timestamp_to_seq(frames_meta_path):
    with open(frames_meta_path) as f:
        meta = json.load(f)

    cam_params = meta["camera_params_id_to_camera_params"]
    left_sids = {}
    right_sids = set()
    for keyframe in meta["keyframes_metadata"]:
        cam_id = keyframe["camera_params_id"]
        sid = int(keyframe["synced_sample_id"])
        sensor = cam_params[cam_id]["sensor_meta_data"]["sensor_name"]
        if "front_stereo_camera_left" in sensor:
            left_sids[sid] = int(keyframe["timestamp_microseconds"])
        elif "front_stereo_camera_right" in sensor:
            right_sids.add(sid)

    common_sids = sorted(set(left_sids) & right_sids)
    return {left_sids[sid]: idx for idx, sid in enumerate(common_sids)}


def _load_sfm_keyframes(sfm_keyframes_path, frames_meta_path):
    ts_to_seq = _load_timestamp_to_seq(frames_meta_path)

    with open(sfm_keyframes_path) as f:
        sfm = json.load(f)

    frames = []
    unmatched = 0
    for keyframe in sfm["keyframes_metadata"]:
        if "front_stereo_camera_left" not in keyframe.get("image_name", ""):
            continue
        ts_us = int(keyframe["timestamp_microseconds"])
        seq_idx = ts_to_seq.get(ts_us)
        if seq_idx is None:
            unmatched += 1
            continue
        c2w = keyframe["camera_to_world"]
        t = c2w["translation"]
        frames.append({
            "seq_idx": seq_idx,
            "timestamp_us": ts_us,
            "position": np.array([t["x"], t["y"], t["z"]], dtype=float),
            "rotation": _axis_angle_to_rotation(c2w["axis_angle"]),
        })

    frames.sort(key=lambda item: item["seq_idx"])
    return frames, unmatched


def _fit_projected_orbit(positions):
    centroid = positions.mean(axis=0)
    _, _, vt = np.linalg.svd(positions - centroid, full_matrices=False)
    projected = np.stack([
        (positions - centroid) @ vt[0],
        (positions - centroid) @ vt[1],
    ], axis=1)
    center_2d = projected.mean(axis=0)
    raw_angles = np.arctan2(
        projected[:, 1] - center_2d[1],
        projected[:, 0] - center_2d[0],
    )
    angles_deg = np.degrees(np.unwrap(raw_angles))
    angles_deg -= angles_deg[0]
    return projected, center_2d, angles_deg


def _robust_threshold(values, sigma, floor):
    if len(values) == 0:
        return float("inf"), 0.0, 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + sigma * 1.4826 * mad
    return max(threshold, floor), median, mad


def _backtracking_fraction(angles_deg):
    deltas = np.diff(angles_deg)
    if len(deltas) == 0:
        return 0.0, 0.0, 0.0

    direction = 1.0 if angles_deg[-1] >= 0 else -1.0
    directed = deltas * direction
    forward = float(np.clip(directed, 0.0, None).sum())
    backward = float(np.clip(-directed, 0.0, None).sum())
    total = forward + backward
    fraction = backward / total if total > 1e-9 else 0.0
    return fraction, forward, backward


def _rotation_deltas_deg(rotations):
    deltas = []
    for prev, curr in zip(rotations[:-1], rotations[1:]):
        deltas.append((prev.inv() * curr).magnitude() * 180.0 / math.pi)
    return np.array(deltas, dtype=float)


def _stats(values):
    if len(values) == 0:
        return {"median": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _add_check(checks, failures, name, passed, message, metrics=None):
    entry = {
        "name": name,
        "passed": bool(passed),
        "message": message,
    }
    if metrics is not None:
        entry["metrics"] = metrics
    checks.append(entry)
    if not passed:
        failures.append(message)


def _plot_diagnostics(out_dir, seq_indices, positions, projected, angles_deg,
                      translation_steps, translation_threshold, rotation_deltas):
    order = np.arange(len(seq_indices))

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(projected[:, 0], projected[:, 1], c=order, s=8, cmap="viridis")
    ax.scatter(projected[0, 0], projected[0, 1], c="green", s=45, marker="o", label="start")
    ax.scatter(projected[-1, 0], projected[-1, 1], c="red", s=45, marker="x", label="end")
    if len(translation_steps):
        jump_idx = np.where(translation_steps > translation_threshold)[0]
        if len(jump_idx):
            bad = np.unique(np.clip(np.concatenate([jump_idx, jump_idx + 1]), 0, len(seq_indices) - 1))
            ax.scatter(
                projected[bad, 0],
                projected[bad, 1],
                s=30,
                facecolors="none",
                edgecolors="red",
                label="large step",
            )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Projected CuSFM Camera Path")
    ax.set_xlabel("PCA axis 0")
    ax.set_ylabel("PCA axis 1")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="SfM keyframe order")
    fig.tight_layout()
    fig.savefig(out_dir / "projected_path.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(seq_indices, angles_deg, lw=1.6)
    ax.axhline(300.0, color="gray", ls="--", lw=1, label="+300 deg")
    ax.axhline(-300.0, color="gray", ls="--", lw=1, label="-300 deg")
    ax.set_title("Cumulative Projected Orbit Angle")
    ax.set_xlabel("seq_idx")
    ax.set_ylabel("degrees")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_angle.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if len(translation_steps):
        axes[0].plot(seq_indices[1:], translation_steps, lw=1.2)
        axes[0].axhline(
            translation_threshold,
            color="red",
            ls="--",
            lw=1,
            label=f"large-step threshold {translation_threshold:.3f} m",
        )
        axes[0].legend(loc="best")
    axes[0].set_title("Consecutive Translation Step")
    axes[0].set_ylabel("meters")

    if len(rotation_deltas):
        axes[1].plot(seq_indices[1:], rotation_deltas, lw=1.2)
    axes[1].set_title("Consecutive Rotation Delta")
    axes[1].set_xlabel("seq_idx")
    axes[1].set_ylabel("degrees")
    fig.tight_layout()
    fig.savefig(out_dir / "pose_steps.png", dpi=130)
    plt.close(fig)


def check_quality(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, unmatched = _load_sfm_keyframes(args.sfm_keyframes, args.frames_meta)
    checks = []
    failures = []

    _add_check(
        checks,
        failures,
        "min_keyframes",
        len(frames) >= args.min_keyframes,
        f"CuSFM left-camera keyframes: {len(frames)}; required >= {args.min_keyframes}",
        {"num_keyframes": len(frames), "unmatched_keyframes": unmatched},
    )

    if len(frames) < 2:
        result = {
            "passed": False,
            "method": "sfm_scan_quality",
            "failure_reasons": failures or ["Need at least two matched CuSFM keyframes"],
            "checks": checks,
            "metrics": {"num_keyframes": len(frames), "unmatched_keyframes": unmatched},
        }
        with open(out_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    seq_indices = np.array([item["seq_idx"] for item in frames], dtype=int)
    timestamps_s = np.array([item["timestamp_us"] for item in frames], dtype=float) / 1e6
    positions = np.stack([item["position"] for item in frames], axis=0)
    rotations = [item["rotation"] for item in frames]

    translation_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    rotation_deltas = _rotation_deltas_deg(rotations)
    dt = np.diff(timestamps_s)
    translation_speeds = np.divide(
        translation_steps,
        dt,
        out=np.full_like(translation_steps, np.nan),
        where=dt > 1e-9,
    )
    step_threshold, step_median, step_mad = _robust_threshold(
        translation_steps,
        args.robust_step_sigma,
        args.large_step_floor_m,
    )
    large_step_count = int(np.count_nonzero(translation_steps > step_threshold))
    large_step_fraction = large_step_count / max(1, len(translation_steps))
    max_translation_step = float(np.max(translation_steps)) if len(translation_steps) else 0.0
    max_rotation_delta = float(np.max(rotation_deltas)) if len(rotation_deltas) else 0.0

    projected, center_2d, angles_deg = _fit_projected_orbit(positions)
    angle_span = float(np.max(angles_deg) - np.min(angles_deg))
    signed_total_angle = float(angles_deg[-1])
    total_abs_angle = float(np.abs(np.diff(angles_deg)).sum())
    backtrack_fraction, forward_angle, backward_angle = _backtracking_fraction(angles_deg)

    _add_check(
        checks,
        failures,
        "max_translation_step",
        args.max_translation_step_m <= 0.0 or max_translation_step <= args.max_translation_step_m,
        (
            f"Max consecutive translation step is {max_translation_step:.3f} m; "
            f"allowed <= {args.max_translation_step_m:.3f} m"
        ),
        {"max_translation_step_m": max_translation_step},
    )
    _add_check(
        checks,
        failures,
        "large_translation_step_fraction",
        large_step_fraction <= args.max_large_step_fraction,
        (
            f"Large translation step fraction is {large_step_fraction:.3f}; "
            f"allowed <= {args.max_large_step_fraction:.3f}"
        ),
        {
            "large_step_count": large_step_count,
            "num_steps": int(len(translation_steps)),
            "large_step_fraction": large_step_fraction,
            "large_step_threshold_m": float(step_threshold),
        },
    )
    _add_check(
        checks,
        failures,
        "max_rotation_step",
        args.max_rotation_step_deg <= 0.0 or max_rotation_delta <= args.max_rotation_step_deg,
        (
            f"Max consecutive rotation step is {max_rotation_delta:.1f} deg; "
            f"allowed <= {args.max_rotation_step_deg:.1f} deg"
        ),
        {"max_rotation_step_deg": max_rotation_delta},
    )
    _add_check(
        checks,
        failures,
        "two_loop_angle_span",
        angle_span >= args.min_angle_span_deg,
        (
            f"Projected trajectory angle span is {angle_span:.1f} deg; "
            f"two-loop scan requires >= {args.min_angle_span_deg:.1f} deg"
        ),
        {
            "angle_span_deg": angle_span,
            "signed_total_angle_deg": signed_total_angle,
            "total_abs_angle_deg": total_abs_angle,
        },
    )
    _add_check(
        checks,
        failures,
        "backtracking_fraction",
        backtrack_fraction <= args.max_backtracking_fraction,
        (
            f"Projected-angle backtracking fraction is {backtrack_fraction:.3f}; "
            f"allowed <= {args.max_backtracking_fraction:.3f}"
        ),
        {
            "backtracking_fraction": backtrack_fraction,
            "forward_angle_deg": forward_angle,
            "backward_angle_deg": backward_angle,
        },
    )

    metrics = {
        "num_keyframes": int(len(frames)),
        "unmatched_keyframes": int(unmatched),
        "seq_first": int(seq_indices[0]),
        "seq_last": int(seq_indices[-1]),
        "translation_step_m": _stats(translation_steps),
        "translation_speed_mps": _stats(translation_speeds[np.isfinite(translation_speeds)]),
        "rotation_step_deg": _stats(rotation_deltas),
        "large_step_threshold_m": float(step_threshold),
        "large_step_median_m": float(step_median),
        "large_step_mad_m": float(step_mad),
        "large_step_count": large_step_count,
        "large_step_fraction": float(large_step_fraction),
        "projected_center_2d": [float(center_2d[0]), float(center_2d[1])],
        "angle_span_deg": angle_span,
        "signed_total_angle_deg": signed_total_angle,
        "total_abs_angle_deg": total_abs_angle,
        "backtracking_fraction": backtrack_fraction,
        "forward_angle_deg": forward_angle,
        "backward_angle_deg": backward_angle,
    }
    result = {
        "passed": not failures,
        "method": "sfm_scan_quality",
        "failure_reasons": failures,
        "checks": checks,
        "metrics": metrics,
    }

    _plot_diagnostics(
        out_dir,
        seq_indices,
        positions,
        projected,
        angles_deg,
        translation_steps,
        step_threshold,
        rotation_deltas,
    )
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check whether CuSFM poses satisfy the expected two-loop scan pattern"
    )
    parser.add_argument("--sfm_keyframes", required=True)
    parser.add_argument("--frames_meta", required=True)
    parser.add_argument("--output_dir", default="/tmp/sfm_scan_quality")
    parser.add_argument("--min_keyframes", type=int, default=30)
    parser.add_argument("--min_angle_span_deg", type=float, default=600.0)
    parser.add_argument("--max_backtracking_fraction", type=float, default=0.25)
    parser.add_argument("--max_translation_step_m", type=float, default=2.0)
    parser.add_argument("--max_rotation_step_deg", type=float, default=90.0)
    parser.add_argument("--robust_step_sigma", type=float, default=6.0)
    parser.add_argument("--large_step_floor_m", type=float, default=0.25)
    parser.add_argument("--max_large_step_fraction", type=float, default=0.25)
    parser.add_argument("--warn_only", action="store_true",
                        help="Always exit 0 after writing diagnostics")
    args = parser.parse_args()

    result = check_quality(args)
    print(json.dumps({
        "passed": result["passed"],
        "failure_reasons": result["failure_reasons"],
        "metrics": {
            "num_keyframes": result["metrics"]["num_keyframes"],
            "angle_span_deg": result["metrics"].get("angle_span_deg"),
            "signed_total_angle_deg": result["metrics"].get("signed_total_angle_deg"),
            "total_abs_angle_deg": result["metrics"].get("total_abs_angle_deg"),
            "backtracking_fraction": result["metrics"].get("backtracking_fraction"),
            "large_step_fraction": result["metrics"].get("large_step_fraction"),
            "max_translation_step_m": result["metrics"].get("translation_step_m", {}).get("max"),
        },
    }, indent=2))

    if result["passed"] or args.warn_only:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
