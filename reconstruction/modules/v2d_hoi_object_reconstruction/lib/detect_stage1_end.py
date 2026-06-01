#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Auto-detect the end of Stage 1 (first scanning loop) from CuSFM camera trajectory.

Algorithm:
  1. Load CuSFM keyframe camera positions.
  2. Fit a plane to all positions via PCA (handles non-horizontal scanning planes).
  3. Project positions onto the fitted plane, find centroid (≈ object center).
  4. Compute per-frame azimuthal angle of camera around centroid in the plane.
  5. Unwrap to cumulative angle; smooth with a rolling window.
  6. Compute local slope (angular velocity); find the longest sustained low-slope
     plateau across the entire sequence (the transition period where the person
     stops and rotates the object).
  7. Stage-1 end = frame just before the longest plateau starts (backed off by
     buffer_deg for safety).

  If no qualifying plateau is found the script exits with code 1 and writes
  result.json with stage1_end_frame=null.  The pipeline should treat this as a
  fatal error and stop with a clear message rather than continuing with an
  incorrect split point.

Usage:
  python detect_stage1_end.py \\
      --sfm_keyframes /data/hoi_obj_recon/<job>/sfm/keyframes/frames_meta.json \\
      --frames_meta   /data/hoi_obj_recon/<job>/frames_meta.json \\
      [--smooth_window 15]         rolling window for slope smoothing (keyframes)
      [--slope_drop_frac 0.35]    slope fraction of median below which = plateau
      [--min_plateau_len 5]       min consecutive keyframes below threshold
      [--tail_exclude_frac 0.15]  ignore plateaus starting in last X% of sequence
      [--buffer_deg 10]           back off this many degrees before plateau start
      [--output_dir /tmp/stage1_debug]

  --max_angle_deg is accepted but ignored (deprecated).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as _Rotation


def _axis_angle_to_matrix(ax, ay, az, angle_degrees):
    axis = np.array([ax, ay, az])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    return _Rotation.from_rotvec((axis / norm) * np.deg2rad(angle_degrees)).as_matrix()


def cam_to_world_to_matrix(c2w):
    aa = c2w['axis_angle']
    t  = c2w['translation']
    T  = np.eye(4)
    T[:3, :3] = _axis_angle_to_matrix(aa['x'], aa['y'], aa['z'], aa['angle_degrees'])
    T[:3,  3] = [t['x'], t['y'], t['z']]
    return T


# ── Plane fitting via PCA ──────────────────────────────────────────────────────

def fit_plane_pca(points):
    """
    Fit a plane to 3D points via PCA.
    Returns (normal, basis_u, basis_v, centroid).
      normal   — unit normal (smallest variance direction)
      basis_u  — first in-plane axis (largest variance)
      basis_v  — second in-plane axis
      centroid — mean of points
    """
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    return Vt[2], Vt[0], Vt[1], centroid


def project_onto_plane(points, centroid, basis_u, basis_v):
    centered = points - centroid
    return np.stack([centered @ basis_u, centered @ basis_v], axis=1)


# ── Rolling statistics ─────────────────────────────────────────────────────────

def sliding_window_slope(angles_deg, window):
    """
    Compute local slope (deg/keyframe) as a centered finite difference
    over a sliding window: slope[i] = (angle[i+W//2] - angle[i-W//2]) / W.
    Edge frames use the available span.
    """
    N = len(angles_deg)
    half = window // 2
    slopes = np.zeros(N)
    for i in range(N):
        lo = max(0, i - half)
        hi = min(N - 1, i + half)
        span = hi - lo
        if span > 0:
            slopes[i] = (angles_deg[hi] - angles_deg[lo]) / span
    return slopes


# ── Plateau detection ──────────────────────────────────────────────────────────

def detect_longest_plateau(slopes, median_slope, drop_frac, min_len,
                           tail_exclude_frac=0.15, qualify_frac=0.65,
                           spread_frac=1.5):
    """
    Find the longest contiguous region where the rolling-mean absolute slope
    (window = min_len frames) is significantly lower than the active-scanning
    median slope.  This is the transition pause where the person stops moving
    and rotates the object between Stage 1 and Stage 2.

    Algorithm:
      1. Compute rolling-mean of |slope| over min_len frames → window_mean[i].
      2. Find the global minimum of window_mean (before the tail exclusion zone).
         This is the "quietest" part of the scan.
      3. Qualify check: min_wm must be < median * qualify_frac (default 0.65).
         Scans with no real transition have no deep minimum and fail here.
      4. Adaptive threshold = max(min_wm * spread_frac, |median| * drop_frac).
         • For a clean, crisp pause  : spread term is tiny → strict drop_frac wins.
         • For a noisy / partial pause: spread term gives a threshold relative to
           the actual minimum, catching the whole neighbourhood even when the
           camera never fully stops.
      5. Find the longest contiguous run of positions below this threshold
         (start must be before tail_start).

    Returns (plateau_start_idx, plateau_len, method_str):
      plateau_start_idx — keyframe index of the first frame of the longest plateau
      plateau_len       — length of the plateau in frames
      method_str        — 'longest_plateau' if found, 'no_plateau' otherwise
    """
    N = len(slopes)
    if N < min_len:
        return None, 0, 'no_plateau'

    tail_start = int(N * (1.0 - tail_exclude_frac))

    abs_slopes  = np.abs(slopes)
    cumsum      = np.concatenate([[0.0], np.cumsum(abs_slopes)])
    window_mean = (cumsum[min_len:] - cumsum[:-min_len]) / min_len  # length N-min_len+1

    # Step 2: global minimum before tail
    search_end = min(tail_start, len(window_mean))
    if search_end <= 0:
        return None, 0, 'no_plateau'
    i_min  = int(np.argmin(window_mean[:search_end]))
    min_wm = window_mean[i_min]

    # Step 3: qualify check — the minimum must be a meaningful slowdown
    abs_median = abs(median_slope)
    if min_wm >= abs_median * qualify_frac:
        return None, 0, 'no_plateau'

    # Step 4: adaptive threshold
    threshold = max(min_wm * spread_frac, abs_median * drop_frac)

    # Step 5: longest contiguous qualifying run (start < tail_start)
    best_start    = None
    best_len      = 0
    run_start_pos = None
    run_len_pos   = 0

    for i in range(min(len(window_mean), tail_start)):
        if window_mean[i] < threshold:
            if run_start_pos is None:
                run_start_pos = i
            run_len_pos += 1
        else:
            if run_start_pos is not None:
                actual_len = run_len_pos + min_len - 1
                if actual_len > best_len:
                    best_start = run_start_pos
                    best_len   = actual_len
            run_start_pos = None
            run_len_pos   = 0

    if run_start_pos is not None:
        actual_len = run_len_pos + min_len - 1
        if actual_len > best_len:
            best_start = run_start_pos
            best_len   = actual_len

    if best_start is not None:
        return best_start, best_len, 'longest_plateau'
    return None, 0, 'no_plateau'


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect Stage-1 end from CuSFM trajectory slope change")
    parser.add_argument('--sfm_keyframes', required=True)
    parser.add_argument('--frames_meta',   required=True)
    parser.add_argument('--smooth_window',      type=int,   default=15,
                        help='Rolling window (keyframes) for slope smoothing (default: 15)')
    parser.add_argument('--slope_drop_frac',    type=float, default=0.35,
                        help='Slope below median*frac triggers plateau (default: 0.35)')
    parser.add_argument('--min_plateau_len',    type=int,   default=5,
                        help='Min consecutive low-slope keyframes for plateau (default: 5)')
    parser.add_argument('--tail_exclude_frac',  type=float, default=0.15,
                        help='Ignore plateaus starting in last X fraction of sequence (default: 0.15)')
    parser.add_argument('--buffer_deg',         type=float, default=10.0,
                        help='Back off by this many degrees before the plateau start (default: 10°)')
    parser.add_argument('--max_angle_deg',      type=float, default=None,
                        help='[DEPRECATED — ignored] Previously limited the search window.')
    parser.add_argument('--output_dir', default='/tmp/stage1_debug')
    args = parser.parse_args()

    if args.max_angle_deg is not None:
        print(f"WARNING: --max_angle_deg is deprecated and will be ignored. "
              f"The search now covers the full sequence.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build timestamp → seq_idx ─────────────────────────────────────────────
    with open(args.frames_meta) as f:
        meta = json.load(f)
    cam_params = meta['camera_params_id_to_camera_params']
    left_sids, right_sids = {}, set()
    for kf in meta['keyframes_metadata']:
        cam_id = kf['camera_params_id']
        sid    = int(kf['synced_sample_id'])
        sensor = cam_params[cam_id]['sensor_meta_data']['sensor_name']
        if 'front_stereo_camera_left' in sensor:
            left_sids[sid] = int(kf['timestamp_microseconds'])
        elif 'front_stereo_camera_right' in sensor:
            right_sids.add(sid)
    common_sids   = sorted(set(left_sids) & right_sids)
    ts_to_seq_idx = {left_sids[sid]: i for i, sid in enumerate(common_sids)}

    # ── Load CuSFM keyframes (left camera only) ───────────────────────────────
    with open(args.sfm_keyframes) as f:
        sfm = json.load(f)

    frames = []
    for kf in sfm['keyframes_metadata']:
        if 'front_stereo_camera_left' not in kf.get('image_name', ''):
            continue
        ts_us   = int(kf['timestamp_microseconds'])
        seq_idx = ts_to_seq_idx.get(ts_us)
        if seq_idx is None:
            continue
        T = cam_to_world_to_matrix(kf['camera_to_world'])
        frames.append((seq_idx, T[:3, 3]))

    frames.sort(key=lambda x: x[0])
    seq_indices = np.array([f[0] for f in frames])
    positions   = np.array([f[1] for f in frames])
    N = len(frames)
    print(f"Loaded {N} CuSFM keyframes, seq_idx {seq_indices[0]}–{seq_indices[-1]}")

    # ── Fit plane ─────────────────────────────────────────────────────────────
    normal, basis_u, basis_v, centroid_3d = fit_plane_pca(positions)
    print(f"Plane normal:   {normal}")
    print(f"Plane centroid: {centroid_3d}")

    # ── Project onto plane ────────────────────────────────────────────────────
    pts_2d      = project_onto_plane(positions, centroid_3d, basis_u, basis_v)
    centroid_2d = pts_2d.mean(axis=0)
    pts_c       = pts_2d - centroid_2d
    print(f"2D centroid (in-plane): {centroid_2d}")

    # ── Cumulative angle ──────────────────────────────────────────────────────
    angles_raw       = np.arctan2(pts_c[:, 1], pts_c[:, 0])
    angles_unwrapped = np.unwrap(angles_raw)
    angles_unwrapped -= angles_unwrapped[0]
    angles_deg = np.rad2deg(angles_unwrapped)

    # ── Slope with rolling linear fit + smoothing ─────────────────────────────
    W = args.smooth_window
    slopes_smooth = sliding_window_slope(angles_deg, W)

    # Median slope computed over the middle 60% of frames (avoid edges)
    lo, hi       = int(0.2 * N), int(0.8 * N)
    median_slope  = np.median(slopes_smooth[lo:hi])
    print(f"Median slope (active scanning): {median_slope:.3f} °/keyframe")
    print(f"Strict drop threshold:          < {median_slope * args.slope_drop_frac:.3f} °/keyframe  "
          f"({args.slope_drop_frac*100:.0f}% of median)")
    print(f"Qualify threshold:              global min must be < {median_slope * 0.65:.3f} °/keyframe  "
          f"(65% of median)")

    # ── Detect longest sustained low-slope plateau (full sequence) ───────────
    plateau_idx, plateau_len, method = detect_longest_plateau(
        slopes_smooth, median_slope,
        args.slope_drop_frac, args.min_plateau_len,
        tail_exclude_frac=args.tail_exclude_frac,
    )

    if plateau_idx is not None:
        # Apply angle buffer: back off toward 0 by buffer_deg from the plateau start.
        # Works correctly for both positive (CCW) and negative (CW) cumulative angles.
        transition_angle = angles_deg[plateau_idx]
        if transition_angle >= 0:
            target_angle = transition_angle - args.buffer_deg
            buffered_idx = plateau_idx
            for i in range(plateau_idx, -1, -1):
                if angles_deg[i] <= target_angle:
                    buffered_idx = i
                    break
        else:
            target_angle = transition_angle + args.buffer_deg  # less negative → closer to 0
            buffered_idx = plateau_idx
            for i in range(plateau_idx, -1, -1):
                if angles_deg[i] >= target_angle:
                    buffered_idx = i
                    break
        stage1_end_seq = seq_indices[buffered_idx]
        print(f"\n*** Stage-1 end detected ({method}) ***")
        print(f"  Plateau start keyframe index: {plateau_idx}  (seq={seq_indices[plateau_idx]}, angle={transition_angle:.1f}°)")
        print(f"  Plateau length:               {plateau_len} keyframes")
        print(f"  Buffer:                       {args.buffer_deg:.0f}° toward 0 → target angle {target_angle:.1f}°")
        print(f"  Buffered keyframe index:      {buffered_idx}  (seq={stage1_end_seq}, angle={angles_deg[buffered_idx]:.1f}°)")
        print(f"  Slope at plateau start:       {slopes_smooth[plateau_idx]:.3f} °/keyframe  (median: {median_slope:.3f})")
    else:
        buffered_idx   = None
        stage1_end_seq = None
        target_angle   = None
        print(
            f"\n{'='*60}\n"
            f"WARNING: Stage-1 end could NOT be detected.\n"
            f"  No sustained low-slope plateau (>= {args.min_plateau_len} keyframes, "
            f"slope < {args.slope_drop_frac*100:.0f}% of median) was found in the\n"
            f"  trajectory (excluding the last {args.tail_exclude_frac*100:.0f}% of frames).\n"
            f"\n"
            f"  Likely causes:\n"
            f"    • The scan has no clear Stage-1 → Stage-2 transition pause.\n"
            f"    • The object was not manually repositioned between the two scan stages.\n"
            f"    • CuSFM may have failed to reconstruct the transition segment.\n"
            f"\n"
            f"  Action required:\n"
            f"    Inspect the debug plots in {out_dir}/ and either:\n"
            f"      a) Re-collect the data following the two-stage scan protocol, or\n"
            f"      b) Specify --stage1_end_frame <frame> manually.\n"
            f"{'='*60}"
        )

    # ── Plot 1: 3D camera positions + fitted plane ────────────────────────────
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                     c=np.arange(N), cmap='viridis', s=10)
    plt.colorbar(sc, ax=ax, label='Keyframe index')
    r = np.ptp(positions, axis=0).max() * 0.6
    gu, gv = np.linspace(-r, r, 5), np.linspace(-r, r, 5)
    GU, GV = np.meshgrid(gu, gv)
    pp = centroid_3d + GU[..., None] * basis_u + GV[..., None] * basis_v
    ax.plot_surface(pp[..., 0], pp[..., 1], pp[..., 2], alpha=0.15, color='cyan')
    ax.scatter(*centroid_3d, color='red', s=80, zorder=5, label='centroid')
    if plateau_idx is not None:
        ax.scatter(*positions[plateau_idx], color='red', s=100, marker='^', zorder=6,
                   label=f'plateau start seq={seq_indices[plateau_idx]} [{method}]')
        ax.scatter(*positions[buffered_idx], color='orange', s=150, marker='*', zorder=7,
                   label=f'stage1 end seq={stage1_end_seq} (-{args.buffer_deg:.0f}°)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D camera positions + fitted plane')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / '1_camera_positions_3d.png', dpi=120)
    plt.close(fig)
    print(f"Saved: {out_dir}/1_camera_positions_3d.png")

    # ── Plot 2: 2D projected positions ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(pts_c[:, 0], pts_c[:, 1], c=np.arange(N), cmap='viridis', s=15)
    plt.colorbar(sc, ax=ax, label='Keyframe index')
    ax.scatter(0, 0, color='red', s=80, zorder=5, label='centroid')
    ax.plot(pts_c[:, 0], pts_c[:, 1], 'k-', lw=0.4, alpha=0.4)
    if plateau_idx is not None:
        ax.scatter(pts_c[plateau_idx, 0], pts_c[plateau_idx, 1],
                   color='red', s=100, marker='^', zorder=6,
                   label=f'plateau start seq={seq_indices[plateau_idx]} [{method}]')
        ax.scatter(pts_c[buffered_idx, 0], pts_c[buffered_idx, 1],
                   color='orange', s=200, marker='*', zorder=7,
                   label=f'stage1 end seq={stage1_end_seq} (-{args.buffer_deg:.0f}°)')
    ax.set_aspect('equal')
    ax.set_xlabel(f'basis_u  {np.round(basis_u, 2)}')
    ax.set_ylabel(f'basis_v  {np.round(basis_v, 2)}')
    ax.set_title('Camera positions projected onto fitted plane')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / '2_projected_2d.png', dpi=120)
    plt.close(fig)
    print(f"Saved: {out_dir}/2_projected_2d.png")

    # ── Plot 3: Cumulative angle ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(seq_indices, angles_deg, 'b-', lw=1.5, label='cumulative angle')
    if plateau_idx is not None:
        ax.axvline(seq_indices[plateau_idx], color='red', lw=1.5, ls='--',
                   label=f'plateau start (seq={seq_indices[plateau_idx]}, {angles_deg[plateau_idx]:.1f}°, len={plateau_len}kf)')
        ax.axhline(target_angle, color='orange', lw=1.5, ls=':',
                   label=f'buffer target ({target_angle:.1f}°)')
        ax.axvline(stage1_end_seq, color='orange', lw=2,
                   label=f'stage1 end (seq={stage1_end_seq}, {angles_deg[buffered_idx]:.1f}°)')
    ax.set_xlabel('seq_idx'); ax.set_ylabel('Cumulative angle (°)')
    ax.set_title('Cumulative azimuthal angle around object centroid')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / '3_cumulative_angle.png', dpi=120)
    plt.close(fig)
    print(f"Saved: {out_dir}/3_cumulative_angle.png")

    # ── Plot 4: Slope (raw + smoothed) + threshold ────────────────────────────
    strict_thr = abs(median_slope) * args.slope_drop_frac
    # Recompute the adaptive threshold used by detect_longest_plateau for display
    abs_slopes_disp = np.abs(slopes_smooth)
    cumsum_disp = np.concatenate([[0.0], np.cumsum(abs_slopes_disp)])
    wm_disp = (cumsum_disp[args.min_plateau_len:] - cumsum_disp[:-args.min_plateau_len]) / args.min_plateau_len
    tail_s = int(N * (1.0 - args.tail_exclude_frac))
    min_wm_disp = float(np.min(wm_disp[:min(tail_s, len(wm_disp))]))
    adapt_thr = max(min_wm_disp * 1.5, strict_thr)
    thr_line = adapt_thr
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(seq_indices, slopes_smooth, color='blue', lw=1.8, label=f'slope (sliding window, w={W})')
    ax.axhline(thr_line,  color='green', ls='--', lw=1.5,
               label=f'adaptive threshold ({thr_line:.2f}°/kf = max(min×1.5, {strict_thr:.2f}))')
    ax.axhline(-thr_line, color='green', ls='--', lw=1.5)   # mirror for CW scans
    ax.axhline(median_slope, color='gray', ls=':', lw=1, label=f'median slope ({median_slope:.2f}°/kf)')
    ax.axhline(0, color='k', lw=0.5)
    # Shade the tail-exclusion zone
    tail_start_seq = seq_indices[int(N * (1.0 - args.tail_exclude_frac))]
    ax.axvspan(tail_start_seq, seq_indices[-1], alpha=0.08, color='red',
               label=f'tail exclusion (last {args.tail_exclude_frac*100:.0f}%)')
    if plateau_idx is not None:
        # Shade the detected longest plateau
        plateau_end_idx = min(plateau_idx + plateau_len - 1, N - 1)
        ax.axvspan(seq_indices[plateau_idx], seq_indices[plateau_end_idx],
                   alpha=0.15, color='cyan',
                   label=f'longest plateau ({plateau_len} kf)')
        ax.axvline(seq_indices[plateau_idx], color='red', lw=1.5, ls='--',
                   label=f'plateau start (seq={seq_indices[plateau_idx]})')
        ax.axvline(stage1_end_seq, color='orange', lw=2,
                   label=f'stage1 end (seq={stage1_end_seq}, -{args.buffer_deg:.0f}°)')
    ax.set_xlabel('seq_idx'); ax.set_ylabel('Angular velocity (°/keyframe)')
    ax.set_title(f'Slope of cumulative angle  [smooth_window={W}, drop_frac={args.slope_drop_frac}, min_len={args.min_plateau_len}, tail_excl={args.tail_exclude_frac}]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / '4_slope.png', dpi=120)
    plt.close(fig)
    print(f"Saved: {out_dir}/4_slope.png")

    # ── Plot 5: Raw angle (sanity check for unwrap) ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(seq_indices, np.rad2deg(angles_raw), 'g-', lw=1.2, label='raw angle (atan2)')
    if plateau_idx is not None:
        ax.axvline(seq_indices[plateau_idx], color='red', lw=1.5, ls='--',
                   label=f'plateau start (seq={seq_indices[plateau_idx]})')
        ax.axvline(stage1_end_seq, color='orange', lw=2,
                   label=f'stage1 end (seq={stage1_end_seq}, -{args.buffer_deg:.0f}°)')
    ax.set_xlabel('seq_idx'); ax.set_ylabel('Angle (°)')
    ax.set_title('Raw azimuthal angle (before unwrap) — check for discontinuities')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / '5_raw_angle.png', dpi=120)
    plt.close(fig)
    print(f"Saved: {out_dir}/5_raw_angle.png")

    print(f"\nAll plots saved to {out_dir}/")
    if stage1_end_seq is not None:
        print(f"\nSuggested --stage1_end_frame: {stage1_end_seq}"
              f"  (plateau start seq={seq_indices[plateau_idx]}, len={plateau_len} kf,"
              f" buffer={args.buffer_deg:.0f}°)")
        (out_dir / "result.json").write_text(
            json.dumps({
                "stage1_end_frame": int(stage1_end_seq),
                "plateau_start_seq": int(seq_indices[plateau_idx]),
                "plateau_len_keyframes": int(plateau_len),
                "method": method,
            })
        )
    else:
        # Write a result.json with null so the pipeline can emit a clear error
        # message rather than a generic "file not found" failure.
        (out_dir / "result.json").write_text(
            json.dumps({
                "stage1_end_frame": None,
                "method": "no_plateau",
                "reason": (
                    "No sustained low-slope plateau found. "
                    "The scan may lack a Stage-1 to Stage-2 transition. "
                    "Use --stage1_end_frame to specify manually."
                ),
            })
        )
        print(f"result.json written with stage1_end_frame=null — pipeline will stop.")
        sys.exit(0)  # exit 0: script ran correctly; the data has no detectable plateau


if __name__ == '__main__':
    main()
