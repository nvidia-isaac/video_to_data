#!/usr/bin/env python3
"""
Auto-detect the end of Stage 1 (first scanning loop) from CuSFM camera trajectory.

Algorithm:
  1. Load CuSFM keyframe camera positions.
  2. Fit a plane to all positions via PCA (handles non-horizontal scanning planes).
  3. Project positions onto the fitted plane, find centroid (≈ object center).
  4. Compute per-frame azimuthal angle of camera around centroid in the plane.
  5. Unwrap to cumulative angle; smooth with a rolling window.
  6. Compute local slope (angular velocity); detect a sustained drop in slope
     (the transition period where the person stops and rotates the object).
  7. Stage-1 end = start of the first sustained low-slope plateau.

Usage:
  python detect_stage1_end.py \\
      --sfm_keyframes /data/hoi_obj_recon/<job>/sfm/keyframes/frames_meta.json \\
      --frames_meta   /data/hoi_obj_recon/<job>/frames_meta.json \\
      [--smooth_window 15]        rolling window for slope smoothing (keyframes)
      [--slope_drop_frac 0.35]   slope fraction of median below which = plateau
      [--min_plateau_len 5]      min consecutive keyframes below threshold
      [--output_dir /tmp/stage1_debug]
"""

import argparse
import json
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

def detect_plateau(slopes, angles_deg, median_slope, drop_frac, min_len,
                   max_angle=350.0):
    """
    Find the start of the first sustained low-slope period within the
    search window (cumulative angle <= max_angle).

    Primary: first run of >= min_len frames with slope < median*drop_frac.
    Fallback: frame with the global minimum slope in the search window
              (used when no hard threshold is crossed — gradual transitions).

    Returns (plateau_idx, method_str).
    """
    threshold = abs(median_slope) * drop_frac
    run_start = None
    run_len   = 0
    window_indices = []

    for i in range(len(slopes)):
        if abs(angles_deg[i]) > max_angle:
            break
        window_indices.append(i)
        if abs(slopes[i]) < threshold:
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len >= min_len:
                return run_start, 'threshold'
        else:
            run_start = None
            run_len   = 0

    # Fallback: minimum |slope| in search window (closest to zero angular velocity)
    if window_indices:
        win = np.array(window_indices)
        win_slopes = slopes[win]
        best = win[np.argmin(np.abs(win_slopes))]
        return best, 'min_slope_fallback'
    return None, None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect Stage-1 end from CuSFM trajectory slope change")
    parser.add_argument('--sfm_keyframes', required=True)
    parser.add_argument('--frames_meta',   required=True)
    parser.add_argument('--smooth_window',    type=int,   default=15,
                        help='Rolling window (keyframes) for slope smoothing (default: 15)')
    parser.add_argument('--slope_drop_frac',  type=float, default=0.35,
                        help='Slope below median*frac triggers plateau (default: 0.35)')
    parser.add_argument('--min_plateau_len',  type=int,   default=5,
                        help='Min consecutive low-slope keyframes for plateau (default: 5)')
    parser.add_argument('--max_angle_deg',    type=float, default=350.0,
                        help='Stop searching for plateau after this cumulative angle (default: 350°)')
    parser.add_argument('--buffer_deg',       type=float, default=10.0,
                        help='Back off by this many degrees before the detected transition (default: 10°)')
    parser.add_argument('--output_dir', default='/tmp/stage1_debug')
    args = parser.parse_args()

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
    print(f"Plateau threshold:              < {median_slope * args.slope_drop_frac:.3f} °/keyframe  "
          f"({args.slope_drop_frac*100:.0f}% of median)")

    # ── Detect first sustained low-slope plateau ──────────────────────────────
    plateau_idx, method = detect_plateau(slopes_smooth, angles_deg, median_slope,
                                         args.slope_drop_frac, args.min_plateau_len,
                                         max_angle=args.max_angle_deg)

    if plateau_idx is not None:
        # Apply angle buffer: back off to min(max_angle_deg, detected_angle - buffer_deg)
        target_angle = min(args.max_angle_deg, angles_deg[plateau_idx] - args.buffer_deg)
        buffered_idx = plateau_idx
        for i in range(plateau_idx, -1, -1):
            if angles_deg[i] <= target_angle:
                buffered_idx = i
                break
        stage1_end_seq = seq_indices[buffered_idx]
        print(f"\n*** Stage-1 end detected ({method}) ***")
        print(f"  Transition keyframe index:    {plateau_idx}  (seq={seq_indices[plateau_idx]}, angle={angles_deg[plateau_idx]:.1f}°)")
        print(f"  Buffer:                       -{args.buffer_deg:.0f}° → target angle {target_angle:.1f}°")
        print(f"  Buffered keyframe index:      {buffered_idx}  (seq={stage1_end_seq}, angle={angles_deg[buffered_idx]:.1f}°)")
        print(f"  Slope at transition: {slopes_smooth[plateau_idx]:.3f} °/keyframe  (median: {median_slope:.3f})")
    else:
        buffered_idx = None
        stage1_end_seq = None
        method = None
        print("\nStage-1 end NOT detected")

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
                   label=f'transition seq={seq_indices[plateau_idx]} [{method}]')
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
                   label=f'transition seq={seq_indices[plateau_idx]} [{method}]')
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
                   label=f'transition (seq={seq_indices[plateau_idx]}, {angles_deg[plateau_idx]:.1f}°)')
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
    thr_line = abs(median_slope) * args.slope_drop_frac
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(seq_indices, slopes_smooth, color='blue', lw=1.8, label=f'slope (sliding window, w={W})')
    ax.axhline(thr_line,  color='green', ls='--', lw=1.5,
               label=f'plateau threshold ({args.slope_drop_frac*100:.0f}% of median = {thr_line:.2f}°/kf)')
    ax.axhline(median_slope, color='gray', ls=':', lw=1, label=f'median slope ({median_slope:.2f}°/kf)')
    ax.axhline(0, color='k', lw=0.5)
    # Shade the search window [min_angle, max_angle] on the angle axis
    in_window = np.abs(angles_deg) <= args.max_angle_deg
    if in_window.any():
        win_seq = seq_indices[in_window]
        ax.axvspan(win_seq[0], win_seq[-1], alpha=0.08, color='green',
                   label=f'search window (0°–{args.max_angle_deg:.0f}°)')
    if plateau_idx is not None:
        ax.axvline(seq_indices[plateau_idx], color='red', lw=1.5, ls='--',
                   label=f'transition (seq={seq_indices[plateau_idx]})')
        ax.axvline(stage1_end_seq, color='orange', lw=2,
                   label=f'stage1 end (seq={stage1_end_seq}, -{args.buffer_deg:.0f}°)')
    ax.set_xlabel('seq_idx'); ax.set_ylabel('Angular velocity (°/keyframe)')
    ax.set_title(f'Slope of cumulative angle  [smooth_window={W}, drop_frac={args.slope_drop_frac}, min_len={args.min_plateau_len}]')
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
                   label=f'transition (seq={seq_indices[plateau_idx]})')
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
        print(f"\nSuggested --stage1_end_frame: {stage1_end_seq}  (transition at seq={seq_indices[plateau_idx]}, buffer={args.buffer_deg:.0f}°)")
        (out_dir / "result.json").write_text(
            json.dumps({"stage1_end_frame": int(stage1_end_seq)})
        )


if __name__ == '__main__':
    main()
