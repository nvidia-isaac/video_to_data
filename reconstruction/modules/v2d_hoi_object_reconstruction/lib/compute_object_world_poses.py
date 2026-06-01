# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Compute object-to-world transforms by composing:
  T_world_from_obj = T_world_from_cam  @  T_cam_from_obj

Inputs:
  - poses_dir: per-frame object-to-camera 4×4 matrices (from FoundationPose)
  - frame_metadata: maps frame_id → original image filenames (from mapping_data)
  - sfm_keyframes: camera-to-world poses for SfM keyframes (frames_meta.json)

Camera-to-world poses exist only for keyframes (~405 of 2021 frames).
Non-keyframe frames are handled by SLERP (rotation) + linear (translation)
interpolation between the nearest keyframes.

Two-stage detection:
  The object is stationary during each scan stage, so T_world_from_obj should be
  approximately constant within each stage.  A sudden jump in object translation
  or orientation in the world frame indicates the object was physically moved
  between stage 1 and stage 2.  We detect this by finding the frame index that
  maximises the within-stage compactness of the object's world position (a 1-D
  change-point search on the object translation).

Output:
  - JSON file with per-frame T_world_from_obj
  - Diagnostic plots (camera trajectory, object position, rotation angle over time)
  - Stage analysis printed to stdout

Usage:
    python compute_object_world_poses.py \\
        --poses_dir /data/basketball_full/poses \\
        --frame_metadata /path/to/mapping_data/frame_metadata.jsonl \\
        --sfm_keyframes /path/to/processed/kpmap/keyframes/frames_meta.json \\
        --camera left \\
        --output /data/basketball_full/poses_world.json \\
        --plot /data/basketball_full/poses_world_analysis.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.spatial.transform import Rotation, Slerp

from v2d_common.datatypes import Transform3d

from v2d_hoi_object_reconstruction.lib.recon_utils import (
    cam_to_world_to_matrix,
    build_timestamp_to_seq_idx,
)


# ── Pose helpers ─────────────────────────────────────────────────────────────

def rotation_angle_between(R1, R2):
    """Geodesic angle (radians) between two 3×3 rotation matrices."""
    dR = R1.T @ R2
    cos_val = (np.trace(dR) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))


# ── SfM loading & interpolation ───────────────────────────────────────────────

def build_cam_to_world_map(sfm_path, ts_us_to_seq_idx, camera='left'):
    """dict: seq_idx (int) → 4×4 T_world_from_cam, matched by timestamp_us."""
    with open(sfm_path) as f:
        data = json.load(f)
    camera_key = f'front_stereo_camera_{camera}'
    kf_map = {}
    unmatched = 0
    for kf in data['keyframes_metadata']:
        if not kf['image_name'].startswith(camera_key):
            continue
        ts_us   = int(kf['timestamp_microseconds'])
        seq_idx = ts_us_to_seq_idx.get(ts_us)
        if seq_idx is None:
            unmatched += 1
            continue
        kf_map[seq_idx] = cam_to_world_to_matrix(kf['camera_to_world'])
    if unmatched:
        print(f"  WARNING: {unmatched} SfM keyframes had no timestamp match")
    return kf_map


def interpolate_cam_to_world(kf_map, frame_ids):
    """Interpolate T_world_from_cam for all frame_ids; None if out of range."""
    sorted_ids = sorted(kf_map.keys())
    kf_min, kf_max = sorted_ids[0], sorted_ids[-1]
    ids_arr  = np.array(sorted_ids)
    rots     = Rotation.from_matrix(np.stack([kf_map[i][:3, :3] for i in sorted_ids]))
    trans    = np.stack([kf_map[i][:3, 3] for i in sorted_ids])
    slerp    = Slerp(ids_arr, rots)

    result = {}
    for fid in frame_ids:
        if fid < kf_min or fid > kf_max:
            result[fid] = None
            continue
        if fid in kf_map:
            result[fid] = kf_map[fid]
            continue
        idx_hi  = int(np.searchsorted(ids_arr, fid))
        idx_lo  = idx_hi - 1
        alpha   = (fid - ids_arr[idx_lo]) / (ids_arr[idx_hi] - ids_arr[idx_lo])
        R_interp = slerp(np.array([fid], dtype=float)).as_matrix()[0]
        t_interp = (1 - alpha) * trans[idx_lo] + alpha * trans[idx_hi]
        T = np.eye(4)
        T[:3, :3] = R_interp
        T[:3,  3] = t_interp
        result[fid] = T
    return result


# ── Stage detection ───────────────────────────────────────────────────────────

def _seg_var(prefix_sum, prefix_sum_sq, lo, hi):
    """Variance * length for segment [lo, hi) using prefix sums. O(1)."""
    length = hi - lo
    if length <= 0:
        return 0.0
    s  = prefix_sum[hi]    - prefix_sum[lo]     # (3,)
    s2 = prefix_sum_sq[hi] - prefix_sum_sq[lo]  # (3,)
    # sum of squared deviations = sum(x²) - n*mean² = s2 - s²/n
    return float(np.sum(s2 - s * s / length))


def filter_tracking_failures(frame_ids, obj_translations,
                              spike_factor=10.0, density_thresh=0.3, density_window=30):
    """
    Detect FP tracker failure using per-frame translation velocity.

    Two failure modes handled:
      1. Isolated spikes (momentary occlusion): delta_t > spike_factor × median.
      2. Sustained wandering (object leaves frame, tracker latches onto background):
         spike density in a sliding window exceeds density_thresh — everything from
         that onset onward is excluded.

    Returns:
        inlier_mask : boolean array, True = frame is reliable
        onset_idx   : index of first unreliable frame in frame_ids (None if clean)
    """
    n = len(frame_ids)
    trans_arr = np.array(obj_translations)
    delta_t = np.concatenate([[0.0], np.linalg.norm(np.diff(trans_arr, axis=0), axis=1)])

    v_median = float(np.median(delta_t[1:]))
    v_thresh = spike_factor * max(v_median, 1e-4)
    spike = delta_t >= v_thresh
    spike[0] = False

    density = uniform_filter1d(spike.astype(float), size=density_window, mode='nearest')

    onset_idx = next((i for i in range(n) if density[i] > density_thresh), None)

    inlier_mask = ~spike
    if onset_idx is not None:
        inlier_mask[onset_idx:] = False

    n_removed = int((~inlier_mask).sum())
    if n_removed:
        onset_fid = frame_ids[onset_idx] if onset_idx is not None else None
        print(f"  Velocity filter: median={v_median*100:.2f}cm/frame  "
              f"threshold={v_thresh*100:.2f}cm/frame  "
              f"onset=frame {onset_fid}  removed {n_removed}/{n} frames")

    return inlier_mask, onset_idx


def detect_three_stages(frame_ids, obj_translations, obj_rotations, smooth_kernel=15):
    """
    Find two split indices that separate:
      [stage1 stationary] [transition: object being rotated] [stage2 stationary]

    Expects pre-filtered frame_ids (tracker failure frames already removed by
    filter_tracking_failures).

    Returns:
        split1_idx, split2_idx : indices into frame_ids where transition starts/ends
        angular_jumps          : per-frame geodesic angle change (deg), for plotting
    """
    n = len(frame_ids)

    # Frame-to-frame angular change (for plotting)
    angular_jumps = np.zeros(n)
    for i in range(1, n):
        angular_jumps[i] = np.rad2deg(
            rotation_angle_between(obj_rotations[i - 1], obj_rotations[i])
        )

    # Cumulative rotation angle from the stage-1 reference
    n_ref = max(1, n // 10)
    R_ref = Rotation.from_matrix(obj_rotations[:n_ref]).mean().as_matrix()
    angles = np.array([np.rad2deg(rotation_angle_between(R_ref, R))
                       for R in obj_rotations])
    angles_s = median_filter(angles, size=smooth_kernel * 3)

    span = float(np.median(angles_s[-n_ref:])) - float(np.median(angles_s[:n_ref]))

    W = max(10, smooth_kernel)
    trans_s = np.stack([median_filter(np.array(obj_translations)[:, k], size=smooth_kernel)
                        for k in range(3)], axis=1)

    rot_thresh   = 5.0    # degrees over 2W frames
    trans_thresh = 0.05   # metres over 2W frames

    slope_rot_fwd   = np.zeros(n)
    slope_trans_fwd = np.zeros(n)
    slope_rot_fwd[:-2*W]   = angles_s[2*W:] - angles_s[:-2*W]
    slope_trans_fwd[:-2*W] = np.linalg.norm(trans_s[2*W:] - trans_s[:-2*W], axis=1)
    score_fwd = slope_rot_fwd / rot_thresh + slope_trans_fwd / trans_thresh

    slope_rot_bwd   = np.zeros(n)
    slope_trans_bwd = np.zeros(n)
    slope_rot_bwd[2*W:]   = angles_s[2*W:] - angles_s[:-2*W]
    slope_trans_bwd[2*W:] = np.linalg.norm(trans_s[2*W:] - trans_s[:-2*W], axis=1)
    score_bwd = slope_rot_bwd / rot_thresh + slope_trans_bwd / trans_thresh

    split1 = next((i for i in range(n) if score_fwd[i] > 1.0), n // 3)
    split2 = n - 1 - next((i for i in range(n) if score_bwd[n - 1 - i] > 1.0), n // 3)

    split1 = max(1, min(split1, n - 2))
    split2 = max(split1 + 1, min(split2, n - 1))

    print(f"  Asymmetric sliding-window slope W={W}  "
          f"rot_thresh={rot_thresh}°  trans_thresh={trans_thresh}m")
    print(f"  Peak fwd: rot={slope_rot_fwd.max():.1f}°  trans={slope_trans_fwd.max()*100:.1f}cm  "
          f"score={score_fwd.max():.2f}")
    print(f"  Peak bwd: rot={slope_rot_bwd.max():.1f}°  trans={slope_trans_bwd.max()*100:.1f}cm  "
          f"score={score_bwd.max():.2f}  span={span:.1f}°")

    return split1, split2, angular_jumps


def representative_pose(T_list, mad_threshold=3.0):
    """Mean pose over inlier frames (outliers removed via MAD on translation distance)."""
    trans_all = np.array([T[:3, 3] for T in T_list])
    t_median  = np.median(trans_all, axis=0)
    dists     = np.linalg.norm(trans_all - t_median, axis=1)
    mad       = np.median(np.abs(dists - np.median(dists)))
    if mad < 1e-9:
        inliers = np.ones(len(T_list), dtype=bool)
    else:
        inliers = dists < np.median(dists) + mad_threshold * mad
    T_inliers = [T for T, keep in zip(T_list, inliers) if keep]
    trans = np.mean([T[:3, 3] for T in T_inliers], axis=0)
    R_mean = Rotation.from_matrix([T[:3, :3] for T in T_inliers]).mean().as_matrix()
    T_rep = np.eye(4)
    T_rep[:3, :3] = R_mean
    T_rep[:3,  3] = trans
    return T_rep


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plots(frame_ids, cam_from_obj_list, obj_world_list, angular_jumps,
               split1_idx, split2_idx, plot_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fids  = np.array(frame_ids)
    # Camera position in object frame: T_obj_from_cam = inv(T_cam_from_obj)
    cam_in_obj = np.array([np.linalg.inv(T)[:3, 3] for T in cam_from_obj_list])
    obj_t = np.array([T[:3, 3] for T in obj_world_list])

    colors = {
        'stage1':      'steelblue',
        'transition':  'gray',
        'stage2':      'darkorange',
        'split1':      'green',
        'split2':      'red',
    }

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Object-World Pose Analysis (3 stages)', fontsize=14, fontweight='bold')

    # ── 1. Camera-to-object trajectory 3-D ───────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    sc = ax1.scatter(cam_in_obj[:, 0], cam_in_obj[:, 1], cam_in_obj[:, 2],
                     c=fids, cmap='viridis', s=4)
    ax1.scatter(*cam_in_obj[split1_idx], color=colors['split1'], s=60, zorder=5, label='transition start')
    ax1.scatter(*cam_in_obj[split2_idx], color=colors['split2'], s=60, zorder=5, label='transition end')
    ax1.set_title('Camera trajectory (object frame)')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.legend(fontsize=7)
    plt.colorbar(sc, ax=ax1, label='frame id', shrink=0.6)

    # ── 2. Camera-to-object trajectory top-down (X-Y) ────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(cam_in_obj[:split1_idx, 0],              cam_in_obj[:split1_idx, 1],
                c=colors['stage1'],      s=4, label='stage 1')
    ax2.scatter(cam_in_obj[split1_idx:split2_idx, 0],    cam_in_obj[split1_idx:split2_idx, 1],
                c=colors['transition'],  s=4, label='transition')
    ax2.scatter(cam_in_obj[split2_idx:, 0],              cam_in_obj[split2_idx:, 1],
                c=colors['stage2'],      s=4, label='stage 2')
    ax2.scatter(*cam_in_obj[split1_idx, :2], color=colors['split1'], s=80, zorder=5, marker='*')
    ax2.scatter(*cam_in_obj[split2_idx, :2], color=colors['split2'], s=80, zorder=5, marker='*')
    ax2.set_title('Camera trajectory top-down (object frame, X-Y)')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal'); ax2.legend(fontsize=8); ax2.grid(True)

    # ── 3. Object translation in cm ──────────────────────────────────────────
    obj_t_cm = obj_t * 100   # metres → centimetres

    ax3 = fig.add_subplot(2, 3, 3)
    for k, label, color in zip(range(3), ['X', 'Y', 'Z'],
                                ['tab:red', 'tab:green', 'tab:blue']):
        ax3.plot(fids, obj_t_cm[:, k], label=label, color=color, lw=0.8)
    ax3.axvline(fids[split1_idx], color=colors['split1'], ls='--', lw=1.5, label='transition start')
    ax3.axvline(fids[split2_idx], color=colors['split2'], ls='--', lw=1.5, label='transition end')
    ax3.set_title('Object position (cm)')
    ax3.set_xlabel('frame id'); ax3.set_ylabel('position (cm)')
    ax3.legend(fontsize=8); ax3.grid(True)

    # ── 4. Object rotation angle from stage-1 reference ──────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    R_ref = obj_world_list[0][:3, :3]
    angles_from_ref = [np.rad2deg(rotation_angle_between(R_ref, T[:3, :3]))
                       for T in obj_world_list]
    ax4.plot(fids, angles_from_ref, lw=0.8, color='purple')
    ax4.axvline(fids[split1_idx], color=colors['split1'], ls='--', lw=1.5, label='transition start')
    ax4.axvline(fids[split2_idx], color=colors['split2'], ls='--', lw=1.5, label='transition end')
    ax4.set_title('Object rotation from stage-1 reference (°)')
    ax4.set_xlabel('frame id'); ax4.set_ylabel('angle (°)')
    ax4.legend(fontsize=8); ax4.grid(True)

    # ── 5. Frame-to-frame angular velocity ───────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(fids, angular_jumps, lw=0.7, color='darkorange')
    ax5.axvline(fids[split1_idx], color=colors['split1'], ls='--', lw=1.5, label='transition start')
    ax5.axvline(fids[split2_idx], color=colors['split2'], ls='--', lw=1.5, label='transition end')
    ax5.set_title('Frame-to-frame rotation jump (°)')
    ax5.set_xlabel('frame id'); ax5.set_ylabel('Δangle (°)')
    ax5.legend(fontsize=8); ax5.grid(True)

    # ── 6. Camera + object positions top-down (world frame) ──────────────────
    # Camera world positions: T_world_from_cam = T_world_from_obj @ inv(T_cam_from_obj)
    cam_w = np.array([(T_wo @ np.linalg.inv(T_co))[:3, 3]
                      for T_wo, T_co in zip(obj_world_list, cam_from_obj_list)])
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(cam_w[:, 0], cam_w[:, 1], c='lightgray', s=3, label='camera')
    ax6.scatter(obj_t[:split1_idx, 0],           obj_t[:split1_idx, 1],
                c=colors['stage1'],     s=8, label='obj stage 1', alpha=0.7)
    ax6.scatter(obj_t[split1_idx:split2_idx, 0], obj_t[split1_idx:split2_idx, 1],
                c=colors['transition'], s=8, label='obj transition', alpha=0.5)
    ax6.scatter(obj_t[split2_idx:, 0],           obj_t[split2_idx:, 1],
                c=colors['stage2'],     s=8, label='obj stage 2', alpha=0.7)
    ax6.set_title('Camera + Object positions top-down (world)')
    ax6.set_xlabel('X (m)'); ax6.set_ylabel('Y (m)')
    ax6.set_aspect('equal'); ax6.legend(fontsize=8); ax6.grid(True)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute object-to-world poses + two-stage analysis')
    parser.add_argument('--poses_dir', required=True)
    parser.add_argument('--frames_meta', required=True,
                        help='frames_meta.json from mapping_data (covers all frames)')
    parser.add_argument('--sfm_keyframes', required=True)
    parser.add_argument('--camera', default='left', choices=['left', 'right'])
    parser.add_argument('--output', required=True,
                        help='Output JSON path (e.g. poses_world.json)')
    parser.add_argument('--plot', default=None,
                        help='Save diagnostic plots to this PNG path')
    parser.add_argument('--transform_output', default=None,
                        help='Save T_pose2_from_pose1 as a space-separated 4×4 txt file')
    parser.add_argument('--smooth', type=int, default=15,
                        help='Median filter half-width for stage detection (frames)')
    args = parser.parse_args()

    # ── Build timestamp → seq_idx index ──────────────────────────────────────
    print("Building timestamp → seq_idx index from frames_meta.json...")
    ts_us_to_seq_idx = build_timestamp_to_seq_idx(args.frames_meta, camera=args.camera)
    print(f"  {len(ts_us_to_seq_idx)} frames indexed")

    # ── Load SfM keyframes ────────────────────────────────────────────────────
    print(f"Loading SfM keyframes ({args.camera} camera)...")
    kf_map = build_cam_to_world_map(args.sfm_keyframes, ts_us_to_seq_idx, camera=args.camera)
    print(f"  {len(kf_map)} keyframes, frame_id range: "
          f"{min(kf_map)} – {max(kf_map)}")

    # ── Load FoundationPose poses ─────────────────────────────────────────────
    poses_dir = Path(args.poses_dir)
    pose_files = sorted(poses_dir.glob('*.json'))
    frame_ids  = [int(p.stem) for p in pose_files]
    print(f"Pose files: {len(frame_ids)} frames (0 – {max(frame_ids)})")

    # ── Interpolate camera-to-world ───────────────────────────────────────────
    print("Interpolating camera-to-world poses...")
    c2w_interp = interpolate_cam_to_world(kf_map, frame_ids)

    # ── Compose T_world_from_obj ──────────────────────────────────────────────
    print("Composing object-to-world transforms...")
    results        = []
    valid_fids     = []
    valid_c2w      = []
    valid_obj_world  = []
    valid_cam_from_obj = []
    skipped        = 0

    for fid, pose_file in zip(frame_ids, pose_files):
        raw = json.loads(pose_file.read_text())
        if isinstance(raw, dict):
            T_cam_from_obj = Transform3d.from_dict(raw).to_matrix()
        else:
            T_cam_from_obj = np.array(raw)
        T_world_from_cam = c2w_interp.get(fid)
        if T_world_from_cam is None:
            skipped += 1
            continue
        T_world_from_obj = T_world_from_cam @ T_cam_from_obj
        results.append({'frame_id': fid,
                        'T_world_from_obj': T_world_from_obj.tolist()})
        valid_fids.append(fid)
        valid_c2w.append(T_world_from_cam)
        valid_obj_world.append(T_world_from_obj)
        valid_cam_from_obj.append(T_cam_from_obj)

    print(f"  Computed {len(results)} transforms, skipped {skipped} "
          f"(outside SfM range)")

    # ── Three-stage detection ─────────────────────────────────────────────────
    print("\n── Three-stage analysis ────────────────────────────────────────")
    print("  (stage1 stationary → transition rotating → stage2 stationary)")
    obj_trans = [T[:3, 3] for T in valid_obj_world]
    obj_rots  = [T[:3, :3] for T in valid_obj_world]

    # Filter tracker failures first; stage detection runs on clean frames only.
    # onset_idx caps stage2 so outlier frames never reach downstream steps.
    inlier_mask, onset_idx = filter_tracking_failures(valid_fids, obj_trans)
    fids_clean  = [f for f, ok in zip(valid_fids,       inlier_mask) if ok]
    trans_clean = [t for t, ok in zip(obj_trans,        inlier_mask) if ok]
    rots_clean  = [r for r, ok in zip(obj_rots,         inlier_mask) if ok]

    split1_idx, split2_idx, angular_jumps_clean = detect_three_stages(
        fids_clean, trans_clean, rots_clean, smooth_kernel=args.smooth)

    # Map split frame IDs back to indices in the full valid_fids list
    fids_arr = np.array(valid_fids)
    split1_full = int(np.argmin(np.abs(fids_arr - fids_clean[split1_idx])))
    split2_full = int(np.argmin(np.abs(fids_arr - fids_clean[split2_idx])))
    # Cap stage2 end at the tracker failure onset (excludes bad frames from output)
    stage2_end = onset_idx if onset_idx is not None else len(valid_fids)

    # angular_jumps for full frame list (for plotting)
    angular_jumps = np.zeros(len(valid_fids))
    for i in range(1, len(valid_fids)):
        angular_jumps[i] = np.rad2deg(
            rotation_angle_between(obj_rots[i - 1], obj_rots[i]))

    stage1_fids = valid_fids[:split1_full]
    trans_fids  = valid_fids[split1_full:split2_full]
    stage2_fids = valid_fids[split2_full:stage2_end]

    print(f"  Stage 1 (stationary):  frames {stage1_fids[0]} – {stage1_fids[-1]}"
          f"  ({len(stage1_fids)} frames)")
    print(f"  Transition (rotating): frames {trans_fids[0]} – {trans_fids[-1]}"
          f"  ({len(trans_fids)} frames)")
    print(f"  Stage 2 (stationary):  frames {stage2_fids[0]} – {stage2_fids[-1]}"
          f"  ({len(stage2_fids)} frames)")

    # Representative object pose per stationary stage
    T_obj_stage1 = representative_pose(valid_obj_world[:split1_full])
    T_obj_stage2 = representative_pose(valid_obj_world[split2_full:stage2_end])

    # Representative camera-to-world pose per stationary stage
    T_cam_stage1 = representative_pose(valid_c2w[:split1_full])
    T_cam_stage2 = representative_pose(valid_c2w[split2_full:stage2_end])

    # Transform from stage-1 camera frame to stage-2 camera frame
    # (used by downstream merge pipeline to align cam_in_ob poses)
    T_pose2_from_pose1 = np.linalg.inv(T_cam_stage2) @ T_cam_stage1
    R_delta   = T_pose2_from_pose1[:3, :3]
    t_delta   = T_pose2_from_pose1[:3,  3]
    angle_deg = np.rad2deg(rotation_angle_between(np.eye(3), R_delta))

    print(f"\n  T_pose2_from_pose1 (stage1 cam frame → stage2 cam frame):")
    for row in T_pose2_from_pose1:
        print(f"    [{', '.join(f'{v:+.6f}' for v in row)}]")
    print(f"  Rotation angle: {angle_deg:.2f}°")
    print(f"  Translation:    [{t_delta[0]:+.4f}, {t_delta[1]:+.4f}, {t_delta[2]:+.4f}] m")

    pos1 = T_obj_stage1[:3, 3]
    pos2 = T_obj_stage2[:3, 3]
    print(f"\n  Stage-1 object position (world): [{pos1[0]:+.4f}, {pos1[1]:+.4f}, {pos1[2]:+.4f}] m")
    print(f"  Stage-2 object position (world): [{pos2[0]:+.4f}, {pos2[1]:+.4f}, {pos2[2]:+.4f}] m")

    # ── Per-stage stability (std of translation and rotation) ─────────────────
    def stage_std(obj_world_slice):
        ts = np.array([T[:3, 3] for T in obj_world_slice])
        Rs = [T[:3, :3] for T in obj_world_slice]
        R_mean = Rotation.from_matrix(Rs).mean().as_matrix()
        angles = [rotation_angle_between(R_mean, R) for R in Rs]
        t_std = ts.std(axis=0) * 100   # metres → cm
        return t_std, np.std(angles)

    t_std1, r_std1 = stage_std(valid_obj_world[:split1_full])
    t_std2, r_std2 = stage_std(valid_obj_world[split2_full:stage2_end])
    print(f"\n  Stability (std):")
    print(f"    Stage 1: translation X={t_std1[0]:.3f} Y={t_std1[1]:.3f} Z={t_std1[2]:.3f} cm,  rotation {r_std1:.4f}°")
    print(f"    Stage 2: translation X={t_std2[0]:.3f} Y={t_std2[1]:.3f} Z={t_std2[2]:.3f} cm,  rotation {r_std2:.4f}°")

    # ── Write JSON output ─────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        'frames': results,
        'stage_analysis': {
            'stage1': {
                'start_frame':      stage1_fids[0],
                'end_frame':        stage1_fids[-1],
                'num_frames':       len(stage1_fids),
                'T_world_from_obj': T_obj_stage1.tolist(),
                'T_world_from_cam': T_cam_stage1.tolist(),
            },
            'transition': {
                'start_frame': trans_fids[0],
                'end_frame':   trans_fids[-1],
                'num_frames':  len(trans_fids),
            },
            'stage2': {
                'start_frame':      stage2_fids[0],
                'end_frame':        stage2_fids[-1],
                'num_frames':       len(stage2_fids),
                'T_world_from_obj': T_obj_stage2.tolist(),
                'T_world_from_cam': T_cam_stage2.tolist(),
            },
            'T_pose2_from_pose1':     T_pose2_from_pose1.tolist(),
            'rotation_angle_degrees': float(angle_deg),
            'translation_m':          t_delta.tolist(),
        },
    }
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {output_path}")

    # ── Transform txt output ──────────────────────────────────────────────────
    if args.transform_output:
        transform_path = Path(args.transform_output)
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transform_path, 'w') as f:
            for row in T_pose2_from_pose1:
                f.write(' '.join(f'{v:.18e}' for v in row) + '\n')
        print(f"Transform saved to {transform_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        make_plots(valid_fids, valid_cam_from_obj, valid_obj_world,
                   angular_jumps, split1_full, split2_full, args.plot)


if __name__ == '__main__':
    main()
