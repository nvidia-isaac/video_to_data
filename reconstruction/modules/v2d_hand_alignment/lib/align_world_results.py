"""
Align DynHaMR world_results.npz to monocular depth and attach object poses.

Uses HandObjectAlignment.compute_offset per frame to find the camera-space
depth shift needed, back-projects it to world units, and stores the result as
a new 'trans_aligned' field alongside the original 'trans'.

Optionally loads per-frame FoundationPose object transforms (Transform3d JSON)
and stores them in both camera and world frame.

Output schema: all original world_results fields plus:
  trans_aligned     (B, T, 3)  depth-aligned translation, same world units as trans
  intrins_aligned   (4,)       [fx,fy,cx,cy] of the depth image used for alignment
  hand_scale        ()         constant uniform mesh scale that makes the
                               aligned hand silhouette match the image under
                               the depth intrinsics (n_pixels-weighted median
                               of per-frame depth_image / rendered_depth)
  object_pose_cam   (T, 4, 4)  object-to-camera SE(3) [present if --object_poses_dir]
  object_pose_world (T, 4, 4)  object-to-world SE(3)  [present if --object_poses_dir]

Usage:
    python -m v2d.hand_alignment.lib.align_world_results \\
        --input_hand_data   data/.../world_results.npz \\
        --depth_dir         data/.../depth_moge \\
        --depth_intrinsics  data/.../intrinsics_moge_stable.json \\
        --mano_model_dir    data/weights/hand \\
        --output_hand_data  data/.../world_results_aligned.npz \\
        --object_masks_dir  data/.../masks/1 \\
        --object_poses_dir  data/.../poses_moge_smoothed \\
        --smooth_sigma      5.0
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from v2d.common.datatypes import Transform3d
from v2d.hand_alignment.lib.hand_object_alignment import HandObjectAlignment


def align_world_results(
    input_hand_data: str,
    depth_dir: str,
    depth_intrinsics_path: str,
    mano_model_dir: str,
    output_hand_data: str,
    object_masks_dir: str | None = None,
    object_poses_dir: str | None = None,
    smooth_sigma: float = 5.0,
) -> None:
    wr = np.load(input_hand_data, allow_pickle=True)

    trans       = wr['trans'].astype(np.float64)          # (B, T, 3)
    cam_R       = wr['cam_R'].astype(np.float64)          # (B, T, 3, 3)
    cam_t       = wr['cam_t'].astype(np.float64)          # (B, T, 3)
    is_right    = wr['is_right']                           # (B, T)
    world_scale = float(wr['world_scale'].flat[0]) if 'world_scale' in wr else 1.0

    B, T = trans.shape[:2]
    is_right_track = is_right.mean(axis=1) > 0.5          # (B,) — per-track majority vote

    print(f"world_scale: {world_scale:.4f}  hands: {B}  frames: {T}")

    # ------------------------------------------------------------------
    # Build alignment helper (pre-computes MANO FK for all frames)
    # ------------------------------------------------------------------
    alignment = HandObjectAlignment(
        pose_data_path        = input_hand_data,
        depth_folder          = depth_dir,
        depth_intrinsics_path = depth_intrinsics_path,
        mano_assets_root      = mano_model_dir,
        occlusion_mask_folder = object_masks_dir,
    )

    # ------------------------------------------------------------------
    # Per-hand per-frame depth offset → trans_aligned + per-frame scales
    # ------------------------------------------------------------------
    # One render pass per (hand, frame) yields both the offset (used for
    # trans_aligned) and the per-frame depth-vs-render scale (aggregated into
    # a single global hand_scale below).
    print("\nComputing per-frame depth alignment (offset + scale)...")
    trans_aligned = trans.copy()
    offsets       = np.full((B, T), np.nan)
    scales        = np.full((B, T), np.nan)
    pixels        = np.zeros((B, T), dtype=np.int64)

    for h in range(B):
        side = 1 if is_right_track[h] else 0
        # The renderer mirrors v_world.x AFTER adding trans for the left hand
        # (render_dynhamr_video.py: `v_world[:, :, 0] = -v_world[:, :, 0]`),
        # so for left hands we must mirror the world-space delta to keep the
        # cam-space shift equal to delta_cam.  Derivation:
        #   v_cam = cam_R · M · (verts_local + trans) + cam_t      # left
        #   ⇒ delta_world = M · cam_R.T · delta_cam,   M = diag(-1, 1, 1)
        for f in tqdm(range(T), desc=f"  hand {h} ({'right' if is_right_track[h] else 'left'})",
                      unit="frame", ncols=80):
            try:
                a = alignment.compute_alignment(side, f)
                delta_world = cam_R[h, f].T @ a["offset_reprojected"]
                if side == 0:
                    delta_world[0] = -delta_world[0]
                trans_aligned[h, f] += delta_world
                offsets[h, f] = a["offset_reprojected"][2]
                scales[h, f]  = a["scale"]
                pixels[h, f]  = a["n_pixels"]
            except Exception as e:
                tqdm.write(f"  hand {h} frame {f}: alignment failed ({e})")

    for h in range(B):
        side = 'right' if is_right_track[h] else 'left'
        n_valid = int(np.sum(np.isfinite(offsets[h])))
        med_dz  = float(np.nanmedian(offsets[h])) if n_valid > 0 else float('nan')
        med_s   = float(np.nanmedian(scales[h]))  if n_valid > 0 else float('nan')
        print(f"  hand {h} ({side}): {n_valid}/{T} valid  median dz={med_dz:.4f}  median scale={med_s:.4f}")

    # ------------------------------------------------------------------
    # Aggregate per-frame scales → single global hand_scale (n_pixels-weighted)
    # ------------------------------------------------------------------
    # Reject frames with very small masks (occluded / clipped) or absurd
    # scale values (a fully-occluded hand can yield wild ratios).
    valid = (pixels >= 256) & np.isfinite(scales) & (scales > 0.2) & (scales < 5.0)
    if valid.any():
        s_vals = scales[valid].astype(np.float64)
        w_vals = pixels[valid].astype(np.float64)
        order  = np.argsort(s_vals)
        s_sorted = s_vals[order]
        w_cum    = np.cumsum(w_vals[order])
        cutoff   = w_cum[-1] / 2.0
        hand_scale = float(s_sorted[int(np.searchsorted(w_cum, cutoff))])
        print(f"\nGlobal hand_scale = {hand_scale:.4f}  "
              f"(weighted median over {int(valid.sum())}/{B*T} frames)")
    else:
        hand_scale = 1.0
        print("\nGlobal hand_scale = 1.0  (no valid frames; alignment will not rescale)")

    # ------------------------------------------------------------------
    # Optional temporal smoothing of trans_aligned
    # ------------------------------------------------------------------
    if smooth_sigma > 0.0:
        print(f"\nSmoothing trans_aligned with sigma={smooth_sigma} frames...")
        frames = np.arange(T)
        for h in range(B):
            valid = np.isfinite(offsets[h])
            if valid.sum() < 2:
                print(f"  hand {h}: too few valid frames, skipping smoothing")
                continue
            filled = trans_aligned[h].copy()
            for dim in range(3):
                filled[:, dim] = np.interp(frames, frames[valid], trans_aligned[h, valid, dim])
            smoothed = np.stack([
                gaussian_filter1d(filled[:, dim], sigma=smooth_sigma) for dim in range(3)
            ], axis=-1)
            shift = np.linalg.norm(smoothed[valid] - trans_aligned[h, valid], axis=-1)
            print(f"  hand {h}: max_shift={shift.max():.4f}  mean_shift={shift.mean():.4f}")
            trans_aligned[h] = smoothed

    # ------------------------------------------------------------------
    # Object poses in camera and world frame
    # ------------------------------------------------------------------
    out = {k: wr[k] for k in wr.files}
    out['trans_aligned'] = trans_aligned.astype(np.float32)
    out['hand_scale']    = np.float32(hand_scale)

    # Store the depth intrinsics used for alignment so renderers can use them
    depth_intr = alignment.get_depth_intrinsics()   # [fx, fy, cx, cy]
    out['intrins_aligned'] = depth_intr.astype(np.float64)

    if object_poses_dir is not None:
        pose_files = sorted(glob.glob(os.path.join(object_poses_dir, '*.json')))
        if not pose_files:
            print(f"Warning: no JSON files found in {object_poses_dir}")
        else:
            print(f"\nLoading {len(pose_files)} object poses from {object_poses_dir}...")
            T_cam_all   = []
            T_world_all = []
            for f in range(T):
                path = os.path.join(object_poses_dir, f"{f:06d}.json")
                if not os.path.exists(path):
                    T_cam_all.append(np.eye(4))
                    T_world_all.append(np.eye(4))
                    continue

                T_cam = Transform3d.load(path).to_matrix()   # (4, 4) object→camera

                # cam→world from hand 0's extrinsics (same camera for all hands)
                R = cam_R[0, f]
                t = cam_t[0, f]
                cam_to_world       = np.eye(4)
                cam_to_world[:3, :3] = R.T
                cam_to_world[:3, 3]  = -(R.T @ t)

                T_world = cam_to_world @ T_cam               # (4, 4) object→world

                T_cam_all.append(T_cam)
                T_world_all.append(T_world)

            out['object_pose_cam']   = np.stack(T_cam_all).astype(np.float32)    # (T, 4, 4)
            out['object_pose_world'] = np.stack(T_world_all).astype(np.float32)  # (T, 4, 4)
            print(f"  Added object_pose_cam and object_pose_world  shape: {out['object_pose_cam'].shape}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_hand_data)), exist_ok=True)
    np.savez_compressed(output_hand_data, **out)
    print(f"\nSaved → {output_hand_data}")
    print(f"  trans_aligned: depth-aligned world units"
          + (f", smoothed (sigma={smooth_sigma})" if smooth_sigma > 0 else ""))
    print(f"  hand_scale:    {hand_scale:.4f}")
    if 'object_pose_cam' in out:
        print(f"  object_pose_cam / object_pose_world: {out['object_pose_cam'].shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Align DynHaMR world_results.npz to monocular depth.'
    )
    parser.add_argument('--input_hand_data',  required=True,
                        help='DynHaMR world_results.npz')
    parser.add_argument('--depth_dir',        required=True,
                        help='Folder of depth PNGs (000000.png, …)')
    parser.add_argument('--depth_intrinsics', required=True,
                        help='Depth intrinsics JSON {fx,fy,cx,cy,width,height}')
    parser.add_argument('--mano_model_dir',   required=True,
                        help='Dir containing MANO_RIGHT.pkl (or models/ subdir)')
    parser.add_argument('--output_hand_data', required=True,
                        help='Output path for world_results_aligned.npz')
    parser.add_argument('--object_masks_dir', default=None,
                        help='Per-frame object mask PNGs (SAM2); excluded from depth comparison')
    parser.add_argument('--object_poses_dir', default=None,
                        help='Per-frame FoundationPose Transform3d JSON files; '
                             'written to output as object_pose_cam / object_pose_world')
    parser.add_argument('--smooth_sigma',     type=float, default=5.0,
                        help='Gaussian sigma (frames) for smoothing trans_aligned. '
                             '0 = disable. (default: 5.0)')
    args = parser.parse_args()

    align_world_results(
        input_hand_data       = args.input_hand_data,
        depth_dir             = args.depth_dir,
        depth_intrinsics_path = args.depth_intrinsics,
        mano_model_dir        = args.mano_model_dir,
        output_hand_data      = args.output_hand_data,
        object_masks_dir      = args.object_masks_dir,
        object_poses_dir      = args.object_poses_dir,
        smooth_sigma          = args.smooth_sigma,
    )


if __name__ == '__main__':
    main()
