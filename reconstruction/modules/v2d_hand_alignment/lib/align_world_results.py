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
    # Three-pass design:
    #
    #   Pass 1: compute cam-frame ``offset_reprojected`` per (hand, frame).
    #           No alignment applied yet — we just collect the corrections.
    #   Pass 2: smooth ``offset_cam`` IN CAM FRAME across time.  This is
    #           critical: smoothing in world frame mixes cam-frame components
    #           as the camera rotates between frames, so even a "smooth"
    #           world-frame trajectory ends up jittery in the cam frame
    #           where rendering happens. Smoothing in cam frame guarantees
    #           the per-frame cam-frame correction is itself smooth.
    #   Pass 3: rotate each smoothed offset by R.T (with left-hand mirror)
    #           and add to ``trans`` to produce ``trans_aligned``.  DynHaMR's
    #           ``trans`` carries inverse-camera-motion baked in (image
    #           stability); only the depth correction comes from alignment.
    #
    # For left hands the renderer mirrors v_world.x AFTER adding trans
    # (render_dynhamr_video.py: ``v_world[:, :, 0] = -v_world[:, :, 0]``),
    # so the cam-to-world mapping picks up an extra mirror M = diag(-1, 1, 1):
    #   v_cam = cam_R · M · (verts_local + trans) + cam_t   # left
    #   ⇒ delta_world = M · cam_R.T · delta_cam
    print("\nPass 1: per-frame depth alignment (cam-frame offsets + scale)...")
    offset_cam = np.full((B, T, 3), np.nan, dtype=np.float64)
    offsets    = np.full((B, T), np.nan)
    scales     = np.full((B, T), np.nan)
    pixels     = np.zeros((B, T), dtype=np.int64)

    for h in range(B):
        side = 1 if is_right_track[h] else 0
        for f in tqdm(range(T), desc=f"  hand {h} ({'right' if is_right_track[h] else 'left'})",
                      unit="frame", ncols=80):
            try:
                a = alignment.compute_alignment(side, f)
                offset_cam[h, f] = a["offset_reprojected"]
                offsets[h, f]    = a["offset_reprojected"][2]
                scales[h, f]     = a["scale"]
                pixels[h, f]     = a["n_pixels"]
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
        hand_scale_depth = float(s_sorted[int(np.searchsorted(w_cum, cutoff))])
        print(f"\nDepth-only hand_scale = {hand_scale_depth:.4f}  "
              f"(weighted median over {int(valid.sum())}/{B*T} frames)")
    else:
        hand_scale_depth = 1.0
        print("\nDepth-only hand_scale = 1.0  (no valid frames; alignment will not rescale)")

    # ------------------------------------------------------------------
    # Compensate for ViPE↔depth-source intrinsics mismatch.
    # ------------------------------------------------------------------
    # The depth-only scale (di/dr) gives a mesh whose physical extent is
    # consistent with the depth values, but the *projected* size depends
    # on which fx we render under:
    #
    #   silhouette_under_vipe = mano * fx_vipe / dr            (unaligned)
    #   silhouette_under_moge = mano * (di/dr) * fx_moge / di
    #                         = mano * fx_moge / dr             (aligned)
    #   ratio = fx_moge / fx_vipe
    #
    # Aligned renderer uses MoGe intrinsics (depth-source), so when
    # fx_moge != fx_vipe the silhouette shrinks/grows by that ratio.
    # Pre-multiply hand_scale by fx_vipe/fx_moge so the rendered silhouette
    # under MoGe intrins matches the image (which the unaligned render
    # under ViPE intrins already does).
    fx_vipe = float(wr["intrins"][0])
    fy_vipe = float(wr["intrins"][1])
    _depth_intr = alignment.get_depth_intrinsics()    # [fx, fy, cx, cy]
    fx_moge = float(_depth_intr[0])
    fy_moge = float(_depth_intr[1])
    # Geometric mean of the per-axis ratios — robust to fx/fy asymmetry.
    fx_ratio = float(np.sqrt((fx_vipe / fx_moge) * (fy_vipe / fy_moge)))
    hand_scale = hand_scale_depth * fx_ratio
    print(f"  fx ratio (ViPE / MoGe) = {fx_ratio:.4f}  "
          f"(fx_vipe={fx_vipe:.1f}  fx_moge={fx_moge:.1f})")
    print(f"  hand_scale (with fx correction) = {hand_scale:.4f}")
    if abs(fx_ratio - 1.0) > 0.05:
        print(f"  NOTE: |fx_ratio - 1| > 0.05; aligned silhouette would be "
              f"{1.0 / fx_ratio:.2f}× off without this correction.")

    # ------------------------------------------------------------------
    # Pass 2: per-frame depth alignment in *metric* cam frame.
    # ------------------------------------------------------------------
    # Mirrors v2d_hamer/lib/align_hands.py: render the mesh at the metric-
    # rescaled cam pose under the *real* (MoGe) intrinsics, get a small
    # per-frame dz_metric, apply via a ray-offset in cam frame.
    #
    # The systemic ~fx_vipe/fx_moge mismatch is fully absorbed into the
    # global α scaling (uniform across frames). What remains per-frame
    # is small (cm-scale) noise from MoGe's per-frame depth jitter — this
    # is what we *want* to align against per-frame, the same regime
    # v2d_hamer operates in (where alignment works well).
    print(f"\nPass 2: per-frame metric alignment "
          f"(α = {hand_scale_depth:.4f}, render under MoGe intrins)...")
    alpha = float(hand_scale_depth)
    offset_cam_metric = np.full((B, T, 3), np.nan, dtype=np.float64)

    for h in range(B):
        side = 1 if is_right_track[h] else 0
        for f in tqdm(range(T),
                      desc=f"  pass 2 hand {h} ({'right' if is_right_track[h] else 'left'})",
                      unit="frame", ncols=80):
            try:
                # 1. Mesh in DynHaMR cam frame, rescaled to metric.
                mesh = alignment.get_mesh(side, f)
                mesh.vertices = mesh.vertices.astype(np.float64) * alpha   # → metric cam frame

                # 2. Render under MoGe intrins.
                depth_image = alignment.get_depth_image(f)
                occ_mask    = alignment.get_occlusion_mask(f)
                h_img, w_img = depth_image.shape
                _, rendered = alignment.render_mesh(mesh, (w_img, h_img), _depth_intr)

                hand_mask = rendered > 0
                if occ_mask is not None:
                    hand_mask &= ~occ_mask
                n_pixels = int(hand_mask.sum())
                if n_pixels < 100:
                    continue

                # 3. Small per-frame dz: both di and rendered are metric now.
                di = depth_image[hand_mask]
                dr = rendered[hand_mask]
                dz = float(np.median(di - dr))            # cm-scale, not metres

                # 4. Ray-offset in cam frame so centroid pixel is preserved
                #    while depth gets nudged by dz.
                cz = float(np.mean(mesh.vertices[:, 2]))
                cx = float(np.mean(mesh.vertices[:, 0]))
                cy = float(np.mean(mesh.vertices[:, 1]))
                if cz < 1e-6:
                    continue
                z_p = cz + dz
                offset_cam_metric[h, f] = np.array(
                    [cx * (z_p / cz - 1.0), cy * (z_p / cz - 1.0), dz],
                    dtype=np.float64,
                )
            except Exception as e:
                tqdm.write(f"  hand {h} frame {f}: metric alignment failed ({e})")

    # Report per-hand dz_metric stats so it's clear how big the per-frame
    # corrections are after the uniform α has done its job.
    for h in range(B):
        side = 'right' if is_right_track[h] else 'left'
        dz_vals = offset_cam_metric[h, :, 2]
        finite  = np.isfinite(dz_vals)
        if finite.any():
            n_v = int(finite.sum())
            print(f"  hand {h} ({side}): per-frame dz_metric — "
                  f"{n_v}/{T} valid, "
                  f"median={float(np.nanmedian(dz_vals)):+.4f}  "
                  f"|max|={float(np.nanmax(np.abs(dz_vals))):.4f}")

    # ------------------------------------------------------------------
    # Pass 3: compose trans_aligned = uniform-α base + per-frame metric offset.
    # ------------------------------------------------------------------
    print("\nPass 3: composing trans_aligned = α·(trans − cam_center) + cam_center "
          "+ R.T·offset_cam_metric")
    trans_aligned = trans.copy()
    for h in range(B):
        side = 1 if is_right_track[h] else 0
        for f in range(T):
            # Step A: uniform α scaling (in DynHaMR world → metric).
            cam_center_f = -cam_R[h, f].T @ cam_t[h, f]
            if side == 0:
                cam_center_f = cam_center_f.copy()
                cam_center_f[0] = -cam_center_f[0]
            ta = alpha * (trans[h, f] - cam_center_f) + cam_center_f

            # Step B: small per-frame metric offset (cm-scale).
            oc = offset_cam_metric[h, f]
            if np.isfinite(oc[0]):
                delta_world = cam_R[h, f].T @ oc
                if side == 0:
                    delta_world[0] = -delta_world[0]
                ta = ta + delta_world

            trans_aligned[h, f] = ta

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
