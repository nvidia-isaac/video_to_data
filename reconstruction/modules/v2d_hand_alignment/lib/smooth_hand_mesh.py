"""
Smooth hand mesh center position temporally.

For each hand track, the centroid (mean vertex position) is smoothed with a
Gaussian filter. Per-frame finger shapes are preserved by applying only the
smoothed centroid delta to all vertices.

Algorithm per hand:
  1. Compute centroid[f] = mean(verts[h, f]) over visible frames
  2. Gaussian-smooth centroid over time → centroid_smooth[f]
  3. verts_out[h, f] = verts[h, f] + (centroid_smooth[f] - centroid[f])

Invisible frames are gap-filled by linear interpolation before smoothing,
then restored to their original values so only visible frames are affected.

Usage:
    python -m v2d.hand_alignment.lib.smooth_hand_mesh \\
        --input  data/.../hand_mesh_traj_000300_moge_aligned_perhand.npz \\
        --output data/.../hand_mesh_traj_000300_moge_aligned_perhand_smooth.npz \\
        --sigma  5.0
"""

import argparse

import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_hand_mesh(
    input_path: str,
    output_path: str,
    sigma: float = 5.0,
) -> None:
    data = np.load(input_path, allow_pickle=True)
    out  = dict(data)

    verts    = data['verts'].copy().astype(np.float64)   # (n_hands, n_frames, n_verts,  3)
    joints   = data['joints'].copy().astype(np.float64)  # (n_hands, n_frames, n_joints, 3)
    vis_mask = data.get('vis_mask', np.ones(verts.shape[:2]))  # (n_hands, n_frames)

    n_hands, n_frames = verts.shape[:2]

    for h in range(n_hands):
        visible = vis_mask[h].astype(bool)   # (n_frames,)
        if visible.sum() < 2:
            print(f"  hand {h}: too few visible frames ({visible.sum()}), skipping")
            continue

        # Centroid per frame: (n_frames, 3)
        centroid = verts[h].mean(axis=1)     # mean over vertices

        # Gap-fill invisible frames via linear interpolation so the Gaussian
        # filter doesn't see hard zeros at the edges.
        frames = np.arange(n_frames)
        centroid_filled = centroid.copy()
        for dim in range(3):
            centroid_filled[:, dim] = np.interp(
                frames,
                frames[visible],
                centroid[visible, dim],
            )

        # Gaussian smooth (each spatial dim independently)
        centroid_smooth = np.stack([
            gaussian_filter1d(centroid_filled[:, dim], sigma=sigma)
            for dim in range(3)
        ], axis=-1)   # (n_frames, 3)

        # Delta: only apply to visible frames
        delta = np.zeros((n_frames, 3))
        delta[visible] = (centroid_smooth - centroid)[visible]   # (n_frames, 3)

        # Shift verts and joints by delta (broadcast over vertices/joints)
        verts[h]  += delta[:, None, :]   # (n_frames, 1, 3) broadcast
        joints[h] += delta[:, None, :]

        applied = np.linalg.norm(delta[visible], axis=-1)
        print(f"  hand {h}: sigma={sigma:.1f}  "
              f"max_shift={applied.max():.4f}m  mean_shift={applied.mean():.4f}m")

    out['verts']  = verts.astype(np.float32)
    out['joints'] = joints.astype(np.float32)
    np.savez(output_path, **out)
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Temporally smooth hand mesh center position")
    parser.add_argument('--input_path',  required=True, help='Input NPZ hand mesh')
    parser.add_argument('--output_path', required=True, help='Output NPZ path')
    parser.add_argument('--sigma',  type=float, default=5.0,
                        help='Gaussian sigma in frames (default: 5.0)')
    args = parser.parse_args()

    print(f"Smoothing hand centroid with sigma={args.sigma} frames...")
    smooth_hand_mesh(args.input_path, args.output_path, sigma=args.sigma)


if __name__ == '__main__':
    main()
