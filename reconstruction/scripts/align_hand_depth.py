"""
Align hand mesh depth to MoGe depth by estimating a z-scale factor.

For each frame with visible hands:
  1. Rasterize the hand mesh into a z-buffer (rendered hand depth)
  2. At pixels covered by the hand mesh, sample the MoGe depth
  3. Compute ratio: moge_z / hand_z  →  robust median gives the scale

A global scale is estimated (median over all frames × hands) and applied to
all vert/joint z values.  Use --per_frame to apply a per-frame scale instead.

Usage:
    python scripts/align_hand_depth.py \\
        --input   data/.../hand/hand_mesh/airplane_hand_mesh_traj_000300_aligned.npz \\
        --depth   data/.../outputs/depth_moge \\
        --intrinsics data/.../outputs/intrinsics_moge_stable.json \\
        --output  data/.../hand/hand_mesh/airplane_hand_mesh_traj_000300_depth_aligned.npz

    # Diagnostic: save per-frame overlay images
    python scripts/align_hand_depth.py ... --diag_dir /tmp/hand_depth_diag

    # Per-frame scale instead of global:
    python scripts/align_hand_depth.py ... --per_frame

    # Per-frame per-hand scale (each hand aligned independently each frame):
    python scripts/align_hand_depth.py ... --per_hand
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Depth utilities
# ---------------------------------------------------------------------------

def load_moge_depth(path: str) -> np.ndarray:
    """Load uint16/int32 depth PNG → float64 metres.
    Encoding: pixel = 65535 * 1/(depth_m + 1)  →  depth_m = 65535/pixel - 1
    """
    from PIL import Image
    px = np.array(Image.open(path)).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(px > 0, 65535.0 / px - 1.0, np.nan)
    return depth


def load_intrinsics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Z-buffer rasterizer
# ---------------------------------------------------------------------------

def render_zbuffer(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """
    Rasterize hand mesh triangles into a z-buffer.

    Args:
        verts_cam: (N, 3) vertices in camera space (metres)
        faces:     (F, 3) integer face indices
        width/height: image dimensions
        fx/fy/cx/cy: camera intrinsics

    Returns:
        (H, W) float64 array — depth in metres, nan where no mesh.
    """
    z_buf = np.full((height, width), np.inf)

    z = verts_cam[:, 2].astype(np.float64)
    safe_z = np.where(z > 1e-4, z, 1e-4)
    pu = (verts_cam[:, 0] * fx / safe_z + cx).astype(np.float64)
    pv = (verts_cam[:, 1] * fy / safe_z + cy).astype(np.float64)

    for tri in faces:
        ia, ib, ic = int(tri[0]), int(tri[1]), int(tri[2])
        za, zb, zc = z[ia], z[ib], z[ic]
        if za <= 0 or zb <= 0 or zc <= 0:
            continue

        ax, ay = pu[ia], pv[ia]
        bx, by = pu[ib], pv[ib]
        dx, dy = pu[ic], pv[ic]   # 'd' to avoid shadowing 'cx'

        x0 = max(0,      int(np.floor(min(ax, bx, dx))))
        x1 = min(width,  int(np.ceil( max(ax, bx, dx))) + 1)
        y0 = max(0,      int(np.floor(min(ay, by, dy))))
        y1 = min(height, int(np.ceil( max(ay, by, dy))) + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        # Pixel centres
        xs = np.arange(x0, x1) + 0.5
        ys = np.arange(y0, y1) + 0.5
        gx, gy = np.meshgrid(xs, ys)  # (rows, cols)

        # Barycentric weights w.r.t. A=a, B=b, C=d
        v0x, v0y = bx - ax, by - ay   # edge A→B
        v1x, v1y = dx - ax, dy - ay   # edge A→C
        v2x = gx - ax
        v2y = gy - ay

        d00 = v0x*v0x + v0y*v0y
        d01 = v0x*v1x + v0y*v1y
        d11 = v1x*v1x + v1y*v1y
        d20 = v2x*v0x + v2y*v0y
        d21 = v2x*v1x + v2y*v1y

        denom = d00*d11 - d01*d01
        if abs(denom) < 1e-10:
            continue

        wb = (d11*d20 - d01*d21) / denom   # weight for B
        wc = (d00*d21 - d01*d20) / denom   # weight for C
        wa = 1.0 - wb - wc

        inside  = (wa >= 0) & (wb >= 0) & (wc >= 0)
        zinterp = wa*za + wb*zb + wc*zc

        patch  = z_buf[y0:y1, x0:x1]
        update = inside & (zinterp < patch)
        patch[update] = zinterp[update]
        z_buf[y0:y1, x0:x1] = patch

    return np.where(np.isfinite(z_buf), z_buf, np.nan)


# ---------------------------------------------------------------------------
# Alignment estimation
# ---------------------------------------------------------------------------

def estimate_scale(hand_depth: np.ndarray, moge_depth: np.ndarray) -> tuple[float | None, int]:
    """
    Estimate scale s such that  s * hand_depth ≈ moge_depth.
    Returns (median_ratio, n_inliers).
    """
    mask = (
        np.isfinite(hand_depth) & (hand_depth > 0.01) &
        np.isfinite(moge_depth) & (moge_depth > 0.01)
    )
    n = int(mask.sum())
    if n < 20:
        return None, n

    ratios = moge_depth[mask] / hand_depth[mask]

    # RANSAC-lite: keep ratios within 2× of the median
    med = np.median(ratios)
    inliers = np.abs(ratios - med) < 0.5 * med
    if inliers.sum() < 10:
        return float(med), n

    return float(np.median(ratios[inliers])), int(inliers.sum())


def estimate_offset(hand_depth: np.ndarray, moge_depth: np.ndarray) -> tuple[float | None, int]:
    """
    Estimate offset d such that  hand_depth + d ≈ moge_depth.
    Returns (median_diff, n_inliers).
    """
    mask = (
        np.isfinite(hand_depth) & (hand_depth > 0.01) &
        np.isfinite(moge_depth) & (moge_depth > 0.01)
    )
    n = int(mask.sum())
    if n < 20:
        return None, n

    diffs = moge_depth[mask] - hand_depth[mask]

    # RANSAC-lite: keep diffs within 0.5m of the median
    med = np.median(diffs)
    inliers = np.abs(diffs - med) < 0.5
    if inliers.sum() < 10:
        return float(med), n

    return float(np.median(diffs[inliers])), int(inliers.sum())


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def save_diag(
    frame_id: int,
    frames_folder: str | None,
    hand_depth: np.ndarray,
    moge_depth: np.ndarray,
    scale_before: float,
    scale_after: float,
    diag_dir: str,
):
    """Save side-by-side depth comparison image."""
    from PIL import Image, ImageDraw

    def depth_to_rgb(d: np.ndarray) -> np.ndarray:
        valid = np.isfinite(d) & (d > 0)
        out = np.zeros((*d.shape, 3), dtype=np.uint8)
        if valid.any():
            lo, hi = d[valid].min(), d[valid].max()
            norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            t = (norm * 255).astype(np.uint8)
            out[..., 0] = np.where(valid, t, 0)
            out[..., 2] = np.where(valid, 255 - t, 0)
        return out

    hand_rgb = depth_to_rgb(hand_depth)
    hand_sc  = depth_to_rgb(hand_depth * scale_after)
    moge_rgb = depth_to_rgb(moge_depth)

    # Mask overlay: where both are valid
    mask = np.isfinite(hand_depth) & (hand_depth > 0) & np.isfinite(moge_depth) & (moge_depth > 0)
    diff_before = np.abs(hand_depth * scale_before - moge_depth)
    diff_after  = np.abs(hand_depth * scale_after  - moge_depth)

    H, W = hand_depth.shape
    panel_w = W * 3
    canvas = np.zeros((H, panel_w, 3), dtype=np.uint8)
    canvas[:, :W]     = moge_rgb
    canvas[:, W:2*W]  = hand_rgb
    canvas[:, 2*W:]   = hand_sc

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((5,   5), f"MoGe depth", fill=(255,255,255))
    draw.text((W+5, 5), f"hand depth (raw,  scale={scale_before:.3f})", fill=(255,255,255))
    draw.text((2*W+5, 5), f"hand depth (×{scale_after:.3f})", fill=(255,255,255))

    if mask.any():
        me_before = float(np.median(diff_before[mask]))
        me_after  = float(np.median(diff_after[mask]))
        draw.text((5, 20), f"median |err| before={me_before:.4f}m  after={me_after:.4f}m", fill=(255,255,0))

    os.makedirs(diag_dir, exist_ok=True)
    img.save(os.path.join(diag_dir, f"depth_align_{frame_id:06d}.jpg"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Align hand mesh depth to MoGe depth")
    parser.add_argument('--input',           required=True, help='Aligned hand mesh NPZ (camera-space verts)')
    parser.add_argument('--depth',           required=True, help='Folder of MoGe depth PNGs (000000.png, ...)')
    parser.add_argument('--intrinsics',      required=True, help='Target intrinsics JSON (depth image space)')
    parser.add_argument('--mesh_intrinsics', default=None,
                        help='Intrinsics the hand mesh was aligned with (for rasterization). '
                             'Defaults to --intrinsics. Use when hand mesh was not reprojected to target.')
    parser.add_argument('--output',          required=True, help='Output NPZ path')
    parser.add_argument('--per_frame',  action='store_true',
                        help='Apply per-frame scale instead of a single global scale')
    parser.add_argument('--per_hand',   action='store_true',
                        help='Apply per-frame per-hand scale (each hand aligned independently each frame)')
    parser.add_argument('--align', choices=['scale', 'offset'], default='scale',
                        help='Alignment mode: "scale" (multiplicative, s*z≈z_moge) or '
                             '"offset" (additive, z+d≈z_moge, rigid ray translation). Default: scale')
    parser.add_argument('--smooth_sigma', type=float, default=0.0,
                        help='Gaussian sigma (frames) for temporal smoothing of per-frame offsets/scales '
                             'before applying them. Smoothing fills gaps with the per-hand global median '
                             'first, then smooths, making it robust to frames with bad image alignment.')
    parser.add_argument('--diag_dir',   default=None,
                        help='If set, save depth comparison images here')
    parser.add_argument('--diag_frames', type=int, nargs='+', default=None,
                        help='Frame indices to save diagnostics for (default: 0, 25%%, 50%%, 75%%, 100%%)')
    args = parser.parse_args()

    intr = load_intrinsics(args.intrinsics)
    width  = intr.get('width',  776)
    height = intr.get('height', 1032)

    # mesh_intr: used for rasterization (pixel positions must match the depth image)
    # If the hand mesh was aligned to a different intrinsics than the depth image,
    # pass those here so projected pixels land correctly.
    mesh_intr = load_intrinsics(args.mesh_intrinsics) if args.mesh_intrinsics else intr
    fx, fy = mesh_intr['fx'], mesh_intr['fy']
    cx, cy = mesh_intr['cx'], mesh_intr['cy']
    if args.mesh_intrinsics:
        print(f"Rasterizing with mesh intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"Comparing against depth intrinsics: fx={intr['fx']:.2f} fy={intr['fy']:.2f}")

    mesh_data  = np.load(args.input, allow_pickle=True)
    verts      = mesh_data['verts'].copy()    # (n_hands, n_frames, n_verts, 3)
    joints     = mesh_data['joints'].copy()   # (n_hands, n_frames, n_joints, 3)
    faces_r    = mesh_data['faces_right']
    faces_l    = mesh_data['faces_left']
    is_right   = mesh_data['is_right']        # (n_hands, n_frames)
    vis_mask   = mesh_data.get('vis_mask', np.ones(is_right.shape))

    n_hands, n_frames = verts.shape[:2]

    depth_files = sorted(glob.glob(os.path.join(args.depth, '*.png')))
    if len(depth_files) != n_frames:
        print(f"Warning: {len(depth_files)} depth files but {n_frames} hand frames")

    diag_frames = args.diag_frames
    if diag_frames is None and args.diag_dir:
        diag_frames = sorted({0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1})

    use_offset = (args.align == 'offset')
    print(f"Alignment mode: {'offset (additive)' if use_offset else 'scale (multiplicative)'}")

    # ------------------------------------------------------------------
    # Pass 1: collect per-frame and per-hand alignment estimates
    # ------------------------------------------------------------------
    per_frame_scales = []        # list of (frame_id, value) — median across hands
    per_hand_frame_scales = {}   # (fid, h) -> value
    per_hand_all_scales = [[] for _ in range(n_hands)]  # h -> [values across frames]

    for fid in range(min(n_frames, len(depth_files))):
        moge_depth = load_moge_depth(depth_files[fid])
        frame_vals = []

        for h in range(n_hands):
            if vis_mask[h, fid] < 0.5:
                continue

            v_cam = verts[h, fid].astype(np.float64)   # (N, 3)
            faces = faces_r if is_right[h, fid] > 0.5 else faces_l

            zb = render_zbuffer(v_cam, faces, width, height, fx, fy, cx, cy)

            if use_offset:
                val, n_in = estimate_offset(zb, moge_depth)
                valid = val is not None and -5.0 < val < 5.0
            else:
                val, n_in = estimate_scale(zb, moge_depth)
                valid = val is not None and 0.1 < val < 10.0

            if valid:
                frame_vals.append(val)
                per_hand_frame_scales[(fid, h)] = val
                per_hand_all_scales[h].append(val)

                if args.diag_dir and diag_frames and fid in diag_frames:
                    diag_scale = (1.0 + val / zb[np.isfinite(zb)].mean()) if use_offset else val
                    save_diag(fid, None, zb, moge_depth, 1.0, diag_scale, args.diag_dir)

        if frame_vals:
            per_frame_scales.append((fid, float(np.median(frame_vals))))

        if fid % 50 == 0 and per_frame_scales:
            last = per_frame_scales[-1][1] if per_frame_scales else float('nan')
            label = 'offset' if use_offset else 'scale'
            print(f"  frame {fid:4d}  {label}={last:.4f}  ({len(per_frame_scales)} frames collected)")

    if not per_frame_scales:
        print("ERROR: no valid frames found — check input files and visibility mask")
        return

    all_scales = [s for _, s in per_frame_scales]
    global_scale = float(np.median(all_scales))
    print(f"\nPer-frame scale stats:  median={global_scale:.4f}  "
          f"std={np.std(all_scales):.4f}  "
          f"min={min(all_scales):.4f}  max={max(all_scales):.4f}  "
          f"n_frames={len(all_scales)}")

    # Per-hand global fallback (median of per-hand observations)
    per_hand_global = {}
    for h in range(n_hands):
        if per_hand_all_scales[h]:
            per_hand_global[h] = float(np.median(per_hand_all_scales[h]))
            print(f"  hand {h}: global median scale={per_hand_global[h]:.4f}  "
                  f"n_frames={len(per_hand_all_scales[h])}")

    # ------------------------------------------------------------------
    # Optional: temporally smooth per-frame per-hand estimates
    # Fill missing frames with per-hand global median, then Gaussian-smooth.
    # This makes alignment robust to frames with poor image alignment.
    # ------------------------------------------------------------------
    if args.smooth_sigma > 0 and args.per_hand:
        print(f"\nSmoothing per-frame per-hand offsets with sigma={args.smooth_sigma} frames...")
        for h in range(n_hands):
            fallback = per_hand_global.get(h, global_scale)
            raw = np.array([
                per_hand_frame_scales.get((fid, h), fallback)
                for fid in range(n_frames)
            ])
            smoothed = gaussian_filter1d(raw, sigma=args.smooth_sigma)
            for fid in range(n_frames):
                per_hand_frame_scales[(fid, h)] = float(smoothed[fid])
            print(f"  hand {h}: raw std={raw.std():.4f}  smoothed std={smoothed.std():.4f}")

    # ------------------------------------------------------------------
    # Pass 2: apply alignment
    #
    # Scale mode (multiplicative):
    #   verts *= s  — scales xyz uniformly, preserving pixel projections:
    #   u = fx*(x*s)/(z*s) + cx = fx*x/z + cx  (unchanged)
    #
    # Offset mode (additive):
    #   Translates the mesh rigidly along the centroid ray by d in z.
    #   delta = centroid * (d / centroid_z)  so centroid stays on its ray.
    #   u_centroid preserved; other vertices shift rigidly (correct for a
    #   true 3D translation of a rigid hand).
    # ------------------------------------------------------------------
    def _apply(v: np.ndarray, j: np.ndarray, val: float) -> None:
        """Apply alignment value to verts/joints arrays (both shape (..., 3))."""
        if use_offset:
            centroid = v.mean(axis=0)          # (3,)
            z_c = centroid[2]
            if abs(z_c) < 1e-6:
                return
            delta = centroid * (val / z_c)     # translate along centroid ray
            v += delta
            j += delta
        else:
            v *= val
            j *= val

    if args.per_hand:
        mode = 'offset' if use_offset else 'scale'
        print(f"Applying per-frame per-hand {mode} (ray-preserving)...")
        for fid in range(n_frames):
            for h in range(n_hands):
                val = per_hand_frame_scales.get(
                    (fid, h),
                    per_hand_global.get(h, global_scale),
                )
                _apply(verts[h, fid], joints[h, fid], val)
    elif args.per_frame:
        mode = 'offset' if use_offset else 'scale'
        print(f"Applying per-frame {mode} (ray-preserving)...")
        scale_lookup = dict(per_frame_scales)
        for fid in range(n_frames):
            val = scale_lookup.get(fid, global_scale)
            for h in range(n_hands):
                _apply(verts[h, fid], joints[h, fid], val)
    else:
        mode = 'offset' if use_offset else 'scale'
        print(f"Applying global {mode} {global_scale:.4f} to all frames (ray-preserving)...")
        for fid in range(n_frames):
            for h in range(n_hands):
                _apply(verts[h, fid], joints[h, fid], global_scale)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = dict(mesh_data)
    out['verts']  = verts.astype(np.float32)
    out['joints'] = joints.astype(np.float32)
    np.savez(args.output, **out)
    print(f"Saved → {args.output}")

    # Sample vert before/after
    h, f = 0, 0
    before = mesh_data['verts'][h, f, 0]
    after  = verts[h, f, 0]
    print(f"\nSample vert [hand=0, frame=0, vert=0]:")
    print(f"  before: x={before[0]:.4f}  y={before[1]:.4f}  z={before[2]:.4f}m")
    print(f"  after:  x={after[0]:.4f}  y={after[1]:.4f}  z={after[2]:.4f}m  (×{global_scale:.4f})")


if __name__ == '__main__':
    main()
