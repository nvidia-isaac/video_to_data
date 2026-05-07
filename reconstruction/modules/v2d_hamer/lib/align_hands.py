"""Per-frame depth alignment of HaMeR predictions.

Takes raw HaMeR JSONs (virtual-pinhole, scaled_focal_length ≈ 25000) plus a
real depth source (depth PNG + intrinsics JSON) and writes per-frame JSONs
in real-camera units, with cam_t shifted to match the depth image along the
centroid ray.

For each (track, frame):
  1. Reconstruct MANO mesh from saved axis-angle (global_orient, hand_pose,
     betas) via manotorch. Mirror x for left hand.
  2. Rescale HaMeR's pred_cam_t_full to real intrinsics:
        cam_t_real = pred_cam_t_full · (fx_real / scaled_focal_length)
  3. Render mesh depth under real intrinsics with pyrender.
  4. Take mask = (rendered_depth > 0)
                  ∧ sam2_hand_mask  (if --hand_masks_dir provided)
                  ∧ ¬object_mask    (if --object_masks_dir provided).
     Intersecting with SAM2's per-track hand mask restricts the depth
     comparison to pixels that are *both* rendered-MANO and image-grounded
     hand, which is more reliable than the rendered silhouette alone.
  5. dz = median(depth_image[mask] − rendered_depth[mask]).
  6. Shift cam_t_real along the centroid ray by dz so the silhouette stays
     on the same image pixel but lands at depth_image's z.

Output schema (per file, single detection):
    {
      "track_id":   int,
      "is_right":   bool,
      "frame_idx":  int,
      "image_size": [W, H],
      "intrinsics": {"fx":..., "fy":..., "cx":..., "cy":...},
      "mano": {
        "betas":         [10 floats],
        "global_orient": [3 floats],
        "hand_pose":     [45 floats]
      },
      "cam_t":      [tx, ty, tz],   # aligned, in real intrinsics
      "diagnostics": {
        "dz":            float,    # depth shift applied
        "n_pixels":      int,      # hand-mask area used
        "scale":         float,    # median(depth_image / rendered_depth)
        "cam_t_pre_dz":  [tx,ty,tz],
        "scaled_focal":  float     # HaMeR virtual focal (pre-rescale)
      }
    }

Frames whose mask is below ``mask_min_pixels`` get no shift (cam_t_real saved
as-is, dz=0, scale=1) and ``n_pixels`` reports the actual count so downstream
code can treat them as low-confidence.

Usage:
    python -m v2d.hamer.lib.align_hands \\
        --hamer_dir       /data/hamer \\
        --depth_dir       /data/depth \\
        --intrinsics_path /data/intrinsics.json \\
        --mano_assets_root /data/weights/hamer/_DATA/data \\
        --output_dir      /data/hamer_aligned \\
        [--object_masks_dir /data/object_masks]   # optional occlusion exclusion
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Tuple

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import torch
import trimesh
from manotorch.manolayer import ManoLayer
from PIL import Image
from tqdm import tqdm


_CV_TO_GL_4X4 = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def _load_depth(path: str) -> np.ndarray:
    """uint16 PNG (px = 65535/(d+1)) → float32 metres."""
    px = np.asarray(Image.open(path)).astype(np.float32)
    return 1.0 / (px / 65535.0) - 1.0


def _load_intrinsics(path: str) -> Tuple[float, float, float, float, int, int]:
    with open(path) as f:
        d = json.load(f)
    return (
        float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]),
        int(d["width"]), int(d["height"]),
    )


def _make_mano(mano_assets_root: str) -> ManoLayer:
    return ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="right",
        center_idx=None,
        mano_assets_root=mano_assets_root,
    )


def _mesh_for_record(rec: dict, mano: ManoLayer) -> trimesh.Trimesh:
    pose_aa = np.concatenate([
        np.array(rec["mano"]["global_orient"], dtype=np.float32),
        np.array(rec["mano"]["hand_pose"],     dtype=np.float32),
    ])
    betas = np.array(rec["mano"]["betas"], dtype=np.float32)
    out = mano(torch.from_numpy(pose_aa)[None], torch.from_numpy(betas)[None])
    verts_local = out.verts[0].detach().numpy()        # (778, 3) — right-frame
    if not rec["is_right"]:
        verts_local[:, 0] *= -1
    faces = mano.th_faces.numpy()
    if not rec["is_right"]:
        faces = faces[:, [0, 2, 1]]                    # reverse winding
    return trimesh.Trimesh(verts_local, faces, process=False)


def _render_depth(
    renderer: pyrender.OffscreenRenderer,
    cam: pyrender.IntrinsicsCamera,
    mesh: trimesh.Trimesh,
    cam_t: np.ndarray,
) -> np.ndarray:
    """Render `mesh` translated by `cam_t` in CV cam space, return depth (H, W).

    The caller owns ``renderer`` and ``cam``: creating an ``OffscreenRenderer``
    spins up an EGL context (~100 ms+), so we share one across all frames.
    The Scene itself is cheap to rebuild.
    """
    posed = mesh.copy()
    posed.vertices = posed.vertices + cam_t[None, :]

    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(posed, smooth=False))
    scene.add(cam, pose=_CV_TO_GL_4X4)

    _, depth = renderer.render(scene)
    return depth.astype(np.float32)


def _ray_offset(centroid: np.ndarray, dz: float) -> np.ndarray:
    """[dx, dy, dz] that shifts a centroid along its camera ray by dz in z."""
    x, y, z = float(centroid[0]), float(centroid[1]), float(centroid[2])
    z_p = z + dz
    return np.array([x / z * z_p - x, y / z * z_p - y, dz], dtype=np.float64)


def align_hands(
    hamer_dir: str,
    depth_dir: str,
    intrinsics_path: str,
    mano_assets_root: str,
    output_dir: str,
    hand_masks_dir: str | None = None,
    object_masks_dir: str | None = None,
    mask_min_pixels: int = 256,
) -> None:
    fx, fy, cx, cy, W, H = _load_intrinsics(intrinsics_path)
    print(f"Real intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  {W}×{H}")

    mano = _make_mano(mano_assets_root)

    track_dirs = sorted(d for d in glob.glob(os.path.join(hamer_dir, "*")) if os.path.isdir(d))
    if not track_dirs:
        raise FileNotFoundError(f"No track subdirs in {hamer_dir}")

    # Pyrender renderer + camera are reused for every frame. Creating an
    # OffscreenRenderer spins up an EGL context (~100 ms+), so the per-frame
    # version of this loop was dominated by context init/teardown. Camera
    # intrinsics are constant, so the IntrinsicsCamera instance is shared too.
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                    znear=0.01, zfar=100.0)
    renderer = pyrender.OffscreenRenderer(W, H)
    try:
        _align_loop(
            track_dirs=track_dirs,
            output_dir=output_dir,
            depth_dir=depth_dir,
            hand_masks_dir=hand_masks_dir,
            object_masks_dir=object_masks_dir,
            mask_min_pixels=mask_min_pixels,
            mano=mano,
            renderer=renderer,
            cam=cam,
            intrinsics=(fx, fy, cx, cy),
            image_size=(W, H),
        )
    finally:
        renderer.delete()


def _align_loop(
    track_dirs: list[str],
    output_dir: str,
    depth_dir: str,
    hand_masks_dir: str | None,
    object_masks_dir: str | None,
    mask_min_pixels: int,
    mano: ManoLayer,
    renderer: pyrender.OffscreenRenderer,
    cam: pyrender.IntrinsicsCamera,
    intrinsics: tuple,        # (fx, fy, cx, cy)
    image_size: tuple,        # (W, H)
) -> None:
    fx, fy, cx, cy = intrinsics
    W, H = image_size

    for track_dir in track_dirs:
        oid = os.path.basename(track_dir)
        out_track = os.path.join(output_dir, oid)
        os.makedirs(out_track, exist_ok=True)
        files = sorted(glob.glob(os.path.join(track_dir, "*.json")))
        print(f"\nTrack {oid}: {len(files)} frames")

        for src in tqdm(files, desc=f"  track {oid}", ncols=80, unit="frame"):
            frame_idx = int(os.path.splitext(os.path.basename(src))[0])
            out_path = os.path.join(out_track, f"{frame_idx:06d}.json")
            if os.path.exists(out_path):
                continue

            with open(src) as f:
                rec = json.load(f)
            scaled_focal = float(rec["camera"]["scaled_focal_length"])
            pred_cam_t_full = np.array(rec["camera"]["pred_cam_t_full"], dtype=np.float64)

            # (2) Rescale to real intrinsics — preserve the centroid pixel.
            #
            # Naive scalar scaling cam_t · (fx_real/scaled_focal) only works
            # when cx_real = W/2 AND the centroid happens to lie on the
            # optical axis. In general we project the centroid under the
            # virtual pinhole, choose a real metric depth, then unproject
            # through the real pinhole at that depth.
            cx_v = W / 2.0
            cy_v = H / 2.0
            u_pix = scaled_focal * pred_cam_t_full[0] / pred_cam_t_full[2] + cx_v
            v_pix = scaled_focal * pred_cam_t_full[1] / pred_cam_t_full[2] + cy_v
            z_real = pred_cam_t_full[2] * (fx / scaled_focal)
            x_real = (u_pix - cx) * z_real / fx
            y_real = (v_pix - cy) * z_real / fy
            cam_t_real_pre = np.array([x_real, y_real, z_real], dtype=np.float64)

            # (1) Render MANO depth at cam_t_real_pre, compare to depth image.
            mesh = _mesh_for_record(rec, mano)
            rendered_depth = _render_depth(renderer, cam, mesh, cam_t_real_pre)
            hand_mask = rendered_depth > 0

            depth_path = os.path.join(depth_dir, f"{frame_idx:06d}.png")
            depth_present = os.path.exists(depth_path)
            depth_image = _load_depth(depth_path) if depth_present else None
            if depth_image is not None and depth_image.shape != (H, W):
                # depth image at a different resolution — skip alignment for this frame
                depth_image = None

            # Intersect with SAM2 hand mask for this track (if provided).
            # Looks up <hand_masks_dir>/<track_id>/<frame:06d>.png.
            if hand_masks_dir is not None:
                hp = os.path.join(hand_masks_dir, oid, f"{frame_idx:06d}.png")
                if os.path.exists(hp):
                    sam_mask = np.asarray(Image.open(hp)) > 0
                    if sam_mask.shape == (H, W):
                        hand_mask &= sam_mask

            occ_mask = None
            if object_masks_dir is not None:
                op = os.path.join(object_masks_dir, f"{frame_idx:06d}.png")
                if os.path.exists(op):
                    occ_mask = np.asarray(Image.open(op)) > 0
                    if occ_mask.shape != (H, W):
                        occ_mask = None
            if occ_mask is not None:
                hand_mask &= ~occ_mask

            n_pixels = int(hand_mask.sum())

            if depth_image is not None and n_pixels >= mask_min_pixels:
                di = depth_image[hand_mask]
                dr = rendered_depth[hand_mask]
                dz = float(np.median(di - dr))
                scale = float(np.median(di / dr))
                centroid_pre = mesh.vertices.mean(axis=0) + cam_t_real_pre
                offset = _ray_offset(centroid_pre, dz)
                cam_t_aligned = cam_t_real_pre + offset
            else:
                dz = 0.0
                scale = 1.0
                cam_t_aligned = cam_t_real_pre

            out_rec = {
                "track_id":   rec["track_id"],
                "is_right":   rec["is_right"],
                "frame_idx":  frame_idx,
                "image_size": [int(W), int(H)],
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "mano":       rec["mano"],
                "cam_t":      cam_t_aligned.tolist(),
                "diagnostics": {
                    "dz":           dz,
                    "n_pixels":     n_pixels,
                    "scale":        scale,
                    "cam_t_pre_dz": cam_t_real_pre.tolist(),
                    "scaled_focal": scaled_focal,
                },
            }
            with open(out_path, "w") as f:
                json.dump(out_rec, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamer_dir",        required=True)
    parser.add_argument("--depth_dir",        required=True)
    parser.add_argument("--intrinsics_path",  required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--hand_masks_dir", default=None,
                        help="Optional root folder of SAM2 per-track hand "
                             "masks (subdirs named <track_id>/). Restricts "
                             "the depth-comparison region to pixels that are "
                             "BOTH rendered-MANO and image-grounded hand.")
    parser.add_argument("--object_masks_dir", default=None,
                        help="Optional folder of object-occlusion masks "
                             "(per-frame PNGs). Pixels in this mask are "
                             "excluded from the depth-comparison region.")
    parser.add_argument("--mask_min_pixels",  type=int, default=256)
    args = parser.parse_args()
    align_hands(
        hamer_dir        = args.hamer_dir,
        depth_dir        = args.depth_dir,
        intrinsics_path  = args.intrinsics_path,
        mano_assets_root = args.mano_assets_root,
        output_dir       = args.output_dir,
        hand_masks_dir   = args.hand_masks_dir,
        object_masks_dir = args.object_masks_dir,
        mask_min_pixels  = args.mask_min_pixels,
    )


if __name__ == "__main__":
    main()
