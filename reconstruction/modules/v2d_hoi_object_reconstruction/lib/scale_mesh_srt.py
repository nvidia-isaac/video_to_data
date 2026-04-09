"""Scale + Rotation + Translation (SRT) estimation for SAM3D meshes.

Multi-view silhouette alignment: optimises scale, orientation, and translation
so that the SAM3D mesh projects consistently onto the observed object masks.

Ported and adapted from robomem/robomem/sam3d_post/estimate_srt.py.
Key adaptations vs the original:
  - Poses come from CuSFM keyframes (sfm/keyframes/frames_meta.json)
    rather than FoundationPose tracking output.
  - Depth maps are read from FoundationStereo 16-bit inverse-depth PNG
    (job_dir/depth/) instead of .npy files.
  - No external robomem / edex dependencies: intrinsics are loaded from
    job_dir/intrinsics/<frame>.json (v2d_common CameraIntrinsics format).
"""

from __future__ import annotations

import itertools
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# Internal camera intrinsics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def _load_intrinsics(intrinsics_json: Path) -> _Intrinsics:
    with open(intrinsics_json) as f:
        d = json.load(f)
    return _Intrinsics(
        fx=float(d["fx"]), fy=float(d["fy"]),
        cx=float(d["cx"]), cy=float(d["cy"]),
        width=int(d["width"]), height=int(d["height"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FrameView
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FrameView:
    """One frame's data for silhouette alignment."""
    frame_name: str
    intrinsics: _Intrinsics
    # world-to-camera 4×4 matrix (T_cam_from_world)
    world_to_cam: NDArray[np.float64]
    mask_u8: NDArray[np.uint8]
    dt_to_fg: NDArray[np.float32]
    mask_boundary_uv: NDArray[np.int32]
    mask_extent: float = 0.0
    depth: Optional[NDArray[np.float32]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_dt(mask_u8: NDArray[np.uint8]) -> NDArray[np.float32]:
    fg = mask_u8 > 127
    inv = (~fg).astype(np.uint8)
    return cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)


def _mask_boundary(mask_u8: NDArray[np.uint8]) -> NDArray[np.int32]:
    binary = (mask_u8 > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.int32)
    canvas = np.zeros_like(binary)
    cv2.drawContours(canvas, contours, -1, 255, 1)
    ys, xs = np.nonzero(canvas)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)


def _project(
    pts_cam: NDArray[np.float64], intr: _Intrinsics
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    ok = z > 1e-3
    u = (intr.fx * (x / np.where(ok, z, 1.0)) + intr.cx).astype(np.float32)
    v = (intr.fy * (y / np.where(ok, z, 1.0)) + intr.cy).astype(np.float32)
    inb = ok & (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    return u, v, inb


def _bilinear(
    img: NDArray[np.float32], u: NDArray[np.float32], v: NDArray[np.float32]
) -> NDArray[np.float32]:
    h, w = img.shape[:2]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, w - 1)
    v1 = np.clip(v0 + 1, 0, h - 1)
    du = (u - u0.astype(np.float32)).astype(np.float32)
    dv = (v - v0.astype(np.float32)).astype(np.float32)
    return (
        (1 - du) * (1 - dv) * img[v0, u0]
        + du * (1 - dv) * img[v0, u1]
        + (1 - du) * dv * img[v1, u0]
        + du * dv * img[v1, u1]
    ).astype(np.float32)


def _huber(x: NDArray[np.float32], delta: float) -> NDArray[np.float32]:
    absx = np.abs(x)
    q = absx <= delta
    out = np.empty_like(x, dtype=np.float32)
    out[q] = 0.5 * x[q] ** 2
    out[~q] = delta * (absx[~q] - 0.5 * delta)
    return out


def _raster_iou(
    u: NDArray[np.float32],
    v: NDArray[np.float32],
    mask_u8: NDArray[np.uint8],
    downsample: int = 4,
) -> float:
    h, w = mask_u8.shape
    hs, ws = h // downsample, w // downsample
    if hs < 2 or ws < 2:
        return 1.0
    gt = cv2.resize(mask_u8, (ws, hs), interpolation=cv2.INTER_AREA)
    gt_bin = gt > 127
    us = np.clip((u / downsample).astype(np.int32), 0, ws - 1)
    vs = np.clip((v / downsample).astype(np.int32), 0, hs - 1)
    proj = np.zeros((hs, ws), dtype=np.uint8)
    proj[vs, us] = 1
    n_proj = int(np.sum(proj))
    n_gt = int(np.sum(gt_bin))
    if n_proj < 1 or n_gt < 1:
        return 1.0
    r = max(1, min(int(round(math.sqrt(n_gt / n_proj / math.pi))), 8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    proj = cv2.dilate(proj, kernel, iterations=1)
    proj_bin = proj > 0
    intersection = float(np.sum(gt_bin & proj_bin))
    union = float(np.sum(gt_bin | proj_bin))
    return 1.0 - intersection / union if union >= 1 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _aa_to_matrix(aa: dict) -> np.ndarray:
    axis = np.array([aa["x"], aa["y"], aa["z"]])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec((axis / norm) * np.deg2rad(aa["angle_degrees"])).as_matrix()


def load_sfm_keyframe_poses(job_dir: Path) -> Dict[str, NDArray[np.float64]]:
    """Load CuSFM left-camera keyframe poses as world-to-camera 4×4 matrices.

    Returns {frame_stem: T_cam_from_world(4×4)} keyed by zero-padded frame ID.
    """
    sfm_kf_path = job_dir / "sfm" / "keyframes" / "frames_meta.json"
    frames_meta_path = job_dir / "frames_meta.json"
    if not sfm_kf_path.exists() or not frames_meta_path.exists():
        raise FileNotFoundError(
            f"SfM keyframe data not found under {job_dir}/sfm/keyframes/"
        )

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

    with open(sfm_kf_path) as f:
        sfm = json.load(f)

    poses: Dict[str, NDArray[np.float64]] = {}
    for kf in sfm["keyframes_metadata"]:
        if "front_stereo_camera_left" not in kf.get("image_name", ""):
            continue
        ts_us = int(kf["timestamp_microseconds"])
        seq_idx = ts_to_seq_idx.get(ts_us)
        if seq_idx is None:
            continue
        aa = kf["camera_to_world"]["axis_angle"]
        t_d = kf["camera_to_world"]["translation"]
        R_c2w = _aa_to_matrix(aa)
        t_c2w = np.array([t_d["x"], t_d["y"], t_d["z"]])
        # Build camera-to-world 4×4, then invert → world-to-camera
        T_c2w = np.eye(4, dtype=np.float64)
        T_c2w[:3, :3] = R_c2w
        T_c2w[:3, 3] = t_c2w
        T_w2c = np.linalg.inv(T_c2w)
        poses[f"{seq_idx:06d}"] = T_w2c

    if not poses:
        raise ValueError("No left-camera keyframes found in SfM output")
    return poses


def load_depth_png(depth_path: Path) -> NDArray[np.float32]:
    """Load a FoundationStereo 16-bit inverse-depth PNG → float32 depth in metres."""
    img = Image.open(depth_path)
    inv_depth = np.array(img).astype(np.float32)
    # Encoding: pixel = 65535 * (1 / (depth_m + 1))  →  depth_m = 65535/pixel - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_m = np.where(inv_depth > 0, 65535.0 / inv_depth - 1.0, 0.0)
    return depth_m.astype(np.float32)


def load_glb_vertices(glb_path: Path) -> NDArray[np.float64]:
    """Parse a binary GLB and return POSITION vertices as (N, 3) float64."""
    data = glb_path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Bad GLB magic in {glb_path}")
    c0_len, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20: 20 + c0_len].decode("utf-8").strip().rstrip("\x00"))
    pos_idx = gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    acc = gltf["accessors"][pos_idx]
    bv = gltf["bufferViews"][acc.get("bufferView", 0)]
    bin_start = 20 + c0_len
    c1_len, _ = struct.unpack_from("<II", data, bin_start)
    bin_data = data[bin_start + 8: bin_start + 8 + c1_len]
    byte_off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    verts = np.frombuffer(bin_data, dtype=np.float32, count=count * 3, offset=byte_off).reshape(-1, 3)
    return verts.astype(np.float64)


def export_scaled_glb(
    glb_path: Path,
    out_path: Path,
    *,
    orientation_matrix: NDArray[np.float64],
    rotation_matrix: NDArray[np.float64],
    scale: float,
    mesh_center: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> None:
    """Write a scaled/aligned GLB by binary-patching POSITION (and NORMAL) vertices."""
    data = bytearray(glb_path.read_bytes())
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Bad GLB magic in {glb_path}")
    c0_len, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(bytes(data[20: 20 + c0_len]).decode("utf-8").strip().rstrip("\x00"))
    bin_chunk_offset = 20 + c0_len
    bin_start = bin_chunk_offset + 8

    mc = np.asarray(mesh_center, dtype=np.float64).reshape(1, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(1, 3)
    s = float(scale)
    R_full = np.asarray(rotation_matrix, dtype=np.float64) @ np.asarray(orientation_matrix, dtype=np.float64)

    def _patch_vec3(accessor_idx: int, transform_fn) -> None:
        acc = gltf["accessors"][accessor_idx]
        if acc["type"] != "VEC3" or acc["componentType"] != 5126:
            return
        bv = gltf["bufferViews"][acc["bufferView"]]
        byte_off = bin_start + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride", 12)
        count = acc["count"]
        pts = np.empty((count, 3), dtype=np.float32)
        for i in range(count):
            pts[i] = struct.unpack_from("<3f", data, byte_off + i * stride)
        pts_out = transform_fn(pts.astype(np.float64)).astype(np.float32)
        for i in range(count):
            struct.pack_into("<3f", data, byte_off + i * stride, *pts_out[i].tolist())
        acc["min"] = pts_out.min(axis=0).tolist()
        acc["max"] = pts_out.max(axis=0).tolist()

    def _xform_pos(pts):
        return (rotation_matrix @ (orientation_matrix @ (pts - mc).T)).T * s + t

    def _xform_nrm(nrm):
        n = (R_full @ nrm.T).T
        norms = np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return n / norms

    for mesh_obj in gltf.get("meshes", []):
        for prim in mesh_obj.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "POSITION" in attrs:
                _patch_vec3(attrs["POSITION"], _xform_pos)
            if "NORMAL" in attrs:
                _patch_vec3(attrs["NORMAL"], _xform_nrm)

    new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad = (4 - len(new_json) % 4) % 4
    new_json += b" " * pad
    out_buf = bytearray()
    out_buf += struct.pack("<III", magic, version, 0)
    out_buf += struct.pack("<II", len(new_json), 0x4E4F534A)
    out_buf += new_json
    out_buf += bytes(data[bin_chunk_offset:])
    struct.pack_into("<I", out_buf, 8, len(out_buf))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(out_buf))


# ─────────────────────────────────────────────────────────────────────────────
# View building
# ─────────────────────────────────────────────────────────────────────────────

def _filter_pose_drift(
    names: List[str],
    poses: Dict[str, NDArray[np.float64]],
    jump_factor: float = 10.0,
) -> List[str]:
    if len(names) <= 2:
        return names
    # Camera position in world = -R.T @ t  (where pose is T_cam_from_world)
    positions = np.array(
        [-poses[n][:3, :3].T @ poses[n][:3, 3] for n in names],
        dtype=np.float64,
    )
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    med = float(np.median(diffs))
    if med < 1e-9:
        return names
    threshold = max(jump_factor * med, 0.5)
    for i, d in enumerate(diffs):
        if d > threshold:
            removed = len(names) - (i + 1)
            if removed > 0:
                print(f"  [pose-drift] Removed {removed} frames from index {i + 1} "
                      f"(jump {d:.3f}m, threshold {threshold:.3f}m)")
            return names[: i + 1]
    return names


def build_frame_views(
    intrinsics: _Intrinsics,
    poses: Dict[str, NDArray[np.float64]],
    masks_dir: Path,
    depth_dir: Optional[Path],
    frame_step: int = 1,
    max_views: int = 0,
) -> List[FrameView]:
    """Build FrameView list from SfM poses and mask directory."""
    mask_files = {p.stem: p for p in sorted(masks_dir.glob("*.png"))}
    common = sorted(set(poses.keys()) & set(mask_files.keys()))
    if not common:
        raise ValueError("No frames in common between SfM poses and masks")

    common = _filter_pose_drift(common, poses)
    if frame_step > 1:
        common = common[::frame_step]
    if max_views > 0:
        common = common[:max_views]

    views: List[FrameView] = []
    skipped = 0
    for name in common:
        img = cv2.imread(str(mask_files[name]), cv2.IMREAD_UNCHANGED)
        if img is None:
            skipped += 1
            continue
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.asarray(img, dtype=np.uint8)
        fg_count = int(np.count_nonzero(mask > 127))
        if fg_count < 50:
            skipped += 1
            continue
        ys, xs = np.nonzero(mask > 127)
        mask_ext = float(max(int(np.ptp(xs)), int(np.ptp(ys))))

        depth: Optional[NDArray[np.float32]] = None
        if depth_dir is not None:
            depth_path = depth_dir / f"{name}.png"
            if depth_path.exists():
                depth = load_depth_png(depth_path)

        views.append(FrameView(
            frame_name=name,
            intrinsics=intrinsics,
            world_to_cam=poses[name],
            mask_u8=mask,
            dt_to_fg=_compute_dt(mask),
            mask_boundary_uv=_mask_boundary(mask),
            mask_extent=mask_ext,
            depth=depth,
        ))

    if skipped:
        print(f"  Skipped {skipped} frames (missing mask or <50 fg pixels)")
    if not views:
        raise ValueError("No valid views after filtering")
    return views


# ─────────────────────────────────────────────────────────────────────────────
# SRT evaluation
# ─────────────────────────────────────────────────────────────────────────────

def generate_axis_orientations() -> Dict[str, NDArray[np.float64]]:
    """All 48 signed axis permutations."""
    mats: Dict[str, NDArray[np.float64]] = {}
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3), dtype=np.float64)
            for out_ax, (in_ax, sgn) in enumerate(zip(perm, signs)):
                m[out_ax, in_ax] = float(sgn)
            det = int(round(float(np.linalg.det(m))))
            s_str = "".join(str(s) for s in signs)
            key = f"perm({perm[0]},{perm[1]},{perm[2]})_s{s_str}_det{det}"
            mats[key] = m
    return mats


def evaluate_srt(
    vertices: NDArray[np.float64],
    views: Sequence[FrameView],
    *,
    scale: float,
    orient: NDArray[np.float64],
    rotvec: NDArray[np.float64],
    translation: NDArray[np.float64],
    mesh_center: NDArray[np.float64],
    dt_clip: float = 20.0,
    huber_delta: float = 3.0,
    coverage_weight: float = 0.5,
    iou_weight: float = 2.0,
    depth_weight: float = 1.0,
    max_pts: int = 25000,
    max_boundary: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(0)

    v = vertices - mesh_center.reshape(1, 3)
    v = v @ orient.T
    if np.any(rotvec != 0):
        v = (Rotation.from_rotvec(rotvec).as_matrix() @ v.T).T
    pts_w = v * float(scale) + translation.reshape(1, 3)
    pts_h = np.hstack([pts_w, np.ones((pts_w.shape[0], 1), dtype=np.float64)])

    losses: List[float] = []
    for view in views:
        view_dt_clip = max(dt_clip, 0.1 * view.mask_extent) if view.mask_extent > 0 else dt_clip
        pts_c = (view.world_to_cam @ pts_h.T).T[:, :3]
        u, vv, inb = _project(pts_c, view.intrinsics)

        if not np.any(inb):
            losses.append(float(view_dt_clip))
            continue

        ui_all, vi_all = u[inb], vv[inb]
        zi_all = pts_c[inb, 2].astype(np.float32)

        ui, vi, zi = ui_all, vi_all, zi_all
        if ui.size > max_pts:
            idx = rng.choice(ui.size, size=max_pts, replace=False)
            ui, vi, zi = ui[idx], vi[idx], zi[idx]

        dt_vals = np.minimum(_bilinear(view.dt_to_fg, ui, vi), view_dt_clip)
        view_loss = float(np.mean(_huber(dt_vals, huber_delta)))

        if coverage_weight > 0 and view.mask_boundary_uv.shape[0] > 0 and ui.size > 0:
            bnd = view.mask_boundary_uv
            if bnd.shape[0] > max_boundary:
                bnd = bnd[rng.choice(bnd.shape[0], size=max_boundary, replace=False)]
            tree = cKDTree(np.stack([ui, vi], axis=1).astype(np.float64))
            dists, _ = tree.query(bnd.astype(np.float64), k=1, workers=-1)
            cov_dt = np.minimum(dists.astype(np.float32), view_dt_clip)
            view_loss += coverage_weight * float(np.mean(_huber(cov_dt, huber_delta)))

        if iou_weight > 0 and ui_all.size > 0:
            view_loss += iou_weight * _raster_iou(ui_all, vi_all, view.mask_u8)

        if depth_weight > 0 and view.depth is not None and ui.size > 0:
            depth_obs = _bilinear(view.depth, ui, vi)
            valid_depth = (depth_obs > 0.01) & np.isfinite(depth_obs)
            if np.any(valid_depth):
                focal = (view.intrinsics.fx + view.intrinsics.fy) * 0.5
                depth_err_px = (
                    np.abs(zi[valid_depth] - depth_obs[valid_depth])
                    * focal
                    / np.maximum(depth_obs[valid_depth], 0.01)
                )
                depth_err_px = np.minimum(depth_err_px, view_dt_clip)
                view_loss += depth_weight * float(np.mean(_huber(depth_err_px, huber_delta)))

        losses.append(view_loss)

    return float(np.mean(losses))


# ─────────────────────────────────────────────────────────────────────────────
# Initialization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_object_center(views: Sequence[FrameView]) -> NDArray[np.float64]:
    """Estimate 3D object centre via ray triangulation (with depth fallback)."""
    cam_origins = np.array(
        [-v.world_to_cam[:3, :3].T @ v.world_to_cam[:3, 3] for v in views],
        dtype=np.float64,
    )
    cam_centroid = np.mean(cam_origins, axis=0)

    # Depth back-projection (when available)
    has_depth = any(v.depth is not None for v in views)
    if has_depth:
        rng = np.random.default_rng(0)
        pts_3d: List[NDArray] = []
        for view in views:
            if view.depth is None:
                continue
            fg = view.mask_u8 > 127
            ys, xs = np.nonzero(fg)
            if xs.size < 10:
                continue
            if xs.size > 2000:
                idx = rng.choice(xs.size, size=2000, replace=False)
                xs, ys = xs[idx], ys[idx]
            d = view.depth[ys, xs].astype(np.float64)
            valid = (d > 0.01) & np.isfinite(d)
            if not np.any(valid):
                continue
            xs_v, ys_v, d_v = xs[valid].astype(np.float64), ys[valid].astype(np.float64), d[valid]
            intr = view.intrinsics
            x_cam = (xs_v - intr.cx) / intr.fx * d_v
            y_cam = (ys_v - intr.cy) / intr.fy * d_v
            pts_cam = np.column_stack([x_cam, y_cam, d_v, np.ones_like(d_v)])
            T_c2w = np.linalg.inv(view.world_to_cam)
            pts_w = (T_c2w @ pts_cam.T).T[:, :3]
            pts_3d.append(pts_w)
        if pts_3d:
            all_pts = np.concatenate(pts_3d, axis=0)
            center = np.median(all_pts, axis=0).astype(np.float64)
            dist = float(np.median(np.linalg.norm(cam_origins - center, axis=1)))
            if dist < 20.0:
                return center

    # Ray triangulation
    A = np.zeros((3, 3), dtype=np.float64)
    b_vec = np.zeros(3, dtype=np.float64)
    for view in views:
        ys, xs = np.nonzero(view.mask_u8 > 127)
        if xs.size == 0:
            continue
        cu, cv = float(np.mean(xs)), float(np.mean(ys))
        intr = view.intrinsics
        ray_cam = np.array([(cu - intr.cx) / intr.fx, (cv - intr.cy) / intr.fy, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        T_c2w = np.linalg.inv(view.world_to_cam)
        ray_w = T_c2w[:3, :3] @ ray_cam
        ray_w /= np.linalg.norm(ray_w)
        origin = T_c2w[:3, 3]
        P = np.eye(3) - np.outer(ray_w, ray_w)
        A += P
        b_vec += P @ origin

    try:
        center = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        center = cam_centroid.copy()

    cam_spread = float(np.max(np.ptp(cam_origins, axis=0)))
    dist = float(np.median(np.linalg.norm(cam_origins - center, axis=1)))
    if dist > max(5.0, cam_spread * 5.0):
        mean_dir = np.zeros(3)
        for view in views:
            T_c2w = np.linalg.inv(view.world_to_cam)
            mean_dir += T_c2w[:3, :3] @ np.array([0, 0, 1.0])
        mean_dir /= max(np.linalg.norm(mean_dir), 1e-12)
        center = cam_centroid + mean_dir * cam_spread * 0.5

    return center


def _estimate_initial_scale(
    views: Sequence[FrameView],
    mesh_extent: float,
    object_center: NDArray[np.float64],
) -> float:
    scales: List[float] = []
    for view in views:
        ys, xs = np.nonzero(view.mask_u8 > 127)
        if xs.size < 10:
            continue
        mask_px = float(max(np.ptp(xs), np.ptp(ys)))
        cam_pos = -view.world_to_cam[:3, :3].T @ view.world_to_cam[:3, 3]
        dist = float(np.linalg.norm(cam_pos - object_center))
        if dist < 1e-4:
            continue
        focal = (view.intrinsics.fx + view.intrinsics.fy) * 0.5
        obj_m = mask_px * dist / focal
        if mesh_extent > 1e-9:
            scales.append(obj_m / mesh_extent)
    return float(np.median(scales)) if scales else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Axis search + optimization
# ─────────────────────────────────────────────────────────────────────────────

def run_axis_search(
    vertices: NDArray[np.float64],
    views: Sequence[FrameView],
    mesh_center: NDArray[np.float64],
    translation: NDArray[np.float64],
    s0: float,
    top_k: int = 5,
    max_views: int = 15,
) -> List[Tuple[str, NDArray[np.float64], float]]:
    all_orients = generate_axis_orientations()
    rng = np.random.default_rng(0)
    view_subset = list(views)
    if len(view_subset) > max_views:
        idxs = np.linspace(0, len(view_subset) - 1, max_views, dtype=int)
        view_subset = [view_subset[i] for i in idxs]

    rv0 = np.zeros(3)
    results: List[Tuple[str, NDArray[np.float64], float]] = []
    best_key, best_loss = "", float("inf")
    total = len(all_orients)
    for i, (key, mat) in enumerate(all_orients.items()):
        loss = evaluate_srt(
            vertices, view_subset, scale=s0, orient=mat, rotvec=rv0,
            translation=translation, mesh_center=mesh_center,
            coverage_weight=0.5, iou_weight=1.0, depth_weight=0.0,
            max_pts=10000, rng=rng,
        )
        results.append((key, mat, loss))
        if loss < best_loss:
            best_loss = loss
            best_key = key
        if (i + 1) % 12 == 0:
            print(f"  axis search: {i + 1}/{total}  best={best_key}  loss={best_loss:.4f}")

    results.sort(key=lambda x: x[2])
    top = results[:top_k]
    print(f"  Top-{top_k} orientations:")
    for rank, (k, _, l) in enumerate(top):
        print(f"    #{rank + 1}  {k}  loss={l:.4f}")
    return top


def optimize_srt(
    vertices: NDArray[np.float64],
    views: Sequence[FrameView],
    orient: NDArray[np.float64],
    mesh_center: NDArray[np.float64],
    t_init: NDArray[np.float64],
    s_init: float,
    *,
    mode: str = "str",
    s_bounds: Tuple[float, float] = (0.001, 100.0),
    dt_clip: float = 20.0,
    huber_delta: float = 3.0,
    coverage_weight: float = 0.5,
    iou_weight: float = 2.0,
    depth_weight: float = 1.0,
    maxiter: int = 120,
) -> Dict[str, object]:
    rng = np.random.default_rng(0)
    log_s0 = math.log(max(float(s_init), 1e-12))
    log_s_min = math.log(max(float(s_bounds[0]), 1e-12))
    log_s_max = math.log(float(s_bounds[1]))

    cam_positions = np.array(
        [-v.world_to_cam[:3, :3].T @ v.world_to_cam[:3, 3] for v in views],
        dtype=np.float64,
    )
    t_bound = float(np.max(np.ptp(cam_positions, axis=0))) * 0.5
    rot_bound = 45.0 * math.pi / 180.0

    tracker: Dict = {"n": 0, "best_loss": float("inf"), "best_x": None}

    def _obj(x):
        n = tracker["n"]
        tracker["n"] = n + 1
        s_val = math.exp(float(x[0]))
        t_val = t_init.copy()
        rv_val = np.zeros(3)
        if mode in ("st", "str"):
            t_val = np.array([float(x[1]), float(x[2]), float(x[3])])
        if mode == "str":
            rv_val = np.array([float(x[4]), float(x[5]), float(x[6])])
        loss = evaluate_srt(
            vertices, views, scale=s_val, orient=orient, rotvec=rv_val,
            translation=t_val, mesh_center=mesh_center, dt_clip=dt_clip,
            huber_delta=huber_delta, coverage_weight=coverage_weight,
            iou_weight=iou_weight, depth_weight=depth_weight, rng=rng,
        )
        if loss < tracker["best_loss"]:
            tracker["best_loss"] = loss
            tracker["best_x"] = x.copy()
        if n % 20 == 0:
            print(f"  [opt_{mode}] eval={n:04d}  s={s_val:.4f}  loss={loss:.6f}  best={tracker['best_loss']:.6f}")
        return loss

    t0 = t_init.astype(np.float64)
    if mode == "s":
        x0 = np.array([log_s0])
        bounds = [(log_s_min, log_s_max)]
    elif mode == "st":
        x0 = np.array([log_s0, t0[0], t0[1], t0[2]])
        bounds = [(log_s_min, log_s_max),
                  (t0[0] - t_bound, t0[0] + t_bound),
                  (t0[1] - t_bound, t0[1] + t_bound),
                  (t0[2] - t_bound, t0[2] + t_bound)]
    else:
        x0 = np.array([log_s0, t0[0], t0[1], t0[2], 0.0, 0.0, 0.0])
        bounds = [(log_s_min, log_s_max),
                  (t0[0] - t_bound, t0[0] + t_bound),
                  (t0[1] - t_bound, t0[1] + t_bound),
                  (t0[2] - t_bound, t0[2] + t_bound),
                  (-rot_bound, rot_bound),
                  (-rot_bound, rot_bound),
                  (-rot_bound, rot_bound)]

    res = minimize(_obj, x0, method="Powell", bounds=bounds, options={"maxiter": maxiter})
    best_x = tracker["best_x"] if tracker["best_x"] is not None else res.x

    s_best = math.exp(float(best_x[0]))
    t_best = t_init.copy()
    rv_best = np.zeros(3)
    if mode in ("st", "str"):
        t_best = np.array([float(best_x[1]), float(best_x[2]), float(best_x[3])])
    if mode == "str":
        rv_best = np.array([float(best_x[4]), float(best_x[5]), float(best_x[6])])

    R_mat = Rotation.from_rotvec(rv_best).as_matrix()
    final_loss = evaluate_srt(
        vertices, views, scale=s_best, orient=orient, rotvec=rv_best,
        translation=t_best, mesh_center=mesh_center, dt_clip=dt_clip,
        huber_delta=huber_delta, coverage_weight=coverage_weight,
        iou_weight=iou_weight, depth_weight=depth_weight, rng=rng,
    )
    return {
        "scale": s_best,
        "translation": t_best.tolist(),
        "rotvec": rv_best.tolist(),
        "rotation_matrix": R_mat.tolist(),
        "total_loss": final_loss,
        "optimizer_nfev": int(res.nfev),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Debug overlays
# ─────────────────────────────────────────────────────────────────────────────

def _write_debug_overlays(
    vertices: NDArray[np.float64],
    views: Sequence[FrameView],
    orient: NDArray[np.float64],
    rotvec: NDArray[np.float64],
    scale: float,
    translation: NDArray[np.float64],
    mesh_center: NDArray[np.float64],
    output_dir: Path,
    images_dir: Optional[Path] = None,
) -> None:
    rng = np.random.default_rng(0)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    v = vertices - mesh_center.reshape(1, 3)
    v = v @ orient.T
    if np.any(rotvec != 0):
        v = (Rotation.from_rotvec(rotvec).as_matrix() @ v.T).T
    pts_w = v * float(scale) + translation.reshape(1, 3)
    pts_h = np.hstack([pts_w, np.ones((pts_w.shape[0], 1))])

    MAX_PTS = 30000
    for view in views:
        pts_c = (view.world_to_cam @ pts_h.T).T[:, :3]
        u, vv, inb = _project(pts_c, view.intrinsics)
        h, w = view.mask_u8.shape[:2]

        # Base: image if available, else dark canvas
        if images_dir is not None:
            img_path = images_dir / f"{view.frame_name}.jpg"
            canvas = cv2.imread(str(img_path)) if img_path.exists() else None
        else:
            canvas = None
        if canvas is None:
            canvas = np.full((h, w, 3), 40, dtype=np.uint8)

        binary = (view.mask_u8 > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2)

        ui, vi_vis = u[inb], vv[inb]
        if ui.size > MAX_PTS:
            idx = rng.choice(ui.size, MAX_PTS, replace=False)
            ui, vi_vis = ui[idx], vi_vis[idx]
        for uu, vvi in zip(ui, vi_vis):
            cv2.circle(canvas, (int(uu), int(vvi)), 1, (255, 255, 0), -1)

        cv2.putText(canvas, f"{view.frame_name}  N={int(ui.size)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(debug_dir / f"{view.frame_name}.jpg"), canvas)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def estimate_srt_for_frame(
    job_dir: Path,
    glb_path: Path,
    output_dir: Path,
    *,
    frame_step: int = 5,
    max_views: int = 100,
    mode: str = "str",
    maxiter: int = 120,
    iou_weight: float = 2.0,
    depth_weight: float = 1.0,
    use_depth: bool = False,
    debug: bool = True,
) -> dict:
    """Estimate scale+rotation+translation for a SAM3D mesh using SfM poses + masks.

    Args:
        job_dir:      HOI reconstruction job directory (must have sfm/, masks/, intrinsics/).
        glb_path:     SAM3D output GLB mesh.
        output_dir:   Where to write srt_result.json and output_scaled.glb.
        frame_step:   Use every Nth keyframe (speed vs. accuracy).
        max_views:    Cap on number of views (0 = all).
        mode:         's' scale-only, 'st' scale+translation, 'str' all DOF.
        maxiter:      Powell optimiser iterations.
        iou_weight:   Weight for rasterised silhouette IoU loss.
        depth_weight: Weight for depth loss (only when use_depth=True and depth available).
        use_depth:    If True, load depth maps from job_dir/depth/.
        debug:        Write per-view overlay images.

    Returns:
        The srt_result dict (also written to output_dir/srt_result.json).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load intrinsics from the first available file in job_dir/intrinsics/
    intrinsics_dir = job_dir / "intrinsics"
    intr_files = sorted(intrinsics_dir.glob("*.json"))
    if not intr_files:
        raise FileNotFoundError(f"No intrinsics JSON found in {intrinsics_dir}")
    intrinsics = _load_intrinsics(intr_files[0])

    # Load SfM keyframe poses
    print(f"[srt] Loading SfM keyframe poses from {job_dir}/sfm/keyframes/ …")
    poses = load_sfm_keyframe_poses(job_dir)
    print(f"[srt] {len(poses)} keyframe poses loaded")

    # Build views
    masks_dir = job_dir / "masks" / "0"
    depth_dir: Optional[Path] = (job_dir / "depth") if use_depth else None
    views = build_frame_views(
        intrinsics, poses, masks_dir, depth_dir,
        frame_step=frame_step, max_views=max_views,
    )
    print(f"[srt] {len(views)} views built (frame_step={frame_step}, max_views={max_views})")

    # Load mesh vertices
    print(f"[srt] Loading mesh vertices from {glb_path} …")
    vertices = load_glb_vertices(glb_path)
    print(f"[srt] {len(vertices)} vertices")

    mesh_center = vertices.mean(axis=0)
    mesh_extent = float(np.max(np.ptp(vertices - mesh_center, axis=0)))

    # Initial estimates
    object_center = _estimate_object_center(views)
    s_init = _estimate_initial_scale(views, mesh_extent, object_center)
    print(f"[srt] Initial scale estimate: {s_init:.4f}")
    print(f"[srt] Estimated object center: {object_center}")

    # Axis search
    print("[srt] Running axis orientation search …")
    top_orients = run_axis_search(vertices, views, mesh_center, object_center, s_init)

    # Optimise for each top orientation, pick best
    best_result: Optional[dict] = None
    best_loss = float("inf")
    best_orient: Optional[NDArray] = None

    for rank, (key, orient_mat, _) in enumerate(top_orients):
        print(f"\n[srt] Optimising orientation #{rank + 1}: {key}")
        result = optimize_srt(
            vertices, views, orient_mat, mesh_center, object_center, s_init,
            mode=mode, iou_weight=iou_weight, depth_weight=depth_weight,
            maxiter=maxiter,
        )
        print(f"  → loss={result['total_loss']:.6f}  scale={result['scale']:.4f}")
        if result["total_loss"] < best_loss:
            best_loss = result["total_loss"]
            best_result = result
            best_orient = orient_mat

    assert best_result is not None
    best_result["orientation_key"] = top_orients[0][0]
    best_result["mesh_center"] = mesh_center.tolist()
    best_result["num_views"] = len(views)

    # Save result JSON
    result_path = output_dir / "srt_result.json"
    with open(result_path, "w") as f:
        json.dump(best_result, f, indent=2)
    print(f"\n[srt] Result saved → {result_path}")

    # Export scaled GLB
    out_glb = output_dir / "output_scaled.glb"
    export_scaled_glb(
        glb_path, out_glb,
        orientation_matrix=best_orient,
        rotation_matrix=np.array(best_result["rotation_matrix"]),
        scale=best_result["scale"],
        mesh_center=mesh_center,
        translation=np.array(best_result["translation"]),
    )
    print(f"[srt] Scaled mesh saved → {out_glb}")

    # Debug overlays
    if debug:
        images_dir = job_dir / "left"
        _write_debug_overlays(
            vertices, views, best_orient,
            np.array(best_result["rotvec"]),
            best_result["scale"],
            np.array(best_result["translation"]),
            mesh_center, output_dir,
            images_dir=images_dir if images_dir.is_dir() else None,
        )
        print(f"[srt] Debug overlays → {output_dir / 'debug'}/")

    return best_result
