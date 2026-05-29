# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
3D depth viewer — Flask backend

Serves an interactive Three.js viewer to inspect unprojected depth point clouds
and FP mesh poses for a given session.

Usage (from reconstruction/):
    python -m v2d.viz.run_3d_viewer \
        --mesh_path       data/objects/electric_drill_toy/mesh/textured_mesh.obj \
        --depth_folder    data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/depth_moge_aligned \
        --intrinsics_path data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/intrinsics_moge_stable.json \
        --poses_folder    data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/poses_moge_aligned_smoothed \
        [--frames_folder  data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/frames] \
        [--masks_folder   data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/masks/1] \
        [--hand_mesh_path data/objects/.../hand/hand_mesh/traj.npz] \
        [--port 5000]
"""

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Config (populated from CLI args)
# ---------------------------------------------------------------------------
MESH_PATH       = None
DEPTH_FOLDER    = None
INTRINSICS_PATH = None
POSES_FOLDER    = None
FRAMES_FOLDER   = None
MASKS_FOLDER    = None
HAND_MESH_PATH  = None

_hand_data = None       # lazy-loaded NPZ cache
_obj_verts = None       # parsed OBJ vertices (N, 3)
_obj_faces = None       # parsed OBJ faces    (M, 3) int indices


def _load_hand_data():
    global _hand_data
    if _hand_data is None and HAND_MESH_PATH:
        _hand_data = np.load(HAND_MESH_PATH, allow_pickle=True)
    return _hand_data


def _load_obj_geometry() -> tuple[np.ndarray | None, np.ndarray | None]:
    """Parse OBJ file once and cache (verts, faces) as numpy arrays."""
    global _obj_verts, _obj_faces
    if _obj_verts is None and MESH_PATH and os.path.exists(MESH_PATH):
        verts, faces = [], []
        with open(MESH_PATH) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'v' and len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f' and len(parts) >= 4:
                    faces.append([int(p.split('/')[0]) - 1 for p in parts[1:4]])
        _obj_verts = np.array(verts, dtype=np.float32)
        _obj_faces = np.array(faces, dtype=np.int32)
    return _obj_verts, _obj_faces


def _project_to_image(pts_cv: np.ndarray, intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    """(N,3) CV-space points → (N,2) pixel coords + boolean valid mask (z > 0.01)."""
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    z = pts_cv[:, 2]
    valid = z > 0.01
    z_safe = np.where(valid, z, 1.0)
    u = fx * pts_cv[:, 0] / z_safe + cx
    v = fy * pts_cv[:, 1] / z_safe + cy
    return np.stack([u, v], axis=-1), valid

app = Flask(__name__, static_folder="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_depth(path: Path) -> np.ndarray:
    """uint16 PNG → metric depth in metres."""
    raw = np.array(Image.open(path), dtype=np.float32)
    return 65535.0 / (raw + 1.0) - 1.0


def unproject(depth: np.ndarray, intrinsics: dict, mask: np.ndarray = None,
              max_points: int = 80_000) -> tuple[np.ndarray, np.ndarray | None]:
    """Depth map → (N,3) points in CV camera space (X right, Y down, Z forward).
    Returns (positions, depth_values_for_coloring).
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    valid = depth > 0.01
    if mask is not None:
        valid &= mask > 128

    # Subsample if too many points
    n_valid = valid.sum()
    if n_valid > max_points:
        idx = np.where(valid.ravel())[0]
        chosen = np.random.choice(idx, max_points, replace=False)
        valid = np.zeros(h * w, bool)
        valid[chosen] = True
        valid = valid.reshape(h, w)

    z = depth[valid]
    x = (u[valid] - cx) / fx * z
    y = (v[valid] - cy) / fy * z

    pts = np.stack([x, y, z], axis=-1)          # CV camera space
    return pts, z


def cv_to_threejs(pts: np.ndarray) -> np.ndarray:
    """CV (X right, Y down, Z forward) → Three.js (X right, Y up, Z back)."""
    out = pts.copy()
    out[:, 1] = -pts[:, 1]
    out[:, 2] = -pts[:, 2]
    return out


def pose_to_threejs_matrix(pose_path: Path) -> list[float] | None:
    """Load Transform3d pose → column-major 4×4 for Three.js set from matrix.
    Pose is object-to-camera (CV convention): v_cam_cv = M_cv * v_obj.
    To display in Three.js world space (Y up, Z back):
        v_world = F * v_cam_cv = F * M_cv * v_obj  →  M_three = F * M_cv
    where F = diag(1,-1,-1,1) flips Y and Z.
    Returns flat list of 16 floats (column-major, for Three.js Matrix4.fromArray).
    """
    if not pose_path.exists():
        return None

    with open(pose_path) as f:
        d = json.load(f)

    w, x, y, z = d["rotation"]
    tx, ty, tz = d["translation"]
    sx, sy, sz = d["scale"]

    R = np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,      2*x*z + 2*w*y],
        [2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,      2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R @ np.diag([sx, sy, sz])
    M[:3, 3] = [tx, ty, tz]

    # Convert from CV to Three.js: M_three = F * M_cv
    # F flips Y and Z: maps (X right, Y down, Z fwd) → (X right, Y up, Z back)
    F = np.diag([1.0, -1.0, -1.0, 1.0])
    M_three = F @ M

    # Three.js Matrix4.set() takes row-major args, but fromArray() is column-major
    # Return column-major (transpose)
    return M_three.T.ravel().tolist()


def depth_to_color(z: np.ndarray) -> np.ndarray:
    """Depth values → uint8 RGB using a simple plasma-ish colormap."""
    zmin, zmax = np.percentile(z, 2), np.percentile(z, 98)
    t = np.clip((z - zmin) / (zmax - zmin + 1e-8), 0, 1)
    r = np.clip(1.5 - abs(t - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - abs(t - 0.5)  * 4, 0, 1)
    b = np.clip(1.5 - abs(t - 0.25) * 4, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/frames")
def get_frames():
    frames = sorted(
        int(p.stem) for p in Path(DEPTH_FOLDER).glob("*.png")
        if p.stem.isdigit()
    )
    return jsonify(frames)


@app.route("/api/pointcloud/<int:frame_id>")
def get_pointcloud(frame_id):
    depth_path = Path(DEPTH_FOLDER) / f"{frame_id:06d}.png"
    if not depth_path.exists():
        return jsonify({"error": "not found"}), 404

    use_mask = request.args.get("use_mask", "false").lower() == "true"
    max_points = int(request.args.get("max_points", 80_000))

    with open(INTRINSICS_PATH) as f:
        intrinsics = json.load(f)

    depth = load_depth(depth_path)

    # Resize depth to intrinsics resolution if needed
    iw, ih = intrinsics["width"], intrinsics["height"]
    if depth.shape != (ih, iw):
        depth = np.array(Image.fromarray(depth).resize((iw, ih), Image.NEAREST))

    mask = None
    if use_mask and MASKS_FOLDER:
        mp = Path(MASKS_FOLDER) / f"{frame_id:06d}.png"
        if mp.exists():
            mask = np.array(Image.open(mp).resize((iw, ih), Image.NEAREST))

    pts_cv, z_vals = unproject(depth, intrinsics, mask=mask, max_points=max_points)
    pts_three = cv_to_threejs(pts_cv)

    # Colors: try RGB frame first, fall back to depth colormap
    colors = None
    if FRAMES_FOLDER:
        fp = Path(FRAMES_FOLDER) / f"{frame_id:06d}.png"
        if fp.exists():
            rgb = np.array(Image.open(fp).resize((iw, ih)))

            # Reconstruct which pixels were selected
            # Re-run unproject to get valid mask (same random seed not guaranteed;
            # easier to just use depth colormap if subsampled)
            # Only use RGB if we're not subsampling (all valid points kept)
            d = load_depth(depth_path)
            if depth.shape != (ih, iw):
                d = np.array(Image.fromarray(d).resize((iw, ih), Image.NEAREST))
            valid = d > 0.01
            if mask is not None:
                valid &= mask > 128
            if valid.sum() <= 80_000:
                u_idx = np.where(valid.ravel())[0]
                rows = u_idx // iw
                cols = u_idx % iw
                colors = rgb[rows, cols, :3].tolist()

    if colors is None:
        colors = depth_to_color(z_vals).tolist()

    return jsonify({
        "positions": pts_three.tolist(),
        "colors": colors,
    })


@app.route("/api/pose/<int:frame_id>")
def get_pose(frame_id):
    if not POSES_FOLDER:
        return jsonify(None)
    pose_path = Path(POSES_FOLDER) / f"{frame_id:06d}.json"
    matrix = pose_to_threejs_matrix(pose_path)
    return jsonify({"matrix": matrix})


@app.route("/api/mesh")
def get_mesh():
    return send_file(os.path.abspath(MESH_PATH))


@app.route("/api/mesh/<path:filename>")
def get_mesh_asset(filename):
    """Serve MTL and texture files relative to the mesh directory."""
    mesh_dir = os.path.dirname(os.path.abspath(MESH_PATH))
    return send_from_directory(mesh_dir, filename)


@app.route("/api/hand/<int:frame_id>")
def get_hand(frame_id):
    """Return hand mesh vertices and faces for a frame, in Three.js space."""
    data = _load_hand_data()
    if data is None:
        return jsonify({"hands": []})

    verts     = data["verts"]      # (n_hands, n_frames, n_verts, 3) — CV camera space, metres
    is_right  = data["is_right"]   # (n_hands, n_frames)
    vis_mask  = data["vis_mask"]   # (n_hands, n_frames)
    faces_left  = data["faces_left"].tolist()
    faces_right = data["faces_right"].tolist()

    n_frames = verts.shape[1]
    if frame_id >= n_frames:
        return jsonify({"hands": []})

    hands = []
    for h in range(verts.shape[0]):
        if not vis_mask[h, frame_id]:
            continue
        v_cv    = verts[h, frame_id]          # (n_verts, 3)
        v_three = cv_to_threejs(v_cv)
        right   = int(is_right[h, frame_id])
        hands.append({
            "is_right": right,
            "vertices": v_three.tolist(),
            "faces":    faces_right if right else faces_left,
        })

    return jsonify({"hands": hands})


@app.route("/api/overlay/<int:frame_id>")
def get_overlay(frame_id):
    """Return the RGB frame with hand + object mesh overlaid as a JPEG."""
    if not FRAMES_FOLDER:
        return jsonify({"error": "no frames folder"}), 404
    frame_path = Path(FRAMES_FOLDER) / f"{frame_id:06d}.png"
    if not frame_path.exists():
        return jsonify({"error": "frame not found"}), 404

    with open(INTRINSICS_PATH) as f:
        intrinsics = json.load(f)
    iw, ih = intrinsics['width'], intrinsics['height']

    img = Image.open(frame_path).convert('RGB')
    if img.size != (iw, ih):
        img = img.resize((iw, ih))

    overlay = Image.new('RGBA', (iw, ih), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # --- Hand meshes ---
    hand_data = _load_hand_data()
    if hand_data is not None:
        verts    = hand_data['verts']     # (n_hands, n_frames, n_verts, 3)
        is_right = hand_data['is_right']  # (n_hands, n_frames)
        vis_mask = hand_data['vis_mask']  # (n_hands, n_frames)
        n_frames = verts.shape[1]
        if frame_id < n_frames:
            for h in range(verts.shape[0]):
                if not vis_mask[h, frame_id]:
                    continue
                v_cv   = verts[h, frame_id]  # (n_verts, 3) — already in CV camera space
                uv, ok = _project_to_image(v_cv, intrinsics)
                right  = bool(is_right[h, frame_id])
                faces  = hand_data['faces_right'] if right else hand_data['faces_left']
                fill   = (255, 100,  50,  60) if right else ( 50, 150, 255,  60)
                edge   = (255, 100,  50, 200) if right else ( 50, 150, 255, 200)
                for tri in faces:
                    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                    if not (ok[a] and ok[b] and ok[c]):
                        continue
                    pts = [(float(uv[a, 0]), float(uv[a, 1])),
                           (float(uv[b, 0]), float(uv[b, 1])),
                           (float(uv[c, 0]), float(uv[c, 1]))]
                    draw.polygon(pts, fill=fill, outline=edge)

    # --- Object mesh ---
    obj_verts, obj_faces = _load_obj_geometry()
    if obj_verts is not None and POSES_FOLDER:
        pose_path = Path(POSES_FOLDER) / f"{frame_id:06d}.json"
        if pose_path.exists():
            with open(pose_path) as f:
                pd = json.load(f)
            qw, qx, qy, qz = pd['rotation']
            tx, ty, tz      = pd['translation']
            sx, sy, sz      = pd['scale']
            R = np.array([
                [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qw*qz,   2*qx*qz+2*qw*qy],
                [2*qx*qy+2*qw*qz,   1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qw*qx],
                [2*qx*qz-2*qw*qy,   2*qy*qz+2*qw*qx,   1-2*qx*qx-2*qy*qy],
            ], dtype=np.float64)
            v_cam = (R @ np.diag([sx, sy, sz]) @ obj_verts.T).T + np.array([tx, ty, tz])
            uv, ok = _project_to_image(v_cam, intrinsics)
            edge   = (80, 255, 80, 200)
            for tri in obj_faces:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                if not (ok[a] and ok[b] and ok[c]):
                    continue
                # Back-face cull: cross product z > 0 means facing away (CV convention)
                pa, pb, pc = v_cam[a], v_cam[b], v_cam[c]
                if np.cross(pb - pa, pc - pa)[2] > 0:
                    continue
                pts = [(float(uv[a, 0]), float(uv[a, 1])),
                       (float(uv[b, 0]), float(uv[b, 1])),
                       (float(uv[c, 0]), float(uv[c, 1]))]
                draw.polygon(pts, fill=None, outline=edge)

    result = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    buf = BytesIO()
    result.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')


@app.route("/api/info")
def get_info():
    with open(INTRINSICS_PATH) as f:
        intrinsics = json.load(f)
    return jsonify({
        "mesh_path":       MESH_PATH,
        "depth_folder":    DEPTH_FOLDER,
        "poses_folder":    POSES_FOLDER,
        "has_frames":      FRAMES_FOLDER is not None,
        "has_masks":       MASKS_FOLDER is not None,
        "has_hands":       HAND_MESH_PATH is not None,
        "intrinsics":      intrinsics,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D depth viewer")
    parser.add_argument("--mesh_path",       required=True)
    parser.add_argument("--depth_folder",    required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--poses_folder",    default=None)
    parser.add_argument("--frames_folder",   default=None)
    parser.add_argument("--masks_folder",    default=None)
    parser.add_argument("--hand_mesh_path", default=None)
    parser.add_argument("--port",            type=int, default=5000)
    args = parser.parse_args()

    MESH_PATH       = args.mesh_path
    DEPTH_FOLDER    = args.depth_folder
    INTRINSICS_PATH = args.intrinsics_path
    POSES_FOLDER    = args.poses_folder
    FRAMES_FOLDER   = args.frames_folder
    MASKS_FOLDER    = args.masks_folder
    HAND_MESH_PATH  = args.hand_mesh_path

    print(f"Serving at http://localhost:{args.port}")
    app.run(port=args.port, debug=False)
