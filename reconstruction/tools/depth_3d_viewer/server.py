"""
3D depth viewer — Flask backend

Serves an interactive Three.js viewer to inspect unprojected depth point clouds
and FP mesh poses for a given session.

Usage (from reconstruction/):
    python tools/depth_3d_viewer/server.py \
        --mesh_path       data/objects/electric_drill_toy/mesh/textured_mesh.obj \
        --depth_folder    data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/depth_moge_aligned \
        --intrinsics_path data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/intrinsics_moge_stable.json \
        --poses_folder    data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/poses_moge_aligned_smoothed \
        [--frames_folder  data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/frames] \
        [--masks_folder   data/objects/electric_drill_toy/sessions/Session_20260310_133326_f50/outputs/masks/1] \
        [--port 5000]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

# ---------------------------------------------------------------------------
# Config (populated from CLI args)
# ---------------------------------------------------------------------------
MESH_PATH      = None
DEPTH_FOLDER   = None
INTRINSICS_PATH = None
POSES_FOLDER   = None
FRAMES_FOLDER  = None
MASKS_FOLDER   = None

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
    parser.add_argument("--port",            type=int, default=5000)
    args = parser.parse_args()

    MESH_PATH       = args.mesh_path
    DEPTH_FOLDER    = args.depth_folder
    INTRINSICS_PATH = args.intrinsics_path
    POSES_FOLDER    = args.poses_folder
    FRAMES_FOLDER   = args.frames_folder
    MASKS_FOLDER    = args.masks_folder

    print(f"Serving at http://localhost:{args.port}")
    app.run(port=args.port, debug=False)
