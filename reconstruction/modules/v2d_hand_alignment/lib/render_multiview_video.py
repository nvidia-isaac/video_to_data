"""
Render a 4-panel video of object mesh + hand meshes:
  top-left:     original camera perspective
  top-right:    top-down view (centered on object)
  bottom-left:  left-side view
  bottom-right: right-side view

Coordinate conventions:
  - CV camera space: +X right, +Y down, +Z forward
  - GL (pyrender) space: +X right, +Y up, -Z forward
  - Conversion: flip Y and Z  →  gl = cv * [1, -1, -1]

Usage:
    python -m v2d.hand_alignment.lib.render_multiview_video \\
        --mesh      data/.../sam3d/mesh_scaled.obj \\
        --poses     data/.../poses_sam3d_moge \\
        --hand_mesh data/.../hand_mesh_traj_000300_depth_aligned.npz \\
        --intrinsics data/.../intrinsics_moge_stable.json \\
        --output    /tmp/multiview.mp4 \\
        --fps 30
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse
import glob
import json
import subprocess
import tempfile

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])   # CV → GL (homogeneous)


def cv_to_gl(T: np.ndarray) -> np.ndarray:
    """Convert 4×4 pose from CV convention to GL convention."""
    return _FLIP @ T @ _FLIP


def verts_cv_to_gl(v: np.ndarray) -> np.ndarray:
    """Flip (N,3) verts from CV to GL."""
    return v * np.array([1.0, -1.0, -1.0])


def load_pose_cv(path: str) -> np.ndarray:
    """Load object-to-camera 4×4 pose (CV convention) from JSON.
    Rotation is stored as [w, x, y, z]; scipy expects [x, y, z, w].
    """
    with open(path) as f:
        d = json.load(f)
    w, x, y, z = d['rotation']
    sx, sy, sz  = d['scale']
    R = Rotation.from_quat([x, y, z, w]).as_matrix() @ np.diag([sx, sy, sz])
    t = np.array(d['translation'], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def make_look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4×4 camera-to-world pose in GL convention (camera looks along -Z)."""
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    # GL: columns = [right, true_up, -fwd]
    R = np.stack([right, true_up, -fwd], axis=1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


# ---------------------------------------------------------------------------
# Intrinsics / camera setup
# ---------------------------------------------------------------------------

def load_intrinsics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def make_intrinsics_camera(intr: dict) -> pyrender.IntrinsicsCamera:
    return pyrender.IntrinsicsCamera(
        fx=intr['fx'], fy=intr['fy'],
        cx=intr['cx'], cy=intr['cy'],
        znear=0.01, zfar=10.0,
    )


def make_persp_camera(fov_deg: float = 50.0) -> pyrender.PerspectiveCamera:
    return pyrender.PerspectiveCamera(yfov=np.radians(fov_deg), znear=0.01, zfar=10.0)


# ---------------------------------------------------------------------------
# Scene building
# ---------------------------------------------------------------------------

def make_hand_trimesh(verts: np.ndarray, faces: np.ndarray, color_rgba) -> trimesh.Trimesh:
    """Create a trimesh for a single hand (verts already in render space)."""
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tm.visual.face_colors = color_rgba
    return tm


HAND_COLORS = [
    np.array([80, 140, 255, 200], dtype=np.uint8),   # left  – blue
    np.array([255, 100, 50,  200], dtype=np.uint8),   # right – orange
]


def build_scene(
    obj_mesh_gl: trimesh.Trimesh,
    obj_pose_gl: np.ndarray,
    hand_verts_list_gl: list[np.ndarray | None],
    faces_list: list[np.ndarray],
    camera: pyrender.Camera,
    camera_pose_gl: np.ndarray,
    panel_w: int,
    panel_h: int,
    transparent_bg: bool = False,
) -> tuple[pyrender.Scene, pyrender.OffscreenRenderer]:
    """Build a pyrender scene and return (scene, renderer)."""
    bg = np.array([0, 0, 0, 0], dtype=np.uint8) if transparent_bg else np.array([30, 30, 30, 255], dtype=np.uint8)
    scene = pyrender.Scene(
        ambient_light=np.array([0.4, 0.4, 0.4]),
        bg_color=bg,
    )

    # Object mesh
    pr_obj = pyrender.Mesh.from_trimesh(obj_mesh_gl, smooth=False)
    scene.add(pr_obj, pose=obj_pose_gl)

    # Hand meshes
    for h, (v_gl, faces) in enumerate(zip(hand_verts_list_gl, faces_list)):
        if v_gl is None:
            continue
        tm = make_hand_trimesh(v_gl, faces, HAND_COLORS[h % 2])
        pr_hand = pyrender.Mesh.from_trimesh(tm, smooth=False)
        scene.add(pr_hand)

    # Lights (two directional + one point to fill shadows)
    dl1 = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(dl1, pose=camera_pose_gl)
    dl2 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    scene.add(dl2, pose=np.eye(4))

    # Camera
    scene.add(camera, pose=camera_pose_gl)

    return scene


def render_scene(
    obj_mesh_gl: trimesh.Trimesh,
    obj_pose_gl: np.ndarray,
    hand_verts_list_gl: list,
    faces_list: list,
    camera: pyrender.Camera,
    camera_pose_gl: np.ndarray,
    renderer: pyrender.OffscreenRenderer,
    transparent_bg: bool = False,
) -> np.ndarray:
    """Build scene, render, return (H, W, 4) RGBA."""
    scene = build_scene(
        obj_mesh_gl, obj_pose_gl,
        hand_verts_list_gl, faces_list,
        camera, camera_pose_gl,
        renderer.viewport_width, renderer.viewport_height,
        transparent_bg=transparent_bg,
    )
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return color   # (H, W, 4) RGBA


# ---------------------------------------------------------------------------
# Fixed-view camera poses (GL object space)
# ---------------------------------------------------------------------------

def make_fixed_cameras_gl(obj_verts_gl: np.ndarray) -> dict[str, np.ndarray]:
    """
    Three fixed camera-to-world poses in GL object space.
    Object centroid is the look-at target.
    """
    ctr = obj_verts_gl.mean(axis=0)
    ext = obj_verts_gl.max(axis=0) - obj_verts_gl.min(axis=0)
    dist = float(np.linalg.norm(ext)) * 1.8  # camera distance

    # In GL object space: +Y is up, +X is right, -Z is scene depth
    cameras = {
        'top':   make_look_at(ctr + np.array([0,  dist, 0]), ctr, np.array([0, 0, -1])),
        'left':  make_look_at(ctr + np.array([-dist, 0, 0]), ctr, np.array([0, 1,  0])),
        'right': make_look_at(ctr + np.array([ dist, 0, 0]), ctr, np.array([0, 1,  0])),
    }
    return cameras


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def render_multiview_video(
    mesh_path: str,
    poses_dir: str,
    hand_mesh_path: str,
    intrinsics_path: str,
    output_path: str,
    fps: float = 30.0,
    panel_w: int = 388,
    panel_h: int = 516,
    start: int = 0,
    end: int | None = None,
    frames_folder: str | None = None,
) -> None:
    """Render a 4-panel multiview video of object mesh + hand meshes."""
    intr     = load_intrinsics(intrinsics_path)
    W, H     = panel_w, panel_h
    out_w    = W * 2
    out_h    = H * 2

    # --- Load object mesh (strip textures for EGL compatibility) ---
    obj_tm_raw = trimesh.load(mesh_path, force='mesh')
    obj_verts  = np.array(obj_tm_raw.vertices, dtype=np.float64)
    obj_faces  = np.array(obj_tm_raw.faces)
    OBJ_COLOR  = np.array([180, 180, 180, 255], dtype=np.uint8)

    def make_obj_tm(verts_gl):
        tm = trimesh.Trimesh(vertices=verts_gl, faces=obj_faces, process=False)
        tm.visual.face_colors = OBJ_COLOR
        return tm

    # Object mesh in GL object space (for fixed views)
    obj_verts_gl = verts_cv_to_gl(obj_verts)
    obj_tm_gl = make_obj_tm(obj_verts_gl)

    fov_cam = make_persp_camera(fov_deg=55.0)

    # --- Load hand mesh ---
    hand_data = np.load(hand_mesh_path, allow_pickle=True)
    verts_all  = hand_data['verts']    # (n_hands, n_frames, n_verts, 3)  CV camera space
    faces_r    = hand_data['faces_right']
    faces_l    = hand_data['faces_left']
    is_right   = hand_data['is_right']
    vis_mask   = hand_data.get('vis_mask', np.ones(is_right.shape))
    n_hands, n_frames = verts_all.shape[:2]

    # --- Load pose files ---
    pose_files = sorted(glob.glob(os.path.join(poses_dir, '*.json')))
    n_pose_files = len(pose_files)

    # --- RGB frames (optional underlay for camera view) ---
    frame_files = []
    if frames_folder:
        frame_files = sorted(
            glob.glob(os.path.join(frames_folder, '*.png')) +
            glob.glob(os.path.join(frames_folder, '*.jpg'))
        )
        print(f"RGB underlay: {len(frame_files)} frames from {frames_folder}")

    start_frame = start
    end_frame   = end if end is not None else min(n_frames, n_pose_files)

    # --- Fixed-view world frame: frame 0's camera space ---
    T_cam0_obj_cv = load_pose_cv(pose_files[start_frame])
    obj_verts_cam0_cv = (T_cam0_obj_cv[:3, :3] @ obj_verts.T).T + T_cam0_obj_cv[:3, 3]
    obj_verts_cam0_gl = verts_cv_to_gl(obj_verts_cam0_cv)
    obj_tm_cam0 = make_obj_tm(obj_verts_cam0_gl)

    # Fixed cameras centered on object in cam0 GL space
    fixed_cams_gl = make_fixed_cameras_gl(obj_verts_cam0_gl)

    print(f"Rendering {end_frame - start_frame} frames at {W}×{H} × 4 panels → {out_w}×{out_h}")
    print(f"Fixed views centred on frame-{start_frame} object pose")

    # --- Pyrender setup ---
    cam_renderer   = pyrender.OffscreenRenderer(W, H)
    fixed_renderer = pyrender.OffscreenRenderer(W, H)

    # Camera-view camera: intrinsics, scaled to panel size
    scale_x = W / intr.get('width',  776)
    scale_y = H / intr.get('height', 1032)
    cam_intr_scaled = pyrender.IntrinsicsCamera(
        fx=intr['fx'] * scale_x, fy=intr['fy'] * scale_y,
        cx=intr['cx'] * scale_x, cy=intr['cy'] * scale_y,
        znear=0.01, zfar=10.0,
    )
    cam_view_pose_gl = np.eye(4)

    # --- Write frames ---
    with tempfile.TemporaryDirectory() as tmp:
        for fid in range(start_frame, end_frame):
            T_cam_obj_cv = load_pose_cv(pose_files[fid])   # object→camera_f (CV)

            # == Panel 1: camera view ==
            obj_verts_cam_cv = (T_cam_obj_cv[:3, :3] @ obj_verts.T).T + T_cam_obj_cv[:3, 3]
            obj_verts_cam_gl = verts_cv_to_gl(obj_verts_cam_cv)
            obj_tm_cam = make_obj_tm(obj_verts_cam_gl)

            # Hands in GL camera space
            hand_cam_list = []
            hand_faces_cam = []
            for h in range(n_hands):
                if fid >= n_frames or vis_mask[h, fid] < 0.5:
                    hand_cam_list.append(None)
                    hand_faces_cam.append(faces_r)
                    continue
                v_cv = verts_all[h, fid].astype(np.float64)
                v_gl = verts_cv_to_gl(v_cv)
                hand_cam_list.append(v_gl)
                hand_faces_cam.append(faces_r if is_right[h, fid] > 0.5 else faces_l)

            rgba_cam = render_scene(
                obj_tm_cam, np.eye(4),
                hand_cam_list, hand_faces_cam,
                cam_intr_scaled, cam_view_pose_gl,
                cam_renderer,
                transparent_bg=bool(frame_files),
            )  # (H, W, 4)

            # Composite mesh over RGB underlay if available
            if frame_files and fid < len(frame_files):
                rgb_bg = np.array(Image.open(frame_files[fid]).convert('RGB').resize((W, H), Image.BILINEAR))
                alpha = rgba_cam[:, :, 3:4].astype(np.float32) / 255.0
                panel_cam = (rgb_bg * (1.0 - alpha) + rgba_cam[:, :, :3] * alpha).astype(np.uint8)
            else:
                panel_cam = rgba_cam[:, :, :3]

            # == Panels 2/3/4: fixed views ==
            obj_verts_fix_cv = (T_cam_obj_cv[:3, :3] @ obj_verts.T).T + T_cam_obj_cv[:3, 3]
            obj_tm_fix = make_obj_tm(verts_cv_to_gl(obj_verts_fix_cv))

            hand_fix_list = []
            hand_faces_fix = []
            for h in range(n_hands):
                if fid >= n_frames or vis_mask[h, fid] < 0.5:
                    hand_fix_list.append(None)
                    hand_faces_fix.append(faces_r)
                    continue
                v_gl = verts_cv_to_gl(verts_all[h, fid].astype(np.float64))
                hand_fix_list.append(v_gl)
                hand_faces_fix.append(faces_r if is_right[h, fid] > 0.5 else faces_l)

            panels_fixed = {}
            for view_name, cam_pose_gl in fixed_cams_gl.items():
                panels_fixed[view_name] = render_scene(
                    obj_tm_fix, np.eye(4),
                    hand_fix_list, hand_faces_fix,
                    fov_cam, cam_pose_gl,
                    fixed_renderer,
                )[:, :, :3]

            # == Assemble 2×2 grid ==
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            canvas[:H, :W]      = panel_cam               # top-left:     camera view
            canvas[:H, W:]      = panels_fixed['top']     # top-right:    top-down
            canvas[H:, :W]      = panels_fixed['left']    # bottom-left:  left side
            canvas[H:, W:]      = panels_fixed['right']   # bottom-right: right side

            # Labels
            img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img)
            for (tx, ty, label) in [
                (4, 4, "camera"),
                (W + 4, 4, "top-down"),
                (4, H + 4, "left"),
                (W + 4, H + 4, "right"),
            ]:
                draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0))
                draw.text((tx, ty), label, fill=(255, 255, 255))

            img.save(os.path.join(tmp, f'{fid:06d}.jpg'), quality=92)

            if fid % 50 == 0:
                print(f"  frame {fid}/{end_frame}")

        # --- Encode with ffmpeg ---
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(tmp, '%06d.jpg'),
            '-c:v', 'libx264',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path,
        ]
        subprocess.run(cmd, check=True)
        print(f"\nSaved → {output_path}")

    cam_renderer.delete()
    fixed_renderer.delete()


def main():
    parser = argparse.ArgumentParser(description="Render 4-panel multiview video")
    parser.add_argument('--mesh_path',       required=True, help='Object mesh OBJ')
    parser.add_argument('--poses_dir',       required=True, help='Folder of per-frame pose JSONs')
    parser.add_argument('--hand_mesh_path',  required=True, help='Aligned hand mesh NPZ')
    parser.add_argument('--intrinsics_path', required=True, help='Camera intrinsics JSON')
    parser.add_argument('--frames_folder',   default=None,  help='RGB frame images folder (for camera-view underlay)')
    parser.add_argument('--output_path',     required=True, help='Output MP4 path')
    parser.add_argument('--fps',         type=float, default=30.0)
    parser.add_argument('--panel_w',     type=int,   default=388, help='Width of each panel')
    parser.add_argument('--panel_h',     type=int,   default=516, help='Height of each panel')
    parser.add_argument('--start',       type=int,   default=0)
    parser.add_argument('--end',         type=int,   default=None)
    args = parser.parse_args()

    render_multiview_video(
        mesh_path=args.mesh_path,
        poses_dir=args.poses_dir,
        hand_mesh_path=args.hand_mesh_path,
        intrinsics_path=args.intrinsics_path,
        output_path=args.output_path,
        fps=args.fps,
        panel_w=args.panel_w,
        panel_h=args.panel_h,
        start=args.start,
        end=args.end,
        frames_folder=args.frames_folder,
    )


if __name__ == '__main__':
    main()
