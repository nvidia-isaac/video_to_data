"""
Render a video with MANO FK hands projected onto the camera view.

Loads mano_params_*.npz (output of recover_mano_params), runs forward
kinematics entirely in numpy/scipy, and renders the resulting hand meshes
overlaid on the original frames using pyrender.

This is a diagnostic tool: if the rendered hands align with the real hands in
the video, the recovered MANO parameters are correct.

Usage:
    python -m v2d.hand_alignment.lib.render_mano_params_video \\
        --mano_params /data/hand_mesh/mano_params_moge.npz \\
        --intrinsics  /data/intrinsics_moge_stable.json \\
        --mano_model_dir /data/mano \\
        --output      /tmp/mano_check.mp4 \\
        --frames_folder /data/frames          # optional video underlay
        --mesh         /data/mesh_scaled.obj  # optional object overlay
        --poses_dir    /data/poses_moge_smoothed  # required if --mesh given
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import pickle
import pyrender
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Minimal numpy MANO LBS
# ---------------------------------------------------------------------------

def _rodrigues(r: np.ndarray) -> np.ndarray:
    """(3,) axis-angle → (3,3) rotation matrix."""
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        return np.eye(3)
    n = r / theta
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _load_mano_model(pkl_path: str) -> dict:
    """Load and cache a MANO pkl file, converting sparse J_regressor."""
    # Stub out chumpy so pkl files load without it installed.
    import sys, types
    if 'chumpy' not in sys.modules:
        class _Ch:
            """Minimal chumpy array stub — lazily resolves Select/Ch to numpy."""
            def __new__(cls, *args, **kwargs):
                return object.__new__(cls)
            def __init__(self, x=None, *args, **kwargs):
                self._state = {'x': np.asarray(x)} if x is not None else {}
            def __setstate__(self, state):
                self._state = state if isinstance(state, dict) else {'x': state}
            def _resolve(self) -> np.ndarray:
                state = self._state
                if 'x' in state:
                    v = state['x']
                    return v._resolve() if isinstance(v, _Ch) else np.asarray(v)
                if 'a' in state and 'idxs' in state:
                    a = state['a']
                    a = a._resolve() if isinstance(a, _Ch) else np.asarray(a)
                    idxs = np.asarray(state['idxs'])
                    result = a.flatten()[idxs]
                    ps = state.get('preferred_shape')
                    return result.reshape(ps) if ps is not None else result
                return np.array([])
            def __array__(self, dtype=None):
                return np.asarray(self._resolve(), dtype=dtype)
            @property
            def r(self):
                return self._resolve()
        class _ChumMod(types.ModuleType):
            def __getattr__(self, name: str):
                return _Ch
        stub = _ChumMod('chumpy')
        stub.__path__ = []
        stub.Ch = _Ch
        sys.modules['chumpy'] = stub
        for _sub in ['reordering', 'utils', 'ch', 'logic']:
            _m = _ChumMod(f'chumpy.{_sub}')
            sys.modules[f'chumpy.{_sub}'] = _m
            setattr(stub, _sub, _m)
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
    model = {
        'v_template':    np.array(raw['v_template'],    dtype=np.float64),  # (778, 3)
        'shapedirs':     np.array(raw['shapedirs'],     dtype=np.float64),  # (778, 3, 10)
        'posedirs':      np.array(raw['posedirs'],      dtype=np.float64),  # (778, 3, 135)
        'weights':       np.array(raw['weights'],       dtype=np.float64),  # (778, 16)
        'hands_mean':    np.array(raw['hands_mean'],    dtype=np.float64),  # (45,)
        'faces':         np.array(raw['f'],             dtype=np.int32),    # (1538, 3)
        'parents':       np.array(raw['kintree_table'][0], dtype=np.int32), # (16,)
    }
    # J_regressor is a sparse scipy matrix
    jr = raw['J_regressor']
    model['J_regressor'] = np.array(jr.todense(), dtype=np.float64)  # (16, 778)
    return model


def mano_forward(
    model: dict,
    global_orient: np.ndarray,   # (3,) axis-angle in camera space
    hand_pose: np.ndarray,        # (45,) axis-angle — already in model space (no mean added)
    betas: np.ndarray,            # (10,)
    transl: np.ndarray,           # (3,) camera space
) -> np.ndarray:
    """Returns posed vertices (778, 3) in camera space."""
    full_pose = np.concatenate([global_orient, hand_pose])  # (48,)

    # Shape blend shapes: v_shaped = v_template + S @ betas
    v_shaped = (model['v_template']
                + np.einsum('ijk,k->ij', model['shapedirs'], betas))  # (778, 3)

    # Joint locations in rest pose
    J = model['J_regressor'] @ v_shaped  # (16, 3)

    # Per-joint rotation matrices
    R = np.stack([_rodrigues(full_pose[3*i:3*i+3]) for i in range(16)])  # (16, 3, 3)

    # Pose blend shapes
    pose_feature = (R[1:] - np.eye(3)).reshape(-1)  # (135,)
    v_posed = v_shaped + np.einsum('ijk,k->ij', model['posedirs'], pose_feature)  # (778, 3)

    # Global joint transforms via forward kinematics
    G = np.zeros((16, 4, 4))
    parents = model['parents']
    for k in range(16):
        local = np.eye(4)
        local[:3, :3] = R[k]
        local[:3, 3] = J[k] if k == 0 else J[k] - J[parents[k]]
        G[k] = G[parents[k]] @ local if k > 0 else local

    # Subtract rest-pose joint offset so transform is relative
    G_final = np.zeros((16, 4, 4))
    for k in range(16):
        offset = np.eye(4)
        offset[:3, 3] = -J[k]
        G_final[k] = G[k] @ offset

    # LBS: weighted sum of transforms
    T = np.einsum('vk,kij->vij', model['weights'], G_final)  # (778, 4, 4)
    v_homo = np.concatenate([v_posed, np.ones((len(v_posed), 1))], axis=1)  # (778, 4)
    v_out = np.einsum('vij,vj->vi', T, v_homo)[:, :3]  # (778, 3)

    return (v_out + transl).astype(np.float32)


# ---------------------------------------------------------------------------
# Rendering helpers (same conventions as render_multiview_video.py)
# ---------------------------------------------------------------------------

_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])   # CV → GL


def _cv_to_gl(v: np.ndarray) -> np.ndarray:
    """Flip (N,3) verts from OpenCV to OpenGL space."""
    return v * np.array([1.0, -1.0, -1.0])


def _make_intrinsics_camera(intr: dict, scale_x: float = 1.0, scale_y: float = 1.0):
    return pyrender.IntrinsicsCamera(
        fx=intr['fx'] * scale_x, fy=intr['fy'] * scale_y,
        cx=intr['cx'] * scale_x, cy=intr['cy'] * scale_y,
        znear=0.01, zfar=10.0,
    )


def _make_hand_mesh(verts_cv: np.ndarray, faces: np.ndarray,
                    color: tuple) -> trimesh.Trimesh:
    tm = trimesh.Trimesh(vertices=_cv_to_gl(verts_cv), faces=faces, process=False)
    tm.visual.face_colors = np.array(list(color) + [200], dtype=np.uint8)
    return tm


HAND_COLORS = [
    (100, 180, 255),   # hand 0 — blue
    (255, 160,  80),   # hand 1 — orange
]


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def render_mano_params_video(
    mano_params_path: str,
    intrinsics_path: str,
    mano_model_dir: str,
    output_path: str,
    fps: float = 25.0,
    frames_folder: str | None = None,
    mesh_path: str | None = None,
    poses_dir: str | None = None,
    panel_w: int = 388,
    panel_h: int = 516,
    start: int = 0,
    end: int | None = None,
) -> None:
    with open(intrinsics_path) as f:
        intr = json.load(f)
    W, H = panel_w, panel_h
    scale_x = W / intr.get('width',  W)
    scale_y = H / intr.get('height', H)

    params = np.load(mano_params_path, allow_pickle=True)
    global_orient = params['global_orient'].astype(np.float64)  # (B, T, 3)
    transl        = params['transl'].astype(np.float64)          # (B, T, 3)
    hand_pose     = params['hand_pose'].astype(np.float64)       # (B, T, 45)
    betas         = params['betas'].astype(np.float64)           # (B, 10)
    is_right      = params['is_right']                           # (B, T)

    B, T = global_orient.shape[:2]
    is_right_track = is_right.mean(axis=1) > 0.5

    # Load MANO models (right + left)
    models = {
        'right': _load_mano_model(os.path.join(mano_model_dir, 'MANO_RIGHT.pkl')),
        'left':  _load_mano_model(os.path.join(mano_model_dir, 'MANO_LEFT.pkl')),
    }

    # Optional object mesh
    obj_faces = obj_verts_gl = None
    pose_files = []
    if mesh_path and poses_dir:
        obj_tm = trimesh.load(mesh_path, force='mesh')
        obj_faces  = np.array(obj_tm.faces)
        obj_verts  = np.array(obj_tm.vertices)
        pose_files = sorted(glob.glob(os.path.join(poses_dir, '*.json')))

    # Optional video underlay frames
    frame_files = []
    if frames_folder:
        frame_files = sorted(
            glob.glob(os.path.join(frames_folder, '*.png')) +
            glob.glob(os.path.join(frames_folder, '*.jpg'))
        )

    n_frames = end if end is not None else T
    if pose_files:
        n_frames = min(n_frames, len(pose_files))

    cam = _make_intrinsics_camera(intr, scale_x, scale_y)
    # Identity pose: pyrender GL camera looks down -Z; CV→GL flip puts CV Z+ at GL Z-.
    cam_pose_gl = np.eye(4)

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in range(start, n_frames):
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
            scene.add(cam, pose=cam_pose_gl)
            scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.0),
                      pose=cam_pose_gl)

            # Object
            if pose_files and obj_faces is not None:
                with open(pose_files[f]) as fh:
                    pd = json.load(fh)
                w, x, y, z = pd['rotation']
                R_obj = Rotation.from_quat([x, y, z, w]).as_matrix()
                t_obj = np.array(pd['translation'])
                ov = (R_obj @ obj_verts.T).T + t_obj
                ov_gl = _cv_to_gl(ov)
                obj_tm = trimesh.Trimesh(vertices=ov_gl, faces=obj_faces, process=False)
                obj_tm.visual.face_colors = [180, 180, 180, 220]
                scene.add(pyrender.Mesh.from_trimesh(obj_tm, smooth=False))

            # MANO hands
            for h in range(B):
                side = 'right' if is_right_track[h] else 'left'
                verts_cv = mano_forward(
                    models[side],
                    global_orient[h, f],
                    hand_pose[h, f],
                    betas[h],
                    transl[h, f],
                )
                tm = _make_hand_mesh(verts_cv, models[side]['faces'], HAND_COLORS[h % 2])
                scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))

            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            render_img = Image.fromarray(color, 'RGBA')

            # Compose over video frame if available
            if frame_files and f < len(frame_files):
                bg = Image.open(frame_files[f]).convert('RGBA').resize((W, H))
                out_img = Image.alpha_composite(bg, render_img).convert('RGB')
            else:
                out_img = render_img.convert('RGB')

            out_img.save(os.path.join(tmpdir, f'{f:06d}.png'))

        renderer.delete()

        cmd = [
            'ffmpeg', '-y', '-r', str(fps),
            '-i', os.path.join(tmpdir, '%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '18', output_path,
        ]
        subprocess.run(cmd, check=True)

    print(f'Saved → {output_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Render MANO FK hand mesh projected onto camera view.'
    )
    parser.add_argument('--mano_params',    required=True,
                        help='Path to mano_params_*.npz')
    parser.add_argument('--intrinsics',     required=True,
                        help='Camera intrinsics JSON {fx,fy,cx,cy,width,height}')
    parser.add_argument('--mano_model_dir', required=True,
                        help='Directory containing MANO_RIGHT.pkl and MANO_LEFT.pkl')
    parser.add_argument('--output',         required=True,
                        help='Output video path (.mp4)')
    parser.add_argument('--fps',            type=float, default=25.0)
    parser.add_argument('--frames_folder',  default=None,
                        help='Folder of video frames to use as background')
    parser.add_argument('--mesh',           default=None,
                        help='Object mesh .obj to overlay')
    parser.add_argument('--poses_dir',      default=None,
                        help='Per-frame object pose JSONs (required with --mesh)')
    parser.add_argument('--panel_w',        type=int, default=388)
    parser.add_argument('--panel_h',        type=int, default=516)
    parser.add_argument('--start',          type=int, default=0)
    parser.add_argument('--end',            type=int, default=None)
    args = parser.parse_args()

    render_mano_params_video(
        mano_params_path = args.mano_params,
        intrinsics_path  = args.intrinsics,
        mano_model_dir   = args.mano_model_dir,
        output_path      = args.output,
        fps              = args.fps,
        frames_folder    = args.frames_folder,
        mesh_path        = args.mesh,
        poses_dir        = args.poses_dir,
        panel_w          = args.panel_w,
        panel_h          = args.panel_h,
        start            = args.start,
        end              = args.end,
    )


if __name__ == '__main__':
    main()
