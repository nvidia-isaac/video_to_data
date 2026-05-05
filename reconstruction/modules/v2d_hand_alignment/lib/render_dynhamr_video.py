"""
Render DynHaMR world_results.npz hand meshes in a 2×2 grid:

  [camera view + image underlay]  |  [world top-down view]
  [world side view]               |  [world front view]

World views use fixed cameras aimed at the hand centroid, making it easy to
inspect 3D trajectory quality independent of the camera projection.

Usage:
    python -m v2d.hand_alignment.lib.render_dynhamr_video \\
        --world_results /path/to/world_results.npz \\
        --frames_folder /path/to/frames/ \\
        --mano_assets_root /path/to/dir_with_MANO_RIGHT.pkl \\
        --output /tmp/dynhamr_check.mp4 \\
        [--use_trans_aligned / --no_trans_aligned]
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import pyrender
import torch
import json
import trimesh
from manotorch.manolayer import ManoLayer
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


HAND_COLORS = [
    ( 60, 160, 255, 255),   # hand 0 — vivid blue
    (255, 120,  30, 255),   # hand 1 — vivid orange
]
OBJ_COLOR_CAM   = ( 40, 230,  40, 255)  # vivid green
OBJ_COLOR_WORLD = ( 40, 210,  40, 255)  # vivid green

# Camera panel: background dimming factor where mesh is rendered (0=black, 1=full bright)
_BG_DARKEN = 0.45

_CV_TO_GL = np.array([1.0, -1.0, -1.0])


def _lookat_pose(eye: np.ndarray, target: np.ndarray,
                 up: np.ndarray = np.array([0., 1., 0.])) -> np.ndarray:
    """4×4 camera-to-world pose for pyrender (camera looks along -Z, Y-up GL convention)."""
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        up = np.array([0., 0., 1.])
        x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def _add_label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([4, 4, 4 + len(text) * 12, 26], fill=(0, 0, 0, 160))
    draw.text((6, 6), text, fill=(255, 255, 255), font=font)
    return img


def _quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1-2*qy*qy-2*qz*qz,  2*qx*qy-2*qw*qz,   2*qx*qz+2*qw*qy],
        [2*qx*qy+2*qw*qz,    1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qw*qx],
        [2*qx*qz-2*qw*qy,    2*qy*qz+2*qw*qx,   1-2*qx*qx-2*qy*qy],
    ], dtype=np.float64)


def render_dynhamr_video(
    world_results_path: str,
    frames_folder: str,
    mano_assets_root: str,
    output_path: str,
    fps: float = 25.0,
    start: int = 0,
    end: int | None = None,
    use_trans_aligned: bool = True,
    object_mesh_path: str | None = None,
    object_poses_dir: str | None = None,
    intrinsics_path: str | None = None,
) -> None:
    wr = np.load(world_results_path, allow_pickle=True)

    root_orient = wr['root_orient'].astype(np.float32)   # (B, T, 3)
    pose_body   = wr['pose_body'].astype(np.float32)     # (B, T, 15, 3)
    betas       = wr['betas'].astype(np.float32)         # (B, 10)
    cam_R       = wr['cam_R'].astype(np.float64)         # (B, T, 3, 3)
    cam_t       = wr['cam_t'].astype(np.float64)         # (B, T, 3)
    is_right    = wr['is_right']                         # (B, T)

    if intrinsics_path is not None:
        with open(intrinsics_path) as f:
            intr = json.load(f)
        fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
        print(f"  Using intrinsics override (file): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    elif 'intrins_aligned' in wr:
        fx, fy, cx, cy = [float(x) for x in wr['intrins_aligned']]
        print(f"  Using intrins_aligned: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    else:
        fx, fy, cx, cy = [float(x) for x in wr['intrins']]

    if use_trans_aligned and 'trans_aligned' in wr:
        trans       = wr['trans_aligned'].astype(np.float32)
        trans_label = 'trans_aligned'
    else:
        trans       = wr['trans'].astype(np.float32)
        trans_label = 'trans'

    B, T = root_orient.shape[:2]
    is_right_track = is_right.mean(axis=1) > 0.5

    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="right",
        center_idx=None,
        mano_assets_root=mano_assets_root,
    )
    faces_right = mano_layer.th_faces.numpy()
    faces_left  = faces_right[:, [0, 2, 1]]

    print(f"Running MANO FK for {B} hands × {T} frames  (using {trans_label})...")

    # all_verts_cam_gl  — CV cam-space, GL-flipped (for camera panel)
    # all_verts_world_gl — metric world-space, GL-flipped (for world panels)
    all_verts_cam_gl   = np.zeros((B, T, 778, 3), dtype=np.float32)
    all_verts_world_gl = np.zeros((B, T, 778, 3), dtype=np.float32)

    for h in range(B):
        left = not is_right_track[h]

        finger_pose = pose_body[h].reshape(T, 45)
        hand_pose   = torch.from_numpy(
            np.concatenate([root_orient[h], finger_pose], axis=1)
        )
        hand_betas  = torch.from_numpy(np.tile(betas[h], (T, 1)))
        mano_out    = mano_layer(hand_pose, hand_betas)
        verts_local = mano_out.verts.detach().numpy()  # (T, 778, 3)

        v_world = verts_local + trans[h, :, None, :]  # DynHaMR world units (no world_scale)
        if left:
            v_world[:, :, 0] = -v_world[:, :, 0]

        all_verts_world_gl[h] = v_world * _CV_TO_GL

        for f in range(T):
            v_cam = (cam_R[h, f] @ v_world[f].T).T + cam_t[h, f]
            all_verts_cam_gl[h, f] = (v_cam * _CV_TO_GL).astype(np.float32)

        print(f"  hand {h} ({'right' if is_right_track[h] else 'left'}): "
              f"world GL centroid frame 0 = {all_verts_world_gl[h, 0].mean(0)}")

    # Object mesh — pre-compute per-frame vertices in cam GL and world GL space
    obj_verts_base  = None   # (N, 3) float64 — mesh in local OBJ space
    obj_faces       = None   # (M, 3) int
    obj_cam_gl      = [None] * T   # list of (N,3) float32 | None
    obj_world_gl    = [None] * T   # list of (N,3) float32 | None

    if object_mesh_path and object_poses_dir:
        tm_obj = trimesh.load(object_mesh_path, force='mesh', process=False)
        obj_verts_base = np.array(tm_obj.vertices, dtype=np.float64)
        obj_faces      = np.array(tm_obj.faces,    dtype=np.int32)
        print(f"Object mesh: {len(obj_verts_base)} verts, {len(obj_faces)} faces")

        for f in range(T):
            pose_path = os.path.join(object_poses_dir, f"{f:06d}.json")
            if not os.path.exists(pose_path):
                continue
            with open(pose_path) as fp:
                pd = json.load(fp)
            R  = _quat_to_rot(*pd['rotation'])
            t  = np.array(pd['translation'], dtype=np.float64)
            s  = np.array(pd['scale'],       dtype=np.float64)
            RS = R @ np.diag(s)
            v_cam_cv = (RS @ obj_verts_base.T).T + t          # CV camera space
            obj_cam_gl[f] = (v_cam_cv * _CV_TO_GL).astype(np.float32)

            # Invert DynHaMR cam transform to get world space (use hand-0's cam_R/cam_t)
            v_world_cv = (cam_R[0, f].T @ (v_cam_cv - cam_t[0, f]).T).T
            obj_world_gl[f] = (v_world_cv * _CV_TO_GL).astype(np.float32)

    # World-space fixed camera setup — look at hand centroid
    scene_center = all_verts_world_gl[:, 0, :, :].reshape(-1, 3).mean(0)
    r = float(np.linalg.norm(scene_center)) * 0.5 + 0.4

    # Three orbiting cameras (GL world space: Y-up, -Z forward)
    world_cam_poses = [
        _lookat_pose(scene_center + np.array([0.,  r,  0.]),  scene_center,
                     up=np.array([0., 0., -1.])),   # top-down
        _lookat_pose(scene_center + np.array([ r,  0.,  0.]),  scene_center),  # right side
        _lookat_pose(scene_center + np.array([0.,  0.,  r]),  scene_center),   # front (behind camera)
    ]
    panel_labels = [f'camera ({trans_label})', 'world top', 'world side', 'world front']

    # Load frames
    frame_files = sorted(
        glob.glob(os.path.join(frames_folder, '*.png')) +
        glob.glob(os.path.join(frames_folder, '*.jpg'))
    )
    if not frame_files:
        raise FileNotFoundError(f"No images found in {frames_folder}")

    W_full, H_full = Image.open(frame_files[0]).size
    pw = (W_full // 2) & ~1    # panel width, forced even
    ph = (H_full // 2) & ~1    # panel height, forced even

    print(f"Frames {W_full}×{H_full} → panel {pw}×{ph}, output {pw*2}×{ph*2}, "
          f"n_frames={T}")

    # Camera-view renderer (scaled intrinsics)
    scale_x = pw / (cx * 2)
    scale_y = ph / (cy * 2)
    cam_cv = pyrender.IntrinsicsCamera(
        fx=fx * scale_x, fy=fy * scale_y,
        cx=cx * scale_x, cy=cy * scale_y,
        znear=0.01, zfar=20.0,
    )
    world_cam = pyrender.PerspectiveCamera(yfov=np.radians(60.0), znear=0.01, zfar=20.0)
    renderer  = pyrender.OffscreenRenderer(viewport_width=pw, viewport_height=ph)

    n_frames = min(end if end is not None else T, len(frame_files))

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in tqdm(range(start, n_frames), desc="rendering", unit="frame", ncols=80):
            panels: list[Image.Image] = []

            # ---- Panel 0: camera view with image underlay ----
            scene0 = pyrender.Scene(bg_color=[0., 0., 0., 0.], ambient_light=[0.4, 0.4, 0.4])
            scene0.add(cam_cv, pose=np.eye(4))
            scene0.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=np.eye(4))
            for h in range(B):
                faces = faces_right if is_right_track[h] else faces_left
                color = HAND_COLORS[h % len(HAND_COLORS)]
                tm = trimesh.Trimesh(vertices=all_verts_cam_gl[h, f], faces=faces, process=False)
                tm.visual.face_colors = np.array(color, dtype=np.uint8)
                scene0.add(pyrender.Mesh.from_trimesh(tm, smooth=False))
            if obj_faces is not None and obj_cam_gl[f] is not None:
                tm_o = trimesh.Trimesh(vertices=obj_cam_gl[f], faces=obj_faces, process=False)
                tm_o.visual.face_colors = np.array(OBJ_COLOR_CAM, dtype=np.uint8)
                scene0.add(pyrender.Mesh.from_trimesh(tm_o, smooth=False))
            rgba0, _ = renderer.render(scene0, flags=pyrender.RenderFlags.RGBA)
            bg_rgb  = np.array(Image.open(frame_files[f]).convert('RGB').resize((pw, ph)),
                               dtype=np.float32)
            ov      = np.array(Image.fromarray(rgba0, 'RGBA'), dtype=np.float32)
            mask    = ov[:, :, 3:4] / 255.0                     # coverage [0,1]
            bg_out  = bg_rgb * (1.0 - mask * (1.0 - _BG_DARKEN))  # dim bg where mesh is
            result  = (mask * ov[:, :, :3] + (1.0 - mask) * bg_out).clip(0, 255).astype(np.uint8)
            panels.append(Image.fromarray(result))

            # ---- Panels 1-3: world-space views (gray background) ----
            for world_pose in world_cam_poses:
                scene_w = pyrender.Scene(bg_color=[0.78, 0.78, 0.78, 1.0],
                                         ambient_light=[0.5, 0.5, 0.5])
                scene_w.add(world_cam, pose=world_pose)
                scene_w.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0),
                            pose=world_pose)
                for h in range(B):
                    faces = faces_right if is_right_track[h] else faces_left
                    color = HAND_COLORS[h % len(HAND_COLORS)]
                    tm = trimesh.Trimesh(
                        vertices=all_verts_world_gl[h, f], faces=faces, process=False
                    )
                    tm.visual.face_colors = np.array(color, dtype=np.uint8)
                    scene_w.add(pyrender.Mesh.from_trimesh(tm, smooth=False))
                if obj_faces is not None and obj_world_gl[f] is not None:
                    tm_o = trimesh.Trimesh(
                        vertices=obj_world_gl[f], faces=obj_faces, process=False
                    )
                    tm_o.visual.face_colors = np.array(OBJ_COLOR_WORLD, dtype=np.uint8)
                    scene_w.add(pyrender.Mesh.from_trimesh(tm_o, smooth=False))
                rgb_w, _ = renderer.render(scene_w)
                panels.append(Image.fromarray(rgb_w, 'RGB'))

            # ---- Labels ----
            for i, (panel, label) in enumerate(zip(panels, panel_labels)):
                panels[i] = _add_label(panel, label)

            # ---- 2×2 grid ----
            grid = Image.new('RGB', (pw * 2, ph * 2))
            for panel, pos in zip(panels, [(0, 0), (pw, 0), (0, ph), (pw, ph)]):
                grid.paste(panel, pos)
            grid.save(os.path.join(tmpdir, f'{f:06d}.png'))


        renderer.delete()
        subprocess.run([
            'ffmpeg', '-y', '-r', str(fps),
            '-i', os.path.join(tmpdir, '%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', output_path,
        ], check=True)

    print(f'Saved → {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Render DynHaMR hands in a 2×2 grid (camera + 3 world views).'
    )
    parser.add_argument('--world_results',    required=True)
    parser.add_argument('--frames_folder',    required=True)
    parser.add_argument('--mano_assets_root', required=True)
    parser.add_argument('--output',           required=True)
    parser.add_argument('--fps',   type=float, default=25.0)
    parser.add_argument('--start', type=int,   default=0)
    parser.add_argument('--end',   type=int,   default=None)
    parser.add_argument('--use_trans_aligned',  dest='use_trans_aligned',
                        action='store_true',  default=True)
    parser.add_argument('--no_trans_aligned',   dest='use_trans_aligned',
                        action='store_false')
    parser.add_argument('--object_mesh_path',  default=None)
    parser.add_argument('--object_poses_dir',  default=None)
    parser.add_argument('--intrinsics_path',   default=None,
                        help='Optional JSON {fx,fy,cx,cy} to override world_results intrinsics.')
    args = parser.parse_args()

    render_dynhamr_video(
        world_results_path = args.world_results,
        frames_folder      = args.frames_folder,
        mano_assets_root   = args.mano_assets_root,
        output_path        = args.output,
        fps                = args.fps,
        start              = args.start,
        end                = args.end,
        use_trans_aligned  = args.use_trans_aligned,
        object_mesh_path   = args.object_mesh_path,
        object_poses_dir   = args.object_poses_dir,
        intrinsics_path    = args.intrinsics_path,
    )


if __name__ == '__main__':
    main()
