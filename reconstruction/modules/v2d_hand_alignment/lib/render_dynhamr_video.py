"""
Render DynHaMR world_results.npz hand meshes overlaid on camera frames.

Uses manotorch (same as mano.ipynb) to run MANO FK directly from DynHaMR
params, then applies the world→camera transform (cam_R/cam_t) and projects
using the intrinsics stored in world_results.npz.

This is a direct verification of DynHaMR hand quality, bypassing the
alignment pipeline. If hands look correct here but contorted after alignment,
the distortion is introduced by the alignment pipeline's reproject_intrinsics.

Usage:
    python -m v2d.hand_alignment.lib.render_dynhamr_video \\
        --world_results /path/to/world_results.npz \\
        --frames_folder /path/to/frames/ \\
        --mano_assets_root /path/to/dir_with_MANO_RIGHT.pkl \\
        --output /tmp/dynhamr_check.mp4
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
import trimesh
from manotorch.manolayer import ManoLayer
from PIL import Image


HAND_COLORS = [
    (100, 180, 255, 200),   # hand 0 — blue
    (255, 160,  80, 200),   # hand 1 — orange
]

_CV_TO_GL = np.array([1.0, -1.0, -1.0])


def render_dynhamr_video(
    world_results_path: str,
    frames_folder: str,
    mano_assets_root: str,
    output_path: str,
    fps: float = 25.0,
    start: int = 0,
    end: int | None = None,
) -> None:
    wr = np.load(world_results_path, allow_pickle=True)

    root_orient = wr['root_orient'].astype(np.float32)   # (B, T, 3)
    pose_body   = wr['pose_body'].astype(np.float32)     # (B, T, 15, 3)
    trans_key   = 'trans_aligned' if 'trans_aligned' in wr else 'trans'
    trans       = wr[trans_key].astype(np.float32)       # (B, T, 3)
    betas       = wr['betas'].astype(np.float32)         # (B, 10)
    cam_R       = wr['cam_R'].astype(np.float64)         # (B, T, 3, 3)
    cam_t       = wr['cam_t'].astype(np.float64)         # (B, T, 3)
    world_scale = float(wr['world_scale'].flat[0])
    is_right    = wr['is_right']                         # (B, T)
    fx, fy, cx, cy = [float(x) for x in wr['intrins']]

    B, T = root_orient.shape[:2]
    is_right_track = is_right.mean(axis=1) > 0.5

    # Single right-hand ManoLayer — matches notebook exactly.
    # For left hands: flip x of output verts + reverse face winding.
    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="right",
        center_idx=None,
        flat_hand_mean=True,    # DynHaMR pose_body is full axis-angle; don't add hands_mean
        mano_assets_root=mano_assets_root,
    )
    faces_right = mano_layer.th_faces.numpy()           # (F, 3)
    faces_left  = faces_right[:, [0, 2, 1]]             # reversed winding

    # Pre-compute all verts in camera space.
    print(f"Running manotorch MANO FK for {B} hands × {T} frames...")
    all_verts_cam = np.zeros((B, T, 778, 3), dtype=np.float32)

    for h in range(B):
        left = not is_right_track[h]

        # hand_pose: (T, 48) = root_orient (T,3) + pose_body (T,45)
        finger_pose = pose_body[h].reshape(T, 45)
        hand_pose   = torch.from_numpy(
            np.concatenate([root_orient[h], finger_pose], axis=1)  # (T, 48)
        )
        hand_betas = torch.from_numpy(
            np.tile(betas[h], (T, 1))  # (T, 10)
        )

        mano_out = mano_layer(hand_pose, hand_betas)
        verts_local = mano_out.verts.detach().numpy()  # (T, 778, 3) — local, no trans

        # Add world translation, scale to metric
        v = verts_local + trans[h, :, None, :]   # (T, 778, 3)
        v = v * world_scale

        if left:
            v[:, :, 0] = -v[:, :, 0]

        # Apply per-frame cam_R, cam_t → camera space
        for f in range(T):
            v_f = (cam_R[h, f] @ v[f].T).T + cam_t[h, f]
            all_verts_cam[h, f] = v_f.astype(np.float32)

        print(f"  hand {h} ({'right' if is_right_track[h] else 'left'}): "
              f"mean cam-space pos frame 0 = {all_verts_cam[h, 0].mean(0)}")

    # Load frames
    frame_files = sorted(
        glob.glob(os.path.join(frames_folder, '*.png')) +
        glob.glob(os.path.join(frames_folder, '*.jpg'))
    )
    if not frame_files:
        raise FileNotFoundError(f"No images found in {frames_folder}")

    W, H = Image.open(frame_files[0]).size
    print(f"Frame size: {W}×{H}  intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # Scale intrinsics if the stored cx/cy don't match the actual frame dimensions
    scale_x = W / (cx * 2)
    scale_y = H / (cy * 2)

    cam = pyrender.IntrinsicsCamera(
        fx=fx * scale_x, fy=fy * scale_y,
        cx=cx * scale_x, cy=cy * scale_y,
        znear=0.01, zfar=20.0,
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    n_frames = min(end if end is not None else T, len(frame_files))

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in range(start, n_frames):
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
            scene.add(cam, pose=np.eye(4))
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=np.eye(4))

            for h in range(B):
                verts_gl = all_verts_cam[h, f] * _CV_TO_GL
                faces    = faces_right if is_right_track[h] else faces_left
                color    = HAND_COLORS[h % len(HAND_COLORS)]
                tm = trimesh.Trimesh(vertices=verts_gl, faces=faces, process=False)
                tm.visual.face_colors = np.array(color, dtype=np.uint8)
                scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))

            render_img, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            overlay = Image.fromarray(render_img, 'RGBA')
            bg = Image.open(frame_files[f]).convert('RGBA').resize((W, H))
            Image.alpha_composite(bg, overlay).convert('RGB').save(
                os.path.join(tmpdir, f'{f:06d}.png')
            )

            if f % 50 == 0:
                print(f"  rendered frame {f}/{n_frames}")

        renderer.delete()
        subprocess.run([
            'ffmpeg', '-y', '-r', str(fps),
            '-i', os.path.join(tmpdir, '%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', output_path,
        ], check=True)

    print(f'Saved → {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Render DynHaMR hand meshes overlaid on camera frames using manotorch.'
    )
    parser.add_argument('--world_results',    required=True)
    parser.add_argument('--frames_folder',    required=True)
    parser.add_argument('--mano_assets_root', required=True)
    parser.add_argument('--output',           required=True)
    parser.add_argument('--fps',   type=float, default=25.0)
    parser.add_argument('--start', type=int,   default=0)
    parser.add_argument('--end',   type=int,   default=None)
    args = parser.parse_args()

    render_dynhamr_video(
        world_results_path = args.world_results,
        frames_folder      = args.frames_folder,
        mano_assets_root   = args.mano_assets_root,
        output_path        = args.output,
        fps                = args.fps,
        start              = args.start,
        end                = args.end,
    )


if __name__ == '__main__':
    main()
