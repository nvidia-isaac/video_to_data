"""Render aligned MANO meshes (+ optional object) onto the source video.

Consumes the JSONs written by ``align_hands.py`` (real intrinsics,
depth-shifted cam_t) plus optionally a FoundationPose-tracked object mesh
and per-frame poses. Emits a 2×2 grid:

    [src cam (image underlay)]  [world top-down view]
    [world side view         ]  [world front view  ]

The world cameras orbit the per-frame scene center (mean of hand vertices,
plus object centroid if present). Each frame is independently centered,
because HaMeR has no temporal world frame — every frame's camera defines
its own "world."

Usage:
    python -m v2d.hamer.lib.render_hands_aligned_video \\
        --frames_dir /data/frames \\
        --aligned_dir /data/hamer_aligned \\
        --mano_assets_root /data/weights/hamer/_DATA/data \\
        --output_path /data/hamer_aligned_overlay.mp4 \\
        [--object_mesh_path /data/mesh_scaled.obj] \\
        [--object_poses_dir /data/poses_smoothed]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile
from typing import Tuple

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import torch
import trimesh
from manotorch.manolayer import ManoLayer
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


_TRACK_COLORS = [
    (235,  70,  70),   # red
    ( 60, 150, 235),   # blue
    (245, 175,  40),   # orange
    (190,  80, 220),   # purple
    ( 60, 215, 215),   # cyan
    (245, 100, 180),   # pink
    (245, 230,  60),   # yellow
]
_OBJECT_COLOR = (40, 220, 40)   # vivid green — distinct from hand palette
_BG_DARKEN    = 0.45
_CV_TO_GL_VEC = np.array([1.0, -1.0, -1.0])


def _material(rgb: tuple, alpha: float = 1.0) -> "pyrender.MetallicRoughnessMaterial":
    """PBR material so pyrender's lighting actually shades the surface.

    Setting ``trimesh.visual.face_colors`` bakes flat colors that look
    self-illuminating; a material with non-zero roughness picks up Lambert-
    style shading from the scene lights and reveals 3D shape.
    """
    return pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.7,
        alphaMode="OPAQUE",
        baseColorFactor=(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, alpha),
    )


def _add_lights(scene: pyrender.Scene, cam_pose: np.ndarray) -> None:
    """3-point lighting: key (front-right), fill (front-left), back (behind).

    All positioned relative to the camera so meshes are well-lit regardless
    of which orbital view is rendering.
    """
    R = cam_pose[:3, :3]
    eye = cam_pose[:3, 3]
    fwd   = -R[:, 2]   # camera looks down −Z in its local frame
    right =  R[:, 0]
    up    =  R[:, 1]

    def _at(direction: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        # Light "looks" along its local −Z, so place it at eye + offset and
        # orient it back toward the scene (eye - fwd*r, simple parallel-light
        # works since DirectionalLight is direction-only — only rotation matters).
        pose[:3, :3] = R
        pose[:3, 3]  = eye + direction
        return pose

    # Pyrender's DirectionalLight uses the node's −Z as the light direction.
    # All three lights inherit the camera's rotation so the direction is set
    # by the rotation alone (translation is irrelevant for directional lights
    # but pyrender wants a pose).
    base = np.eye(4); base[:3, :3] = R; base[:3, 3] = eye
    rot_y = np.array([
        [np.cos(np.radians(45)),  0, np.sin(np.radians(45))],
        [0,                        1, 0],
        [-np.sin(np.radians(45)), 0, np.cos(np.radians(45))],
    ])
    rot_y_neg = np.array([
        [np.cos(np.radians(-30)), 0, np.sin(np.radians(-30))],
        [0,                        1, 0],
        [-np.sin(np.radians(-30)), 0, np.cos(np.radians(-30))],
    ])

    key = base.copy(); key[:3, :3] = R @ rot_y_neg
    fill = base.copy(); fill[:3, :3] = R @ rot_y
    back = base.copy()
    # Back light points away from camera (~180° around y from key).
    rot_y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    back[:3, :3] = R @ rot_y_180

    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0), pose=key)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5), pose=fill)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=back)


def _font(size: int = 18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _mano_layer(mano_assets_root: str) -> ManoLayer:
    return ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="right",
        center_idx=None,
        mano_assets_root=mano_assets_root,
    )


def _frame_records(aligned_dir: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(aligned_dir, "*", "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        out.setdefault(int(rec["frame_idx"]), []).append(rec)
    return out


def _build_hand_mesh(rec: dict, mano: ManoLayer) -> Tuple[np.ndarray, np.ndarray]:
    """Hand verts in CV cam space + faces."""
    pose_aa = np.concatenate([
        np.array(rec["mano"]["global_orient"], dtype=np.float32),
        np.array(rec["mano"]["hand_pose"],     dtype=np.float32),
    ])
    betas = np.array(rec["mano"]["betas"], dtype=np.float32)
    out = mano(torch.from_numpy(pose_aa)[None], torch.from_numpy(betas)[None])
    verts_local = out.verts[0].detach().numpy()
    if not rec["is_right"]:
        verts_local[:, 0] *= -1
    cam_t = np.array(rec["cam_t"], dtype=np.float64)
    verts_cam = verts_local + cam_t[None, :]
    faces = mano.th_faces.numpy()
    if not rec["is_right"]:
        faces = faces[:, [0, 2, 1]]
    return verts_cam.astype(np.float32), faces


def _quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1-2*qy*qy-2*qz*qz,  2*qx*qy-2*qw*qz,   2*qx*qz+2*qw*qy],
        [2*qx*qy+2*qw*qz,    1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qw*qx],
        [2*qx*qz-2*qw*qy,    2*qy*qz+2*qw*qx,   1-2*qx*qx-2*qy*qy],
    ], dtype=np.float64)


def _load_object_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    tm = trimesh.load(path, force="mesh", process=False)
    return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int32)


def _object_verts_cam(verts_base: np.ndarray, pose_path: str) -> np.ndarray:
    with open(pose_path) as f:
        pd = json.load(f)
    R  = _quat_to_rot(*pd["rotation"])
    t  = np.array(pd["translation"], dtype=np.float64)
    s  = np.array(pd["scale"],       dtype=np.float64)
    RS = R @ np.diag(s)
    return ((RS @ verts_base.T).T + t).astype(np.float32)


def _lookat_pose(eye: np.ndarray, target: np.ndarray,
                 up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """4×4 cam-to-world (pyrender GL convention: cam looks down −Z, up = +Y)."""
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
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
    font = _font(18)
    tw   = draw.textlength(text, font=font)
    draw.rectangle([4, 4, 4 + tw + 8, 28], fill=(0, 0, 0, 180))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return img


def render_hands_aligned_video(
    frames_dir: str,
    aligned_dir: str,
    mano_assets_root: str,
    output_path: str,
    object_mesh_path: str | None = None,
    object_poses_dir: str | None = None,
    object_scale: float = 1.0,
    fps: float = 30.0,
    alpha: float = 0.55,
) -> None:
    frame_files = sorted(
        glob.glob(os.path.join(frames_dir, "*.png")) +
        glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    if not frame_files:
        raise FileNotFoundError(f"No frames in {frames_dir}")

    records = _frame_records(aligned_dir)
    if not records:
        raise FileNotFoundError(f"No aligned JSONs in {aligned_dir}/<track>/")

    mano = _mano_layer(mano_assets_root)

    obj_verts_base = None
    obj_faces = None
    if object_mesh_path is not None and object_poses_dir is not None:
        obj_verts_base, obj_faces = _load_object_mesh(object_mesh_path)
        if object_scale != 1.0:
            # Apply learned global object scale once to the canonical mesh
            # vertices, before any per-frame transform — Transform3d.scale
            # in the per-frame JSONs is left to do whatever the upstream
            # tracker baked in (typically [1,1,1] post-FoundationPose).
            obj_verts_base = obj_verts_base * float(object_scale)
            print(f"  Applied object_scale={object_scale:.4f} to mesh.")

    W_full, H_full = Image.open(frame_files[0]).size
    pw = (W_full // 2) & ~1
    ph = (H_full // 2) & ~1

    # Real intrinsics from any aligned record (constant across frames in a run).
    first = next(iter(records.values()))[0]
    fx = float(first["intrinsics"]["fx"])
    fy = float(first["intrinsics"]["fy"])
    cx = float(first["intrinsics"]["cx"])
    cy = float(first["intrinsics"]["cy"])
    # Far plane covers the actual cam_t.z range with margin.
    z_max = max(rec["cam_t"][2] for recs in records.values() for rec in recs)
    zfar = max(20.0, float(z_max) * 1.5)

    # src-cam: scaled intrinsics for the half-res panel, principal point shifts
    # to the panel center.
    scale_x = pw / W_full
    scale_y = ph / H_full
    cam_src = pyrender.IntrinsicsCamera(
        fx=fx * scale_x, fy=fy * scale_y,
        cx=cx * scale_x, cy=cy * scale_y,
        znear=0.01, zfar=zfar,
    )
    cam_world = pyrender.PerspectiveCamera(
        yfov=np.radians(60.0), znear=0.01, zfar=zfar,
    )

    renderer = pyrender.OffscreenRenderer(viewport_width=pw, viewport_height=ph)
    font = _font(18)

    panel_labels = ["src cam", "world top", "world side", "world front"]

    with tempfile.TemporaryDirectory() as tmpdir:
        for f_idx, frame_path in enumerate(tqdm(frame_files, desc="render", ncols=80)):
            bg_full = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32)
            recs = records.get(f_idx, [])

            # Per-frame hand meshes (cam space).
            hand_meshes_cv = []
            for rec in recs:
                v_cv, f_ = _build_hand_mesh(rec, mano)
                color = _TRACK_COLORS[(rec["track_id"] - 1) % len(_TRACK_COLORS)]
                hand_meshes_cv.append((v_cv, f_, color, rec))

            # Per-frame object mesh (cam space) — optional.
            obj_cv = None
            if obj_verts_base is not None and object_poses_dir is not None:
                pose_path = os.path.join(object_poses_dir, f"{f_idx:06d}.json")
                if os.path.exists(pose_path):
                    obj_cv = _object_verts_cam(obj_verts_base, pose_path)

            # ---- Panel 0: src cam (with image underlay) ------------------
            scene_src = pyrender.Scene(bg_color=[0, 0, 0, 0],
                                       ambient_light=[0.18, 0.18, 0.18])
            scene_src.add(cam_src, pose=np.eye(4))
            _add_lights(scene_src, cam_pose=np.eye(4))
            for v_cv, f_, color, _rec in hand_meshes_cv:
                v_gl = v_cv * _CV_TO_GL_VEC
                tm = trimesh.Trimesh(v_gl, f_, process=False)
                scene_src.add(pyrender.Mesh.from_trimesh(
                    tm, material=_material(color), smooth=True))
            if obj_cv is not None:
                v_gl = obj_cv * _CV_TO_GL_VEC
                tm_o = trimesh.Trimesh(v_gl, obj_faces, process=False)
                scene_src.add(pyrender.Mesh.from_trimesh(
                    tm_o, material=_material(_OBJECT_COLOR), smooth=True))
            rgba_src, _ = renderer.render(scene_src, flags=pyrender.RenderFlags.RGBA)
            bg_panel = np.asarray(
                Image.fromarray(bg_full.astype(np.uint8)).resize((pw, ph)),
                dtype=np.float32,
            )
            ov   = rgba_src.astype(np.float32)
            mask = ov[:, :, 3:4] / 255.0
            bg_dim = bg_panel * (1.0 - mask * (1.0 - _BG_DARKEN))
            src_panel = (mask * ov[:, :, :3] + (1.0 - mask) * bg_dim).clip(0, 255).astype(np.uint8)

            # ---- World panels (top / side / front, gray background) ------
            # Scene center: hand vertices + object vertices (CV → GL).
            all_verts_gl = []
            for v_cv, _, _, _ in hand_meshes_cv:
                all_verts_gl.append(v_cv * _CV_TO_GL_VEC)
            if obj_cv is not None:
                all_verts_gl.append(obj_cv * _CV_TO_GL_VEC)
            if all_verts_gl:
                stacked = np.concatenate(all_verts_gl, axis=0)
                scene_center = stacked.mean(0)
                scene_radius = float(np.linalg.norm(stacked - scene_center, axis=1).max())
                r = max(scene_radius * 2.5, 0.4)
            else:
                scene_center = np.zeros(3, dtype=np.float32)
                r = 0.5

            world_cam_poses = [
                _lookat_pose(scene_center + np.array([0.0,  r,  0.0]),
                             scene_center, up=np.array([0.0, 0.0, -1.0])),  # top
                _lookat_pose(scene_center + np.array([ r,  0.0,  0.0]),
                             scene_center),                                  # side
                _lookat_pose(scene_center + np.array([0.0,  0.0,  r]),
                             scene_center),                                  # front
            ]

            panels = [Image.fromarray(src_panel)]
            for world_pose in world_cam_poses:
                scene_w = pyrender.Scene(bg_color=[0.78, 0.78, 0.78, 1.0],
                                         ambient_light=[0.20, 0.20, 0.20])
                scene_w.add(cam_world, pose=world_pose)
                _add_lights(scene_w, cam_pose=world_pose)
                for v_cv, f_, color, _rec in hand_meshes_cv:
                    v_gl = v_cv * _CV_TO_GL_VEC
                    tm = trimesh.Trimesh(v_gl, f_, process=False)
                    scene_w.add(pyrender.Mesh.from_trimesh(
                        tm, material=_material(color), smooth=True))
                if obj_cv is not None:
                    v_gl = obj_cv * _CV_TO_GL_VEC
                    tm_o = trimesh.Trimesh(v_gl, obj_faces, process=False)
                    scene_w.add(pyrender.Mesh.from_trimesh(
                        tm_o, material=_material(_OBJECT_COLOR), smooth=True))
                rgb_w, _ = renderer.render(scene_w)
                panels.append(Image.fromarray(rgb_w, "RGB"))

            for i, (panel, label) in enumerate(zip(panels, panel_labels)):
                panels[i] = _add_label(panel, label)

            grid = Image.new("RGB", (pw * 2, ph * 2))
            for panel, pos in zip(panels, [(0, 0), (pw, 0), (0, ph), (pw, ph)]):
                grid.paste(panel, pos)
            grid.save(os.path.join(tmpdir, f"{f_idx:06d}.png"))

        renderer.delete()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(fps),
            "-i", os.path.join(tmpdir, "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            output_path,
        ], check=True)
    print(f"Saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir",       required=True)
    parser.add_argument("--aligned_dir",      required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--output_path",      required=True)
    parser.add_argument("--object_mesh_path", default=None,
                        help="Optional FoundationPose-scaled object mesh.")
    parser.add_argument("--object_poses_dir", default=None,
                        help="Optional FoundationPose smoothed per-frame poses.")
    parser.add_argument("--object_scale", type=float, default=1.0,
                        help="Global multiplier applied to object mesh "
                             "vertices before per-frame transforms. Use "
                             "for learned-scale outputs from the gsplat "
                             "refinement step.")
    parser.add_argument("--fps",   type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()
    render_hands_aligned_video(
        frames_dir       = args.frames_dir,
        aligned_dir      = args.aligned_dir,
        mano_assets_root = args.mano_assets_root,
        output_path      = args.output_path,
        object_mesh_path = args.object_mesh_path,
        object_poses_dir = args.object_poses_dir,
        object_scale     = args.object_scale,
        fps              = args.fps,
        alpha            = args.alpha,
    )


if __name__ == "__main__":
    main()
