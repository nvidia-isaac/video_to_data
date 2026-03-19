#!/usr/bin/env python3
"""
Generate a spinning turntable video of a GLB/OBJ mesh using trimesh + pyrender (EGL offscreen).

Usage:
  python spin_mesh_video.py mesh.glb output.mp4
  python spin_mesh_video.py mesh.obj output.mp4 --frames 120 --fps 30 --width 800 --height 600
"""

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import pyrender
import cv2

# Force EGL headless backend
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def look_at(eye, target, up=np.array([0., 0., 1.])):
    """Build a camera-to-world 4x4 pose matrix (OpenGL convention)."""
    forward = np.array(target, dtype=float) - np.array(eye, dtype=float)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0., 1., 0.])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_vec = np.cross(right, forward)
    # OpenGL: camera looks down -Z
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = up_vec
    T[:3, 2] = -forward
    T[:3, 3] = eye
    return T


def spin_video(mesh_path: str, output_path: str,
               n_frames: int = 120, fps: int = 30,
               width: int = 800, height: int = 600,
               elevation_deg: float = 20.0):

    print(f"Loading mesh: {mesh_path}")
    loaded = trimesh.load(mesh_path)

    # Flatten scene to list of meshes
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
    elif isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    else:
        raise ValueError(f"Unsupported mesh type: {type(loaded)}")

    # Compute combined bounds for centering
    all_verts = np.vstack([np.asarray(m.vertices) for m in meshes])
    centre = (all_verts.max(axis=0) + all_verts.min(axis=0)) / 2.0
    extent = all_verts.max(axis=0) - all_verts.min(axis=0)
    print(f"Mesh extents: {extent.round(4)} m")

    # Build pyrender scene
    pr_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                              ambient_light=[0.3, 0.3, 0.3])
    for m in meshes:
        m.vertices -= centre
        pr_mesh = pyrender.Mesh.from_trimesh(m, smooth=False)
        pr_scene.add(pr_mesh)

    # Camera
    diag = float(np.linalg.norm(extent))
    radius = diag * 1.6
    fov_y = math.radians(60)
    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=width / height)
    cam_node = pr_scene.add(camera, pose=np.eye(4))

    # Lights
    dl = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    pr_scene.add(dl, pose=look_at([1, -1, 2], [0, 0, 0]))
    dl2 = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    pr_scene.add(dl2, pose=look_at([-1, 1, 1], [0, 0, 0]))

    renderer = pyrender.OffscreenRenderer(width, height)
    elevation = math.radians(elevation_deg)

    frames_out = os.environ.get("SPIN_FRAMES_DIR")
    if frames_out:
        frame_dir = Path(frames_out)
        frame_dir.mkdir(parents=True, exist_ok=True)
    else:
        frame_dir = Path(tempfile.mkdtemp(prefix="spin_frames_"))
    print(f"Rendering {n_frames} frames ...")

    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        eye = np.array([
            radius * math.cos(angle) * math.cos(elevation),
            radius * math.sin(angle) * math.cos(elevation),
            radius * math.sin(elevation),
        ])
        cam_pose = look_at(eye, [0, 0, 0])
        pr_scene.set_pose(cam_node, cam_pose)

        color, _ = renderer.render(pr_scene)
        bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"), bgr)

        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"  {i+1}/{n_frames}")

    renderer.delete()

    print("Encoding video with ffmpeg ...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not os.environ.get("SPIN_FRAMES_DIR"):
        shutil.rmtree(frame_dir)
    if result.returncode != 0:
        print("ffmpeg error:", result.stderr)
        print(f"Frames left in: {frame_dir}")
    else:
        print(f"Saved {n_frames} frames @ {fps} fps → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Spinning turntable video of a mesh")
    parser.add_argument("mesh", help="Input mesh file (GLB, OBJ, PLY, …)")
    parser.add_argument("output", help="Output video path (e.g. spin.mp4)")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--elevation", type=float, default=20.0,
                        help="Camera elevation angle in degrees (default: 20)")
    args = parser.parse_args()

    spin_video(args.mesh, args.output,
               n_frames=args.frames, fps=args.fps,
               width=args.width, height=args.height,
               elevation_deg=args.elevation)


if __name__ == "__main__":
    main()
