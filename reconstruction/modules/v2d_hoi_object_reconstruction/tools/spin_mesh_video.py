#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate an appearance-faithful spinning video of a mesh.

The renderer supports textured meshes and SAM3D-style GLBs that store their
appearance in the glTF ``COLOR_0`` vertex attribute.  Vertex-colored meshes are
rendered with flat, opaque shading by default so bright PBR lights do not wash
their colors toward white.

Examples:
  python spin_mesh_video.py mesh.glb output.mp4
  python spin_mesh_video.py mesh.obj output.mp4 --frames 120 --fps 30
  python spin_mesh_video.py mesh.glb output.mp4 --shading lit
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

# Select the headless backend before importing pyrender/OpenGL.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh


DEFAULT_BACKGROUND_RGBA = np.array([194, 196, 196, 255], dtype=float) / 255.0


def look_at(
    eye: Sequence[float],
    target: Sequence[float],
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Build a camera-to-world 4x4 pose matrix (OpenGL convention)."""

    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-12:
        raise ValueError("Camera position and target must be different")
    forward /= forward_norm

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up_vector = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up_vector
    pose[:3, 2] = -forward  # OpenGL cameras look down -Z.
    pose[:3, 3] = eye
    return pose


def load_meshes(mesh_path: str | Path) -> list[trimesh.Trimesh]:
    """Load meshes and apply every node transform from a scene graph.

    Accessing ``Scene.geometry`` directly loses per-node transforms.  Applying
    them here makes the turntable match how a GLB viewer places its geometry.
    ``process=False`` also avoids merging vertices that carry distinct colors.
    """

    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded.copy()]
    elif isinstance(loaded, trimesh.Scene):
        meshes = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry[geometry_name]
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            mesh = geometry.copy()
            mesh.apply_transform(transform)
            meshes.append(mesh)
    else:
        raise ValueError(f"Unsupported mesh type: {type(loaded)}")

    if not meshes:
        raise ValueError(f"No triangle meshes found in {mesh_path}")
    if not any(len(mesh.vertices) and len(mesh.faces) for mesh in meshes):
        raise ValueError(f"No renderable triangles found in {mesh_path}")
    return meshes


def uses_vertex_colors(mesh: trimesh.Trimesh) -> bool:
    """Return whether ``mesh`` carries one RGB/RGBA color per vertex."""

    if getattr(mesh.visual, "kind", None) != "vertex":
        return False
    colors = np.asarray(mesh.visual.vertex_colors)
    return (
        colors.ndim == 2
        and len(colors) == len(mesh.vertices)
        and colors.shape[1] >= 3
    )


def repair_inverted_winding(mesh: trimesh.Trimesh) -> int:
    """Orient a globally inverted closed mesh without guessing for open surfaces.

    A closed mesh may contain negative-volume inner cavity shells. Inverting
    the complete mesh preserves that valid relationship between its outer and
    inner boundaries. Open meshes are left unchanged because their outside
    cannot be inferred robustly from volume.
    """

    if not len(mesh.faces) or not mesh.is_watertight:
        return 0
    mesh_scale = float(mesh.scale)
    volume_tolerance = np.finfo(float).eps
    if math.isfinite(mesh_scale):
        volume_tolerance = max(mesh_scale**3 * 1e-12, volume_tolerance)
    total_volume = float(mesh.volume)
    if not math.isfinite(total_volume) or total_volume >= -volume_tolerance:
        return 0

    mesh.invert()
    return 1


def build_render_mesh(
    mesh: trimesh.Trimesh,
    *,
    force_opaque_vertex_colors: bool,
) -> pyrender.Mesh:
    """Convert a trimesh mesh while preserving opaque vertex-color display."""

    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    if force_opaque_vertex_colors and uses_vertex_colors(mesh):
        for primitive in render_mesh.primitives:
            if primitive.color_0 is None:
                continue
            # Trimesh's default vertex-color material uses alphaMode=BLEND.
            # Make opacity and a neutral base factor explicit in the renderer;
            # this does not rewrite or remove the source GLB's COLOR_0 data.
            primitive.material.alphaMode = "OPAQUE"
            primitive.material.baseColorFactor = np.ones(4)
            primitive.material.metallicFactor = 0.0
            primitive.material.roughnessFactor = 0.9
    return render_mesh


def should_use_flat_shading(
    meshes: Sequence[trimesh.Trimesh], shading: str
) -> bool:
    """Resolve ``auto`` shading from mesh appearance data."""

    if shading == "flat":
        return True
    if shading == "lit":
        return False
    if shading != "auto":
        raise ValueError(f"Unsupported shading mode: {shading}")
    return any(uses_vertex_colors(mesh) for mesh in meshes)


def spin_video(
    mesh_path: str | Path,
    output_path: str | Path,
    n_frames: int = 120,
    fps: int = 30,
    width: int = 800,
    height: int = 600,
    elevation_deg: float = 20.0,
    fit_margin: float = 1.08,
    fov_deg: float = 45.0,
    shading: str = "auto",
    frames_dir: str | Path | None = None,
) -> None:
    """Render a turntable video and encode it as H.264/yuv420p."""

    if n_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("frames, fps, width, and height must all be positive")
    if fit_margin < 1.0 or not 1.0 < fov_deg < 179.0:
        raise ValueError("fit_margin must be at least 1 and fov must be 1..179")

    print(f"Loading mesh: {mesh_path}")
    meshes = load_meshes(mesh_path)
    # The observed inversion defect is specific to SAM3D-style vertex-colored
    # exports. Avoid an expensive connected-component scan for conventional
    # scanner/textured meshes whose appearance path is otherwise unchanged.
    repaired = sum(
        repair_inverted_winding(mesh)
        for mesh in meshes
        if uses_vertex_colors(mesh)
    )
    if repaired:
        print(f"Reoriented {repaired} globally inverted closed mesh(es) for rendering")

    all_vertices = np.vstack(
        [np.asarray(mesh.vertices) for mesh in meshes if len(mesh.vertices)]
    )
    center = (all_vertices.max(axis=0) + all_vertices.min(axis=0)) * 0.5
    extent = all_vertices.max(axis=0) - all_vertices.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError(f"Mesh has invalid or zero-size bounds: {extent}")
    print(f"Mesh extents: {extent.round(4)}")

    flat_shading = should_use_flat_shading(meshes, shading)
    appearance = "flat opaque vertex color" if flat_shading else "lit material"
    print(f"Appearance mode: {appearance}")

    scene = pyrender.Scene(
        bg_color=DEFAULT_BACKGROUND_RGBA,
        ambient_light=[0.3, 0.3, 0.3],
    )
    for mesh in meshes:
        mesh.apply_translation(-center)
        scene.add(
            build_render_mesh(
                mesh,
                force_opaque_vertex_colors=flat_shading,
            )
        )

    centered_vertices = all_vertices - center
    object_radius = float(np.linalg.norm(centered_vertices, axis=1).max())
    vertical_fov = math.radians(fov_deg)
    horizontal_fov = 2.0 * math.atan(
        math.tan(vertical_fov * 0.5) * (width / height)
    )
    fit_fov = min(vertical_fov, horizontal_fov)
    radius = object_radius * fit_margin / math.sin(fit_fov * 0.5)
    camera = pyrender.PerspectiveCamera(
        yfov=vertical_fov, aspectRatio=width / height
    )
    camera_node = scene.add(camera, pose=np.eye(4))

    if not flat_shading:
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=4.0),
            pose=look_at([1.0, -1.0, 2.0], [0.0, 0.0, 0.0]),
        )
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
            pose=look_at([-1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        )

    explicit_frames_dir = frames_dir or os.environ.get("SPIN_FRAMES_DIR")
    keep_frames = explicit_frames_dir is not None
    if explicit_frames_dir is not None:
        frame_dir = Path(explicit_frames_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
    else:
        frame_dir = Path(tempfile.mkdtemp(prefix="spin_frames_"))

    print(f"Rendering {n_frames} frames ...")
    renderer = pyrender.OffscreenRenderer(width, height)
    elevation = math.radians(elevation_deg)
    render_flags = (
        pyrender.RenderFlags.FLAT if flat_shading else pyrender.RenderFlags.NONE
    )
    try:
        for index in range(n_frames):
            angle = 2.0 * math.pi * index / n_frames
            eye = np.array(
                [
                    radius * math.cos(angle) * math.cos(elevation),
                    radius * math.sin(angle) * math.cos(elevation),
                    radius * math.sin(elevation),
                ]
            )
            scene.set_pose(camera_node, look_at(eye, [0.0, 0.0, 0.0]))
            color, _ = renderer.render(scene, flags=render_flags)
            cv2.imwrite(
                str(frame_dir / f"frame_{index:04d}.png"),
                cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
            )
            if (index + 1) % 20 == 0 or index == n_frames - 1:
                print(f"  {index + 1}/{n_frames}")
    finally:
        renderer.delete()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    print("Encoding video with ffmpeg ...")
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Video encoding failed; frames remain in: {frame_dir}")
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr)
        raise
    else:
        if not keep_frames:
            shutil.rmtree(frame_dir)
        print(f"Saved {n_frames} frames @ {fps} fps -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an appearance-faithful spinning video of a mesh"
    )
    parser.add_argument("mesh", help="Input mesh file (GLB, OBJ, PLY, ...)")
    parser.add_argument("output", help="Output video path (for example spin.mp4)")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument(
        "--elevation",
        type=float,
        default=20.0,
        help="Camera elevation angle in degrees (default: 20)",
    )
    parser.add_argument(
        "--fit-margin",
        type=float,
        default=1.08,
        help="Framing margin around the mesh bounding sphere (default: 1.08)",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=45.0,
        help="Vertical camera field of view in degrees (default: 45)",
    )
    parser.add_argument(
        "--shading",
        choices=("auto", "flat", "lit"),
        default="auto",
        help=(
            "auto uses flat opaque shading for vertex colors and lit PBR for "
            "materials/textures (default: auto)"
        ),
    )
    parser.add_argument(
        "--frames-dir",
        help="Optional directory in which to retain rendered PNG frames",
    )
    args = parser.parse_args()

    spin_video(
        args.mesh,
        args.output,
        n_frames=args.frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
        elevation_deg=args.elevation,
        fit_margin=args.fit_margin,
        fov_deg=args.fov,
        shading=args.shading,
        frames_dir=args.frames_dir,
    )


if __name__ == "__main__":
    main()
