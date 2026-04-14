from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from wis3d import Wis3D


class Wis3DScene:
    """Thin wrapper around Wis3D for building per-frame 3D visualizations."""

    def __init__(self, output_dir: Path, name: str = "scene"):
        self._wis3d = Wis3D(
            str(output_dir),
            name,
            xyz_pattern=("x", "-y", "-z"),
        )
        self._frame_idx = 0

    def set_frame(self, frame_idx: int) -> None:
        self._frame_idx = frame_idx
        self._wis3d.set_scene_id(frame_idx)

    _DEFAULT_COLOR = np.array([180, 180, 200], dtype=np.uint8)

    def add_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)
        colors = None

        if hasattr(mesh.visual, "vertex_colors"):
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) == len(vertices):
                colors = np.array(vc[:, :3], dtype=np.uint8)
        elif hasattr(mesh.visual, "to_color"):
            try:
                vc = mesh.visual.to_color().vertex_colors
                if vc is not None and len(vc) == len(vertices):
                    colors = np.array(vc[:, :3], dtype=np.uint8)
            except Exception:
                pass

        if colors is None:
            colors = np.tile(self._DEFAULT_COLOR, (len(vertices), 1))

        self._wis3d.add_mesh(vertices, faces, colors, name=name)

    def add_axes(self, name: str, T: np.ndarray, scale: float = 1.0, radius: float = 0.01) -> None:
        """Add a coordinate frame visualization as 3D cylinders.

        Args:
            name: Display name for this axes set.
            T: (4, 4) pose matrix (world-from-frame).
            scale: Length of each axis.
            radius: Cylinder radius (relative to scene units).
        """
        axis_colors = [
            [255, 0, 0, 255],  # X = red
            [0, 255, 0, 255],  # Y = green
            [0, 0, 255, 255],  # Z = blue
        ]

        meshes = []
        origin = T[:3, 3]
        for axis_idx in range(3):
            direction = T[:3, axis_idx]
            tip = origin + direction * scale
            segment = np.array([origin, tip])
            cyl = trimesh.creation.cylinder(
                radius=radius * scale,
                segment=segment,
                sections=6,
            )
            cyl.visual.vertex_colors = np.tile(axis_colors[axis_idx], (len(cyl.vertices), 1)).astype(np.uint8)
            meshes.append(cyl)

        axes_mesh = trimesh.util.concatenate(meshes)
        self.add_mesh(name, axes_mesh)
