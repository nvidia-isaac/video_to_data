# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Headless MP4 renderer that mirrors the viser scene content.

Companion to ``vis_retargeted.py``.  After the viser recording is written,
the same per-frame data (robot qpos, MANO verts, object poses) is fed into
``OfflineVideoRenderer`` which rasterizes via ``pyrender`` (off-screen EGL)
and writes ``<sequence_id>.mp4`` next to ``<sequence_id>.viser``.

No browser or Isaac Sim required — everything runs inside the existing
container using pyrender + imageio.
"""

from __future__ import annotations

import os

# Force EGL before pyrender / PyOpenGL is imported.  The Isaac Sim image has a
# GPU but no X display; EGL gives us headless off-screen rendering.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pinocchio as pin
import pyrender
import trimesh


def _look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: tuple[float, float, float] = (0, 0, 1),
) -> np.ndarray:
    """Build a 4x4 camera pose in pyrender convention (-z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.column_stack([s, u, -f])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


def _compose(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


class OfflineVideoRenderer:
    """Off-screen pyrender wrapper mirroring ``visualize_one_trajectory``.

    Lifecycle per sequence:

        r = OfflineVideoRenderer(fps=30)
        r.add_robot("right", right_kin)
        r.add_robot("left", left_kin)
        for obj_name, mesh in object_meshes.items():
            r.add_object(obj_name, mesh)
        for t in range(num_frames):
            r.update_robot("right", right_qpos[t])
            r.update_robot("left", left_qpos[t])
            for name, T in object_poses[t].items():
                r.update_object(name, T)
            if has_mano:
                r.update_mano("right", mano_right_verts[t], mano_faces)
                r.update_mano("left", mano_left_verts[t], mano_faces)
            r.capture()
        r.save(Path("out.mp4"))
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        camera_eye: tuple[float, float, float] = (0.8, -0.8, 0.6),
        camera_target: tuple[float, float, float] = (0.0, 0.0, 0.1),
    ) -> None:
        self.fps = max(1, int(fps))
        self._renderer = pyrender.OffscreenRenderer(width, height)
        self._scene = pyrender.Scene(
            bg_color=[0.92, 0.92, 0.95, 1.0],
            ambient_light=[0.35, 0.35, 0.35],
        )

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self._camera_node = self._scene.add(
            camera, pose=_look_at(camera_eye, camera_target)
        )
        self._camera_yfov = np.pi / 3.0

        key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
        key_light_pose = _look_at((1.5, -1.5, 2.0), (0, 0, 0))
        self._scene.add(key_light, pose=key_light_pose)
        fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        fill_light_pose = _look_at((-1.0, 1.0, 1.5), (0, 0, 0))
        self._scene.add(fill_light, pose=fill_light_pose)

        # name -> list of (pyrender.Node, pin.GeometryObject) for robot links.
        self._robot_nodes: dict[str, list[tuple[pyrender.Node, Any]]] = {}
        # name -> pin kinematics handle (captures model / data / visual_model / visual_data)
        self._robot_kins: dict[str, Any] = {}
        # Single node per object, geometry static, pose changes.
        self._object_nodes: dict[str, pyrender.Node] = {}
        # MANO nodes are recreated per frame (vertices change).
        self._mano_nodes: dict[str, pyrender.Node] = {}

        self._frames: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Camera framing
    # ------------------------------------------------------------------
    def fit_camera(
        self,
        points: np.ndarray,
        elevation_deg: float = 30.0,
        azimuth_deg: float = 45.0,
        padding: float = 1.5,
        min_extent: float = 0.3,
        aspect_ratio: float = 16.0 / 9.0,
    ) -> None:
        """Re-pose the camera to frame all ``points`` (N, 3).

        - ``target`` = centroid of the point cloud.
        - ``eye`` = target + direction(elevation, azimuth) × computed distance.
        - Distance is chosen so the bounding sphere fits inside the frustum
          (both vertical and horizontal), with ``padding`` slack.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            return
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        target = 0.5 * (lo + hi)
        diag = max(np.linalg.norm(hi - lo), min_extent)
        # Half-angles: vertical = yfov/2, horizontal = derived from aspect.
        tan_v = np.tan(self._camera_yfov / 2.0)
        tan_h = tan_v * aspect_ratio
        tan_half = min(tan_v, tan_h)
        distance = padding * 0.5 * diag / max(tan_half, 1e-6)

        # Elevation from horizontal, azimuth around +Z up-axis.
        el = np.deg2rad(elevation_deg)
        az = np.deg2rad(azimuth_deg)
        direction = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
            dtype=np.float64,
        )
        eye = target + distance * direction
        self._scene.set_pose(self._camera_node, _look_at(eye, target))

    # ------------------------------------------------------------------
    # Robot
    # ------------------------------------------------------------------
    def add_robot(
        self,
        name: str,
        kinematics: Any,
        color: tuple[int, int, int, int] = (200, 200, 215, 255),
    ) -> None:
        """Load every ``visual_model.geometryObjects[i].meshPath`` as a pyrender node.

        ``kinematics`` is a ``SharpaHandKinematics`` (or compatible); we need
        its ``robot.model``, ``robot.data``, ``robot.visual_model``, and
        ``robot.visual_data`` attributes — same surface the viser visualizer
        uses.
        """
        self._robot_kins[name] = kinematics
        nodes: list[tuple[pyrender.Node, Any]] = []
        for geom in kinematics.robot.visual_model.geometryObjects:
            mesh_path = geom.meshPath
            if not mesh_path:
                continue
            try:
                tm = trimesh.load(mesh_path, force="mesh")
            except Exception as e:  # noqa: BLE001
                print(f"[offline_video] skip {geom.name}: {e}")
                continue
            if not isinstance(tm, trimesh.Trimesh) or len(tm.vertices) == 0:
                continue
            # Apply the geometry's own mesh scale.
            scale = np.asarray(geom.meshScale).reshape(3)
            if not np.allclose(scale, 1.0):
                tm.apply_scale(scale)
            tm.visual.face_colors = np.array(color, dtype=np.uint8)
            pyr = pyrender.Mesh.from_trimesh(tm, smooth=False)
            node = self._scene.add(pyr, pose=np.eye(4), name=f"{name}:{geom.name}")
            nodes.append((node, geom))
        self._robot_nodes[name] = nodes

    def update_robot(self, name: str, qpos: np.ndarray) -> None:
        """Run pinocchio FK on ``qpos`` and set each link's pose."""
        kin = self._robot_kins[name]
        pin.forwardKinematics(kin.robot.model, kin.robot.data, qpos)
        pin.updateGeometryPlacements(
            kin.robot.model,
            kin.robot.data,
            kin.robot.visual_model,
            kin.robot.visual_data,
        )
        for node, geom in self._robot_nodes[name]:
            geom_id = kin.robot.visual_model.getGeometryId(geom.name)
            M = kin.robot.visual_data.oMg[geom_id]
            self._scene.set_pose(node, _compose(M.rotation, M.translation))

    # ------------------------------------------------------------------
    # Rigid objects (geometry static, pose changes)
    # ------------------------------------------------------------------
    def add_object(self, name: str, mesh: trimesh.Trimesh) -> None:
        if name in self._object_nodes:
            self._scene.remove_node(self._object_nodes[name])
        pyr = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        self._object_nodes[name] = self._scene.add(
            pyr, pose=np.eye(4), name=f"object:{name}"
        )

    def update_object(self, name: str, T: np.ndarray) -> None:
        """``T`` is a 4x4 world transform."""
        if name in self._object_nodes:
            self._scene.set_pose(self._object_nodes[name], T)

    # ------------------------------------------------------------------
    # MANO hand meshes (vertices change per frame)
    # ------------------------------------------------------------------
    def update_mano(
        self,
        side: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: tuple[int, int, int, int] = (230, 180, 170, 200),
    ) -> None:
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # Use vertex colours so smooth=True doesn't trip the face/vertex check.
        tm.visual.vertex_colors = np.tile(
            np.array(color, dtype=np.uint8)[None, :], (len(vertices), 1)
        )
        pyr = pyrender.Mesh.from_trimesh(tm, smooth=True)
        if side in self._mano_nodes:
            self._scene.remove_node(self._mano_nodes[side])
        self._mano_nodes[side] = self._scene.add(
            pyr, pose=np.eye(4), name=f"mano:{side}"
        )

    # ------------------------------------------------------------------
    # Frame capture / save
    # ------------------------------------------------------------------
    def capture(self) -> None:
        color, _ = self._renderer.render(self._scene)
        self._frames.append(color.copy())

    def save(self, out_mp4: Path) -> None:
        if not self._frames:
            print(f"[offline_video] no frames captured, skipping {out_mp4}")
            return
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(
            str(out_mp4),
            np.stack(self._frames, axis=0),
            fps=self.fps,
            codec="libx264",
        )
        print(f"  Saved → {out_mp4} ({len(self._frames)} frames @ {self.fps} fps)")
        self._frames.clear()

    def close(self) -> None:
        self._renderer.delete()
