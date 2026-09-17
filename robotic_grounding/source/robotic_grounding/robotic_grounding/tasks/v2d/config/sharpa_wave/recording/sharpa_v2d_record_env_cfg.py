# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sharpa V2D environment with a fixed third-person camera + dataset recorder.

This mirrors the Isaac Lab locomanipulation-SDG pattern: a task-specific env cfg
that (1) adds a camera sensor to the scene and (2) attaches a RecorderManager cfg
so rollouts are written to HDF5. The actual capture is driven by
``scripts/rsl_rl/record_dataset.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robotic_grounding.tasks.v2d.config.sharpa_wave.sharpa_v2d_env_cfg import (
    SharpaV2DEnvCfg,
)

from .recorders_cfg import V2DSDGRecorderManagerCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

    from robotic_grounding.tasks.scene_utils import SceneConfig

# Fixed third-person camera eye, relative to each env's origin. The look-at target is
# derived per-sequence from the mean object position (see ``aim_camera_at_scene``);
# ``CAM_LOOKAT`` is only a fallback when no object positions are available.
CAM_EYE = (-0.5, 0.5, 1.5)
CAM_LOOKAT = (0.1, -0.1, 0.6)

# Egocentric camera eye, relative to each env's origin: above and behind the hand
# workspace, looking down at the objects. Like front_cam, its orientation is derived
# at runtime from the mean object position (aim_camera_at_scene). Iterate on this pos.
EGO_CAM_EYE = (0.3, -0.45, 1.2)

# Grey "cubicle" walls around each env so the cameras don't see neighbouring tiled envs.
# env_spacing is 1.5 m, so tile boundaries are at ±0.75 m; inset slightly to avoid
# z-fighting with the adjacent env's walls. Iterate on extent/height.
WALL_EXTENT = 0.72  # half tile size (m); walls sit at ±this in x and y
WALL_HEIGHT = 1.6  # m; tall enough to occlude neighbours from the camera eye heights
WALL_THICKNESS = 0.02  # m
WALL_COLOR = (0.5, 0.5, 0.5)  # grey
WALL_NAMES = ("wall_px", "wall_nx", "wall_py", "wall_ny")
"""Scene attribute names of the cubicle walls ``add_cubicle_walls`` installs.

Exported so visual-DR configs can target the walls by name without duplicating the
literals -- only envs that actually call ``add_cubicle_walls`` should pass these to
``inject_scene_visual_dr_terms``.
"""


def workspace_center_from_scene(
    scene_config: SceneConfig, fallback: tuple[float, float, float] = CAM_LOOKAT
) -> tuple[float, float, float]:
    """Mean initial position of the scene's manipulable objects (env-relative).

    Uses ``scene_config.scene_objects`` (the tool/target objects loaded from the motion
    data), which excludes the robots and any decorative geometry such as the cubicle
    walls — those are added directly to the scene, not via ``scene_config``, so they
    never enter this mean. Falls back to ``fallback`` if no object positions exist.
    """
    positions = [
        o.init_pos
        for o in getattr(scene_config, "scene_objects", [])
        if getattr(o, "init_pos", None) is not None
    ]
    if not positions:
        return fallback
    center = np.asarray(positions, dtype=float).mean(axis=0)
    return (float(center[0]), float(center[1]), float(center[2]))


def aim_camera_at_scene(
    env_cfg: ManagerBasedRLEnvCfg,
    scene_config: SceneConfig,
    sensor_name: str = "front_cam",
) -> tuple[float, float, float] | None:
    """Point a configured camera at the scene's mean object position.

    Reads the camera's eye from its own ``offset.pos`` and bakes the look-at orientation
    into ``offset.rot`` BEFORE the env is created (so it survives every episode reset).
    Returns the target it aimed at, or ``None`` if the sensor isn't configured.
    """
    camera_cfg = getattr(env_cfg.scene, sensor_name, None)
    if camera_cfg is None:
        return None
    target = workspace_center_from_scene(scene_config)
    camera_cfg.offset.rot = lookat_quat_world(camera_cfg.offset.pos, target)
    return target


def add_cubicle_walls(
    env_cfg: ManagerBasedRLEnvCfg,
    extent: float = WALL_EXTENT,
    height: float = WALL_HEIGHT,
    thickness: float = WALL_THICKNESS,
    color: tuple[float, float, float] = WALL_COLOR,
) -> None:
    """Enclose each env in 4 thin grey walls so cameras don't see neighbouring tiled envs.

    Walls sit at ``±extent`` in x and y (just inside the tile boundary), span the full
    tile width, and rise to ``height`` from the ground. Added directly to the scene (not
    ``scene_config.scene_objects``), so they never enter the camera-aim object mean.
    Collision is explicitly disabled (``CollisionPropertiesCfg(collision_enabled=False)``),
    so the walls are purely visual and never perturb the physics/policy (rather than relying
    on the ``collision_props=None`` spawner default).
    """
    span = 2.0 * extent
    half_h = height / 2.0
    geometry = {
        "wall_px": ((extent, 0.0, half_h), (thickness, span, height)),
        "wall_nx": ((-extent, 0.0, half_h), (thickness, span, height)),
        "wall_py": ((0.0, extent, half_h), (span, thickness, height)),
        "wall_ny": ((0.0, -extent, half_h), (span, thickness, height)),
    }
    walls = {name: geometry[name] for name in WALL_NAMES}
    for name, (pos, size) in walls.items():
        setattr(
            env_cfg.scene,
            name,
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False
                    ),
                ),
            ),
        )


def add_world_axes(
    env_cfg: ManagerBasedRLEnvCfg, length: float = 0.4, radius: float = 0.006
) -> None:
    """Spawn RGB = XYZ world-frame axes at each env origin (debug aid for camera framing).

    Adds three static, colored cuboids to the scene so the cameras render them into the
    RGB images: +X red, +Y green, +Z blue, each spanning ``length`` from the origin. This
    is a DEBUG-only visual; it also alters the instance-segmentation ids, so it must not
    be used for real dataset generation (the record script guards on episode count).
    """
    t = radius * 2.0
    half = length / 2.0
    axes = {
        "debug_axis_x": ((half, 0.0, 0.0), (length, t, t), (1.0, 0.0, 0.0)),
        "debug_axis_y": ((0.0, half, 0.0), (t, length, t), (0.0, 1.0, 0.0)),
        "debug_axis_z": ((0.0, 0.0, half), (t, t, length), (0.0, 0.0, 1.0)),
    }
    for name, (pos, size, color) in axes.items():
        setattr(
            env_cfg.scene,
            name,
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color, emissive_color=color
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False
                    ),
                ),
            ),
        )


def lookat_quat_world(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for a camera at ``eye`` looking at ``target``.

    Returned in Isaac Lab's ``"world"`` camera convention (forward = +X, up = +Z), so it
    can be used directly as a ``TiledCameraCfg.OffsetCfg(rot=..., convention="world")``.
    Baking the orientation into the cfg offset means the camera holds this pose across
    every episode reset (a runtime ``set_world_poses_from_view`` call does NOT survive
    resets), so recorded extrinsics match what is rendered.
    """
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(world_up, dtype=np.float64)

    forward = target_v - eye_v
    fwd_norm = np.linalg.norm(forward)
    if fwd_norm < 1e-8:
        # eye ≈ target: look direction is undefined. Return identity (no rotation) rather
        # than dividing by ~0, which would produce NaNs and corrupt the camera offset.
        return (1.0, 0.0, 0.0, 0.0)
    forward /= fwd_norm  # camera +X
    # Project world up onto the plane orthogonal to forward; fall back if degenerate
    # (camera pointing nearly straight up/down).
    if abs(float(np.dot(up_v, forward))) > 0.999:
        up_v = np.array([1.0, 0.0, 0.0])
    up = up_v - np.dot(up_v, forward) * forward
    up /= np.linalg.norm(up)  # camera +Z
    left = np.cross(up, forward)  # camera +Y = (+Z) x (+X)
    left /= np.linalg.norm(left)

    # Columns are the camera axes (x=forward, y=left, z=up) expressed in world frame.
    rot = np.column_stack([forward, left, up])
    trace = np.trace(rot)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


@configclass
class SharpaV2DRecordEnvCfg(SharpaV2DEnvCfg):
    """Sharpa V2D env that records camera + state/action/reward rollouts to HDF5."""

    # Camera settings (overridable from the recording script).
    camera_width: int = 256
    camera_height: int = 256
    camera_data_types: tuple[str, ...] = (
        "rgb",
        "distance_to_image_plane",
        "instance_id_segmentation_fast",
    )

    # RecorderManager: action/state recorders + camera + rewards.
    recorders: V2DSDGRecorderManagerCfg = V2DSDGRecorderManagerCfg()

    def __post_init__(self) -> None:
        """Add front/ego cameras, cubicle walls and the recorder manager to the base env cfg."""
        super().__post_init__()

        # Modest env count for data generation (override with --num_envs).
        self.scene.num_envs = 16
        # Base cfg points the viewer at env 6 (assumes thousands of envs); data-gen
        # often runs with only a handful, so use env 0 to stay in range.
        self.viewer.env_index = 0

        # Cameras (one per env). The look-at orientation baked here is a fallback aimed at
        # CAM_LOOKAT; record_dataset.py re-bakes each camera's rot at the scene's mean
        # object position via aim_camera_at_scene() before env creation. Baking into the
        # cfg offset (world convention) means it holds across every episode reset, so the
        # recorded extrinsics match what is rendered. (A runtime set_world_poses_from_view
        # aim does NOT survive the per-episode reset.)
        def _make_camera(name: str, eye: tuple[float, float, float]) -> TiledCameraCfg:
            return TiledCameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=eye,
                    rot=lookat_quat_world(eye, CAM_LOOKAT),
                    convention="world",
                ),
                data_types=list(self.camera_data_types),
                width=self.camera_width,
                height=self.camera_height,
                # Store raw id buffers rather than colorized RGBA for segmentation.
                colorize_semantic_segmentation=False,
                colorize_instance_id_segmentation=False,
                colorize_instance_segmentation=False,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.05, 20.0),
                ),
            )

        # Fixed third-person camera + egocentric (above/behind hands, looking down) camera.
        self.scene.front_cam = _make_camera("front_cam", CAM_EYE)
        self.scene.ego_cam = _make_camera("ego_cam", EGO_CAM_EYE)

        # Grey cubicle walls so the cameras don't capture neighbouring tiled envs.
        add_cubicle_walls(self)

        # Keep the camera recorders in sync with the configured sensors/data types.
        camera_sensors = ["front_cam", "ego_cam"]
        self.recorders.record_camera.sensor_names = camera_sensors
        self.recorders.record_camera.data_types = list(self.camera_data_types)
        self.recorders.record_camera_info.sensor_names = camera_sensors
