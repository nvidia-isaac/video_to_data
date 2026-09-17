# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable Isaac Lab visual domain-randomization event terms.

This API originated in Huihua's ``vega-groot-port`` branch and now lives in the
shared module so task packages do not need to vendor their own texture and light terms.

The raw-prim texture term complements Isaac Lab's built-in scene-entity term. Every
trigger re-randomizes all matched prims; ``env_ids`` is accepted but ignored. Consumers
using reset-mode events with multiple environments should set a global debounce interval
or trigger at a synchronized boundary.
"""

# Omniverse imports are intentionally lazy and only valid after AppLauncher.
# ruff: noqa: PLC0415

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any, Callable

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg

__all__ = [
    "prewarm_textures",
    "randomize_distant_light",
    "randomize_dome_light",
    "randomize_visual_texture_on_prims",
]


class randomize_visual_texture_on_prims(ManagerTermBase):  # noqa: N801
    """Randomize textures for prims selected by a raw USD path pattern."""

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv") -> None:
        """Create the term and validate the scene replication contract."""
        super().__init__(cfg, env)
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Visual texture randomization requires scene replication to be disabled. "
                "Set scene.replicate_physics=False."
            )
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.replicator.core")
        self._material_prims: list | None = None
        self._rng: Any = None  # omni.replicator.core ReplicatorRNG, set in _lazy_init
        self._last_trigger_step = -(2**62)

    def _lazy_init(self, env: "ManagerBasedEnv", prim_path_pattern: str) -> None:
        import omni.log
        import omni.replicator.core as rep
        from isaacsim.core.utils.stage import get_current_stage

        pattern = prim_path_pattern.replace("{ENV_REGEX_NS}", env.scene.env_regex_ns)
        prims = rep.functional.get.prims(
            path_pattern=pattern,
            stage=get_current_stage(),
        )
        if not prims:
            omni.log.warn(
                f"[visual_dr] texture pattern {pattern!r} matched no prims; term is a no-op"
            )
            self._material_prims = []
            return
        for prim in prims:
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
        self._material_prims = rep.functional.create_batch.material(
            mdl="OmniPBR.mdl",
            bind_prims=prims,
            count=len(prims),
            project_uvw=True,
        )
        self._rng = rep.rng.ReplicatorRNG()

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor | None,
        prim_path_pattern: str,
        textures: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
        min_global_step_interval: int = 0,
    ) -> None:
        """Re-sample a texture and rotation for every matched prim."""
        del env_ids
        if not textures:
            return
        if env.common_step_counter - self._last_trigger_step < min_global_step_interval:
            return
        self._last_trigger_step = env.common_step_counter

        import omni.replicator.core as rep

        if self._material_prims is None:
            self._lazy_init(env, prim_path_pattern)
        if not self._material_prims:
            return
        count = len(self._material_prims)
        rotation_degrees = tuple(math.degrees(value) for value in texture_rotation)
        generator = self._rng.generator
        rep.functional.modify.attribute(
            self._material_prims,
            "diffuse_texture",
            generator.choice(textures, size=count),
        )
        rep.functional.modify.attribute(
            self._material_prims,
            "texture_rotate",
            generator.uniform(*rotation_degrees, size=count),
        )


def randomize_distant_light(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    light_prim_path: str,
    intensity_range: tuple[float, float] = (1800.0, 2600.0),
    color_temperature_range: tuple[float, float] | None = (3500.0, 6500.0),
    elevation_range_deg: tuple[float, float] = (35.0, 58.0),
    azimuth_range_deg: tuple[float, float] = (0.0, 360.0),
) -> None:
    """Randomize intensity, native color temperature, and direction of a distant light."""
    del env, env_ids
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath(light_prim_path)
    if not light_prim.IsValid():
        return
    light = UsdLux.DistantLight(light_prim)
    light.CreateIntensityAttr().Set(float(random.uniform(*intensity_range)))
    if color_temperature_range is not None:
        light.CreateEnableColorTemperatureAttr().Set(True)
        light.CreateColorTemperatureAttr().Set(
            float(random.uniform(*color_temperature_range))
        )

    elevation = math.radians(random.uniform(*elevation_range_deg))
    azimuth = math.radians(random.uniform(*azimuth_range_deg))
    destination = (
        -math.cos(elevation) * math.cos(azimuth),
        -math.cos(elevation) * math.sin(azimuth),
        -math.sin(elevation),
    )
    source = (0.0, 0.0, -1.0)
    axis = (
        source[1] * destination[2] - source[2] * destination[1],
        source[2] * destination[0] - source[0] * destination[2],
        source[0] * destination[1] - source[1] * destination[0],
    )
    sin_angle = math.sqrt(sum(value * value for value in axis))
    cos_angle = sum(a * b for a, b in zip(source, destination, strict=True))
    if sin_angle < 1e-6:
        quaternion = (1.0, 0.0, 0.0, 0.0) if cos_angle > 0 else (0.0, 1.0, 0.0, 0.0)
    else:
        half_angle = math.atan2(sin_angle, cos_angle) / 2.0
        scale = math.sin(half_angle) / sin_angle
        quaternion = (
            math.cos(half_angle),
            axis[0] * scale,
            axis[1] * scale,
            axis[2] * scale,
        )

    xform = UsdGeom.Xformable(light_prim)
    orient_op = next(
        (
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient
        ),
        None,
    )
    if orient_op is None:
        orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        orient_op.Set(Gf.Quatf(*quaternion))
    else:
        orient_op.Set(Gf.Quatd(*quaternion))


def randomize_dome_light(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    intensity_range: tuple[float, float] = (2500.0, 3500.0),
    textures: list[str] | None = None,
    color_temperature_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Randomize dome-light intensity, HDR background, and optional temperature."""
    del env_ids
    asset_cfg = asset_cfg or SceneEntityCfg("sky_light")
    light_prim = env.scene[asset_cfg.name].prims[0]
    light_prim.GetAttribute("inputs:intensity").Set(random.uniform(*intensity_range))
    if color_temperature_range is not None:
        from pxr import UsdLux

        light = UsdLux.DomeLight(light_prim)
        light.CreateEnableColorTemperatureAttr().Set(True)
        light.CreateColorTemperatureAttr().Set(
            float(random.uniform(*color_temperature_range))
        )
    if textures:
        texture_attr = light_prim.GetAttribute("inputs:texture:file")
        if texture_attr:
            texture_attr.Set(random.choice(textures))


def prewarm_textures(
    env: "ManagerBasedEnv",
    dome_textures: list[str] | None = None,
    material_textures: list[str] | None = None,
    set_material_texture: Callable[[str], None] | None = None,
    renders_per_texture: int = 2,
    dome_asset_name: str = "sky_light",
) -> None:
    """Force-load async textures before capture begins."""
    if dome_textures:
        attr = env.scene[dome_asset_name].prims[0].GetAttribute("inputs:texture:file")
        if attr:
            for texture in dome_textures:
                attr.Set(texture)
                for _ in range(renders_per_texture):
                    env.sim.render()
    if material_textures and set_material_texture is not None:
        for texture in dict.fromkeys(material_textures):
            set_material_texture(texture)
            for _ in range(renders_per_texture):
                env.sim.render()
