# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-episode domain randomization: object materials + lighting.

Randomizes on every episode boundary:
  - Scene object materials — color, roughness, metallic sampled from per-class
    palettes (METAL / CERAMIC / PLASTIC) keyed on obj.name substrings.
  - Support surface material — randomly SURF_WOOD or SURF_STONE palette.
  - Distant light — intensity, colour temperature, and direction.

Usage (from record_dataset.py)::

    dr = SceneMaterialRandomizer(base_env, scene_config)
    # in episode loop:
    obs, _, dones, _ = env.step(actions)
    if dones.any():
        dr.randomize(dones)

Note (support surface): we randomise the existing USDA material in-place.
For datasets where pre-built textured surface USDAs are available (e.g. wood
grain, polished stone), swapping the whole USDA per episode would give richer
DR. Revisit when expanding beyond taco — see claude_datagen.md.

Implementation note: material creation uses ``omni.replicator.core`` functional
API (``rep.functional.create_batch.material`` / ``rep.functional.modify.attribute``)
rather than raw pxr because the replicator pipeline handles MDL path resolution
and render-graph registration that raw USD calls miss.  This mirrors the pattern
in ``isaaclab.envs.mdp.events``.
"""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Material class palette definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MatClass:
    hue_lo: float  # HSV hue degrees [0, 360)
    hue_hi: float
    sat_lo: float  # HSV saturation [0, 1]
    sat_hi: float
    val_lo: float  # HSV value / brightness [0, 1]
    val_hi: float
    roughness_lo: float
    roughness_hi: float
    metallic_lo: float
    metallic_hi: float


METAL = _MatClass(
    hue_lo=0,
    hue_hi=30,
    sat_lo=0.00,
    sat_hi=0.15,
    val_lo=0.5,
    val_hi=0.9,
    roughness_lo=0.1,
    roughness_hi=0.4,
    metallic_lo=0.7,
    metallic_hi=1.0,
)
CERAMIC = _MatClass(
    hue_lo=0,
    hue_hi=360,
    sat_lo=0.00,
    sat_hi=0.30,
    val_lo=0.7,
    val_hi=1.0,
    roughness_lo=0.3,
    roughness_hi=0.6,
    metallic_lo=0.0,
    metallic_hi=0.0,
)
PLASTIC = _MatClass(
    hue_lo=0,
    hue_hi=360,
    sat_lo=0.30,
    sat_hi=0.80,
    val_lo=0.4,
    val_hi=0.9,
    roughness_lo=0.4,
    roughness_hi=0.7,
    metallic_lo=0.0,
    metallic_hi=0.05,
)
SURF_WOOD = _MatClass(
    hue_lo=20,
    hue_hi=40,
    sat_lo=0.30,
    sat_hi=0.60,
    val_lo=0.3,
    val_hi=0.6,
    roughness_lo=0.5,
    roughness_hi=0.8,
    metallic_lo=0.0,
    metallic_hi=0.0,
)
SURF_STONE = _MatClass(
    hue_lo=0,
    hue_hi=30,
    sat_lo=0.00,
    sat_hi=0.15,
    val_lo=0.3,
    val_hi=0.6,
    roughness_lo=0.6,
    roughness_hi=0.9,
    metallic_lo=0.0,
    metallic_hi=0.0,
)

# Object name substring → material class.  Longest matching key wins.
_NAME_TO_CLASS: dict[str, _MatClass] = {
    "knife": METAL,
    "fork": METAL,
    "spoon": METAL,
    "ladle": METAL,
    "kettle": METAL,
    "pan": METAL,
    "wok": METAL,
    "spatula": METAL,
    "tong": METAL,
    "plate": CERAMIC,
    "bowl": CERAMIC,
    "mug": CERAMIC,
    "cup": CERAMIC,
    "dish": CERAMIC,
    "saucer": CERAMIC,
    "brush": PLASTIC,
    "bottle": PLASTIC,
    "roller": PLASTIC,
    "eraser": PLASTIC,
    "jar": PLASTIC,
    "container": PLASTIC,
}

_SURFACE_CLASSES = (SURF_WOOD, SURF_STONE)


def _mat_class_for(name: str) -> _MatClass:
    """Return the material class for an object name, defaulting to PLASTIC."""
    n = name.lower()
    match = max(
        ((k, v) for k, v in _NAME_TO_CLASS.items() if k in n),
        key=lambda kv: len(kv[0]),
        default=(None, PLASTIC),
    )
    return match[1]


def _sample_color(cls: _MatClass) -> tuple[float, float, float]:
    h = (random.uniform(cls.hue_lo, cls.hue_hi) % 360) / 360.0
    s = random.uniform(cls.sat_lo, cls.sat_hi)
    v = random.uniform(cls.val_lo, cls.val_hi)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return float(r), float(g), float(b)


def _kelvin_to_rgb(k: float) -> tuple[float, float, float]:
    """Tanner Helland blackbody approximation → linear sRGB [0,1]."""
    t = k / 100.0
    if t <= 66:
        r = 1.0
        g = max(0.0, min(1.0, (99.4708025861 * math.log(t) - 161.1195681661) / 255.0))
        b = (
            0.0
            if t <= 19
            else max(
                0.0,
                min(1.0, (138.5177312231 * math.log(t - 10) - 305.0447927307) / 255.0),
            )
        )
    else:
        r = max(0.0, min(1.0, (329.698727446 * (t - 60) ** -0.1332047592) / 255.0))
        g = max(0.0, min(1.0, (288.1221695283 * (t - 60) ** -0.0755148492) / 255.0))
        b = 1.0
    return r, g, b


def _parse_object_names_from_scene(scene_config) -> list[str]:
    """Extract real object names (e.g. knife, plate) from the sequence ID.

    TACO sequences encode object names like ``taco_cut__knife__plate_20231013_105``.
    We strip the trailing date+id suffix, split on ``__``, skip the first segment
    (action verb), and return the rest.
    Falls back to ``obj.name`` (tool, target, …) if parsing fails.
    """
    import re

    try:
        seq = getattr(scene_config, "sequence_id", None) or ""
        # Strip trailing _YYYYMMDD_NNN
        seq = re.sub(r"_\d{8}_\d+$", "", seq)
        parts = seq.split("__")
        # parts[0] = action verb, parts[1:] = object names
        if len(parts) >= 2:
            return [p.lower() for p in parts[1:]]
    except Exception:
        pass
    return [obj.name for obj in scene_config.scene_objects]


# ---------------------------------------------------------------------------
# SceneMaterialRandomizer
# ---------------------------------------------------------------------------


class SceneMaterialRandomizer:
    """Creates and randomizes OmniPBR materials for scene objects and lighting.

    Must be instantiated AFTER ``gym.make()`` (USD stage is populated then).
    Call ``randomize(dones)`` after ``env.step()`` whenever dones has True
    entries — materials update before the next step renders.

    Uses ``omni.replicator.core`` functional API for material creation and
    attribute modification, matching the pattern from ``isaaclab.envs.mdp.events``.
    """

    def __init__(self, base_env, scene_config) -> None:
        import omni.replicator.core as rep
        import omni.usd

        self._num_envs: int = base_env.num_envs
        self._light_path = "/World/DR_DistantLight"

        # [(lookup_name, rep_mats_list), ...]
        self._obj_mat_groups: list[tuple[str, list]] = []
        self._surf_mats: list | None = None

        stage = omni.usd.get_context().get_stage()
        real_names = _parse_object_names_from_scene(scene_config)
        print(f"[DR] real object names from motion: {real_names}")

        for j, obj in enumerate(scene_config.scene_objects):
            lookup_name = real_names[j] if j < len(real_names) else obj.name
            # Target the visuals prim directly so we can de-instance it and bind there.
            # RigidObject structure: {name}/object/visuals → {name}/object → {name}
            # path_pattern is Python regex — use .* not glob *.
            found = False
            for path_suffix in ("/object/visuals", "/object", ""):
                path_pattern = rf"/World/envs/env_.*/{obj.name}{path_suffix}$"
                prims = rep.functional.get.prims(path_pattern=path_pattern, stage=stage)
                if prims:
                    for prim in prims:
                        if prim.IsInstanceable():
                            prim.SetInstanceable(False)
                    mats = rep.functional.create_batch.material(
                        mdl="OmniPBR.mdl", bind_prims=prims, count=len(prims)
                    )
                    self._obj_mat_groups.append((lookup_name, mats))
                    print(
                        f"[DR] {lookup_name} ({obj.name}): {len(prims)} prims bound at ...{path_suffix or '/<name>'}"
                    )
                    found = True
                    break
            if not found:
                print(f"[DR] WARNING: no prims found for {obj.name}")

        # Support surface: apply_scene_config iterates prims in the support USDA with
        # enumerate(), producing {name}_0, {name}_1, … (tabletop, legs, etc.).
        # Bind ALL parts so the whole surface changes together.
        for fixed in scene_config.fixed_objects:
            found = False
            for suffix in (r"_\d+", ""):
                path_pattern = rf"/World/envs/env_.*/{fixed.name}{suffix}$"
                prims = rep.functional.get.prims(path_pattern=path_pattern, stage=stage)
                if prims:
                    for prim in prims:
                        if prim.IsInstanceable():
                            prim.SetInstanceable(False)
                    mats = rep.functional.create_batch.material(
                        mdl="OmniPBR.mdl", bind_prims=prims, count=len(prims)
                    )
                    self._surf_mats = mats
                    print(
                        f"[DR] surface ({fixed.name}): {len(prims)} prims bound, pattern={path_pattern}"
                    )
                    found = True
                    break
            if not found:
                print(f"[DR] WARNING: surface prim not found for {fixed.name}")
            break  # only first fixed object

        _setup_distant_light(stage, self._light_path)
        self.randomize()  # seed initial randomization

    def randomize(self, dones=None) -> None:
        """Re-sample materials + lighting.

        Note: replicator attribute modification applies to all envs; the dones
        mask is ignored here (same limitation as isaaclab.envs.mdp.events).
        """
        import omni.replicator.core as rep
        import torch

        env_ids = (
            list(range(self._num_envs))
            if dones is None
            else torch.where(dones)[0].tolist()
        )

        for lookup_name, mats in self._obj_mat_groups:
            cls = _mat_class_for(lookup_name)
            n = len(mats)
            colors = np.array([_sample_color(cls) for _ in range(n)])
            roughness = np.array(
                [random.uniform(cls.roughness_lo, cls.roughness_hi) for _ in range(n)]
            )
            metallic = np.array(
                [random.uniform(cls.metallic_lo, cls.metallic_hi) for _ in range(n)]
            )
            print(
                f"[DR] randomize {lookup_name}: color[0]={colors[0].round(2)} rough={roughness[0]:.2f} metal={metallic[0]:.2f}"
            )
            rep.functional.modify.attribute(mats, "diffuse_color_constant", colors)
            rep.functional.modify.attribute(
                mats, "reflection_roughness_constant", roughness
            )
            rep.functional.modify.attribute(mats, "metallic_constant", metallic)

        if self._surf_mats:
            # Same surface material for all envs in an episode — sample once, broadcast.
            cls = random.choice(_SURFACE_CLASSES)
            color = _sample_color(cls)
            roughness_val = random.uniform(cls.roughness_lo, cls.roughness_hi)
            n = len(self._surf_mats)
            colors = np.array([color] * n)
            roughness = np.full(n, roughness_val)
            rep.functional.modify.attribute(
                self._surf_mats, "diffuse_color_constant", colors
            )
            rep.functional.modify.attribute(
                self._surf_mats, "reflection_roughness_constant", roughness
            )
            rep.functional.modify.attribute(
                self._surf_mats, "metallic_constant", np.zeros(n)
            )

        # Only update the global light when env 0 finishes — prevents mid-episode
        # lighting changes for envs whose episodes started at different times.
        if dones is None or 0 in env_ids:
            _randomize_light(self._light_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _setup_distant_light(stage, path: str) -> None:
    if not stage.GetPrimAtPath(path).IsValid():
        stage.DefinePrim(path, "DistantLight")


def _randomize_light(path: str) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath(path)
    if not light_prim.IsValid():
        return

    light = UsdLux.DistantLight(light_prim)
    intensity = random.uniform(500.0, 2000.0)
    kelvin = random.uniform(3000.0, 7000.0)
    r, g, b = _kelvin_to_rgb(kelvin)

    light.CreateIntensityAttr().Set(float(intensity))
    light.CreateEnableColorTemperatureAttr().Set(True)
    light.CreateColorTemperatureAttr().Set(float(kelvin))
    light.CreateColorAttr().Set(Gf.Vec3f(r, g, b))

    # Orient light to random elevation [30°,70°] + azimuth [0°,360°].
    elev = math.radians(random.uniform(30.0, 70.0))
    az = math.radians(random.uniform(0.0, 360.0))
    to_vec = np.array(
        [
            -math.cos(elev) * math.cos(az),
            -math.cos(elev) * math.sin(az),
            -math.sin(elev),
        ]
    )
    to_vec /= np.linalg.norm(to_vec)
    from_vec = np.array([0.0, 0.0, -1.0])
    axis = np.cross(from_vec, to_vec)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.dot(from_vec, to_vec))
    if sin_a < 1e-6:
        qw, qx, qy, qz = (1.0, 0.0, 0.0, 0.0) if cos_a > 0 else (0.0, 1.0, 0.0, 0.0)
    else:
        axis /= sin_a
        half = math.atan2(sin_a, cos_a) / 2.0
        s = math.sin(half)
        qw = math.cos(half)
        qx, qy, qz = float(axis[0] * s), float(axis[1] * s), float(axis[2] * s)

    xform = UsdGeom.Xformable(light_prim)
    xform.ClearXformOpOrder()
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(qw, qx, qy, qz))
