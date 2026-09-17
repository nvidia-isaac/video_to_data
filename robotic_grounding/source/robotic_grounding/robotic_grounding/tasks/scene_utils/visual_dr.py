# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared, embodiment-agnostic visual-domain-randomization pools and term builders.

This module owns *what* is randomized and *with which values*; the mechanism of how a
texture or light is randomized lives in :mod:`robotic_grounding.rendering.dr`. Task
configs own only *which scene entities* are targeted, by passing them to the builders
below.

It lives in ``scene_utils`` rather than under a task package because both embodiment
families already depend on ``scene_utils`` (``v2d_whole_body/base_env_cfg.py``,
``v2d/config/sharpa_wave/recording/sharpa_v2d_record_env_cfg.py``). A floating-hand env
passes ``robot_entities=("right_robot", "left_robot")``; a whole-body env passes
``("robot",)``. Nothing here is Sharpa-specific.

The event functions are imported from ``robotic_grounding.rendering.dr.isaaclab_events``
directly rather than via ``tasks.v2d.mdp``, so that the whole-body family can use this
module without importing the floating-hand task package.

**Timing contract.** All terms are ``mode="interval"`` with ``is_global_time=True``, so
visuals change *during* an episode roughly every ``VISUAL_INTERVAL_S`` seconds rather than
being held constant per episode. A single recorded episode therefore contains several
visual conditions. This is deliberate -- it maximises visual variety per unit of recording
time. Drivers that need one fixed condition per demo (the re-render augmentation flow)
instead trigger the terms manually between demos.

Every term name carries the ``visual_`` prefix so
``robotic_grounding.rendering.dr.controller`` can filter them generically.

**Requires ``ManagerBasedRLEnv``.** ``randomize_visual_texture_on_prims`` reads
``env.common_step_counter``, which a plain ``ManagerBasedEnv`` does not define.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from robotic_grounding.rendering.dr.isaaclab_events import (
    randomize_distant_light,
    randomize_dome_light,
    randomize_visual_texture_on_prims,
)

# IsaacLab's built-in scene-entity texture term. Scene entities (robots, rigid objects)
# go through this; plain spawned prims (ground, support surfaces, walls) go through the
# prim-pattern term above, which does not require a SceneEntityCfg.
from isaaclab.envs.mdp import randomize_visual_texture_material  # isort: skip

__all__ = [
    "DISTANT_LIGHT_PARAMS",
    "DOME_LIGHT_PARAMS",
    "GROUND_TEXTURE_PARAMS",
    "OBJECT_TEXTURE_PARAMS",
    "ROBOT_TEXTURE_PARAMS",
    "SURFACE_TEXTURE_PARAMS",
    "VISUAL_INTERVAL_S",
    "inject_scene_visual_dr_terms",
    "make_visual_dr_terms",
]


#################################################
# Randomization parameter pools (editable)
#################################################
# All assets are referenced relative to NVIDIA_NUCLEUS_DIR (the asset root configured via
# OMNI_SERVER). The container must be able to reach that root; repoint these to locally
# baked assets for offline runs. An unreachable root makes the terms no-op silently, so
# validate reachability explicitly rather than inferring it from rendered output.

VISUAL_INTERVAL_S = (4.0, 6.0)
"""Seconds between re-randomizations, global time. See the timing contract above."""

DOME_LIGHT_PARAMS: dict[str, Any] = {
    "asset_cfg": SceneEntityCfg("sky_light"),
    "intensity_range": (1500.0, 6000.0),
    "textures": [
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/noon_grass_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/adams_place_bridge_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
    ],
}

# Strong single directional key so the scene has clear, distinct-source shadows; colour
# comes from the light's NATIVE kelvin attribute rather than a hand-rolled RGB tint.
DISTANT_LIGHT_PARAMS: dict[str, Any] = {
    "light_prim_path": "/World/light",
    "intensity_range": (1800.0, 2600.0),
    "color_temperature_range": (3500.0, 6500.0),
    "elevation_range_deg": (35.0, 58.0),
    "azimuth_range_deg": (0.0, 360.0),
}

GROUND_TEXTURE_PARAMS: dict[str, Any] = {
    "prim_path_pattern": "/World/ground",
    "textures": [
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
    ],
    "texture_rotation": (0.0, 2 * math.pi),
}

# Wood/stone/metal set from the franka stack visuomotor example -- the right look for
# boxes and tabletops. Shared by the dynamic-object pool (scene-entity term, keyed
# `texture_paths`) and the support-surface pool (prim-pattern term, keyed `textures`).
_SURFACE_TEXTURES = [
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch/Birch_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks/Mahogany_Planks_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
]

OBJECT_TEXTURE_PARAMS: dict[str, Any] = {
    "texture_paths": _SURFACE_TEXTURES,
    "texture_rotation": (0.0, math.pi),
}

SURFACE_TEXTURE_PARAMS: dict[str, Any] = {
    "textures": _SURFACE_TEXTURES,
    "texture_rotation": (0.0, 2 * math.pi),
}

# Metal set from the franka stack visuomotor example's robot-arm randomization. Named for
# the robot rather than the hand: a whole-body arm uses exactly the same pool.
ROBOT_TEXTURE_PARAMS: dict[str, Any] = {
    "texture_paths": [
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Cast/Aluminum_Cast_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Polished/Aluminum_Polished_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brass/Brass_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Bronze/Bronze_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brushed_Antique_Copper/Brushed_Antique_Copper_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Cast_Metal_Silver_Vein/Cast_Metal_Silver_Vein_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Copper/Copper_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Gold/Gold_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Iron/Iron_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/RustedMetal/RustedMetal_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Silver/Silver_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Carbon/Steel_Carbon_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
    ],
    "texture_rotation": (0.0, 0.0),
}


#################################################
# Term builders
#################################################
# Plain functions rather than a configclass mixin, so each env cfg composes DR explicitly
# and no inheritance diamond forms between the base env cfg and the recording variant.


def _interval_term(func: Any, params: dict[str, Any]) -> EventTerm:
    """An interval-mode, global-time event term carrying ``params``."""
    return EventTerm(
        func=func,
        mode="interval",
        interval_range_s=VISUAL_INTERVAL_S,
        is_global_time=True,
        params=params,
    )


def make_visual_dr_terms(events: Any, *, robot_entities: Sequence[str]) -> None:
    """Attach the scene-level ``visual_*`` terms to an events cfg, in place.

    Sets the lights (dome + distant, each at startup and on an interval), the ground
    texture, and one texture term per robot entity. Scene-derived targets -- the
    manipulated object and any support surfaces -- are not known until the motion file is
    resolved, so they are added later by :func:`inject_scene_visual_dr_terms`.

    Light terms run once at ``startup`` and then on an interval. The texture terms are
    interval-only: the underlying event classes bind one OmniPBR material batch per term
    INSTANCE, so a startup twin would create a second competing bind on the same prims.

    Args:
        events: the events configclass to mutate (typically ``self`` in ``__post_init__``).
        robot_entities: scene-entity names to texture, e.g. ``("right_robot",
            "left_robot")`` for floating hands or ``("robot",)`` for a whole-body robot.
    """
    events.visual_dome_light_startup = EventTerm(
        func=randomize_dome_light, mode="startup", params=DOME_LIGHT_PARAMS
    )
    events.visual_distant_light_startup = EventTerm(
        func=randomize_distant_light, mode="startup", params=DISTANT_LIGHT_PARAMS
    )
    events.visual_dome_light = _interval_term(randomize_dome_light, DOME_LIGHT_PARAMS)
    events.visual_distant_light = _interval_term(
        randomize_distant_light, DISTANT_LIGHT_PARAMS
    )
    events.visual_ground_texture = _interval_term(
        randomize_visual_texture_on_prims, GROUND_TEXTURE_PARAMS
    )

    for entity in robot_entities:
        term_name = f"visual_{entity}_texture"
        setattr(
            events,
            term_name,
            _interval_term(
                randomize_visual_texture_material,
                {
                    "event_name": term_name,
                    "asset_cfg": SceneEntityCfg(entity),
                    **ROBOT_TEXTURE_PARAMS,
                },
            ),
        )


def inject_scene_visual_dr_terms(
    events: Any, *, extra_prim_names: Sequence[str] = ()
) -> None:
    """Add texture terms for scene-derived prims, in place.

    Must run AFTER ``apply_scene_config`` has populated
    ``events.setup_collision_groups.params`` -- object names come from the motion file's
    scene config and are unknown at cfg-definition time. Dynamic objects are scene
    entities (built-in term); fixed objects and ``extra_prim_names`` are plain spawned
    prims addressed by pattern.

    Because injection happens after Hydra override application, these terms cannot be
    nulled from the command line. To run without visual DR use the non-DR task id, or
    ``rendering.dr.controller.disable_visual_event_terms``.

    Args:
        events: the events configclass to mutate.
        extra_prim_names: additional env-relative prim names to texture, e.g. cubicle
            wall names. Pass only names the env actually spawns -- a pattern that matches
            nothing produces a live term that silently does nothing.
    """
    groups = getattr(events, "setup_collision_groups", None)
    if groups is None:
        return

    for name in groups.params.get("object_names", []):
        term_name = f"visual_{name}_texture"
        setattr(
            events,
            term_name,
            _interval_term(
                randomize_visual_texture_material,
                {
                    "event_name": term_name,
                    "asset_cfg": SceneEntityCfg(name),
                    **OBJECT_TEXTURE_PARAMS,
                },
            ),
        )

    fixed = list(groups.params.get("fixed_object_names", [])) + list(extra_prim_names)
    for name in fixed:
        setattr(
            events,
            f"visual_{name}_texture",
            _interval_term(
                randomize_visual_texture_on_prims,
                {
                    "prim_path_pattern": f"{{ENV_REGEX_NS}}/{name}",
                    **SURFACE_TEXTURE_PARAMS,
                },
            ),
        )
