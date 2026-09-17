# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in visual domain randomization for the floating-hand Sharpa task.

Two env configs, registered as ``Sharpa-V2D-DR-v0`` and ``Sharpa-V2D-DR-Record-v0``. The
non-DR tasks are untouched.

The randomization *mechanism* lives in :mod:`robotic_grounding.rendering.dr`; the parameter
pools and term builders live in :mod:`robotic_grounding.tasks.scene_utils.visual_dr`. This
module only says which scene entities each task targets.

**Composition is explicit in both classes.** ``SharpaV2DRecordEnvCfg`` derives from
``SharpaV2DEnvCfg``, *not* from the DR config, so the record variant cannot inherit DR by
subclassing it -- it re-declares ``events`` and re-implements the injection hook. Keeping
that explicit avoids a configclass inheritance diamond.

Visual DR is deliberately **not** enabled for training: the texture terms allocate one
OmniPBR material per matched prim at env-build time, expanded across all envs.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robotic_grounding.tasks.scene_utils.visual_dr import (
    inject_scene_visual_dr_terms,
    make_visual_dr_terms,
)
from robotic_grounding.tasks.v2d import mdp
from robotic_grounding.tasks.v2d.v2d_hand_env_cfg import EventCfg, ObservationsCfg

from .recording.sharpa_v2d_record_env_cfg import WALL_NAMES, SharpaV2DRecordEnvCfg
from .sharpa_v2d_env_cfg import SharpaV2DEnvCfg

_TRACKING_COMMAND = "dual_hands_object_tracking_command"

DR_RECORD_CAMERA = "front_cam"
"""Camera the ``record`` observation group and the re-render driver both read.

``SharpaV2DRecordEnvCfg`` installs ``front_cam`` and ``ego_cam``; this constant keeps the
observation term and ``rerender_demo_visuals.py`` from drifting apart. The live
RecorderManager still captures BOTH cameras -- only the named ``record`` group and the
re-render HDF5 are limited to this one.
"""

_ROBOT_ENTITIES = ("right_robot", "left_robot")


@configclass
class VisualDomainRandomizationEventCfg(EventCfg):
    """Base events plus the scene-level ``visual_*`` terms.

    Inherits ``setup_collision_groups`` (patched at apply time) and the robot
    physics-material terms from the base ``EventCfg``.
    """

    def __post_init__(self) -> None:
        """Attach the light, ground and per-hand texture terms."""
        make_visual_dr_terms(self, robot_entities=_ROBOT_ENTITIES)


@configclass
class SharpaV2DDREnvCfg(SharpaV2DEnvCfg):
    """Box-lift env with visual DR. No cameras -- use ``eval.py --video`` for pixels."""

    events: VisualDomainRandomizationEventCfg = VisualDomainRandomizationEventCfg()

    def __post_init__(self) -> None:
        """Force ``replicate_physics=False``, required by raw-prim material binding."""
        super().__post_init__()
        # Already the base default; set explicitly so this cfg stays correct if that
        # changes. The texture events raise if it is True.
        self.scene.replicate_physics = False

    def register_scene_visual_dr_events(self) -> None:
        """Inject object / support-surface texture terms (called by apply_scene_config).

        No ``extra_prim_names``: this env spawns no cubicle walls, and a pattern matching
        nothing would create a live term that silently does nothing.
        """
        inject_scene_visual_dr_terms(self.events)


@configclass
class RecordObsCfg(ObsGroup):
    """Non-policy observation group captured for VLA data recording.

    ``concatenate_terms=False`` so the group surfaces as a dict of named terms, which the
    re-render driver writes verbatim as ``data/demo_i/obs/<term>``. Does not affect the
    policy or critic.
    """

    image = ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg(DR_RECORD_CAMERA),
            "data_type": "rgb",
            "normalize": False,
        },
    )
    wrist_position_e = ObsTerm(
        func=mdp.wrist_position_e, params={"command_name": _TRACKING_COMMAND}
    )
    wrist_orientation_e = ObsTerm(
        func=mdp.wrist_orientation_e, params={"command_name": _TRACKING_COMMAND}
    )
    finger_joint_pos = ObsTerm(
        func=mdp.finger_joint_pos, params={"command_name": _TRACKING_COMMAND}
    )
    object_position_e = ObsTerm(
        func=mdp.object_position_e, params={"command_name": _TRACKING_COMMAND}
    )
    object_orientation_e = ObsTerm(
        func=mdp.object_orientation_e, params={"command_name": _TRACKING_COMMAND}
    )
    processed_right_action = ObsTerm(
        func=mdp.processed_action,
        params={"action_name": "right_joint_residual_action"},
    )
    processed_left_action = ObsTerm(
        func=mdp.processed_action,
        params={"action_name": "left_joint_residual_action"},
    )

    def __post_init__(self) -> None:
        """Surface as named terms, uncorrupted."""
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class DRRecordObservationsCfg(ObservationsCfg):
    """Base observations plus a ``record`` group (policy group unchanged)."""

    record: RecordObsCfg = RecordObsCfg()


@configclass
class SharpaV2DDRRecordEnvCfg(SharpaV2DRecordEnvCfg):
    """Recording env with visual DR: cameras and recorders from the parent, DR added here.

    The parent supplies ``front_cam``/``ego_cam``, cubicle walls and the RecorderManager
    but no DR, so ``events`` and the injection hook are re-declared rather than inherited.
    """

    events: VisualDomainRandomizationEventCfg = VisualDomainRandomizationEventCfg()

    def __post_init__(self) -> None:
        """Add the ``record`` obs group and force ``replicate_physics=False``."""
        super().__post_init__()
        self.observations = DRRecordObservationsCfg()
        self.scene.replicate_physics = False

    def register_scene_visual_dr_events(self) -> None:
        """Inject scene-derived terms, including the cubicle walls this env installs."""
        inject_scene_visual_dr_terms(self.events, extra_prim_names=WALL_NAMES)
