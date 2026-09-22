# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SharpaWave dual floating-hand embodiment.

Each hand is anchored by a 3-prismatic (wrist x/y/z) + ball (wrist orientation) joint
chain, then the URDF fingers. The wrist is driven by joint PD (prismatic) + effort PD
(ball); fingers by position PD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import newton
import numpy as np

from flash_chord.assets import SHARPAWAVE_URDF_DIR
from flash_chord.data.reference import HandPoseReference
from flash_chord.embodiments.base import (
    BodyFrame,
    Embodiment,
    EmbodimentLayout,
    HandLayout,
    HandLinkGeometry,
    JointControlSpec,
    finger_joint_order,
    register_embodiment,
    resolve_joint_controls,
)
from flash_chord.embodiments.binding import RobotReferenceBinding, build_robot_reference_binding
from flash_chord.embodiments.frames import capture_body_ids, resolve_body_frames
from flash_chord.embodiments.sharpa_collision import (
    capture_sharpa_link_geometry,
    filter_sharpa_collisions,
    sharpa_excluded_contacts,
    sharpa_object_contact_links,
)
from flash_chord.utils.labels import side_of
from flash_chord.utils.quat import wxyz_to_rotvec, wxyz_to_xyzw

_DIGITS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass
class SharpaHandsConfig:
    left_urdf_path: Path = SHARPAWAVE_URDF_DIR / "left_sharpawave.urdf"
    right_urdf_path: Path = SHARPAWAVE_URDF_DIR / "right_sharpawave.urdf"
    left_base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity_compensation: bool = True
    default_joint: JointControlSpec = field(default_factory=JointControlSpec)
    joint_overrides: dict[str, dict] = field(
        default_factory=lambda: {
            r"wrist_[xyz]$": dict(kp=25.0, kd=6.0, effort_limit=45.0),  # prismatic base
            r"right_wrist_y": dict(default_pos=0.25),
            r"left_wrist_y": dict(default_pos=-0.25),
            r"wrist_z$": dict(default_pos=0.5),  # initial lift
            r"wrist_rotvec$": dict(mode="effort", kp=25.0, kd=6.0, effort_limit=45.0),
        }
    )
    exclude_contacts: list[tuple[str, str]] = field(default_factory=sharpa_excluded_contacts)
    object_contact_links: list[str] = field(default_factory=sharpa_object_contact_links)

    @classmethod
    def stiff_tracking(cls) -> "SharpaHandsConfig":
        """A stiff-wrist variant for kinematic-fidelity replay (collision-free): the wrist tracks the
        reference to sub-cm instead of lagging under the soft default gains. Fingers unchanged. Gains tuned
        for 200 Hz physics (kd kept near-critical — over-damping adds tracking lag)."""
        cfg = cls()
        cfg.joint_overrides = dict(cfg.joint_overrides)
        cfg.joint_overrides[r"wrist_[xyz]$"] = dict(kp=6000.0, kd=60.0, effort_limit=2000.0)
        cfg.joint_overrides[r"wrist_rotvec$"] = dict(mode="effort", kp=6000.0, kd=60.0, effort_limit=2000.0)
        return cfg


@register_embodiment("sharpa_hands")
class SharpaHands:
    """Two floating SharpaWave hands (left + right) as a single Newton robot."""

    name = "sharpa_hands"

    def __init__(self, config: SharpaHandsConfig | None = None) -> None:
        self.config = config or SharpaHandsConfig()

    # ---- build -----------------------------------------------------------

    def _add_hand(
        self,
        builder: newton.ModelBuilder,
        side: str,
        urdf_path,
        base_pos,
    ) -> dict[str, int]:
        """Add one floating hand (3 prismatic + ball wrist + URDF fingers) at ``base_pos``."""
        import warp as wp  # noqa: PLC0415

        dummy_mass = 1e-4
        x_body = builder.add_link(
            mass=dummy_mass, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)), label=f"{side}_wrist_px_body"
        )
        y_body = builder.add_link(
            mass=dummy_mass, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)), label=f"{side}_wrist_py_body"
        )
        z_body = builder.add_link(
            mass=dummy_mass, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)), label=f"{side}_wrist_pz_body"
        )

        j_px = builder.add_joint_prismatic(
            parent=-1,
            child=x_body,
            axis=newton.Axis.X,
            parent_xform=wp.transform(wp.vec3(*base_pos), wp.quat_identity()),
            label=f"{side}_wrist_x",
            friction=0.01,
            armature=0.01,
        )
        j_py = builder.add_joint_prismatic(
            parent=x_body, child=y_body, axis=newton.Axis.Y, label=f"{side}_wrist_y", friction=0.01, armature=0.01
        )
        j_pz = builder.add_joint_prismatic(
            parent=y_body, child=z_body, axis=newton.Axis.Z, label=f"{side}_wrist_z", friction=0.01, armature=0.01
        )
        builder.add_articulation([j_px, j_py, j_pz], label=f"{side}_wrist_position")

        builder.add_urdf(
            source=urdf_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)),
            base_joint={
                "joint_type": newton.JointType.BALL,
                "label": f"{side}_wrist_rotvec",
                "angular_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, armature=0.01, friction=0.01),
                    newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, armature=0.01, friction=0.01),
                    newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, armature=0.01, friction=0.01),
                ],
            },
            parent_body=z_body,
            force_show_colliders=False,
            enable_self_collisions=True,
            ignore_inertial_definitions=False,
            collapse_fixed_joints=False,
            force_position_velocity_actuation=True,
        )
        return capture_body_ids(builder, _semantic_frame_names(side))

    def _apply_control_config(self, builder: newton.ModelBuilder) -> None:
        """Resolve + apply per-joint control (gains, mode, gravcomp) per the regex config."""
        cfg = self.config
        passive_damping = builder.custom_attributes["mujoco:dof_passive_damping"].values
        joint_ids = tuple(
            joint
            for joint, joint_type in enumerate(builder.joint_type)
            if joint_type != newton.JointType.FIXED
        )
        specs = resolve_joint_controls(
            tuple(builder.joint_label[joint] for joint in joint_ids),
            cfg.default_joint,
            cfg.joint_overrides,
        )
        for j, spec in zip(joint_ids, specs, strict=True):
            qd_start = builder.joint_qd_start[j]
            q_start = builder.joint_q_start[j]
            lin, ang = builder.joint_dof_dim[j]
            n = lin + ang
            effort = spec.mode == "effort"
            is_ball = builder.joint_type[j] == newton.JointType.BALL
            for k in range(n):
                dof = qd_start + k
                if effort:
                    builder.joint_target_ke[dof] = 0.0
                    builder.joint_target_kd[dof] = 0.0
                    builder.joint_target_mode[dof] = int(newton.JointTargetMode.EFFORT)
                    passive_damping[dof] = spec.kd
                else:
                    builder.joint_target_ke[dof] = spec.kp
                    builder.joint_target_kd[dof] = spec.kd
                    builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
                builder.joint_armature[dof] = spec.armature
                builder.joint_effort_limit[dof] = spec.effort_limit
                if spec.velocity_limit is not None:
                    builder.joint_velocity_limit[dof] = spec.velocity_limit
                if spec.friction is not None:
                    builder.joint_friction[dof] = spec.friction
                builder.joint_target_pos[dof] = spec.default_pos
            if not is_ball:  # seed initial joint_q for single-coord joints
                for k in range(n):
                    builder.joint_q[q_start + k] = spec.default_pos
        if cfg.gravity_compensation:
            actgravcomp = builder.custom_attributes["mujoco:jnt_actgravcomp"].values
            for dof in range(builder.joint_dof_count):
                actgravcomp[dof] = True
            body_gravcomp = builder.custom_attributes["mujoco:gravcomp"].values
            for b in range(len(builder.body_label)):
                body_gravcomp[b] = 1.0

    def build(self, builder: newton.ModelBuilder) -> EmbodimentLayout:
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0

        semantic_body_ids = self._add_hand(
            builder,
            "left",
            self.config.left_urdf_path,
            self.config.left_base_pos,
        )
        semantic_body_ids.update(
            self._add_hand(builder, "right", self.config.right_urdf_path, self.config.right_base_pos)
        )
        palm_body_ids = {
            side: semantic_body_ids[f"{side}_hand_C_MC"] for side in ("left", "right")
        }
        hand_geometry = capture_sharpa_link_geometry(
            builder,
            palm_body_ids=palm_body_ids,
            sides=("left", "right"),
            object_contact_links=self.config.object_contact_links,
        )
        filter_sharpa_collisions(
            builder,
            hand_geometry,
            protected_fixed_bodies=palm_body_ids.values(),
            excluded_contacts=self.config.exclude_contacts,
        )
        shape_count_before_collapse = len(builder.shape_body)
        collapse_result = builder.collapse_fixed_joints()
        if len(builder.shape_body) != shape_count_before_collapse:
            raise ValueError("fixed-joint collapse changed shape IDs required by hand geometry ownership")
        semantic_frames = {
            frame.name: frame for frame in resolve_body_frames(semantic_body_ids, collapse_result)
        }
        self._apply_control_config(builder)
        return self._extract_layout(builder, semantic_frames, hand_geometry)

    def _extract_layout(
        self,
        builder: newton.ModelBuilder,
        semantic_frames: dict[str, BodyFrame],
        hand_geometry: dict[str, tuple[HandLinkGeometry, ...]],
    ) -> EmbodimentLayout:
        """Derive the per-hand DOF/body layout from joint + body labels (no magic indices)."""
        wrist_pos: dict[str, list[tuple[int, int]]] = {"left": [], "right": []}  # (dof, q-coord)
        wrist_orient: dict[str, list[int]] = {"left": [], "right": []}
        wrist_orient_q: dict[str, int] = {}
        finger: dict[str, list[tuple[int, int]]] = {"left": [], "right": []}  # (dof, q-coord)
        finger_names: dict[str, list[str]] = {"left": [], "right": []}
        for j, label in enumerate(builder.joint_label):
            side = side_of(label)
            qd_start = builder.joint_qd_start[j]
            q_start = builder.joint_q_start[j]
            lin, ang = builder.joint_dof_dim[j]
            jtype = builder.joint_type[j]
            if jtype == newton.JointType.PRISMATIC:
                wrist_pos[side].append((qd_start, q_start))
            elif jtype == newton.JointType.BALL:
                wrist_orient[side] = [qd_start + k for k in range(lin + ang)]
                wrist_orient_q[side] = q_start
            else:  # revolute finger joints
                finger[side].extend((qd_start + k, q_start + k) for k in range(lin + ang))
                finger_names[side].extend(label.split("/")[-1] for _ in range(lin + ang))

        hands = tuple(
            HandLayout(
                side=side,
                wrist_pos_dof_ids=tuple(d for d, _ in sorted(wrist_pos[side])),
                wrist_orient_dof_ids=tuple(wrist_orient[side]),
                wrist_orient_q_id=wrist_orient_q[side],
                finger_dof_ids=tuple(d for d, _ in sorted(finger[side])),
                link_geometry=hand_geometry[side],
                wrist_pos_q_ids=tuple(q for _, q in sorted(wrist_pos[side])),
                finger_q_ids=tuple(q for _, q in sorted(finger[side])),
                finger_joint_names=tuple(finger_names[side]),
                palm_frame=semantic_frames[f"{side}_hand_C_MC"],
                dp_frames=tuple(semantic_frames[f"{side}_{digit}_DP"] for digit in _DIGITS),
                fingertip_frames=tuple(semantic_frames[f"{side}_{digit}_fingertip"] for digit in _DIGITS),
            )
            for side in ("left", "right")
        )
        return EmbodimentLayout(num_joint_q=len(builder.joint_q), num_joint_dof=builder.joint_dof_count, hands=hands)

    def bind_reference(
        self,
        builder: newton.ModelBuilder,
        layout: EmbodimentLayout,
        reference,
    ) -> RobotReferenceBinding:
        """Bind a ManoSharpaData wrist/finger reference to the floating-hand simulation layout."""
        if not isinstance(reference, HandPoseReference):
            raise TypeError("SharpaHands requires a HandPoseReference")
        joint_q = np.tile(np.asarray(builder.joint_q, dtype=np.float32), (reference.num_frames, 1))
        joint_target = np.tile(
            np.asarray(builder.joint_target_pos, dtype=np.float32),
            (reference.num_frames, 1),
        )
        for hand in layout.hands:
            wrist_pos = np.asarray(reference.wrist_pos_w(hand.side), dtype=np.float32)
            wrist_quat = np.asarray(reference.wrist_quat_w(hand.side), dtype=np.float32)
            fingers = np.asarray(reference.finger_joint_pos(hand.side), dtype=np.float32)[
                :, list(finger_joint_order(hand, reference))
            ]
            joint_q[:, hand.wrist_pos_q_ids] = wrist_pos
            joint_q[:, hand.wrist_orient_q_id : hand.wrist_orient_q_id + 4] = wxyz_to_xyzw(wrist_quat)
            joint_q[:, hand.finger_q_ids] = fingers
            joint_target[:, hand.wrist_pos_dof_ids] = wrist_pos
            joint_target[:, hand.wrist_orient_dof_ids] = wxyz_to_rotvec(wrist_quat)
            joint_target[:, hand.finger_dof_ids] = fingers
        return build_robot_reference_binding(builder, layout, joint_q, joint_target, reference.fps)

def _semantic_frame_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_hand_C_MC",
        *(f"{side}_{digit}_DP" for digit in _DIGITS),
        *(f"{side}_{digit}_fingertip" for digit in _DIGITS),
    )


# fail fast if the interface drifts from the implementation
assert isinstance(SharpaHands(), Embodiment)
