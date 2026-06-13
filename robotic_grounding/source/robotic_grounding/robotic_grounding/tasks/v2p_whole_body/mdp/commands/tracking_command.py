# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-body tracking command term.

Loads all reference data from a single planner parquet and provides command
targets + current state observations for body tracking, hand keypoint tracking,
contact tracking, and virtual object control.

Follows the same initialization pattern as DualHandsObjectTrackingCommand
but operates on a single-articulation whole-body robot (not dual floating-base hands).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from robotic_grounding.tasks.v2p.mdp.utils import (
    compute_wrench_space,
    compute_wrench_space_support_function,
    sample_wrench_space_basis_scaled,
)
from robotic_grounding.tasks.v2p.mdp.utils_jit import (
    wrench_preprocess_jit,
    wrench_support_one_body_jit,
)

from .tracking_utils import load_motion_data

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .tracking_command_cfg import TrackingCommandCfg


class TrackingCommand(CommandTerm):
    """Whole-body tracking command term.

    Loads reference motion from a planner parquet and exposes:
    - Body tracking: root pose, joint positions/velocities, multi-future frames
    - EE tracking: wrist positions/orientations
    - Hand keypoints: fingertip positions (in object frame, transformed per step)
    - Contact targets: desired contact positions on object surface
    - Object tracking: object body pose trajectory
    - VOC: virtual object control curriculum scale
    """

    cfg: TrackingCommandCfg

    def __init__(self, cfg: TrackingCommandCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize tracking command from planner parquet."""
        super().__init__(cfg, env)

        self._init_scene_references(cfg, env)
        self._load_and_process_motion(cfg)
        self._init_buffers(cfg)
        self._init_metrics()
        self._init_hand_data(cfg)
        self._init_contact_data()
        self._precompute_hand_keypoints_in_object_frame()
        self._precompute_contact_positions_in_object_frame()
        self._init_wrench_data(cfg)
        self._precompute_contact_wrench_support_values()

        # Re-trigger debug vis now that ee_link_names is known
        if cfg.debug_vis:
            self.set_debug_vis(True)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_scene_references(
        self, cfg: TrackingCommandCfg, env: ManagerBasedRLEnv
    ) -> None:
        """Resolve robot, object, and contact sensor assets from the scene."""
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self._env_origins = env.scene.env_origins

        # Object list (for VOC compatibility)
        if cfg.object_body_names:
            self.objects = [env.scene[name] for name in cfg.object_body_names]
        else:
            self.objects = [self.object]
            cfg.object_body_names = [cfg.object_name]

        # Anchor body (pelvis)
        self._anchor_body_ids, _ = self.robot.find_bodies([cfg.anchor_body_name])
        self._anchor_body_id = self._anchor_body_ids[0]

        # Defaults — resolved after motion data is loaded
        self._left_wrist_body_id = None
        self._right_wrist_body_id = None
        self._left_fingertip_body_ids: list[int] = []
        self._right_fingertip_body_ids: list[int] = []
        self._left_finger_joint_ids: list[int] = []
        self._right_finger_joint_ids: list[int] = []

        # Contact sensors — grouped by side, list per body (matching floating hand env)
        self.object_to_right_hand_contact_sensors = []
        self.object_to_left_hand_contact_sensors = []
        for sensor_name in cfg.object_contact_sensor_names:
            if sensor_name in env.scene.sensors:
                sensor = env.scene.sensors[sensor_name]
                if "_to_right_" in sensor_name:
                    self.object_to_right_hand_contact_sensors.append(sensor)
                elif "_to_left_" in sensor_name:
                    self.object_to_left_hand_contact_sensors.append(sensor)

        if self.object_to_right_hand_contact_sensors:
            self.num_robot_contacts_right = len(
                self.object_to_right_hand_contact_sensors[0].cfg.filter_prim_paths_expr
            )
        else:
            self.num_robot_contacts_right = 0
        if self.object_to_left_hand_contact_sensors:
            self.num_robot_contacts_left = len(
                self.object_to_left_hand_contact_sensors[0].cfg.filter_prim_paths_expr
            )
        else:
            self.num_robot_contacts_left = 0

    def _load_and_process_motion(self, cfg: TrackingCommandCfg) -> None:
        """Load motion data from parquet and set up body tracking tensors."""
        motion_data = load_motion_data(cfg, self.robot, self.device)

        # Body motion (already split on-disk and in-memory).
        self.root_pos_w = motion_data.robot_root_position.float() + torch.tensor(
            cfg.robot_anchor_pos_offset, device=self.device
        )
        self.root_quat_w = motion_data.robot_root_wxyz.float()
        self._joint_pos_file = motion_data.robot_joint_positions.float()
        self._object_pos_w = motion_data.object_pos_w.float() + torch.tensor(
            cfg.object_pos_offset, device=self.device
        )
        self._object_quat_w = motion_data.object_quat_w.float()
        # Multi-body reference object trajectories (E, T, B, *) for the command
        # target. Used by the multi-object command-frame observations/metrics.
        self._object_body_pos_w = (
            motion_data.object_body_position.float()
            + torch.tensor(cfg.object_pos_offset, device=self.device)
        )
        self._object_body_quat_w = motion_data.object_body_wxyz.float()
        if motion_data.object_articulation is not None:
            self.retargeted_object_articulation = (
                motion_data.object_articulation.float()
            )
        else:
            self.retargeted_object_articulation = torch.zeros(
                self._object_body_pos_w.shape[0], 0, device=self.device
            )
        self.retargeted_object_body_names = motion_data.object_body_names
        self.num_timesteps = self.root_pos_w.shape[0]

        # EE data
        if motion_data.ee_pos_w is not None:
            self.ee_link_names = motion_data.ee_link_names
            self.ee_link_ids = motion_data.ee_link_ids
            self.ee_pos_w = motion_data.ee_pos_w
            self.ee_quat_w = motion_data.ee_quat_w

        # Joint velocity via finite differences
        self._joint_vel_file = torch.zeros_like(self._joint_pos_file)
        self._joint_vel_file[:-1] = (
            self._joint_pos_file[1:] - self._joint_pos_file[:-1]
        ) / cfg.dt

        # Future frame config
        self.num_future_frames = cfg.num_future_frames
        self.frame_step = int(cfg.dt_future_frames / cfg.dt)
        self._future_frame_offsets = torch.arange(
            0,
            self.num_future_frames * self.frame_step,
            self.frame_step,
            dtype=torch.int32,
            device=self.device,
        )

        # Joint reordering
        self._tracked_joint_ids, self._tracked_joint_names = self.robot.find_joints(
            cfg.joint_names
        )
        file_joint_names = motion_data.file_joint_names or cfg.file_joint_names
        if file_joint_names is not None:
            file_to_isaac = [
                file_joint_names.index(n)
                for n in self._tracked_joint_names
                if n in file_joint_names
            ]
            if len(file_to_isaac) == len(self._tracked_joint_names):
                self.joint_pos = self._joint_pos_file[:, file_to_isaac]
                self.joint_vel = self._joint_vel_file[:, file_to_isaac]
            else:
                self.joint_pos = self._joint_pos_file
                self.joint_vel = self._joint_vel_file
        else:
            self.joint_pos = self._joint_pos_file
            self.joint_vel = self._joint_vel_file

        # Object height peak (for reward phase detection)
        self.object_height_peak_timestep = torch.argmax(self._object_pos_w[:, 2]).item()

        # Hand data flag
        # Resolve wrist body IDs now that ee_link_names is known from parquet
        ee_names = getattr(self, "ee_link_names", None) or []
        if ee_names:
            all_body_ids, all_body_names = self.robot.find_bodies(ee_names)
            for bid, bname in zip(all_body_ids, all_body_names, strict=False):
                if "left" in bname.lower():
                    self._left_wrist_body_id = bid
                elif "right" in bname.lower():
                    self._right_wrist_body_id = bid

        self._motion_data = motion_data

    def _init_buffers(self, cfg: TrackingCommandCfg) -> None:
        """Allocate per-env counters, VOC scale, and encoder mode."""
        self.timestep = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.reset_timestep = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.trajectory_end_timestep = torch.full(
            (self.num_envs,),
            self.num_timesteps - 1,
            dtype=torch.int32,
            device=self.device,
        )
        self.steps_since_last_reset = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._encoder_mode = torch.zeros(self.num_envs, 4, device=self.device)

        # VOC
        self.virtual_object_controller_scale_factor = torch.tensor(
            [cfg.initial_virtual_object_control_curriculum_scale], device=self.device
        )
        self.virtual_object_controller_scale_factor_per_env = (
            cfg.initial_virtual_object_control_curriculum_scale
            * torch.ones(self.num_envs, 1, device=self.device)
        )

        # Tracking lengths (for metrics normalization)
        max_ep_steps = int(self._env.max_episode_length)
        self.tracking_lengths = torch.full(
            (self.num_envs,),
            min(self.num_timesteps, max_ep_steps),
            dtype=torch.int32,
            device=self.device,
        )

        # Action history buffer (lazy init on first update_action_history call)
        self._action_history: torch.Tensor | None = None
        self._action_history_len = cfg.action_history_length

        # Shoulder spread offset (for annealing during freeze period)
        num_joints = self.joint_pos.shape[1]
        self._spread_joint_offset = torch.zeros(
            self.num_envs, num_joints, device=self.device
        )

        self.all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.num_bodies = self.object_position_e.shape[1]
        if (
            self.retargeted_object_body_names is None
            or len(self.retargeted_object_body_names) != self.num_bodies
        ):
            self.retargeted_object_body_names = list(self.object.data.body_names)
        assert len(self.retargeted_object_body_names) == self.num_bodies, (
            "The number of body names in the motion file and the object do not match. "
            f"Find {len(self.retargeted_object_body_names)} in motion file, "
            f"but {self.num_bodies} in the object."
        )
        self.KEYPOINT_VECS = (
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=torch.float32,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(self.num_envs, self.num_bodies, -1, -1)
        )

    def _init_metrics(self) -> None:
        """Allocate per-env tracking metric buffers."""
        self.metrics["anchor_position_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["anchor_wxyz_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["joint_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        for side in ("left", "right"):
            self.metrics[f"{side}_hand_wrist_position_error"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics[f"{side}_hand_wrist_wxyz_error"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics[f"{side}_hand_finger_joints_error"] = torch.zeros(
                self.num_envs, device=self.device
            )
        self.metrics["object_body_position_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["object_body_wxyz_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["virtual_object_controller_scale_factor"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def _init_hand_data(self, cfg: TrackingCommandCfg) -> None:
        """Assign retargeted hand data from motion data. Fields may be None."""
        md = self._motion_data

        # Fingertip body IDs (from cfg, robot-level)
        if cfg.fingertip_body_name:
            tip_ids, tip_names = self.robot.find_bodies(cfg.fingertip_body_name)
            self._left_fingertip_body_ids = [
                i
                for i, n in zip(tip_ids, tip_names, strict=False)
                if "left" in n.lower()
            ]
            self._right_fingertip_body_ids = [
                i
                for i, n in zip(tip_ids, tip_names, strict=False)
                if "right" in n.lower()
            ]

        # Finger joint IDs (from cfg, robot-level). Track names alongside IDs so
        # parquet finger joint values can be reordered to IsaacLab's joint order.
        self._left_finger_joint_names: list[str] = []
        self._right_finger_joint_names: list[str] = []
        if cfg.finger_joint_names:
            all_fj_ids, all_fj_names = self.robot.find_joints(cfg.finger_joint_names)
            for i, n in zip(all_fj_ids, all_fj_names, strict=False):
                if "left" in n.lower():
                    self._left_finger_joint_ids.append(i)
                    self._left_finger_joint_names.append(n)
                elif "right" in n.lower():
                    self._right_finger_joint_ids.append(i)
                    self._right_finger_joint_names.append(n)

        # Retargeted hand data (from parquet, may be None)
        self.retargeted_left_wrist_position = md.left_wrist_position  # (T, 3)
        self.retargeted_left_wrist_wxyz = md.left_wrist_wxyz  # (T, 4)
        self.retargeted_right_wrist_position = md.right_wrist_position
        self.retargeted_right_wrist_wxyz = md.right_wrist_wxyz
        self.retargeted_left_hand_frames = md.left_hand_frames  # (T, K, 7)
        self.retargeted_right_hand_frames = md.right_hand_frames
        self.retargeted_left_hand_frame_names = md.left_hand_frame_names
        self.retargeted_right_hand_frame_names = md.right_hand_frame_names

        # Fingertip indices within hand frames
        for side in ("left", "right"):
            frame_names = getattr(self, f"retargeted_{side}_hand_frame_names") or []
            tip_body_ids = getattr(self, f"_{side}_fingertip_body_ids")
            tip_body_names = (
                [self.robot.body_names[i] for i in tip_body_ids] if tip_body_ids else []
            )
            indices = []
            for tname in tip_body_names:
                if tname in frame_names:
                    indices.append(frame_names.index(tname))
            setattr(
                self,
                f"_retargeted_{side}_fingertip_indices",
                torch.tensor(indices, dtype=torch.long, device=self.device),
            )

        for side in ("left", "right"):
            parquet_vals = getattr(md, f"{side}_finger_joints")
            parquet_names = getattr(md, f"{side}_finger_joint_names")
            sim_names = getattr(self, f"_{side}_finger_joint_names")
            if parquet_vals is not None and parquet_names and sim_names:
                reorder = [parquet_names.index(n) for n in sim_names]
                setattr(
                    self, f"retargeted_{side}_finger_joints", parquet_vals[:, reorder]
                )
            else:
                setattr(self, f"retargeted_{side}_finger_joints", parquet_vals)

        # Binary per-frame contact labels.
        # If a side is absent on disk, we substitute an all-zero mask of length
        # `num_timesteps` so the getters stay branch-free and tensor-typed. The
        # downstream `force_closure_reward` multiplies by these, so a missing
        # mask silently zeros the reward term; warn loudly once per init.
        left_active = md.left_hand_contact_active
        right_active = md.right_hand_contact_active
        missing_sides = [
            side
            for side, tensor in (("left", left_active), ("right", right_active))
            if tensor is None
        ]
        if len(missing_sides) == 2:
            raise ValueError(
                "TrackingCommand: motion data is missing per-frame contact-active "
                "labels for both hands. `force_closure_reward` and any other "
                "consumer of `{left,right}_hand_contact_active_command` would "
                "contribute 0 for the entire motion, which silently disables "
                "contact-driven learning. Re-run the retargeter so "
                "`left_hand_contact_active` / `right_hand_contact_active` are "
                f"populated (motion_file={self.cfg.motion_file})."
            )
        if missing_sides:
            warnings.warn(
                "TrackingCommand: motion data is missing per-frame contact-active "
                f"labels for side(s) {missing_sides}. Falling back to all-zero masks; "
                "`force_closure_reward` and any other consumer of "
                "`{left,right}_hand_contact_active_command` will contribute 0 from "
                f"this motion file ({self.cfg.motion_file}).",
                stacklevel=2,
            )
        zero_mask = torch.zeros(self.num_timesteps, device=self.device)
        self.retargeted_left_contact_active = (
            left_active.to(self.device) if left_active is not None else zero_mask
        )
        self.retargeted_right_contact_active = (
            right_active.to(self.device) if right_active is not None else zero_mask
        )

    def _init_contact_data(self) -> None:
        """Load contact positions, normals, and part IDs from motion data."""
        md = self._motion_data
        for side in ("left", "right"):
            link_contacts = getattr(md, f"{side}_link_contact_positions")  # (T, N, 4)
            obj_contacts = getattr(md, f"{side}_object_contact_positions")  # (T, N, 4)
            link_normals = getattr(
                md, f"{side}_link_contact_normals"
            )  # (T, N, 4) or None
            obj_normals = getattr(
                md, f"{side}_object_contact_normals"
            )  # (T, N, 4) or None
            part_ids = getattr(md, f"{side}_object_contact_part_ids")  # (T, N) or None

            if link_contacts is not None:
                # Normalize link contact normals
                if link_normals is not None:
                    norms = (
                        link_normals[..., :3].norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    )
                    link_normals = link_normals.clone()
                    link_normals[..., :3] = link_normals[..., :3] / norms

                # Extract part IDs from 4th column if not separately provided
                if part_ids is None and link_contacts.shape[-1] > 3:
                    part_ids = link_contacts[..., 3].long()
                is_valid = (
                    part_ids > 0
                    if part_ids is not None
                    else link_contacts[..., :3].abs().sum(dim=-1) > 1e-5
                )
                has_contact = is_valid.any(dim=-1)

                setattr(
                    self, f"retargeted_{side}_link_contact_positions_e", link_contacts
                )
                setattr(
                    self, f"retargeted_{side}_object_contact_positions_e", obj_contacts
                )
                setattr(self, f"retargeted_{side}_link_contact_normals_e", link_normals)
                setattr(
                    self, f"retargeted_{side}_object_contact_normals_e", obj_normals
                )
                setattr(self, f"retargeted_{side}_object_contact_part_ids", part_ids)
                setattr(self, f"retargeted_{side}_object_contact_is_valid", is_valid)
                setattr(self, f"retargeted_{side}_object_has_contact", has_contact)
                setattr(self, f"num_contacts_{side}", link_contacts.shape[1])
                setattr(self, f"num_retargeted_contacts_{side}", link_contacts.shape[1])
            else:
                for attr in (
                    "link_contact_positions_e",
                    "object_contact_positions_e",
                    "link_contact_normals_e",
                    "object_contact_normals_e",
                    "object_contact_part_ids",
                    "object_contact_is_valid",
                    "object_has_contact",
                ):
                    setattr(self, f"retargeted_{side}_{attr}", None)
                setattr(self, f"num_contacts_{side}", 0)
                setattr(self, f"num_retargeted_contacts_{side}", 0)

    def _precompute_hand_keypoints_in_object_frame(self) -> None:
        """Express hand frames and wrist poses in the contacted object's local frame.

        Stored as object-frame tensors so they can be quickly transformed
        to env frame each step using the current object pose.
        """
        horizon = self._object_body_pos_w.shape[0]
        horizon_ids = torch.arange(horizon, device=self.device)

        for side in ("left", "right"):
            frames = getattr(self, f"retargeted_{side}_hand_frames")
            wrist_pos = getattr(self, f"retargeted_{side}_wrist_position")
            wrist_wxyz = getattr(self, f"retargeted_{side}_wrist_wxyz")
            part_ids = getattr(self, f"retargeted_{side}_object_contact_part_ids", None)

            part_ids_per_hand = torch.ones(
                horizon, dtype=torch.int64, device=self.device
            )
            if part_ids is not None:
                T_ids = min(horizon, part_ids.shape[0])
                last_contact_part_id = torch.ones(
                    (), dtype=torch.int64, device=self.device
                )
                for horizon_idx in range(T_ids - 1, -1, -1):
                    hand_contact_part_id = part_ids[horizon_idx].mode().values
                    part_ids_per_hand[horizon_idx] = (
                        hand_contact_part_id
                        if hand_contact_part_id > 0
                        else last_contact_part_id
                    )
                    last_contact_part_id = part_ids_per_hand[horizon_idx]
            part_ids_per_hand = part_ids_per_hand.clamp(min=1, max=self.num_bodies)
            setattr(
                self,
                f"retargeted_{side}_object_contact_part_ids_per_hand",
                part_ids_per_hand,
            )

            contact_body_position = self._object_body_pos_w[
                horizon_ids, part_ids_per_hand - 1
            ]
            contact_body_wxyz = self._object_body_quat_w[
                horizon_ids, part_ids_per_hand - 1
            ]

            if frames is not None and wrist_pos is not None:
                T_frames = min(frames.shape[0], horizon)

                # Hand frame positions → object frame
                frame_pos = frames[:T_frames, :, :3]  # (T, K, 3)
                obj_pos_exp = contact_body_position[:T_frames].unsqueeze(1)
                obj_quat_exp = contact_body_wxyz[:T_frames].unsqueeze(1)
                frame_pos_o = math_utils.quat_apply_inverse(
                    obj_quat_exp.expand_as(frame_pos[..., :1].expand(-1, -1, 4)),
                    frame_pos - obj_pos_exp,
                )
                setattr(self, f"retargeted_{side}_hand_frame_positions_o", frame_pos_o)

                if frames.shape[-1] >= 7:
                    frame_wxyz_o = math_utils.quat_mul(
                        math_utils.quat_conjugate(
                            obj_quat_exp.expand(-1, frame_pos.shape[1], -1)
                        ),
                        frames[:T_frames, :, 3:7],
                    )
                    setattr(
                        self,
                        f"retargeted_{side}_hand_frame_wxyz_o",
                        frame_wxyz_o,
                    )

                # Wrist pose → object frame
                T_wrist = min(wrist_pos.shape[0], horizon)
                wrist_pos_o = math_utils.quat_apply_inverse(
                    contact_body_wxyz[:T_wrist],
                    wrist_pos[:T_wrist] - contact_body_position[:T_wrist],
                )
                wrist_wxyz_o = math_utils.quat_mul(
                    math_utils.quat_conjugate(contact_body_wxyz[:T_wrist]),
                    wrist_wxyz[:T_wrist],
                )
                setattr(self, f"retargeted_{side}_wrist_position_o", wrist_pos_o)
                setattr(self, f"retargeted_{side}_wrist_wxyz_o", wrist_wxyz_o)

    def _precompute_contact_positions_in_object_frame(self) -> None:
        """Transform contact positions and normals into per-body object/COM frames."""
        object_o_t_com = torch.cat(
            [obj.data.body_com_pose_b for obj in self.objects], dim=1
        ).float()
        object_o_p_com = object_o_t_com[..., :3].mean(dim=0)
        object_o_q_com = object_o_t_com[0, :, 3:7]

        for side in ("left", "right"):
            obj_contacts = getattr(
                self, f"retargeted_{side}_object_contact_positions_e"
            )
            link_normals = getattr(self, f"retargeted_{side}_link_contact_normals_e")
            contact_part_ids = getattr(
                self, f"retargeted_{side}_object_contact_part_ids", None
            )

            if obj_contacts is not None:
                T_c = min(obj_contacts.shape[0], self._object_body_pos_w.shape[0])
                contact_pos = obj_contacts[:T_c, :, :3]
                horizon_ids = torch.arange(T_c, device=self.device)
                if contact_part_ids is None:
                    if obj_contacts.shape[-1] > 3:
                        contact_part_ids = obj_contacts[..., 3].long()
                    else:
                        contact_part_ids = torch.ones(
                            obj_contacts.shape[:2],
                            dtype=torch.long,
                            device=self.device,
                        )
                is_valid = getattr(self, f"retargeted_{side}_object_contact_is_valid")[
                    :T_c
                ]

                contact_pos_o = torch.zeros_like(contact_pos)
                contact_pos_com = torch.zeros_like(contact_pos)
                normals_o = None
                normals_com = None
                if link_normals is not None:
                    normals = link_normals[:T_c, :, :3]
                    normals_o = torch.zeros_like(normals)
                    normals_com = torch.zeros_like(normals)

                for link_idx in range(contact_pos.shape[1]):
                    part_id = (contact_part_ids[:T_c, link_idx] - 1).clamp(
                        min=0, max=self.num_bodies - 1
                    )
                    object_e_p_o = self._object_body_pos_w[horizon_ids, part_id]
                    object_e_q_o = self._object_body_quat_w[horizon_ids, part_id]

                    contact_pos_o[:, link_idx] = math_utils.quat_apply_inverse(
                        object_e_q_o,
                        contact_pos[:, link_idx] - object_e_p_o,
                    )

                    _object_o_p_com = object_o_p_com[part_id]
                    _object_o_q_com = object_o_q_com[part_id]
                    contact_pos_com[:, link_idx], _ = (
                        math_utils.subtract_frame_transforms(
                            _object_o_p_com,
                            _object_o_q_com,
                            contact_pos_o[:, link_idx],
                            q02=None,
                        )
                    )

                    if normals_o is not None and normals_com is not None:
                        normals_o[:, link_idx] = math_utils.quat_apply_inverse(
                            object_e_q_o,
                            normals[:, link_idx],
                        )
                        normals_com[:, link_idx], _ = (
                            math_utils.subtract_frame_transforms(
                                torch.zeros_like(_object_o_p_com),
                                _object_o_q_com,
                                normals_o[:, link_idx],
                                q02=None,
                            )
                        )

                contact_pos_o.masked_fill_(~is_valid.unsqueeze(-1), 0.0)
                contact_pos_com.masked_fill_(~is_valid.unsqueeze(-1), 0.0)
                setattr(
                    self,
                    f"retargeted_{side}_object_contact_positions_com",
                    contact_pos_com,
                )
                setattr(
                    self,
                    f"retargeted_{side}_object_contact_positions_o",
                    contact_pos_o,
                )

                if normals_o is not None and normals_com is not None:
                    normals_o.masked_fill_(~is_valid.unsqueeze(-1), 0.0)
                    normals_com.masked_fill_(~is_valid.unsqueeze(-1), 0.0)
                    setattr(
                        self, f"retargeted_{side}_link_contact_normals_o", normals_o
                    )
                    setattr(
                        self, f"retargeted_{side}_link_contact_normals_com", normals_com
                    )

    def _init_wrench_data(self, cfg: TrackingCommandCfg) -> None:
        """Set up wrench computation buffers: basis, friction cone, mesh radius."""
        md = self._motion_data
        self.object_mesh_radius = list(md.object_mesh_radius or [0.05])
        self.object_mesh_radius = self.object_mesh_radius[: self.num_bodies]
        if len(self.object_mesh_radius) < self.num_bodies:
            self.object_mesh_radius = self.object_mesh_radius + [
                self.object_mesh_radius[-1]
            ] * (self.num_bodies - len(self.object_mesh_radius))
        self.retargeted_horizon = min(
            self._object_body_pos_w.shape[0],
            (
                getattr(
                    self, "retargeted_left_link_contact_positions_e", torch.empty(0)
                ).shape[0]
                if getattr(self, "retargeted_left_link_contact_positions_e", None)
                is not None
                else 0
            ),
        )

        self.wrench_space_bases = torch.cat(
            [
                sample_wrench_space_basis_scaled(
                    cfg.num_wrench_space_basis_samples, rc=1.0, device=self.device
                ).unsqueeze(0)
                for _ in self.object_mesh_radius
            ],
            dim=0,
        )

        theta = torch.linspace(
            0, 2 * torch.pi, steps=cfg.num_friction_cone_edges + 1, device=self.device
        )[:-1]
        self.friction_cone_edge_cosines = torch.cos(theta).view(1, -1, 1)
        self.friction_cone_edge_sines = torch.sin(theta).view(1, -1, 1)

        # Buffers for runtime wrench supports
        self.left_contact_wrench_supports = torch.zeros(
            self.num_envs,
            self.num_bodies,
            cfg.num_wrench_space_basis_samples,
            device=self.device,
        )
        self.right_contact_wrench_supports = torch.zeros(
            self.num_envs,
            self.num_bodies,
            cfg.num_wrench_space_basis_samples,
            device=self.device,
        )
        self._tensors_dirty = True

    def _precompute_contact_wrench_support_values(self) -> None:
        """Precompute wrench supports for all timesteps from retargeted contacts."""
        cfg = self.cfg
        T = self.retargeted_horizon
        if T == 0:
            return

        t_idx = torch.arange(T, device=self.device)[:, None]

        for side in ("left", "right"):
            num_contacts = getattr(self, f"num_retargeted_contacts_{side}", 0)
            if num_contacts == 0:
                continue

            c_idx = torch.arange(num_contacts, device=self.device)[None, :]
            contact_pos_com = getattr(
                self, f"retargeted_{side}_object_contact_positions_com"
            )
            contact_normals_com = getattr(
                self, f"retargeted_{side}_link_contact_normals_com", None
            )

            if contact_pos_com is None or contact_normals_com is None:
                continue

            # Expand to (T, num_bodies, num_contacts, 3)
            pos_expanded = torch.zeros(
                T, self.num_bodies, num_contacts, 3, device=self.device
            )
            normals_expanded = torch.zeros(
                T, self.num_bodies, num_contacts, 3, device=self.device
            )

            part_ids = getattr(self, f"retargeted_{side}_object_contact_part_ids")
            if part_ids is not None:
                part_ids_clamped = (part_ids[:T] - 1).clamp(
                    min=0, max=self.num_bodies - 1
                )
            else:
                part_ids_clamped = torch.zeros(
                    T, num_contacts, dtype=torch.long, device=self.device
                )

            is_valid = getattr(self, f"retargeted_{side}_object_contact_is_valid")[
                :T
            ].unsqueeze(-1)
            pos_expanded[t_idx, part_ids_clamped, c_idx] = (
                contact_pos_com[:T] * is_valid
            )
            normals_expanded[t_idx, part_ids_clamped, c_idx] = (
                contact_normals_com[:T] * is_valid
            )

            # Compute wrench supports per body
            supports = torch.zeros(
                T,
                self.num_bodies,
                cfg.num_wrench_space_basis_samples,
                device=self.device,
            )
            for body_idx, body_radius in enumerate(self.object_mesh_radius):
                wrench_space = compute_wrench_space(
                    contact_points=pos_expanded[:, body_idx],
                    contact_normals=normals_expanded[:, body_idx],
                    cos_t=self.friction_cone_edge_cosines,
                    sin_t=self.friction_cone_edge_sines,
                    rc=body_radius,
                    friction_coefficients=cfg.friction_coefficients,
                )
                supports[:, body_idx] = compute_wrench_space_support_function(
                    wrench_space=wrench_space,
                    basis=self.wrench_space_bases[body_idx],
                )

            setattr(self, f"retargeted_{side}_contact_wrench_supports", supports)

    # ------------------------------------------------------------------
    # Properties: current simulation state
    # ------------------------------------------------------------------

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Current anchor (pelvis) position in world frame. (E, 3)."""
        return self.robot.data.body_pos_w[:, self._anchor_body_id]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Current anchor (pelvis) quaternion in world frame. (E, 4)."""
        return self.robot.data.body_quat_w[:, self._anchor_body_id]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """Current tracked joint positions. (E, J)."""
        return self.robot.data.joint_pos[:, self._tracked_joint_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """Current tracked joint velocities. (E, J)."""
        return self.robot.data.joint_vel[:, self._tracked_joint_ids]

    @property
    def robot_ee_pos_w(self) -> torch.Tensor:
        """Current EE positions in world frame. (E, num_ee, 3)."""
        if hasattr(self, "ee_link_ids") and self.ee_link_ids:
            return self.robot.data.body_pos_w[:, self.ee_link_ids]
        return torch.zeros(self.num_envs, 0, 3, device=self.device)

    @property
    def robot_ee_quat_w(self) -> torch.Tensor:
        """Current EE quaternions in world frame. (E, num_ee, 4)."""
        if hasattr(self, "ee_link_ids") and self.ee_link_ids:
            return self.robot.data.body_quat_w[:, self.ee_link_ids]
        return torch.zeros(self.num_envs, 0, 4, device=self.device)

    @property
    def object_pos_w(self) -> torch.Tensor:
        """Current object position in world frame. (E, 3)."""
        return self.object.data.root_pos_w

    @property
    def object_quat_w(self) -> torch.Tensor:
        """Current object quaternion in world frame. (E, 4)."""
        return self.object.data.root_quat_w

    @property
    def encoder_mode(self) -> torch.Tensor:
        """SONIC encoder mode flags. (E, 4)."""
        return self._encoder_mode

    @property
    def timestep_counter(self) -> torch.Tensor:
        """Alias for timestep (backward compat). (E,)."""
        return self.timestep

    # ------------------------------------------------------------------
    # Properties: command targets (indexed by timestep)
    # ------------------------------------------------------------------

    @property
    def future_timesteps(self) -> torch.Tensor:
        """Clamped future frame indices per env. (E, num_future_frames)."""
        future_timesteps = torch.clamp(
            self.timestep[:, None] + self._future_frame_offsets[None, :],
            0,
            self.num_timesteps - 1,
        )
        return torch.minimum(future_timesteps, self.trajectory_end_timestep[:, None])

    def update_action_history(self, actions: torch.Tensor) -> None:
        """Push current processed actions into the history buffer."""
        if self._action_history is None:
            self._action_history = torch.zeros(
                self.num_envs,
                self._action_history_len,
                actions.shape[-1],
                device=self.device,
            )
        self._action_history = torch.roll(self._action_history, -1, dims=1)
        self._action_history[:, -1] = actions

    @property
    def action_history(self) -> torch.Tensor:
        """(num_envs, history_len * action_dim) flattened past actions."""
        if self._action_history is None:
            # First call — allocate from action term's processed action dim
            action_dim = self._env.action_manager.get_term(
                "joint_pos"
            ).processed_actions.shape[-1]
            self._action_history = torch.zeros(
                self.num_envs, self._action_history_len, action_dim, device=self.device
            )
        return self._action_history.reshape(self.num_envs, -1)

    @property
    def _spread_blend_factor(self) -> torch.Tensor:
        """Per-env blend: 1.0 at reset, anneals to 0.0 before VOC decay starts."""
        if self.cfg.reset_shoulder_spread == 0.0 or self.cfg.reset_freeze_steps == 0:
            return torch.zeros(self.num_envs, device=self.device)
        anneal_steps = max(self.cfg.reset_freeze_steps - self.cfg.voc_decay_steps, 1)
        return (1.0 - self.steps_since_last_reset.float() / anneal_steps).clamp(min=0.0)

    @property
    def _spread_offset_blended(self) -> torch.Tensor:
        """Per-env blended spread offset. (num_envs, num_joints)."""
        return self._spread_joint_offset * self._spread_blend_factor.unsqueeze(-1)

    @property
    def command(self) -> torch.Tensor:
        """Current command tensor (joint position targets)."""
        return self.command_joint_pos

    @property
    def command_anchor_pos_w(self) -> torch.Tensor:
        """Target anchor position in world frame. (E, 3)."""
        return self.root_pos_w[self.timestep] + self._env_origins

    @property
    def command_anchor_quat_w(self) -> torch.Tensor:
        """Target anchor quaternion in world frame. (E, 4)."""
        return math_utils.quat_unique(self.root_quat_w[self.timestep])

    @property
    def command_joint_pos(self) -> torch.Tensor:
        """Target joint positions with spread offset. (E, J)."""
        return self.joint_pos[self.timestep] + self._spread_offset_blended

    @property
    def command_object_pos_w(self) -> torch.Tensor:
        """Target object position in world frame. (E, 3)."""
        return self._object_pos_w[self.timestep] + self._env_origins

    @property
    def command_object_quat_w(self) -> torch.Tensor:
        """Target object quaternion in world frame. (E, 4)."""
        return math_utils.quat_unique(self._object_quat_w[self.timestep])

    @property
    def command_ee_pos_w(self) -> torch.Tensor:
        """Target EE positions in world frame. (E, num_ee, 3)."""
        if hasattr(self, "ee_pos_w"):
            return self.ee_pos_w[self.timestep] + self._env_origins.unsqueeze(1)
        return torch.zeros(self.num_envs, 0, 3, device=self.device)

    @property
    def command_ee_quat_w(self) -> torch.Tensor:
        """Target EE quaternions in world frame. (E, num_ee, 4)."""
        ee_quat = getattr(self, "ee_quat_w", None)
        if ee_quat is not None:
            return math_utils.quat_unique(ee_quat[self.timestep])
        return torch.zeros(self.num_envs, 0, 4, device=self.device)

    # Multi-future variants
    @property
    def command_joint_pos_multi_future(self) -> torch.Tensor:
        """Future joint position targets with spread. (E, F, J)."""
        return self.joint_pos[
            self.future_timesteps
        ] + self._spread_offset_blended.unsqueeze(1)

    @property
    def command_joint_vel_multi_future(self) -> torch.Tensor:
        """Future joint velocity targets. (E, F, J)."""
        return self.joint_vel[self.future_timesteps]

    @property
    def command_anchor_pos_w_multi_future(self) -> torch.Tensor:
        """Future anchor positions in world frame. (E, F, 3)."""
        return self.root_pos_w[self.future_timesteps] + self._env_origins.unsqueeze(1)

    @property
    def command_ee_pos_w_multi_future(self) -> torch.Tensor:
        """Future EE positions in world frame. (E, F, num_ee, 3)."""
        if hasattr(self, "ee_pos_w"):
            return self.ee_pos_w[self.future_timesteps] + self._env_origins.unsqueeze(
                1
            ).unsqueeze(1)
        return torch.zeros(
            self.num_envs, self.num_future_frames, 0, 3, device=self.device
        )

    @property
    def command_ee_quat_w_multi_future(self) -> torch.Tensor:
        """Future EE quaternions in world frame. (E, F, num_ee, 4)."""
        ee_quat = getattr(self, "ee_quat_w", None)
        if ee_quat is not None:
            return math_utils.quat_unique(ee_quat[self.future_timesteps])
        return torch.zeros(
            self.num_envs, self.num_future_frames, 0, 4, device=self.device
        )

    @property
    def command_object_pos_w_multi_future(self) -> torch.Tensor:
        """Future object positions in world frame. (E, F, 3)."""
        return self._object_pos_w[self.future_timesteps] + self._env_origins.unsqueeze(
            1
        )

    @property
    def command_anchor_rot_diff_l_multi_future(self) -> torch.Tensor:
        """(E, F, 6) future root rotation differences in 6D rotation format."""
        future_quat = self.root_quat_w[self.future_timesteps]  # (E, F, 4)
        current_quat = self.root_quat_w[self.timestep]  # (E, 4)
        # Compute relative rotation: q_diff = q_future * q_current^-1
        q_inv = math_utils.quat_conjugate(current_quat).unsqueeze(1)  # (E, 1, 4)
        q_diff = math_utils.quat_mul(future_quat, q_inv.expand_as(future_quat))
        # Convert to 6D rotation (first two columns of rotation matrix)
        rot_mat = math_utils.matrix_from_quat(q_diff)  # (E, F, 3, 3)
        return rot_mat[..., :2].reshape(self.num_envs, self.num_future_frames, 6)

    @property
    def command_anchor_z_multi_future(self) -> torch.Tensor:
        """(E, F) future root Z positions."""
        return self.root_pos_w[self.future_timesteps][..., 2]  # (E, F)

    @property
    def command_multi_future(self) -> torch.Tensor:
        """Future joint pos + vel concatenated. (E, F, 2*J)."""
        return torch.cat(
            [self.command_joint_pos_multi_future, self.command_joint_vel_multi_future],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Properties: hand keypoints (object-frame → env-frame per step)
    # ------------------------------------------------------------------

    @property
    def left_hand_wrist_pose_command_e(self) -> torch.Tensor:
        """(E, 7) left wrist [pos(3), wxyz(4)] in env frame."""
        return self._wrist_command_e("left")

    @property
    def right_hand_wrist_pose_command_e(self) -> torch.Tensor:
        """(E, 7) right wrist [pos(3), wxyz(4)] in env frame."""
        return self._wrist_command_e("right")

    @property
    def left_hand_wrist_position_e(self) -> torch.Tensor:
        """(E, 3) current left wrist position in env frame."""
        if self._left_wrist_body_id is not None:
            return (
                self.robot.data.body_pos_w[:, self._left_wrist_body_id]
                - self._env_origins
            )
        return torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def right_hand_wrist_position_e(self) -> torch.Tensor:
        """(E, 3) current right wrist position in env frame."""
        if self._right_wrist_body_id is not None:
            return (
                self.robot.data.body_pos_w[:, self._right_wrist_body_id]
                - self._env_origins
            )
        return torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def left_hand_wrist_wxyz_e(self) -> torch.Tensor:
        """(E, 4) current left wrist quaternion."""
        if self._left_wrist_body_id is not None:
            return math_utils.quat_unique(
                self.robot.data.body_quat_w[:, self._left_wrist_body_id]
            )
        quat = torch.zeros(self.num_envs, 4, device=self.device)
        quat[:, 0] = 1.0
        return quat

    @property
    def right_hand_wrist_wxyz_e(self) -> torch.Tensor:
        """(E, 4) current right wrist quaternion."""
        if self._right_wrist_body_id is not None:
            return math_utils.quat_unique(
                self.robot.data.body_quat_w[:, self._right_wrist_body_id]
            )
        quat = torch.zeros(self.num_envs, 4, device=self.device)
        quat[:, 0] = 1.0
        return quat

    def get_command_contact_object_position_orientation(
        self, side: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the current contacted object body pose for a hand side."""
        contact_part_id = self.get_command_contact_part_id(side)
        object_position = self.object_position_e[self.all_env_ids, contact_part_id]
        object_orientation = self.object_orientation_e[
            self.all_env_ids, contact_part_id
        ]
        return object_position, object_orientation

    @property
    def left_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """(E, K, 3) left fingertip command positions in env frame."""
        return self._fingertip_command_e("left")

    @property
    def right_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """(E, K, 3) right fingertip command positions in env frame."""
        return self._fingertip_command_e("right")

    @property
    def left_hand_fingertip_position_e(self) -> torch.Tensor:
        """(E, K, 3) current left fingertip positions in env frame."""
        if self._left_fingertip_body_ids:
            return self.robot.data.body_pos_w[
                :, self._left_fingertip_body_ids
            ] - self._env_origins.unsqueeze(1)
        return torch.zeros(self.num_envs, 0, 3, device=self.device)

    @property
    def right_hand_fingertip_position_e(self) -> torch.Tensor:
        """(E, K, 3) current right fingertip positions in env frame."""
        if self._right_fingertip_body_ids:
            return self.robot.data.body_pos_w[
                :, self._right_fingertip_body_ids
            ] - self._env_origins.unsqueeze(1)
        return torch.zeros(self.num_envs, 0, 3, device=self.device)

    # ------------------------------------------------------------------
    # Properties: contact tracking
    # ------------------------------------------------------------------

    @property
    def left_hand_object_contact_command_positions_e(self) -> torch.Tensor:
        """(E, N, 3) left hand target contact positions in env frame."""
        return self._contact_command_e("left")

    @property
    def right_hand_object_contact_command_positions_e(self) -> torch.Tensor:
        """(E, N, 3) right hand target contact positions in env frame."""
        return self._contact_command_e("right")

    @property
    def left_hand_object_contact_positions_e(self) -> torch.Tensor:
        """(E, B, N, 3) live left hand contact positions in env frame."""
        return self._live_contact_positions_e("left")

    @property
    def right_hand_object_contact_positions_e(self) -> torch.Tensor:
        """(E, B, N, 3) live right hand contact positions in env frame."""
        return self._live_contact_positions_e("right")

    @property
    def left_hand_object_contact_positions_w(self) -> torch.Tensor:
        """(E, B, N, 3) live left hand contact positions in world frame."""
        return self._live_contact_positions_w("left")

    @property
    def right_hand_object_contact_positions_w(self) -> torch.Tensor:
        """(E, B, N, 3) live right hand contact positions in world frame."""
        return self._live_contact_positions_w("right")

    @property
    def left_hand_object_contact_forces_w(self) -> torch.Tensor:
        """(E, H, B, N, 3) left hand contact force history in world frame."""
        return self._live_contact_forces_w("left")

    @property
    def right_hand_object_contact_forces_w(self) -> torch.Tensor:
        """(E, H, B, N, 3) right hand contact force history in world frame."""
        return self._live_contact_forces_w("right")

    # ------------------------------------------------------------------
    # Properties: VOC
    # ------------------------------------------------------------------

    @property
    def object_position_e(self) -> torch.Tensor:
        """Object body positions in env frame. (E, B, 3) over all object bodies."""
        object_position_w = torch.cat(
            [obj.data.body_link_pos_w for obj in self.objects], dim=1
        )
        return (object_position_w - self._env_origins.unsqueeze(1)).float()

    @property
    def object_orientation_e(self) -> torch.Tensor:
        """Object body quaternions in env frame. (E, B, 4) over all object bodies."""
        return torch.cat(
            [obj.data.body_link_quat_w for obj in self.objects], dim=1
        ).float()

    @property
    def object_body_position_command_e(self) -> torch.Tensor:
        """(E, B, 3) command object body positions in env frame."""
        return self._object_body_pos_w[self.timestep]

    @property
    def object_body_wxyz_command_e(self) -> torch.Tensor:
        """(E, B, 4) command object body quaternions in env frame."""
        return math_utils.quat_unique(self._object_body_quat_w[self.timestep])

    @property
    def object_body_ids(self) -> torch.Tensor:
        """Object body indices. (B,)."""
        return torch.arange(self.num_bodies, device=self.device)

    @property
    def object_com_position_and_wxyz_w(self) -> torch.Tensor:
        """(E, num_bodies, 7) object COM state."""
        return torch.cat(
            [obj.data.body_com_state_w[..., :7] for obj in self.objects], dim=1
        ).float()

    # ------------------------------------------------------------------
    # Properties: finger joints
    # ------------------------------------------------------------------

    @property
    def left_hand_finger_joint_pos(self) -> torch.Tensor:
        """Current left finger joint positions. (E, J_left)."""
        if self._left_finger_joint_ids:
            return self.robot.data.joint_pos[:, self._left_finger_joint_ids]
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def right_hand_finger_joint_pos(self) -> torch.Tensor:
        """Current right finger joint positions. (E, J_right)."""
        if self._right_finger_joint_ids:
            return self.robot.data.joint_pos[:, self._right_finger_joint_ids]
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def left_hand_finger_joint_pos_command(self) -> torch.Tensor:
        """Target left finger joint positions from retargeting. (E, J_left)."""
        if self.retargeted_left_finger_joints is not None:
            t = self.timestep.clamp(max=self.retargeted_left_finger_joints.shape[0] - 1)
            return self.retargeted_left_finger_joints[t]
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def right_hand_finger_joint_pos_command(self) -> torch.Tensor:
        """Target right finger joint positions from retargeting. (E, J_right)."""
        if self.retargeted_right_finger_joints is not None:
            t = self.timestep.clamp(
                max=self.retargeted_right_finger_joints.shape[0] - 1
            )
            return self.retargeted_right_finger_joints[t]
        return torch.zeros(self.num_envs, 0, device=self.device)

    # ------------------------------------------------------------------
    # Properties: contact positions + normals
    # ------------------------------------------------------------------

    @property
    def left_hand_object_contact_command_positions_and_normals_e(self) -> torch.Tensor:
        """(E, N, 6) contact command positions + normals in env frame."""
        return self._contact_command_pos_normals_e("left")

    @property
    def right_hand_object_contact_command_positions_and_normals_e(self) -> torch.Tensor:
        """(E, N, 6) right hand contact command positions + normals in env frame."""
        return self._contact_command_pos_normals_e("right")

    def get_command_contact_part_id(self, side: str) -> torch.Tensor:
        """Get dominant contact body index per env. (E,) — 0-indexed."""
        per_hand = getattr(
            self, f"retargeted_{side}_object_contact_part_ids_per_hand", None
        )
        if per_hand is not None:
            t = self.timestep.clamp(max=per_hand.shape[0] - 1)
            return (per_hand[t] - 1).clamp(min=0, max=self.num_bodies - 1)

        part_ids = getattr(self, f"retargeted_{side}_object_contact_part_ids")
        if part_ids is not None:
            t = self.timestep.clamp(max=part_ids.shape[0] - 1)
            # Mode across contact links, then clamp to valid body range
            per_frame = part_ids[t]  # (E, N)
            dominant = per_frame.mode(dim=-1).values - 1  # 1-indexed → 0-indexed
            return dominant.clamp(min=0, max=self.num_bodies - 1)
        return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Properties: wrench
    # ------------------------------------------------------------------

    @property
    def left_hand_contact_wrench_supports_command(self) -> torch.Tensor:
        """(E, num_bodies, num_basis) precomputed wrench supports from retargeted data."""
        supports = getattr(self, "retargeted_left_contact_wrench_supports", None)
        if supports is not None:
            t = self.timestep.clamp(max=supports.shape[0] - 1)
            return supports[t]
        return torch.zeros(
            self.num_envs,
            self.num_bodies,
            self.cfg.num_wrench_space_basis_samples,
            device=self.device,
        )

    @property
    def right_hand_contact_wrench_supports_command(self) -> torch.Tensor:
        """(E, num_bodies, num_basis) precomputed right hand wrench supports."""
        supports = getattr(self, "retargeted_right_contact_wrench_supports", None)
        if supports is not None:
            t = self.timestep.clamp(max=supports.shape[0] - 1)
            return supports[t]
        return torch.zeros(
            self.num_envs,
            self.num_bodies,
            self.cfg.num_wrench_space_basis_samples,
            device=self.device,
        )

    @property
    def left_hand_contact_wrench_supports(self) -> torch.Tensor:
        """(E, num_bodies, num_basis) live wrench supports from sim contacts."""
        self.refresh_tensors()
        return self.left_contact_wrench_supports

    @property
    def right_hand_contact_wrench_supports(self) -> torch.Tensor:
        """(E, num_bodies, num_basis) live right hand wrench supports from sim."""
        self.refresh_tensors()
        return self.right_contact_wrench_supports

    @property
    def left_hand_contact_active_command(self) -> torch.Tensor:
        """(E,) binary contact label for left hand at current timestep."""
        t = self.timestep.clamp(max=self.retargeted_left_contact_active.shape[0] - 1)
        return self.retargeted_left_contact_active[t]

    @property
    def right_hand_contact_active_command(self) -> torch.Tensor:
        """(E,) binary contact label for right hand at current timestep."""
        t = self.timestep.clamp(max=self.retargeted_right_contact_active.shape[0] - 1)
        return self.retargeted_right_contact_active[t]

    # ------------------------------------------------------------------
    # Command lifecycle
    # ------------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Reset state for specified environments on episode reset.

        Note: timestep and reset_timestep are set by the reset event (events.py),
        which runs BEFORE _resample_command. Do not overwrite them here.
        Similarly, _spread_joint_offset is set by the reset event — not cleared here.
        """
        self.steps_since_last_reset[env_ids] = 0
        if self.cfg.voc_decay_steps > 0:
            self.virtual_object_controller_scale_factor_per_env[env_ids] = (
                self.cfg.voc_reset_scale
            )
        else:
            self.virtual_object_controller_scale_factor_per_env[env_ids] = (
                self.virtual_object_controller_scale_factor.item()
            )
        if self._action_history is not None:
            self._action_history[env_ids] = 0.0
        if hasattr(self, "_tensors_dirty"):
            self._tensors_dirty = True

    def _update_command(self) -> None:
        """Advance timestep and decay VOC."""
        self.steps_since_last_reset += 1

        # VOC decay
        voc_decay_steps = self.cfg.voc_decay_steps
        if voc_decay_steps > 0:
            decay_start = max(self.cfg.reset_freeze_steps - voc_decay_steps, 0)
            steps_into_decay = (
                self.steps_since_last_reset.float() - decay_start
            ).clamp(min=0.0)
            progress = (steps_into_decay / voc_decay_steps).clamp(max=1.0)
            target = self.virtual_object_controller_scale_factor.item()
            start = self.cfg.voc_reset_scale
            decayed = (start + (target - start) * progress).clamp(min=0.0)
            self.virtual_object_controller_scale_factor_per_env[:] = decayed.view(
                self.num_envs, 1
            )

        # Advance timestep — frozen until max(freeze_steps, voc_decay_steps)
        effective_freeze = max(self.cfg.reset_freeze_steps, voc_decay_steps)
        if effective_freeze > 0:
            past_freeze = self.steps_since_last_reset > effective_freeze
            self.timestep[past_freeze] += 1
        else:
            self.timestep += 1
        self.timestep.clamp_(0, self.num_timesteps - 1)
        self.timestep.copy_(torch.minimum(self.timestep, self.trajectory_end_timestep))
        if hasattr(self, "_tensors_dirty"):
            self._tensors_dirty = True

    def _update_metrics(self) -> None:
        """Track per-step errors for logging."""
        self.metrics["anchor_position_error"] = torch.norm(
            self.robot_anchor_pos_w - self.command_anchor_pos_w, dim=-1
        )
        self.metrics["anchor_wxyz_error"] = math_utils.quat_error_magnitude(
            self.robot_anchor_quat_w, self.command_anchor_quat_w
        )
        self.metrics["joint_pos_error"] = torch.norm(
            self.robot_joint_pos - self.command_joint_pos, dim=-1
        )

        self.metrics["object_body_position_error"] = torch.norm(
            self.object_position_e - self.object_body_position_command_e,
            dim=-1,
        ).mean(dim=-1)
        self.metrics["object_body_wxyz_error"] = math_utils.quat_error_magnitude(
            self.object_orientation_e,
            self.object_body_wxyz_command_e,
        ).mean(dim=-1)

        wrist_metrics = (
            (
                "left",
                self._left_wrist_body_id,
                self.left_hand_wrist_position_e,
                self.left_hand_wrist_pose_command_e,
            ),
            (
                "right",
                self._right_wrist_body_id,
                self.right_hand_wrist_position_e,
                self.right_hand_wrist_pose_command_e,
            ),
        )
        for (
            side,
            wrist_body_id,
            wrist_position_e,
            wrist_pose_command_e,
        ) in wrist_metrics:
            self.metrics[f"{side}_hand_wrist_position_error"] = torch.norm(
                wrist_position_e - wrist_pose_command_e[:, :3],
                dim=-1,
            )
            if wrist_body_id is not None:
                self.metrics[f"{side}_hand_wrist_wxyz_error"] = (
                    math_utils.quat_error_magnitude(
                        self.robot.data.body_quat_w[:, wrist_body_id],
                        wrist_pose_command_e[:, 3:],
                    )
                )
            else:
                self.metrics[f"{side}_hand_wrist_wxyz_error"].zero_()

        self.metrics["left_hand_finger_joints_error"] = torch.norm(
            self.left_hand_finger_joint_pos - self.left_hand_finger_joint_pos_command,
            dim=-1,
        )
        self.metrics["right_hand_finger_joints_error"] = torch.norm(
            self.right_hand_finger_joint_pos - self.right_hand_finger_joint_pos_command,
            dim=-1,
        )
        self.metrics["virtual_object_controller_scale_factor"] = (
            self.virtual_object_controller_scale_factor_per_env.squeeze(-1)
        )

    def _set_debug_vis_impl(self, debug_vis: bool = True) -> None:
        """Enable/disable debug visualization of tracking targets."""
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                cfg = self.cfg.pose_visualizer_cfg.replace(
                    prim_path="/Visuals/Command/goal_marker"
                )
                self.goal_pose_visualizer = VisualizationMarkers(cfg)
            if not hasattr(self, "object_pose_visualizer"):
                cfg = self.cfg.pose_visualizer_cfg.replace(
                    prim_path="/Visuals/Command/object_marker"
                )
                self.object_pose_visualizer = VisualizationMarkers(cfg)
            if not hasattr(self, "actual_object_pose_visualizer"):
                cfg = self.cfg.pose_visualizer_cfg.replace(
                    prim_path="/Visuals/Command/actual_object_marker"
                )
                cfg.markers["frame"].scale = (0.07, 0.07, 0.07)
                self.actual_object_pose_visualizer = VisualizationMarkers(cfg)
            ee_names = getattr(self, "ee_link_names", []) or []
            if not getattr(self, "ee_pose_visualizer", None):
                self.ee_pose_visualizer = {}
                for ee_name in ee_names:
                    cfg = self.cfg.pose_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/ee_marker_{ee_name}"
                    )
                    cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
                    self.ee_pose_visualizer[ee_name] = VisualizationMarkers(cfg)
            if not getattr(self, "wrist_pose_visualizer", None):
                self.wrist_pose_visualizer = {}
                for ee_name in ee_names:
                    cfg = self.cfg.pose_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Current/wrist_marker_{ee_name}"
                    )
                    cfg.markers["frame"].scale = (0.07, 0.07, 0.07)
                    self.wrist_pose_visualizer[ee_name] = VisualizationMarkers(cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.object_pose_visualizer.set_visibility(True)
            self.actual_object_pose_visualizer.set_visibility(True)
            for v in self.ee_pose_visualizer.values():
                v.set_visibility(True)
            for v in self.wrist_pose_visualizer.values():
                v.set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)
            self.object_pose_visualizer.set_visibility(False)
            self.actual_object_pose_visualizer.set_visibility(False)
            for v in self.ee_pose_visualizer.values():
                v.set_visibility(False)
            for v in self.wrist_pose_visualizer.values():
                v.set_visibility(False)

    def _debug_vis_callback(self, event: object) -> None:
        """Visualize tracking targets: anchor, object, EE."""
        if not hasattr(self, "goal_pose_visualizer"):
            return
        self.goal_pose_visualizer.visualize(
            translations=self.command_anchor_pos_w,
            orientations=self.command_anchor_quat_w,
        )
        self.object_pose_visualizer.visualize(
            translations=self.command_object_pos_w,
            orientations=self.command_object_quat_w,
        )
        self.actual_object_pose_visualizer.visualize(
            translations=self.object_pos_w,
            orientations=self.object_quat_w,
        )
        ee_names = getattr(self, "ee_link_names", []) or []
        for i, ee_name in enumerate(ee_names):
            if ee_name in self.ee_pose_visualizer:
                self.ee_pose_visualizer[ee_name].visualize(
                    translations=self.command_ee_pos_w[:, i],
                    orientations=self.command_ee_quat_w[:, i],
                )
            if ee_name in self.wrist_pose_visualizer and self.ee_link_ids:
                wrist_pos = self.robot.data.body_pos_w[:, self.ee_link_ids[i]]
                wrist_quat = self.robot.data.body_quat_w[:, self.ee_link_ids[i]]
                self.wrist_pose_visualizer[ee_name].visualize(
                    translations=wrist_pos,
                    orientations=wrist_quat,
                )

    # ------------------------------------------------------------------
    # Internal helpers for object-frame → env-frame transforms
    # ------------------------------------------------------------------

    def _wrist_command_e(self, side: str) -> torch.Tensor:
        """Wrist command in env frame from object-frame precomputation."""
        wrist_pos_o = getattr(self, f"retargeted_{side}_wrist_position_o", None)
        wrist_wxyz_o = getattr(self, f"retargeted_{side}_wrist_wxyz_o", None)
        if wrist_pos_o is None or wrist_wxyz_o is None:
            return torch.zeros(self.num_envs, 7, device=self.device)

        t = self.timestep.clamp(max=wrist_pos_o.shape[0] - 1)
        obj_pos, obj_quat = self.get_command_contact_object_position_orientation(side)

        pos_o = wrist_pos_o[t]  # (E, 3)
        wxyz_o = wrist_wxyz_o[t]  # (E, 4)

        pos_e = math_utils.quat_apply(obj_quat, pos_o) + obj_pos
        wxyz_e = math_utils.quat_mul(obj_quat, wxyz_o)
        return torch.cat([pos_e, wxyz_e], dim=-1)

    def _fingertip_command_e(self, side: str) -> torch.Tensor:
        """Fingertip command positions in env frame."""
        frame_pos_o = getattr(self, f"retargeted_{side}_hand_frame_positions_o", None)
        tip_indices = getattr(self, f"_retargeted_{side}_fingertip_indices", None)
        if frame_pos_o is None or tip_indices is None or len(tip_indices) == 0:
            return torch.zeros(self.num_envs, 0, 3, device=self.device)

        t = self.timestep.clamp(max=frame_pos_o.shape[0] - 1)
        tips_o = frame_pos_o[t][:, tip_indices]  # (E, K, 3)

        obj_pos, obj_quat = self.get_command_contact_object_position_orientation(side)
        obj_pos = obj_pos.unsqueeze(1)
        obj_quat = obj_quat.unsqueeze(1).expand_as(tips_o[..., :1].expand(-1, -1, 4))
        return math_utils.quat_apply(obj_quat, tips_o) + obj_pos

    def _contact_command_e(self, side: str) -> torch.Tensor:
        """Contact command positions in env frame."""
        contact_o = getattr(self, f"retargeted_{side}_object_contact_positions_o", None)
        if contact_o is None:
            n = getattr(self, f"num_contacts_{side}", 0)
            return torch.zeros(self.num_envs, n, 3, device=self.device)

        t = self.timestep.clamp(max=contact_o.shape[0] - 1)
        contacts = contact_o[t]  # (E, N, 3)

        object_position, object_orientation = self._contact_object_pose_e(
            side, contacts.shape[1]
        )
        pos_e = math_utils.quat_apply(object_orientation, contacts) + object_position
        valid = getattr(self, f"retargeted_{side}_object_contact_is_valid", None)
        if valid is not None:
            valid_t = valid[t]
            pos_e.masked_fill_(~valid_t.unsqueeze(-1), 0.0)
        return pos_e

    def _contact_object_pose_e(
        self, side: str, num_contacts: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Current object body poses for each contact link. Shapes: (E, N, 3/4)."""
        part_ids = getattr(self, f"retargeted_{side}_object_contact_part_ids", None)
        if part_ids is not None:
            t = self.timestep.clamp(max=part_ids.shape[0] - 1)
            contact_part_id = (part_ids[t] - 1).clamp(min=0, max=self.num_bodies - 1)
        else:
            contact_part_id = torch.zeros(
                self.num_envs, num_contacts, dtype=torch.long, device=self.device
            )
        object_position = torch.gather(
            self.object_position_e,
            dim=1,
            index=contact_part_id.unsqueeze(-1).expand(-1, -1, 3),
        )
        object_orientation = torch.gather(
            self.object_orientation_e,
            dim=1,
            index=contact_part_id.unsqueeze(-1).expand(-1, -1, 4),
        )
        return object_position, object_orientation

    def _live_contact_positions_w(self, side: str) -> torch.Tensor:
        """(E, B, N, 3) contact positions from per-body sensors."""
        sensors = getattr(self, f"object_to_{side}_hand_contact_sensors", [])
        num_contacts = getattr(self, f"num_robot_contacts_{side}", 0)
        if sensors:
            raw = torch.nan_to_num(
                torch.cat([s.data.contact_pos_w for s in sensors], dim=1), nan=0.0
            )
            return raw.view(self.num_envs, self.num_bodies, num_contacts, 3)
        return torch.zeros(self.num_envs, 0, 0, 3, device=self.device)

    def _live_contact_positions_e(self, side: str) -> torch.Tensor:
        pos_w = self._live_contact_positions_w(side)
        if pos_w.numel() > 0:
            return pos_w - self._env_origins.view(self.num_envs, 1, 1, 3)
        return pos_w

    def _live_contact_forces_w(self, side: str) -> torch.Tensor:
        """(E, H, B, N, 3) contact force history from per-body sensors."""
        sensors = getattr(self, f"object_to_{side}_hand_contact_sensors", [])
        num_contacts = getattr(self, f"num_robot_contacts_{side}", 0)
        if sensors:
            H = sensors[0].data.force_matrix_w_history.shape[1]
            raw = torch.nan_to_num(
                torch.cat([s.data.force_matrix_w_history for s in sensors], dim=2),
                nan=0.0,
            )
            return raw.view(self.num_envs, H, self.num_bodies, num_contacts, 3)
        return torch.zeros(self.num_envs, 0, 0, 0, 3, device=self.device)

    def _contact_command_pos_normals_e(self, side: str) -> torch.Tensor:
        """Combined contact positions + normals in env frame. (E, N, 6)."""
        contact_o = getattr(self, f"retargeted_{side}_object_contact_positions_o", None)
        normals_com = getattr(self, f"retargeted_{side}_link_contact_normals_com", None)
        if contact_o is None:
            n = getattr(self, f"num_contacts_{side}", 0)
            return torch.zeros(self.num_envs, n, 6, device=self.device)

        t = self.timestep.clamp(max=contact_o.shape[0] - 1)
        object_position, object_orientation = self._contact_object_pose_e(
            side, contact_o.shape[1]
        )

        pos_e = (
            math_utils.quat_apply(object_orientation, contact_o[t]) + object_position
        )
        if normals_com is not None:
            normals_e = math_utils.quat_apply(object_orientation, normals_com[t])
        else:
            normals_e = torch.zeros_like(pos_e)
        valid = getattr(self, f"retargeted_{side}_object_contact_is_valid", None)
        if valid is not None:
            valid_t = valid[t]
            pos_e.masked_fill_(~valid_t.unsqueeze(-1), 0.0)
            normals_e.masked_fill_(~valid_t.unsqueeze(-1), 0.0)
        return torch.cat([pos_e, normals_e], dim=-1)

    def refresh_tensors(self) -> None:
        """Refresh shared contact/wrench tensors once per sim step."""
        if not self._tensors_dirty:
            return
        self._tensors_dirty = False

        self._compute_contact_wrench_supports("right")
        self._compute_contact_wrench_supports("left")

        self._cached_right_wrench_cmd_supports = (
            self.right_hand_contact_wrench_supports_command
        )
        self._cached_left_wrench_cmd_supports = (
            self.left_hand_contact_wrench_supports_command
        )
        self._cached_right_wrench_cmd_active = (
            self._cached_right_wrench_cmd_supports > 1e-3
        )
        self._cached_left_wrench_cmd_active = (
            self._cached_left_wrench_cmd_supports > 1e-3
        )
        self._cached_right_wrench_cur_active = self.right_contact_wrench_supports > 1e-3
        self._cached_left_wrench_cur_active = self.left_contact_wrench_supports > 1e-3
        self._cached_right_wrench_cmd_active_per_body = (
            self._cached_right_wrench_cmd_active.any(dim=-1)
        )
        self._cached_left_wrench_cmd_active_per_body = (
            self._cached_left_wrench_cmd_active.any(dim=-1)
        )
        self._cached_right_wrench_cur_active_per_body = (
            self._cached_right_wrench_cur_active.any(dim=-1)
        )
        self._cached_left_wrench_cur_active_per_body = (
            self._cached_left_wrench_cur_active.any(dim=-1)
        )

    def _compute_contact_wrench_supports(self, side: str) -> None:
        """Fill live wrench supports in place for one hand."""
        sensors = getattr(self, f"object_to_{side}_hand_contact_sensors", [])
        supports = getattr(self, f"{side}_contact_wrench_supports")
        supports.zero_()
        if not sensors or not hasattr(self, "wrench_space_bases"):
            return

        num_contacts = getattr(self, f"num_robot_contacts_{side}", 0)
        if num_contacts == 0:
            return

        contact_pos_w = self._live_contact_positions_w(side)  # (E, B, N, 3)
        contact_forces_w = self._live_contact_forces_w(side)  # (E, H, B, N, 3)

        obj_com = self.object_com_position_and_wxyz_w  # (E, B, 7)
        object_com_position_w = obj_com[..., :3].unsqueeze(2)
        object_com_orientation_w = obj_com[..., 3:7].unsqueeze(2)

        contact_positions_com, contact_normals_com = wrench_preprocess_jit(
            contact_positions_w=contact_pos_w,
            contact_forces_first_hist_w=contact_forces_w[:, 0],
            object_com_position_w=object_com_position_w,
            object_com_orientation_w=object_com_orientation_w,
            num_envs=self.num_envs,
            num_bodies=self.num_bodies,
            num_robot_contacts=num_contacts,
        )
        friction_coefficients = float(self.cfg.friction_coefficients)
        for body_idx, body_radius in enumerate(self.object_mesh_radius):
            supports[:, body_idx] = wrench_support_one_body_jit(
                contact_points=contact_positions_com[:, body_idx],
                contact_normals=contact_normals_com[:, body_idx],
                cos_t=self.friction_cone_edge_cosines,
                sin_t=self.friction_cone_edge_sines,
                basis=self.wrench_space_bases[body_idx],
                rc=float(body_radius),
                friction_coefficients=friction_coefficients,
            )
