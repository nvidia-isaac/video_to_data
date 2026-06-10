# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Command definitions for the V2P environment."""

from __future__ import annotations

import collections
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Tuple

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers.visualization_markers import VisualizationMarkers

try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
except ImportError:
    print("[WARNING]: Debug draw is not available. No lines will be drawn.")
    omni_debug_draw = None

from robotic_grounding.retarget.data_logger import ManoSharpaData
from robotic_grounding.tasks.v2p.mdp.commands.maniptrans_metrics import (
    init_maniptrans_body_tables,
    register_maniptrans_metric_keys,
    update_maniptrans_metrics,
)
from robotic_grounding.tasks.v2p.mdp.commands.spider_metrics import (
    register_spider_metric_keys,
    update_spider_metrics,
)
from robotic_grounding.tasks.v2p.mdp.utils import (
    compute_wrench_space,
    compute_wrench_space_support_function,
    interpolate_robot_motion_data,
    sample_wrench_space_basis_scaled,
)
from robotic_grounding.tasks.v2p.mdp.utils_jit import (
    contact_wrench_support_reward_jit,
    refresh_jit,
    resample_compute_tensors_jit,
    wrench_preprocess_jit,
    wrench_support_one_body_jit,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ENABLE_ADDITIONAL_METRICS = False


class DualHandsObjectTrackingCommand(CommandTerm):
    """Command term that generates pose commands for dual-hand object tracking.

    This term is the central data hub for the tracking task. It bridges
    retargeted human hand motion data with the Isaac Lab simulation,
    providing both **command targets** (what the policy should achieve) and
    **current state observations** (where things are now).

    Initialization is split into private helpers — see each for details:

    1. :meth:`_init_scene_references` — resolve robot / object assets and
       cache body/joint indices and contact sensors.
    2. :meth:`_load_motion_data` — load retargeted motion from parquet and
       interpolate to the simulation FPS.
    3. :meth:`_init_buffers` — allocate per-env counters, reset-pose buffers,
       curriculum scale factors, and constant unit vectors.
    4. :meth:`_init_hand_data` — convert per-side hand motion arrays (wrist,
       fingers, frames, fingertip indices) to GPU tensors.
    5. :meth:`_init_object_data` — convert object body motion arrays to GPU
       tensors.
    6. :meth:`_init_contact_data` — load per-link contact positions, normals,
       and part IDs for both hands.
    7. :meth:`_precompute_contact_positions_in_object_frame` — transform
       contact positions into each contacted object body's local frame.
    8. :meth:`_precompute_hand_keypoints_in_object_frame` — express hand
       frame keypoints and wrist poses in the object body's local frame.
    9. :meth:`_init_metrics` — zero-initialize tracking-error metric buffers.

    At runtime the class exposes:

    - **Command properties** (``*_command_e``, ``*_command_o``) — goal poses
      for wrists, fingers, fingertips, object bodies, and contact points.
      Wrist commands are stored in the object's local frame and recomputed
      each step from the current object pose, so hand targets automatically
      follow object drift.
    - **State properties** — current wrist / finger / fingertip / object
      poses, velocities, and contact forces read from the simulation.
    - **Reset / stepping** — :meth:`_resample_command` teleports hands and
      object to a random trajectory frame; :meth:`_update_command` advances
      the timestep counter and decays the virtual-object-control curriculum.
    - **Visualization** — optional debug markers for goal vs. current poses,
      contact points, and fingertip positions.
    """

    cfg: CommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the dual-hand object tracking command term.

        Loads motion data from the configured folder, resolves scene assets
        (object and left/right robots), and sets up command buffers and reset
        state tensors.

        Args:
            cfg: Command term configuration (motion path, asset names, FPS, etc.).
            env: The RL environment instance; used to resolve scene assets.
        """
        super().__init__(cfg, env)
        self.step_dt = env.step_dt

        self._init_scene_references(cfg, env)
        self._load_motion_data(cfg)
        self._init_buffers(cfg)
        self._init_hand_data()
        self._init_object_data()
        self._init_relative_object_data()
        self._init_contact_data()
        self._precompute_contact_positions_normals_in_object_frame()
        self._precompute_hand_keypoints_in_object_frame()
        self._precompute_contact_wrench_support_values()
        if self.cfg.enable_additional_metrics:
            self._precompute_bbox_corner_vecs()
            self._precompute_dexmachina_sampled_verts()
        else:
            # Same banner format as _precompute_dexmachina_sampled_verts so the
            # gate failure mode is observable regardless of which branch trips.
            self._dexmachina_metric_enabled = False
            print(
                "[dexmachina_eval] WARNING: dexmachina_bbox / dexmachina_AUC metric "
                "DISABLED for this experiment — reason: cfg.enable_additional_metrics "
                "is False, so surface-vertex precompute was skipped. Set "
                "enable_additional_metrics=True on the command term cfg if you want "
                "AUC during training/eval."
            )
        self._set_contact_vis_impl(getattr(self.cfg, "debug_vis", False))
        self._init_metrics(cfg)

        # Lazy per-step refresh cache. refresh_tensors() recomputes all cached
        # tensors on first access when _tensors_dirty is True (set here, at end
        # of _update_command, and at end of _resample_command).
        self._tensors_dirty = True

    def __str__(self) -> str:
        """String representation of the command term."""
        msg = f"{self.__class__.__name__}:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    ######################################################################
    # Initialization Helpers
    ######################################################################

    def _init_scene_references(
        self, cfg: CommandTermCfg, env: ManagerBasedRLEnv
    ) -> None:
        """Resolve simulation scene assets and cache body/joint indices.

        Retrieves object assets, and for each hand side (left/right) resolves
        the robot articulation, finger joint IDs, wrist body IDs, fingertip
        body IDs, and object-to-hand contact sensors from the scene.

        Sets per-side attributes via ``setattr``:
          - ``{side}_robot``, ``{side}_finger_joint_names/ids``
          - ``{side}_wrist_body_id/name``, ``{side}_fingertip_body_ids/names``
          - ``object_to_{side}_hand_contact_sensors``
        """
        # Retrieve objects
        self.objects: list[Articulation | RigidObject] = [
            env.scene[body_name] for body_name in cfg.object_body_names
        ]
        self.num_bodies = self.object_position_e.shape[1]

        # Retrieve both side robots and cache useful indices
        for side in ["right", "left"]:
            # Robot asset handle
            side_robot_name = getattr(cfg, f"{side}_robot_name")
            side_robot = env.scene[side_robot_name]
            setattr(self, f"{side}_robot", side_robot)

            # Finger joint names and ids
            side_finger_joint_names = side_robot.data.joint_names
            finger_joint_ids, _ = side_robot.find_joints(side_finger_joint_names)
            finger_joint_ids = torch.tensor(finger_joint_ids, device=self.device)
            setattr(self, f"{side}_finger_joint_names", side_finger_joint_names)
            setattr(self, f"{side}_finger_joint_ids", finger_joint_ids)

            # Wrist body ids and names
            wrist_body_ids, wrist_body_name = side_robot.find_bodies(
                cfg.wrist_body_name
            )
            setattr(
                self,
                f"{side}_wrist_body_id",
                torch.tensor(wrist_body_ids, device=self.device),
            )
            setattr(self, f"{side}_wrist_body_name", wrist_body_name)

            # Fingertip body ids and names
            fingertip_body_ids, fingertip_body_names = side_robot.find_bodies(
                cfg.fingertip_body_name
            )
            setattr(
                self,
                f"{side}_fingertip_body_ids",
                torch.tensor(fingertip_body_ids, device=self.device),
            )
            setattr(self, f"{side}_fingertip_body_names", fingertip_body_names)

            # Object-hand contact sensors
            contact_sensors = [
                env.scene[sensor_name]
                for sensor_name in env.cfg.object_to_hand_contact_sensor_names
                if side in sensor_name
            ]
            setattr(self, f"object_to_{side}_hand_contact_sensors", contact_sensors)

    def _load_motion_data(self, cfg: CommandTermCfg) -> None:
        """Load retargeted motion data from parquet and interpolate to target FPS.

        Reads the ``ManoSharpaData`` parquet dataset specified by
        ``cfg.motion_folder`` / ``cfg.motion_filters`` / ``cfg.motion_id``,
        then resamples it to match the simulation timestep (or an explicit
        ``cfg.motion_speed``).

        Sets:
          - ``self._retargeted_motion_data`` — the loaded + interpolated data.
        """
        try:
            self._retargeted_motion_data = ManoSharpaData.from_parquet(
                root_path=str(cfg.motion_folder),
                filters=cfg.motion_filters,
                trajectory_id=cfg.motion_id,
            )

            # Interpolate the motion data to the target FPS
            target_num_frames = int(1 / (self.step_dt * cfg.motion_speed))
            self._retargeted_motion_data = interpolate_robot_motion_data(
                motion_data=self._retargeted_motion_data,
                target_num_frames=target_num_frames,
            )
        except Exception as e:
            raise ValueError(
                "Failed to load retargeted motion data from "
                f"{cfg.motion_folder} with filters {cfg.motion_filters} and "
                f"trajectory_id {cfg.motion_id} or interpolate the motion data. "
                f"Please check if the data exists and is valid. Error: {e}"
            ) from e

    def _init_buffers(self, cfg: CommandTermCfg) -> None:
        """Allocate per-env bookkeeping tensors and precomputed constant vectors.

        Creates:
          - Timestep / tracking counters: ``timestep_counter``,
            ``tracking_lengths``, ``steps_since_last_reset``, ``all_env_ids``.
          - Reset wrist pose buffers (written by ``_resample_command``, read by
            action terms for PD target initialization).
          - Virtual object controller curriculum scale factors.
          - Constant unit vectors used by reward/visualization code:
            ``X/Y/Z_UNIT_VEC``, ``QUAT_UNIT_VEC``, ``KEYPOINT_VECS``.
          - ``contact_sensor_history_length`` from the first right-hand sensor.
        """
        # Horizon and per-env counters
        self.retargeted_horizon = len(
            self._retargeted_motion_data.robot_right_wrist_position
        )
        self.timestep_counter = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.tracking_lengths = self.retargeted_horizon * torch.ones(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.steps_since_last_reset = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset wrist pose buffers (written by _resample_command,
        # read by action terms for PD target initialization)
        self.reset_right_wrist_position_e = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.reset_right_wrist_wxyz = torch.zeros(self.num_envs, 4, device=self.device)
        self.reset_left_wrist_position_e = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.reset_left_wrist_wxyz = torch.zeros(self.num_envs, 4, device=self.device)

        # Virtual object controller scale factor for curriculum
        self.virtual_object_controller_scale_factor = torch.tensor(
            [cfg.initial_virtual_object_control_curriculum_scale], device=self.device
        )  # (1,)
        self.virtual_object_controller_scale_factor_per_env = (
            cfg.initial_virtual_object_control_curriculum_scale
            * torch.ones(self.num_envs, 1, device=self.device)  # (num_envs, 1)
        )
        # Set by eval passes that need VOC truly frozen at 0. Gates both
        # _resample_command's per-env reset-to-1.0 write and _update_command's
        # per-step schedule rewrite, so eval can't be inflated by VOC assistance.
        self._spider_eval_freeze_voc: bool = False

        # Unit vectors reused by reward computation and visualization
        self.X_UNIT_VEC = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.Y_UNIT_VEC = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.Z_UNIT_VEC = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.QUAT_UNIT_VEC = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat((self.num_envs, 1))

        # 6 axis-aligned directions for object keypoint computation
        self.KEYPOINT_VECS = (
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(self.num_envs, self.num_bodies, -1, -1)
        )  # (num_envs, num_bodies, 6, 3)

        # Contact sensor history length
        self.contact_sensor_history_length = self.object_to_right_hand_contact_sensors[
            0
        ].cfg.history_length

        # Number of robot contacts
        self.num_robot_contacts_left = len(
            self.object_to_left_hand_contact_sensors[0].cfg.filter_prim_paths_expr
        )
        self.num_robot_contacts_right = len(
            self.object_to_right_hand_contact_sensors[0].cfg.filter_prim_paths_expr
        )
        assert (
            self.num_robot_contacts_left == self.num_robot_contacts_right
        ), "The number of robot contacts for the left and right hand must be the same."
        self.total_num_robot_contacts = (
            self.num_robot_contacts_left + self.num_robot_contacts_right
        )

        # Env origin
        self.env_origins_expanded = self._env.scene.env_origins.reshape(
            self.num_envs, 1, 1, 3
        ).expand(-1, self.num_bodies, self.num_robot_contacts_right, -1)

    def _init_hand_data(self) -> None:
        """Convert retargeted hand motion arrays to GPU tensors for both sides.

        For each hand side (left/right), loads from ``_retargeted_motion_data``
        and stores as tensors:
          - ``retargeted_{side}_wrist_position`` — (horizon, 3)
          - ``retargeted_{side}_wrist_wxyz`` — (horizon, 4)
          - ``retargeted_{side}_finger_joints`` — (horizon, N_joints), reordered
            from retarget joint order to Isaac joint order.
          - ``retargeted_{side}_hand_frames`` — (horizon, N_frames, 7)
          - ``retargeted_{side}_hand_frame_names`` — list of frame names
          - ``retargeted_{side}_fingertip_indices`` — indices into hand_frames
            for fingertip bodies.
        """
        for side in ["right", "left"]:
            # Wrist position and orientation
            retargeted_wrist_position = getattr(
                self._retargeted_motion_data, f"robot_{side}_wrist_position"
            )
            retargeted_wrist_wxyz = getattr(
                self._retargeted_motion_data, f"robot_{side}_wrist_wxyz"
            )
            setattr(
                self,
                f"retargeted_{side}_wrist_position",
                torch.tensor(retargeted_wrist_position, device=self.device),
            )
            setattr(
                self,
                f"retargeted_{side}_wrist_wxyz",
                torch.tensor(retargeted_wrist_wxyz, device=self.device),
            )

            # Finger joints reordered to Isaac joint order
            retargeted_finger_joint_names = getattr(
                self._retargeted_motion_data, f"{side}_robot_finger_joint_names"
            )
            isaac_finger_joint_names = getattr(self, f"{side}_finger_joint_names")
            retargeted_to_isaac_joint_order = [
                retargeted_finger_joint_names.index(joint_name)
                for joint_name in isaac_finger_joint_names
            ]
            retargeted_finger_joints = getattr(
                self._retargeted_motion_data, f"robot_{side}_finger_joints"
            )
            retargeted_finger_joints = torch.tensor(
                retargeted_finger_joints, device=self.device
            )[:, retargeted_to_isaac_joint_order]
            setattr(self, f"retargeted_{side}_finger_joints", retargeted_finger_joints)

            # Hand frames (position + quaternion per body)
            retargeted_hand_frames = getattr(
                self._retargeted_motion_data, f"robot_{side}_frames"
            )
            retargeted_hand_frame_names = getattr(
                self._retargeted_motion_data, f"{side}_robot_frame_names"
            )
            setattr(
                self,
                f"retargeted_{side}_hand_frames",
                torch.tensor(retargeted_hand_frames, device=self.device),
            )
            setattr(
                self, f"retargeted_{side}_hand_frame_names", retargeted_hand_frame_names
            )

            # Fingertip indices into the hand frame array
            fingertip_body_names = getattr(self, f"{side}_fingertip_body_names")
            retargeted_fingertip_indices = [
                retargeted_hand_frame_names.index(name) for name in fingertip_body_names
            ]
            setattr(
                self,
                f"retargeted_{side}_fingertip_indices",
                torch.tensor(retargeted_fingertip_indices, device=self.device),
            )

            init_maniptrans_body_tables(self, side, retargeted_hand_frame_names)

    def _init_object_data(self) -> None:
        """Convert retargeted object motion arrays to GPU tensors.

        Stores:
          - ``retargeted_object_body_position`` — (horizon, N_body, 3)
          - ``retargeted_object_body_wxyz`` — (horizon, N_body, 4)
          - ``retargeted_object_articulation`` — (horizon, N_joints)
          - ``retargeted_object_body_names`` — list of body names

        Asserts that the number of body names in the motion file matches the
        number of bodies in the simulation object.
        """
        self.retargeted_object_body_position = torch.tensor(
            self._retargeted_motion_data.object_body_position, device=self.device
        ).float()
        self.retargeted_object_body_wxyz = torch.tensor(
            self._retargeted_motion_data.object_body_wxyz, device=self.device
        ).float()

        # SPIDER-paper-aligned tracking-error precomputations (used in _update_metrics).
        # ref_pos_mean: mean reference position over the full retargeted trajectory,
        # per body — needed to mean-subtract the reference (matches SPIDER's
        # compute_object_tracking_error, postprocess/get_success_rate.py:143-148).
        # static_body_mask: bodies whose ref position varies <1 mm over the whole
        # trajectory — SPIDER excludes these from the success metric (a static body
        # like stir_beaker's beaker would otherwise trivially score 0 error and bias
        # the multi-body mean).
        self._spider_ref_pos_mean_b = self.retargeted_object_body_position.mean(
            dim=0
        )  # (N_body, 3)
        ref_pos_t0 = self.retargeted_object_body_position[0]  # (N_body, 3)
        ref_pos_disp = (
            self.retargeted_object_body_position - ref_pos_t0
        )  # (T, N_body, 3)
        ref_pos_disp_mean_norm = torch.linalg.norm(ref_pos_disp, dim=-1).mean(
            dim=0
        )  # (N_body,)
        self._spider_static_body_mask = ref_pos_disp_mean_norm < 0.001  # (N_body,)
        self.retargeted_object_articulation = torch.tensor(
            self._retargeted_motion_data.object_articulation, device=self.device
        ).float()
        self.retargeted_object_body_names = (
            self._retargeted_motion_data.object_body_names
        )
        assert len(self.retargeted_object_body_names) == self.num_bodies, (
            f"The number of body names in the motion file and the object do not match. "
            f"Find {len(self.retargeted_object_body_names)} in motion file, "
            f"but {self.num_bodies} in the object."
        )

        # Reference linear velocity per body, computed via central finite
        # difference on the retargeted position trajectory. Used by the
        # ``object_velocity_tracking_exp`` reward. Boundary frames (t=0 and
        # t=T-1) use forward/backward differences. Units are m/s.
        if self.retargeted_object_body_position.shape[0] >= 2:
            dt = float(self.step_dt)
            pos = self.retargeted_object_body_position  # (T, N_body, 3)
            vel = torch.zeros_like(pos)
            vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
            vel[0] = (pos[1] - pos[0]) / dt
            vel[-1] = (pos[-1] - pos[-2]) / dt
            self.retargeted_object_body_lin_vel = vel
        else:
            self.retargeted_object_body_lin_vel = torch.zeros_like(
                self.retargeted_object_body_position
            )

    def _init_relative_object_data(self) -> None:
        """Precompute demo relative pose (obj1 in obj0's frame) for multi-object tasks.

        Sets:
          - ``self._has_multi_object`` — False for single-object sequences, skips all relative-pose logic.
          - ``self._obj1_root_body_idx`` — index of obj1's root body in the concatenated body array.
          - ``self._demo_rel_pos`` — (T, 3) demo relative position of obj1 root in obj0 root frame.
          - ``self._demo_rel_quat`` — (T, 4) demo relative quaternion of obj1 root in obj0 root frame.
          - ``self._demo_inter_object_dist`` — (T,) L2 distance between obj0 and obj1 roots in demo.
        """
        if len(self.objects) < 2:
            self._has_multi_object = False
            return
        self._has_multi_object = True
        self._obj1_root_body_idx = self.objects[0].data.body_link_pos_w.shape[1]

        # env-frame positions and world-frame quats (env origin cancels in relative transform)
        demo_obj0_pos = self.retargeted_object_body_position[:, 0, :]
        demo_obj0_quat = self.retargeted_object_body_wxyz[:, 0, :]
        demo_obj1_pos = self.retargeted_object_body_position[
            :, self._obj1_root_body_idx, :
        ]
        demo_obj1_quat = self.retargeted_object_body_wxyz[
            :, self._obj1_root_body_idx, :
        ]

        self._demo_rel_pos, self._demo_rel_quat = math_utils.subtract_frame_transforms(
            demo_obj0_pos, demo_obj0_quat, demo_obj1_pos, demo_obj1_quat
        )  # (T, 3) and (T, 4)
        self._demo_inter_object_dist = torch.norm(
            demo_obj1_pos - demo_obj0_pos, dim=-1
        )  # (T,)

    def _init_contact_data(self) -> None:
        """Load per-link contact data from retargeted motion for both hand sides.

        For each side, stores tensors with shape (horizon, num_contact_links, 3):
          - ``retargeted_{side}_link_contact_positions_e``
          - ``retargeted_{side}_link_contact_normals_e`` pointing outward from the hand (inward to object).
          - ``retargeted_{side}_object_contact_positions_e``
          - ``retargeted_{side}_object_contact_normals_e``
          - ``retargeted_{side}_object_contact_part_ids`` — (horizon, num_links),
            read from ``mano_{side}_object_contact_part_ids`` (0 = no contact,
            1-indexed body IDs).
        """
        data = self._retargeted_motion_data

        for side in ["left", "right"]:
            link_contact_positions = torch.tensor(
                getattr(data, f"mano_{side}_link_contact_positions"),
                dtype=torch.float32,
                device=self.device,
            )  # (horizon, num_contact_links, 3)
            link_contact_normals = torch.tensor(
                getattr(data, f"mano_{side}_link_contact_normals"),
                dtype=torch.float32,
                device=self.device,
            )  # (horizon, num_contact_links, 3)
            link_contact_normals = link_contact_normals / link_contact_normals.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-6)
            object_contact_positions = torch.tensor(
                getattr(data, f"mano_{side}_object_contact_positions"),
                dtype=torch.float32,
                device=self.device,
            )  # (horizon, num_contact_links, 3)
            object_contact_normals = torch.tensor(
                getattr(data, f"mano_{side}_object_contact_normals"),
                dtype=torch.float32,
                device=self.device,
            )  # (horizon, num_contact_links, 3)
            contact_part_ids = torch.tensor(
                getattr(data, f"mano_{side}_object_contact_part_ids"),
                dtype=torch.long,
                device=self.device,
            )  # (horizon, num_contact_links)

            setattr(
                self,
                f"retargeted_{side}_link_contact_positions_e",
                link_contact_positions,
            )
            setattr(
                self,
                f"retargeted_{side}_link_contact_normals_e",
                link_contact_normals,
            )
            setattr(
                self,
                f"retargeted_{side}_object_contact_positions_e",
                object_contact_positions,
            )
            setattr(
                self,
                f"retargeted_{side}_object_contact_normals_e",
                object_contact_normals,
            )
            setattr(
                self, f"retargeted_{side}_object_contact_part_ids", contact_part_ids
            )

    def _compute_contact_positions_normals_in_object_frame(self, side: str) -> None:
        """Transform env-frame contact positions and normals into the contacted object body's local frame.

        For the given hand side, reads ``retargeted_{side}_object_contact_positions_e``
        and ``retargeted_{side}_link_contact_normals_e``
        and ``retargeted_{side}_object_contact_part_ids``, then uses
        ``subtract_frame_transforms`` to express each contact point in the local
        frame of the object body it belongs to.
        retargeted_{side}_link_contact_normals_e is the normal of the human hand,
        pointing outward from the hand.

        Sets:
          - ``retargeted_{side}_object_contact_positions_o`` — (horizon, N_links, 3)
          - ``retargeted_{side}_link_contact_normals_o`` — (horizon, N_links, 3)
          - ``retargeted_{side}_object_contact_positions_com`` — (horizon, N_links, 3)
          - ``retargeted_{side}_link_contact_normals_com`` — (horizon, N_links, 3)
          - ``retargeted_{side}_object_contact_is_valid`` — (horizon, N_links) bool
          - ``retargeted_{side}_object_has_contact`` — (horizon,) bool

        Args:
            side: ``"left"`` or ``"right"``.
        """
        contact_positions_e = getattr(
            self, f"retargeted_{side}_object_contact_positions_e"
        )
        contact_normals_e = getattr(self, f"retargeted_{side}_link_contact_normals_e")
        contact_part_ids = getattr(self, f"retargeted_{side}_object_contact_part_ids")
        horizon_ids = torch.arange(
            self.retargeted_object_body_position.shape[0], device=self.device
        )

        object_o_t_com = torch.cat(
            [object.data.body_com_pose_b for object in self.objects], dim=1
        ).float()
        object_o_p_com = object_o_t_com[..., :3].mean(dim=0)  # (num_bodies, 3)
        object_o_q_com = object_o_t_com[0, :, 3:7]  # (num_bodies, 4)

        contact_positions_o = torch.zeros_like(contact_positions_e[..., :3])
        contact_normals_o = torch.zeros_like(contact_normals_e[..., :3])
        contact_positions_com = torch.zeros_like(contact_positions_e[..., :3])
        contact_normals_com = torch.zeros_like(contact_normals_e[..., :3])
        is_valid = contact_part_ids > 0
        has_contact = is_valid.sum(dim=-1) > 1e-5

        for link_idx in range(contact_positions_e.shape[1]):
            # Part IDs are 1-indexed; 0 means no contact
            part_id = (contact_part_ids[:, link_idx] - 1).clamp(
                min=0, max=self.num_bodies - 1
            )
            # convert environment frame contact to object frame
            object_e_p_o = self.retargeted_object_body_position[horizon_ids, part_id]
            object_e_q_o = self.retargeted_object_body_wxyz[horizon_ids, part_id]
            contact_positions_o[:, link_idx], _ = math_utils.subtract_frame_transforms(
                object_e_p_o,
                object_e_q_o,
                contact_positions_e[:, link_idx, :3],
                q02=None,
            )
            contact_normals_o[:, link_idx], _ = math_utils.subtract_frame_transforms(
                torch.zeros_like(object_e_p_o),
                object_e_q_o,
                contact_normals_e[:, link_idx, :3],
                q02=None,
            )
            # convert object frame contact to object com frame
            _object_o_p_com = object_o_p_com[part_id]
            _object_o_q_com = object_o_q_com[part_id]
            contact_positions_com[:, link_idx], _ = (
                math_utils.subtract_frame_transforms(
                    _object_o_p_com,
                    _object_o_q_com,
                    contact_positions_o[:, link_idx, :3],
                    q02=None,
                )
            )
            contact_normals_com[:, link_idx], _ = math_utils.subtract_frame_transforms(
                torch.zeros_like(_object_o_p_com),
                _object_o_q_com,
                contact_normals_o[:, link_idx, :3],
                q02=None,
            )

        contact_positions_o.masked_fill_(~is_valid.unsqueeze(-1), 0.0)
        contact_normals_o.masked_fill_(~is_valid.unsqueeze(-1), 0.0)
        contact_positions_com.masked_fill_(~is_valid.unsqueeze(-1), 0.0)
        contact_normals_com.masked_fill_(~is_valid.unsqueeze(-1), 0.0)

        setattr(
            self, f"retargeted_{side}_object_contact_positions_o", contact_positions_o
        )
        setattr(self, f"retargeted_{side}_link_contact_normals_o", contact_normals_o)
        setattr(
            self,
            f"retargeted_{side}_object_contact_positions_com",
            contact_positions_com,
        )
        setattr(
            self, f"retargeted_{side}_link_contact_normals_com", contact_normals_com
        )
        setattr(self, f"retargeted_{side}_object_contact_is_valid", is_valid)
        setattr(self, f"retargeted_{side}_object_has_contact", has_contact)

    def _precompute_contact_positions_normals_in_object_frame(self) -> None:
        """Transform contact positions into object frame for both hands and compute contact counts.

        Calls ``_compute_contact_positions_normals_in_object_frame`` for each hand side,
        then ``_get_contact_counts`` to cache the total number of contact points.
        """
        for side in ["left", "right"]:
            self._compute_contact_positions_normals_in_object_frame(side)
        self._get_retargeted_contact_counts()

    def _compute_hand_keypoints_in_object_frame(self, side: str) -> None:
        """Express hand frame keypoints and wrist pose in the contacted object body's local frame.

        For the given hand side:

        1. Determines the dominant contact body per timestep by taking the mode
           of per-link contact part IDs, with backward-fill for frames without
           contact.
        2. Transforms all hand frame positions and orientations into that
           object body's local frame via ``subtract_frame_transforms``.
        3. Extracts the wrist-specific position and orientation as a convenience
           slice.

        Sets:
          - ``retargeted_{side}_object_contact_part_ids_per_hand`` — (horizon,)
          - ``retargeted_{side}_hand_frame_position_o`` — (horizon, N_frames, 3)
          - ``retargeted_{side}_hand_frame_wxyz_o`` — (horizon, N_frames, 4)
          - ``retargeted_{side}_wrist_position_o`` — (horizon, 3)
          - ``retargeted_{side}_wrist_wxyz_o`` — (horizon, 4)

        Args:
            side: ``"left"`` or ``"right"``.
        """
        contact_part_ids = getattr(self, f"retargeted_{side}_object_contact_part_ids")
        hand_frames = getattr(self, f"retargeted_{side}_hand_frames")
        hand_frame_names = getattr(self, f"retargeted_{side}_hand_frame_names")
        wrist_body_name = getattr(self, f"{side}_wrist_body_name")

        horizon_ids = torch.arange(
            self.retargeted_object_body_position.shape[0], device=self.device
        )

        # Determine the dominant contact object body per timestep (backward fill)
        part_ids_per_hand = torch.zeros(
            self.retargeted_horizon, dtype=torch.int64, device=self.device
        )
        last_contact_part_ids = torch.ones(1, dtype=torch.int64, device=self.device)
        for horizon_idx in range(self.retargeted_horizon - 1, -1, -1):
            hand_contact_part_ids = contact_part_ids[horizon_idx].mode().values
            part_ids_per_hand[horizon_idx] = (
                hand_contact_part_ids
                if hand_contact_part_ids > 0
                else last_contact_part_ids
            )
            last_contact_part_ids = part_ids_per_hand[horizon_idx]

        setattr(
            self,
            f"retargeted_{side}_object_contact_part_ids_per_hand",
            part_ids_per_hand,
        )

        # Transform hand frames into the contact object body's local frame
        frame_position_o = torch.zeros_like(hand_frames[..., :3])
        frame_wxyz_o = torch.zeros_like(hand_frames[..., 3:7])

        # SPIDER_RIGID and other rigid-collapse datasets keep the source
        # annotation's articulated body ids in `*_object_contact_part_ids` even
        # though `retargeted_object_body_position` has been collapsed to a
        # single body. Clamp to the available body count so the `- 1` index
        # stays in range without changing semantics for true multi-body objects.
        num_bodies = self.retargeted_object_body_position.shape[1]
        part_ids_per_hand = part_ids_per_hand.clamp(min=1, max=num_bodies)

        contact_body_position = self.retargeted_object_body_position[
            horizon_ids, part_ids_per_hand - 1
        ]
        contact_body_wxyz = self.retargeted_object_body_wxyz[
            horizon_ids, part_ids_per_hand - 1
        ]

        for frame_idx in range(hand_frames.shape[1]):
            frame_position_o[:, frame_idx], frame_wxyz_o[:, frame_idx] = (
                math_utils.subtract_frame_transforms(
                    contact_body_position,
                    contact_body_wxyz,
                    hand_frames[:, frame_idx, :3],
                    hand_frames[:, frame_idx, 3:7],
                )
            )

        setattr(self, f"retargeted_{side}_hand_frame_position_o", frame_position_o)
        setattr(self, f"retargeted_{side}_hand_frame_wxyz_o", frame_wxyz_o)

        # Extract wrist-specific slice for convenience
        wrist_index = hand_frame_names.index(wrist_body_name[0])
        setattr(
            self,
            f"retargeted_{side}_wrist_position_o",
            frame_position_o[:, wrist_index],
        )
        setattr(self, f"retargeted_{side}_wrist_wxyz_o", frame_wxyz_o[:, wrist_index])

    def _precompute_hand_keypoints_in_object_frame(self) -> None:
        """Express hand frame keypoints in object frame for both hand sides.

        Calls ``_compute_hand_keypoints_in_object_frame`` for left and right.
        """
        for side in ["left", "right"]:
            self._compute_hand_keypoints_in_object_frame(side)

    def _precompute_contact_wrench_support_values(self) -> None:
        """Precompute the contact wrench space support function over sampled basis directions for each hand-body pairs."""
        # 1. Sample basis directions for each object body
        self.object_mesh_radius = self._retargeted_motion_data.object_mesh_radius
        self.wrench_space_bases = torch.cat(
            [
                sample_wrench_space_basis_scaled(
                    self.cfg.num_wrench_space_basis_samples, rc=1.0, device=self.device
                ).unsqueeze(0)
                for _ in self.object_mesh_radius
            ],
            dim=0,
        )

        # 2. Expand contact positions and normals to (horizon, num_bodies, num_hand_contact_links, 3)
        t_idx = torch.arange(self.retargeted_horizon, device=self.device)[
            :, None
        ]  # (horizon, 1)
        c_idx = torch.arange(self.num_retargeted_contacts_left, device=self.device)[
            None, :
        ]  # (1, num_hand_contact_links)

        retargeted_left_contact_positions_com = torch.zeros(
            self.retargeted_horizon,
            self.num_bodies,
            self.num_retargeted_contacts_left,
            3,
            device=self.device,
        )
        retargeted_left_contact_normals_com = torch.zeros(
            self.retargeted_horizon,
            self.num_bodies,
            self.num_retargeted_contacts_left,
            3,
            device=self.device,
        )
        left_command_contact_part_ids = (
            self.retargeted_left_object_contact_part_ids - 1
        ).clamp(
            min=0, max=self.num_bodies - 1
        )  # (horizon, num_hand_contact_links), 0-indexed
        left_command_valid = self.retargeted_left_object_contact_is_valid.unsqueeze(
            -1
        )  # (horizon, num_hand_contact_links, 1)
        retargeted_left_contact_positions_com[
            t_idx, left_command_contact_part_ids, c_idx
        ] = (self.retargeted_left_object_contact_positions_com * left_command_valid)
        retargeted_left_contact_normals_com[
            t_idx, left_command_contact_part_ids, c_idx
        ] = (self.retargeted_left_link_contact_normals_com * left_command_valid)

        retargeted_right_contact_positions_com = torch.zeros(
            self.retargeted_horizon,
            self.num_bodies,
            self.num_retargeted_contacts_right,
            3,
            device=self.device,
        )
        retargeted_right_contact_normals_com = torch.zeros(
            self.retargeted_horizon,
            self.num_bodies,
            self.num_retargeted_contacts_right,
            3,
            device=self.device,
        )
        right_command_contact_part_ids = (
            self.retargeted_right_object_contact_part_ids - 1
        ).clamp(
            min=0, max=self.num_bodies - 1
        )  # (horizon, num_hand_contact_links), 0-indexed
        right_command_valid = self.retargeted_right_object_contact_is_valid.unsqueeze(
            -1
        )  # (horizon, num_hand_contact_links, 1)
        retargeted_right_contact_positions_com[
            t_idx, right_command_contact_part_ids, c_idx
        ] = (self.retargeted_right_object_contact_positions_com * right_command_valid)
        retargeted_right_contact_normals_com[
            t_idx, right_command_contact_part_ids, c_idx
        ] = (self.retargeted_right_link_contact_normals_com * right_command_valid)

        # 3. Compute support function over sampled basis directions for each hand-body pairs
        self.retargeted_left_contact_wrench_supports = torch.zeros(
            self.retargeted_horizon,
            self.num_bodies,
            self.cfg.num_wrench_space_basis_samples,
            device=self.device,
        )
        self.retargeted_right_contact_wrench_supports = torch.zeros(
            self.retargeted_horizon,
            self.num_bodies,
            self.cfg.num_wrench_space_basis_samples,
            device=self.device,
        )

        theta = torch.linspace(
            0.0,
            2.0 * torch.pi,
            steps=self.cfg.num_friction_cone_edges + 1,
            device=retargeted_left_contact_positions_com.device,
            dtype=retargeted_left_contact_positions_com.dtype,
        )[:-1]
        self.friction_cone_edge_cosines = torch.cos(theta).view(1, -1, 1)  # (1, K, 1)
        self.friction_cone_edge_sines = torch.sin(theta).view(1, -1, 1)  # (1, K, 1)

        for body_idx, body_radius in enumerate(self.object_mesh_radius):
            left_wrench_space = compute_wrench_space(
                contact_points=retargeted_left_contact_positions_com[:, body_idx],
                contact_normals=retargeted_left_contact_normals_com[:, body_idx],
                cos_t=self.friction_cone_edge_cosines,
                sin_t=self.friction_cone_edge_sines,
                rc=body_radius,
                friction_coefficients=self.cfg.friction_coefficients,
            )  # (horizon, 6, _)
            self.retargeted_left_contact_wrench_supports[:, body_idx] = (
                compute_wrench_space_support_function(
                    wrench_space=left_wrench_space,
                    basis=self.wrench_space_bases[body_idx],
                )
            )  # (horizon, num_wrench_space_basis_samples)

            right_wrench_space = compute_wrench_space(
                contact_points=retargeted_right_contact_positions_com[:, body_idx],
                contact_normals=retargeted_right_contact_normals_com[:, body_idx],
                cos_t=self.friction_cone_edge_cosines,
                sin_t=self.friction_cone_edge_sines,
                rc=body_radius,
                friction_coefficients=self.cfg.friction_coefficients,
            )  # (horizon, _, 6)
            self.retargeted_right_contact_wrench_supports[:, body_idx] = (
                compute_wrench_space_support_function(
                    wrench_space=right_wrench_space,
                    basis=self.wrench_space_bases[body_idx],
                )
            )  # (horizon, num_wrench_space_basis_samples)

        # Buffer for current contact wrench supports
        self.left_contact_wrench_supports = torch.zeros(
            self.num_envs,
            self.num_bodies,
            self.cfg.num_wrench_space_basis_samples,
            device=self.device,
        )
        self.right_contact_wrench_supports = torch.zeros(
            self.num_envs,
            self.num_bodies,
            self.cfg.num_wrench_space_basis_samples,
            device=self.device,
        )

    def _compute_object_body_half_extents(self) -> torch.Tensor:
        """Compute per-body AABB half-extents in local frame from USD geometry.

        Called once at init. Uses UsdGeom.BBoxCache on the live USD stage so no
        precomputed tables or parquet changes are needed — generalizes to any object.
        Falls back to isotropic sphere radius per body if the prim is not found or
        has empty bounds.

        Returns:
            Tensor of shape (num_bodies, 3) with per-body half-extents [hx, hy, hz].
        """
        import isaaclab.sim.utils as sim_utils  # noqa: PLC0415
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        half_extents: list[list[float]] = []
        body_idx = 0

        for obj in self.objects:
            first_prim = sim_utils.find_first_matching_prim(obj.cfg.prim_path)
            root_path = first_prim.GetPath().pathString
            stage = first_prim.GetStage()
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), ["default", "proxy", "render"]
            )

            for link_name in obj.data.body_names:
                body_prim = stage.GetPrimAtPath(f"{root_path}/{link_name}")
                if body_prim.IsValid():
                    bound = bbox_cache.ComputeUntransformedBound(body_prim)
                    r3d = bound.GetRange()
                    if not r3d.IsEmpty():
                        mn, mx = r3d.GetMin(), r3d.GetMax()
                        half_extents.append([(mx[i] - mn[i]) / 2.0 for i in range(3)])
                        body_idx += 1
                        continue
                # Fallback: isotropic sphere radius
                r = (
                    self.object_mesh_radius[body_idx]
                    if body_idx < len(self.object_mesh_radius)
                    else 0.1
                )
                half_extents.append([r, r, r])
                body_idx += 1

        return torch.tensor(half_extents, device=self.device, dtype=torch.float32)

    def _precompute_bbox_corner_vecs(self) -> None:
        """Pre-compute the 8 AABB corner vectors scaled by per-body half-extents.

        Stores ``self.BBOX_CORNER_VECS`` of shape (num_envs, num_bodies, 8, 3).
        Each of the 8 corners is (±hx, ±hy, ±hz) in the body's local frame.
        Used in ``_update_metrics`` to compute a size-normalized pose-tracking error.
        """
        self.object_body_half_extents = self._compute_object_body_half_extents()
        # (num_bodies, 3)

        signs = torch.tensor(
            [
                [s0, s1, s2]
                for s0 in [1.0, -1.0]
                for s1 in [1.0, -1.0]
                for s2 in [1.0, -1.0]
            ],
            device=self.device,
            dtype=torch.float32,
        )  # (8, 3)

        # Scale unit corners by per-body half-extents
        self.BBOX_CORNER_VECS = (
            (signs.unsqueeze(0) * self.object_body_half_extents.unsqueeze(1))
            .unsqueeze(0)
            .expand(self.num_envs, -1, -1, -1)
        )
        # (num_envs, num_bodies, 8, 3)

    def _precompute_dexmachina_sampled_verts(self, num_samples: int = 500) -> None:
        """Load 500 paper-exact uniformly-sampled vertices per body for ADD.

        Reference: dexmachina/eval/compute_add.py:load_part_verts (calls
        dexmachina/envs/object.py:sample_mesh_vertices, which seeds np.random
        with `seed=42` and uniformly samples `num_samples` indices from
        `mesh.vertices` via `np.random.choice`). The paper does NOT use
        area-weighted surface sampling. We mirror this exactly in
        `dexmachina_eval.sample_mesh_surface_vertices` with `seed=42`.

        Stores ``self.DEXMACHINA_SAMPLED_VERTS`` of shape
        ``(num_envs, num_bodies, num_samples, 3)`` in each body's local frame.
        These are used in ``_update_metrics`` to compute the paper's per-vertex
        ADD plus its threshold-sweep AUC.

        Sets ``self._dexmachina_metric_enabled`` to True iff all body meshes
        loaded successfully. Mesh paths come from the parquet's
        ``object_mesh_paths`` field; if any mesh is missing or unloadable we
        disable the metric for this experiment rather than failing init.
        """
        from robotic_grounding.tasks.v2p.mdp.commands.dexmachina_eval import (  # noqa: PLC0415
            sample_mesh_surface_vertices,
        )

        self._dexmachina_metric_enabled = False
        self._dexmachina_num_verts = num_samples

        # Loud, uniform "DISABLED because X" banner — keeps the gate failure
        # observable so post-training eval (--dexmachina_auc, EvalCallback
        # --eval_dexmachina_auc) doesn't silently lack AUC. Reasons are
        # mutually exclusive; each early-return prints exactly one banner.
        _disabled_prefix = (
            "[dexmachina_eval] WARNING: dexmachina_bbox / dexmachina_AUC metric "
            "DISABLED for this experiment — "
        )

        mesh_paths = list(
            getattr(self._retargeted_motion_data, "object_mesh_paths", []) or []
        )
        if not mesh_paths:
            print(
                _disabled_prefix
                + "reason: motion data has no `object_mesh_paths` field (or it is "
                "empty). Re-run retargeting with a pipeline that populates "
                "object_mesh_paths in the parquet, otherwise post-training "
                "--dexmachina_auc / --eval_dexmachina_auc will skip silently."
            )
            return

        num_bodies = self.object_body_half_extents.shape[0]
        if len(mesh_paths) != num_bodies:
            print(
                _disabled_prefix
                + f"reason: `object_mesh_paths` length ({len(mesh_paths)}) != "
                f"num object bodies ({num_bodies}). Parquet is malformed; "
                "regenerate motion data so the two match."
            )
            return

        sampled: list[torch.Tensor] = []
        for body_idx, mp in enumerate(mesh_paths):
            try:
                verts_np = sample_mesh_surface_vertices(mp, num_samples=num_samples)
            except Exception as e:  # noqa: BLE001
                print(
                    _disabled_prefix
                    + f"reason: failed to load/sample mesh for body {body_idx} "
                    f"at path '{mp}' ({type(e).__name__}: {e}). Verify the mesh "
                    "file exists and is readable by trimesh."
                )
                return
            sampled.append(
                torch.tensor(verts_np, device=self.device, dtype=torch.float32)
            )

        verts = torch.stack(sampled, dim=0)  # (num_bodies, num_samples, 3)
        self.DEXMACHINA_SAMPLED_VERTS = verts.unsqueeze(0).expand(
            self.num_envs, -1, -1, -1
        )  # (num_envs, num_bodies, num_samples, 3)
        # Paper-exact in-training dexmachina_AUC: flat sliding pool of per-env
        # per-frame ADD values, one row per body. Each _update_metrics call
        # appends the current step's (num_envs,) ADD values for body b to
        # _dexmachina_pool[b]; the AUC is then computed across the entire pool.
        # This matches dexmachina/eval/compute_add.py exactly — `compute_auc`
        # flattens the (n_envs * n_frames) array for each part before computing
        # accuracy, and the "overall" AUC is the mean across parts.
        #
        # Pool capacity (per body): num_envs * 50 — roughly 50 rollout steps'
        # worth of samples per env. 50 steps is enough for AUC over 9
        # thresholds to be stable (each accuracy is mean over num_envs*50 ≈
        # 50k+ samples for a typical 1024-env rollout) but short enough that
        # the metric tracks policy improvement instead of lagging behind by
        # multiple PPO iterations. The pool drains in FIFO order so the
        # reported AUC is always over the most recent ~50 sim steps.
        num_bodies = self.object_body_half_extents.shape[0]
        self._dexmachina_num_bodies = int(num_bodies)
        _dexmachina_pool_steps = 50
        self._dexmachina_pool_capacity = int(self.num_envs * _dexmachina_pool_steps)
        # One deque per body — appendleft/popleft are O(1) and the .item()
        # cost we previously paid per-env-per-step is gone.
        self._dexmachina_pool: list[collections.deque] = [
            collections.deque(maxlen=self._dexmachina_pool_capacity)
            for _ in range(num_bodies)
        ]
        self._dexmachina_metric_enabled = True

    def _init_metrics(self, cfg: CommandTermCfg) -> None:
        """Initialize all tracking-error metric buffers to zero.

        Allocates per-env zero tensors for wrist position/orientation errors,
        finger joint errors, object body pose errors, object articulation
        errors, and the virtual object controller scale factor.
        """
        for side in ["right", "left"]:
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
        self.metrics["object_articulation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        register_spider_metric_keys(self)
        register_maniptrans_metric_keys(self)
        self.metrics["virtual_object_controller_scale_factor"] = (
            cfg.initial_virtual_object_control_curriculum_scale
            * torch.ones(self.num_envs, device=self.device)
        )

        if self.cfg.enable_additional_metrics:
            # Contact wrench level metrics (left/right separate)
            for side in ["right", "left"]:
                self.metrics[f"contact_wrench_command_support_mean_{side}"] = (
                    torch.zeros(self.num_envs, device=self.device)
                )
                self.metrics[f"contact_wrench_current_support_mean_{side}"] = (
                    torch.zeros(self.num_envs, device=self.device)
                )
                self.metrics[f"contact_wrench_support_ratio_{side}"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics[f"contact_bodies_coverage_frac_{side}"] = torch.zeros(
                    self.num_envs, device=self.device
                )

            # Object bounding-box tracking and lift metrics
            self.metrics["object_bbox_corner_error"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["object_body_z_error"] = torch.zeros(
                self.num_envs, device=self.device
            )
            # Paper-exact DexMachina ADD metric (mean per-vertex displacement over
            # 500 sampled surface vertices per body) + threshold-sweep AUC.
            # See dexmachina_eval.py + dexmachina/eval/compute_add.py.
            self.metrics["dexmachina_bbox"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["dexmachina_AUC"] = torch.zeros(
                self.num_envs, device=self.device
            )

            # DISABLED: object_has_{right,left,any}_contact_frac W&B logging
            # for suffix in [
            #     "right_contact_frac",
            #     "left_contact_frac",
            #     "any_contact_frac",
            # ]:
            #     self.metrics[f"object_has_{suffix}"] = torch.zeros(
            #         self.num_envs, device=self.device
            #     )

            # DISABLED: hand_tracking_good_frac W&B logging
            # self.metrics["hand_tracking_good_frac"] = torch.zeros(
            #     self.num_envs, device=self.device
            # )

            # DISABLED: voc_decay_fired W&B logging (and its prev-scale tracker)
            # self.metrics["voc_decay_fired"] = torch.zeros(
            #     self.num_envs, device=self.device
            # )
            # self._prev_voc_scale = float(
            #     cfg.initial_virtual_object_control_curriculum_scale
            # )
            # Rolling step-level buffer for contact_wrench_support_reward CV (W=200).
            # Initialized to 1.0 (high = not plateaued) until the buffer fills.
            self._cws_reward_step_buf: collections.deque = collections.deque(maxlen=200)
            self.metrics["contact_wrench_support_reward_cv"] = torch.ones(
                self.num_envs, device=self.device
            )

        if self._has_multi_object:
            self.metrics["relative_object_pos_error"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["relative_object_rot_error"] = torch.zeros(
                self.num_envs, device=self.device
            )
            # Rolling step-level buffer for contact_wrench_support_reward CV (W=200).
            # Initialized to 1.0 (high = not plateaued) until the buffer fills.
            self._cws_reward_step_buf: collections.deque = collections.deque(maxlen=200)
            self.metrics["contact_wrench_support_reward_cv"] = torch.ones(
                self.num_envs, device=self.device
            )

    ######################################################################
    # Commands
    ######################################################################

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, -1)."""
        right_hand_wrist_pose_command_e = self.right_hand_wrist_pose_command_e
        right_wrist_current_p_command, right_wrist_current_q_command = (
            math_utils.subtract_frame_transforms(
                self.right_hand_wrist_position_e,
                self.right_hand_wrist_wxyz_e,
                right_hand_wrist_pose_command_e[:, :3],
                right_hand_wrist_pose_command_e[:, 3:],
            )
        )
        left_hand_wrist_pose_command_e = self.left_hand_wrist_pose_command_e
        left_wrist_current_p_command, left_wrist_current_q_command = (
            math_utils.subtract_frame_transforms(
                self.left_hand_wrist_position_e,
                self.left_hand_wrist_wxyz_e,
                left_hand_wrist_pose_command_e[:, :3],
                left_hand_wrist_pose_command_e[:, 3:],
            )
        )

        right_joint_pos_delta = (
            self.right_hand_finger_joint_pos_command - self.right_robot.data.joint_pos
        )
        left_joint_pos_delta = (
            self.left_hand_finger_joint_pos_command - self.left_robot.data.joint_pos
        )

        object_current_p_command, object_current_q_command = (
            math_utils.subtract_frame_transforms(
                self.object_position_e,
                self.object_orientation_e,
                self.object_body_position_command_e,
                self.object_body_wxyz_command_e,
            )
        )

        return torch.cat(
            (
                right_wrist_current_p_command,
                right_wrist_current_q_command,
                left_wrist_current_p_command,
                left_wrist_current_q_command,
                right_joint_pos_delta,
                left_joint_pos_delta,
                object_current_p_command.reshape(self.num_envs, -1),
                object_current_q_command.reshape(self.num_envs, -1),
            ),
            dim=-1,
        )

    def get_command_contact_part_id(self, side: str) -> torch.Tensor:
        """Get the contact part id for the entire hand. Shape is (num_envs,)."""
        contact_part_id = (
            # object contact part id start from 1
            self.retargeted_left_object_contact_part_ids_per_hand[self.timestep_counter]
            - 1
            if side == "left"
            else self.retargeted_right_object_contact_part_ids_per_hand[
                self.timestep_counter
            ]
            - 1
        )
        contact_part_id = contact_part_id.clamp(min=0, max=self.num_bodies - 1)
        return contact_part_id

    def get_command_contact_object_position_orientation(
        self, side: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the contact object position and orientation in the environment frame for the given side. Shape is (num_envs, 3)."""
        contact_part_id = self.get_command_contact_part_id(side)
        object_position = self.object_position_e[self.all_env_ids, contact_part_id]
        object_orientation = self.object_orientation_e[
            self.all_env_ids, contact_part_id
        ]

        return object_position, object_orientation

    @property
    def right_hand_wrist_pose_command_e(self) -> torch.Tensor:
        """The desired goal position and wxyz in the environment frame for the right hand wrist. Shape is (num_envs, 7)."""
        object_position, object_orientation = (
            self.get_command_contact_object_position_orientation("right")
        )
        position, wxyz = math_utils.combine_frame_transforms(
            object_position,
            object_orientation,
            self.retargeted_right_wrist_position_o[self.timestep_counter],
            self.retargeted_right_wrist_wxyz_o[self.timestep_counter],
        )
        wxyz = math_utils.quat_unique(wxyz) if self.cfg.make_quat_unique else wxyz
        return torch.cat((position, wxyz), dim=-1)

    @property
    def left_hand_wrist_pose_command_e(self) -> torch.Tensor:
        """The desired goal position and wxyz in the environment frame for the left hand wrist. Shape is (num_envs, 7)."""
        object_position, object_orientation = (
            self.get_command_contact_object_position_orientation("left")
        )
        position, wxyz = math_utils.combine_frame_transforms(
            object_position,
            object_orientation,
            self.retargeted_left_wrist_position_o[self.timestep_counter],
            self.retargeted_left_wrist_wxyz_o[self.timestep_counter],
        )
        wxyz = math_utils.quat_unique(wxyz) if self.cfg.make_quat_unique else wxyz
        return torch.cat((position, wxyz), dim=-1)

    @property
    def right_hand_finger_joint_pos_command(self) -> torch.Tensor:
        """The desired goal finger joint position for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_FINGER_JOINTS)."""
        return self.retargeted_right_finger_joints[self.timestep_counter].float()

    @property
    def left_hand_finger_joint_pos_command(self) -> torch.Tensor:
        """The desired goal finger joint position for the left hand. Shape is (num_envs, NUM_LEFT_HAND_FINGER_JOINTS)."""
        return self.retargeted_left_finger_joints[self.timestep_counter].float()

    @property
    def right_hand_fingertip_position_command_o(self) -> torch.Tensor:
        """The desired goal fingertip position in the object frame for the right hand. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return self.retargeted_right_hand_frame_position_o[self.timestep_counter][
            :, self.retargeted_right_fingertip_indices
        ].float()

    @property
    def right_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """The desired goal fingertip position in the environment frame for the right hand. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        if self.cfg.recompute_hand_keypoints_from_object:
            object_position, object_orientation = (
                self.get_command_contact_object_position_orientation("right")
            )
            return math_utils.combine_frame_transforms(
                object_position.unsqueeze(1).expand(
                    -1, len(self.retargeted_right_fingertip_indices), -1
                ),
                object_orientation.unsqueeze(1).expand(
                    -1, len(self.retargeted_right_fingertip_indices), -1
                ),
                self.right_hand_fingertip_position_command_o,
                q12=None,
            )[0]
        else:
            return self.retargeted_right_hand_frames[self.timestep_counter][
                :, self.retargeted_right_fingertip_indices, :3
            ].float()

    @property
    def left_hand_fingertip_position_command_o(self) -> torch.Tensor:
        """The desired goal fingertip position in the object frame for the left hand. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return self.retargeted_left_hand_frame_position_o[self.timestep_counter][
            :, self.retargeted_left_fingertip_indices
        ].float()

    @property
    def left_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """The desired goal fingertip position in the environment frame for the left hand. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        if self.cfg.recompute_hand_keypoints_from_object:
            object_position, object_orientation = (
                self.get_command_contact_object_position_orientation("left")
            )
            return math_utils.combine_frame_transforms(
                object_position.unsqueeze(1).expand(
                    -1, len(self.retargeted_left_fingertip_indices), -1
                ),
                object_orientation.unsqueeze(1).expand(
                    -1, len(self.retargeted_left_fingertip_indices), -1
                ),
                self.left_hand_fingertip_position_command_o,
                q12=None,
            )[0]
        else:
            return self.retargeted_left_hand_frames[self.timestep_counter][
                :, self.retargeted_left_fingertip_indices, :3
            ].float()

    @property
    def object_body_position_command_e(self) -> torch.Tensor:
        """The desired goal position in the environment frame for the object. Shape is (num_envs, NUM_BODY, 3)."""
        return self.retargeted_object_body_position[self.timestep_counter].float()

    @property
    def object_body_wxyz_command_e(self) -> torch.Tensor:
        """The desired goal orientation in the environment frame for the object. Shape is (num_envs, NUM_BODY, 4)."""
        retargeted_object_wxyz = self.retargeted_object_body_wxyz[
            self.timestep_counter
        ].float()
        retargeted_object_wxyz = (
            math_utils.quat_unique(retargeted_object_wxyz)
            if self.cfg.make_quat_unique
            else retargeted_object_wxyz
        )
        return retargeted_object_wxyz.float()

    @property
    def right_hand_object_contact_command_positions_o(self) -> torch.Tensor:
        """The target contact positions in the object frame for the right hand. Shape is (num_envs, num_retargeted_contacts_right, 3)."""
        return self.retargeted_right_object_contact_positions_o[self.timestep_counter]

    @property
    def right_hand_link_contact_command_normals_o(self) -> torch.Tensor:
        """The target contact normals in the object frame for the right hand. Shape is (num_envs, num_retargeted_contacts_right, 3)."""
        return self.retargeted_right_link_contact_normals_o[self.timestep_counter]

    @property
    def right_hand_object_contact_command_positions_and_normals_e(self) -> torch.Tensor:
        """The target contact positions and normals in the environment frame for the right hand.

        retargeted_{side}_link_contact_normals_e is the normal of the human hand, pointing outward from the hand (inward to object).
        Shape is (num_envs, num_retargeted_contacts_right, 6).
        """
        valid_contact_mask = self.retargeted_right_object_contact_is_valid[
            self.timestep_counter
        ]  # (num_envs, num_retargeted_contacts_right)

        contact_part_id = (
            # 0 for no contact, object contact part id start from 1
            self.retargeted_right_object_contact_part_ids[self.timestep_counter]
            - 1
        )
        contact_part_id = contact_part_id.clamp(min=0, max=self.num_bodies - 1)
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

        right_hand_object_contact_command_positions_e, _ = (
            math_utils.combine_frame_transforms(
                object_position,
                object_orientation,
                self.right_hand_object_contact_command_positions_o,
                q12=None,
            )
        )
        right_hand_object_contact_command_normals_e, _ = (
            math_utils.combine_frame_transforms(
                torch.zeros_like(object_position),
                object_orientation,
                self.right_hand_link_contact_command_normals_o,
                q12=None,
            )
        )

        right_hand_object_contact_command_positions_e.masked_fill_(
            ~valid_contact_mask.unsqueeze(-1), 0.0
        )
        right_hand_object_contact_command_normals_e.masked_fill_(
            ~valid_contact_mask.unsqueeze(-1), 0.0
        )
        return torch.cat(
            [
                right_hand_object_contact_command_positions_e,
                right_hand_object_contact_command_normals_e,
            ],
            dim=-1,
        )

    @property
    def left_hand_object_contact_command_positions_o(self) -> torch.Tensor:
        """The target contact positions in the object frame for the left hand. Shape is (num_envs, num_retargeted_contacts_left, 3)."""
        return self.retargeted_left_object_contact_positions_o[self.timestep_counter]

    @property
    def left_hand_link_contact_command_normals_o(self) -> torch.Tensor:
        """The target contact normals in the object frame for the left hand. Shape is (num_envs, num_retargeted_contacts_left, 3)."""
        return self.retargeted_left_link_contact_normals_o[self.timestep_counter]

    @property
    def left_hand_object_contact_command_positions_and_normals_e(self) -> torch.Tensor:
        """The target contact positions and normals in the environment frame for the left hand.

        retargeted_{side}_link_contact_normals_e is the normal of the human hand, pointing outward from the hand (inward to object).
        Shape is (num_envs, num_retargeted_contacts_left, 6).
        """
        valid_contact_mask = self.retargeted_left_object_contact_is_valid[
            self.timestep_counter
        ]  # (num_envs, num_retargeted_contacts_left)
        contact_part_id = (
            # 0 for no contact, object contact part id start from 1
            self.retargeted_left_object_contact_part_ids[self.timestep_counter]
            - 1
        )
        contact_part_id = contact_part_id.clamp(min=0, max=self.num_bodies - 1)
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

        left_hand_object_contact_command_positions_e, _ = (
            math_utils.combine_frame_transforms(
                object_position,
                object_orientation,
                self.left_hand_object_contact_command_positions_o,
                q12=None,
            )
        )
        left_hand_object_contact_command_normals_e, _ = (
            math_utils.combine_frame_transforms(
                torch.zeros_like(object_position),
                object_orientation,
                self.left_hand_link_contact_command_normals_o,
                q12=None,
            )
        )

        left_hand_object_contact_command_positions_e.masked_fill_(
            ~valid_contact_mask.unsqueeze(-1), 0.0
        )
        left_hand_object_contact_command_normals_e.masked_fill_(
            ~valid_contact_mask.unsqueeze(-1), 0.0
        )
        return torch.cat(
            [
                left_hand_object_contact_command_positions_e,
                left_hand_object_contact_command_normals_e,
            ],
            dim=-1,
        )

    @property
    def right_hand_contact_wrench_supports_command(self) -> torch.Tensor:
        """The contact wrench supports for the right hand contact. Shape is (num_envs, num_bodies, num_wrench_space_basis_samples)."""
        # Alias of the cached ref_right (same retargeted[timestep_counter] indexing).
        return self.ref_right

    @property
    def left_hand_contact_wrench_supports_command(self) -> torch.Tensor:
        """The contact wrench supports for the left hand contact. Shape is (num_envs, num_bodies, num_wrench_space_basis_samples)."""
        return self.ref_left

    ######################################################################
    # Observations
    ######################################################################

    @property
    def right_hand_wrist_position_w(self) -> torch.Tensor:
        """The current position in the environment frame for the right hand wrist. Shape is (num_envs, 3)."""
        return self.right_robot.data.root_link_pos_w

    @property
    def left_hand_wrist_position_w(self) -> torch.Tensor:
        """The current position in the environment frame for the left hand wrist. Shape is (num_envs, 3)."""
        return self.left_robot.data.root_link_pos_w

    @property
    def right_hand_wrist_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the right hand wrist. Shape is (num_envs, 3)."""
        return (self.right_hand_wrist_position_w - self._env.scene.env_origins).float()

    @property
    def left_hand_wrist_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the left hand wrist. Shape is (num_envs, 3)."""
        return (self.left_hand_wrist_position_w - self._env.scene.env_origins).float()

    @property
    def right_hand_wrist_wxyz_e(self) -> torch.Tensor:
        """The current wxyz in the environment frame for the right hand wrist. Shape is (num_envs, 4)."""
        right_wrist_wxyz = self.right_robot.data.root_link_quat_w.float()
        return (
            math_utils.quat_unique(right_wrist_wxyz)
            if self.cfg.make_quat_unique
            else right_wrist_wxyz
        )

    @property
    def left_hand_wrist_wxyz_e(self) -> torch.Tensor:
        """The current wxyz in the environment frame for the left hand wrist. Shape is (num_envs, 4)."""
        left_wrist_wxyz = self.left_robot.data.root_link_quat_w.float()
        return (
            math_utils.quat_unique(left_wrist_wxyz)
            if self.cfg.make_quat_unique
            else left_wrist_wxyz
        )

    @property
    def right_hand_wrist_velocity_b(self) -> torch.Tensor:
        """The current velocity in the body frame for the right hand wrist. Shape is (num_envs, 6)."""
        return torch.cat(
            [
                self.right_robot.data.root_lin_vel_b,
                self.right_robot.data.root_ang_vel_b,
            ],
            dim=-1,
        ).float()

    @property
    def left_hand_wrist_velocity_b(self) -> torch.Tensor:
        """The current velocity in the body frame for the left hand wrist. Shape is (num_envs, 6)."""
        return torch.cat(
            [
                self.left_robot.data.root_lin_vel_b,
                self.left_robot.data.root_ang_vel_b,
            ],
            dim=-1,
        ).float()

    @property
    def right_hand_finger_joint_pos(self) -> torch.Tensor:
        """The current joint position for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_FINGER_JOINTS)."""
        return self.right_robot.data.joint_pos.float()

    @property
    def left_hand_finger_joint_pos(self) -> torch.Tensor:
        """The current joint position for the left hand. Shape is (num_envs, NUM_LEFT_HAND_FINGER_JOINTS)."""
        return self.left_robot.data.joint_pos.float()

    @property
    def right_hand_finger_joint_vel(self) -> torch.Tensor:
        """The current joint velocity for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_FINGER_JOINTS)."""
        return self.right_robot.data.joint_vel.float()

    @property
    def left_hand_finger_joint_vel(self) -> torch.Tensor:
        """The current joint velocity for the left hand. Shape is (num_envs, NUM_LEFT_HAND_FINGER_JOINTS)."""
        return self.left_robot.data.joint_vel.float()

    @property
    def right_hand_fingertip_position_w(self) -> torch.Tensor:
        """The current position in the environment frame for the right fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return self.right_robot.data.body_link_pos_w[:, self.right_fingertip_body_ids]

    @property
    def left_hand_fingertip_position_w(self) -> torch.Tensor:
        """The current position in the environment frame for the left fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return self.left_robot.data.body_link_pos_w[:, self.left_fingertip_body_ids]

    @property
    def right_hand_fingertip_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the right fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        fingertip_position_e = (
            self.right_hand_fingertip_position_w
            - self._env.scene.env_origins.unsqueeze(1)
        )
        return fingertip_position_e.float()

    @property
    def left_hand_fingertip_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the left fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        fingertip_position_e = (
            self.left_hand_fingertip_position_w
            - self._env.scene.env_origins.unsqueeze(1)
        )
        return fingertip_position_e.float()

    @property
    def right_hand_fingertip_orientation_e(self) -> torch.Tensor:
        """The current orientation in the environment frame for the left and right fingertip. Shape is (num_envs, NUM_FINGERTIPS, 4)."""
        return self.right_robot.data.body_link_quat_w[
            :, self.right_fingertip_body_ids
        ].float()

    @property
    def left_hand_fingertip_orientation_e(self) -> torch.Tensor:
        """The current orientation in the environment frame for the left fingertip. Shape is (num_envs, NUM_FINGERTIPS, 4)."""
        return self.left_robot.data.body_link_quat_w[
            :, self.left_fingertip_body_ids
        ].float()

    @property
    def object_position_w(self) -> torch.Tensor:
        """The current position in the environment frame for the object. Shape is (num_envs, NUM_BODY, 3)."""
        return torch.cat(
            [object.data.body_link_pos_w for object in self.objects], dim=1
        ).float()

    @property
    def object_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the object. Shape is (num_envs, NUM_BODY, 3)."""
        object_position_e = (
            self.object_position_w - self._env.scene.env_origins.unsqueeze(1)
        )
        return object_position_e.float()

    @property
    def object_orientation_e(self) -> torch.Tensor:
        """The current orientation in the environment frame for the object. Shape is (num_envs, NUM_BODY, 4)."""
        return torch.cat(
            [object.data.body_link_quat_w for object in self.objects], dim=1
        ).float()

    @property
    def object_body_lin_vel_e(self) -> torch.Tensor:
        """Current linear velocity per body in env frame. Shape ``(num_envs, NUM_BODY, 3)``.

        Velocity is translation-invariant, so this is identical to the world-frame
        value — the ``_e`` suffix is kept for naming symmetry with the position
        properties.
        """
        return torch.cat(
            [object.data.body_link_lin_vel_w for object in self.objects], dim=1
        ).float()

    @property
    def object_body_lin_vel_command_e(self) -> torch.Tensor:
        """Target linear velocity per body at the current timestep. Shape ``(num_envs, NUM_BODY, 3)``.

        Precomputed via central finite differences over the retargeted position
        trajectory in :meth:`_init_object_data`.
        """
        return self.retargeted_object_body_lin_vel[self.timestep_counter].float()

    @property
    def relative_object_pose_error(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pose error between current and demo relative pose of obj1 in obj0's frame.

        Returns:
            pos_err: (num_envs,) L2 position error in metres.
            rot_err: (num_envs,) geodesic rotation error in radians.

        Only call when self._has_multi_object is True.
        """
        obj0_pos = self.object_position_e[:, 0, :]
        obj0_quat = self.object_orientation_e[:, 0, :]
        obj1_pos = self.object_position_e[:, self._obj1_root_body_idx, :]
        obj1_quat = self.object_orientation_e[:, self._obj1_root_body_idx, :]
        cur_rel_pos, cur_rel_quat = math_utils.subtract_frame_transforms(
            obj0_pos, obj0_quat, obj1_pos, obj1_quat
        )
        demo_rel_pos = self._demo_rel_pos[self.timestep_counter]
        demo_rel_quat = self._demo_rel_quat[self.timestep_counter]
        pos_err = torch.norm(cur_rel_pos - demo_rel_pos, dim=-1)
        rot_err = math_utils.quat_error_magnitude(cur_rel_quat, demo_rel_quat)
        return pos_err, rot_err

    @property
    def relative_object_proximity_mask(self) -> torch.Tensor:
        """(num_envs,) bool — True when demo inter-object distance < proximity_threshold."""
        demo_dist = self._demo_inter_object_dist[self.timestep_counter]
        return demo_dist < self.cfg.relative_object_proximity_threshold

    @property
    def object_com_position_and_wxyz_w(self) -> torch.Tensor:
        """The current position and orientation in the environment frame for the object com. Shape is (num_envs, num_bodies, 7)."""
        return torch.cat(
            [object.data.body_com_state_w[..., :7] for object in self.objects], dim=1
        ).float()

    @property
    def right_hand_object_contact_positions_w(self) -> torch.Tensor:
        """The current contact positions in the world frame for the right hand. Shape is (num_envs, num_bodies, num_hand_link_w_sensor, 3)."""
        contact_positions = torch.cat(
            [
                contact_sensor.data.contact_pos_w
                for contact_sensor in self.object_to_right_hand_contact_sensors
            ],
            dim=1,
        )  # (num_envs, num_bodies, num_robot_links, 3)

        return torch.nan_to_num(contact_positions, nan=0.0)

    @property
    def right_hand_object_contact_positions_e(self) -> torch.Tensor:
        """The current contact positions in the environment frame for the right hand. Shape is (num_envs, num_bodies, num_hand_link_w_sensor, 3)."""
        return self.right_hand_object_contact_positions_w - self.env_origins_expanded

    @property
    def right_hand_object_contact_forces_w(self) -> torch.Tensor:
        """The current contact forces in the world frame for the right hand.

        Force is hand to object, the direction is inward to object.
        Shape is (num_envs, contact_sensor_history_length, num_bodies, num_hand_link_w_sensor, 3).
        """
        return torch.cat(
            [
                contact_sensor.data.force_matrix_w_history
                for contact_sensor in self.object_to_right_hand_contact_sensors
            ],
            dim=2,
        ).view(
            self.num_envs,
            self.contact_sensor_history_length,
            self.num_bodies,
            self.num_robot_contacts_right,
            3,
        )

    @property
    def left_hand_object_contact_positions_w(self) -> torch.Tensor:
        """The current contact positions in the environment frame for the left hand. Shape is (num_envs, num_bodies, num_hand_link_w_sensor, 3)."""
        contact_positions = torch.cat(
            [
                contact_sensor.data.contact_pos_w
                for contact_sensor in self.object_to_left_hand_contact_sensors
            ],
            dim=1,
        )  # (num_envs, num_bodies, num_robot_links, 3)

        return torch.nan_to_num(contact_positions, nan=0.0)

    @property
    def left_hand_object_contact_positions_e(self) -> torch.Tensor:
        """The current contact positions in the environment frame for the left hand. Shape is (num_envs, num_bodies, num_hand_link_w_sensor, 3)."""
        return self.left_hand_object_contact_positions_w - self.env_origins_expanded

    @property
    def left_hand_object_contact_forces_w(self) -> torch.Tensor:
        """The current contact forces in the world frame for the left hand.

        Force is hand to object, the direction is inward to object.
        Shape is (num_envs, contact_sensor_history_length, num_bodies, num_hand_link_w_sensor, 3).
        """
        return torch.cat(
            [
                contact_sensor.data.force_matrix_w_history
                for contact_sensor in self.object_to_left_hand_contact_sensors
            ],
            dim=2,
        ).view(
            self.num_envs,
            self.contact_sensor_history_length,
            self.num_bodies,
            self.num_robot_contacts_left,
            3,
        )

    @property
    def right_hand_contact_wrench_supports(self) -> torch.Tensor:
        """The wrench representation for the right hand contact. Shape is (num_envs, num_bodies, num_wrench_space_basis_samples)."""
        self.refresh_tensors()
        return self.right_contact_wrench_supports

    @property
    def left_hand_contact_wrench_supports(self) -> torch.Tensor:
        """The wrench representation for the left hand contact. Shape is (num_envs, num_bodies, num_wrench_space_basis_samples)."""
        self.refresh_tensors()
        return self.left_contact_wrench_supports

    ######################################################################
    # Cached refresh accessors (populated by refresh_tensors).
    ######################################################################

    @property
    def right_force_sq_per_link(self) -> torch.Tensor:
        """Per-link squared force magnitude for the right hand.

        Shape ``(num_envs, num_right_hand_links_w_sensor)``. Computed as
        ``forces_w.square().sum(-1).mean(dim=history).sum(dim=bodies)``.
        """
        self.refresh_tensors()
        return self._cached_right_force_sq_per_link

    @property
    def left_force_sq_per_link(self) -> torch.Tensor:
        """Per-link squared force magnitude for the left hand. See :meth:`right_force_sq_per_link`."""
        self.refresh_tensors()
        return self._cached_left_force_sq_per_link

    @property
    def right_link_in_contact(self) -> torch.Tensor:
        """Per-link in-contact boolean for the right hand (threshold 1e-3). Shape ``(num_envs, num_links)``."""
        self.refresh_tensors()
        return self._cached_right_link_in_contact

    @property
    def left_link_in_contact(self) -> torch.Tensor:
        """Per-link in-contact boolean for the left hand (threshold 1e-3). Shape ``(num_envs, num_links)``."""
        self.refresh_tensors()
        return self._cached_left_link_in_contact

    @property
    def right_in_contact(self) -> torch.Tensor:
        """Per-env right-hand in-contact boolean (threshold 1e-3). Shape ``(num_envs,)``."""
        self.refresh_tensors()
        return self._cached_right_in_contact

    @property
    def left_in_contact(self) -> torch.Tensor:
        """Per-env left-hand in-contact boolean (threshold 1e-3). Shape ``(num_envs,)``."""
        self.refresh_tensors()
        return self._cached_left_in_contact

    @property
    def in_contact(self) -> torch.Tensor:
        """Per-env either-hand in-contact boolean. Shape ``(num_envs,)``."""
        self.refresh_tensors()
        return self._cached_in_contact

    @property
    def ref_left(self) -> torch.Tensor:
        """Retargeted left wrench supports at the current timestep. Shape ``(num_envs, num_bodies, num_wrench_basis)``."""
        self.refresh_tensors()
        return self._cached_ref_L

    @property
    def ref_right(self) -> torch.Tensor:
        """Retargeted right wrench supports at the current timestep. Shape ``(num_envs, num_bodies, num_wrench_basis)``."""
        self.refresh_tensors()
        return self._cached_ref_R

    @property
    def mask_left(self) -> torch.Tensor:
        """``ref_left > 1e-6``. Same shape as ``ref_left``."""
        self.refresh_tensors()
        return self._cached_mask_L

    @property
    def mask_right(self) -> torch.Tensor:
        """``ref_right > 1e-6``. Same shape as ``ref_right``."""
        self.refresh_tensors()
        return self._cached_mask_R

    @property
    def ref_active_per_cell(self) -> torch.Tensor:
        """Per-body per-basis reference-active mask: ``mask_left | mask_right``. Shape ``(num_envs, num_bodies, num_wrench_basis)``."""
        self.refresh_tensors()
        return self._cached_ref_active_per_cell

    @property
    def ref_active_per_body(self) -> torch.Tensor:
        """Per-body reference-active mask (any basis). Shape ``(num_envs, num_bodies)``."""
        self.refresh_tensors()
        return self._cached_ref_active_per_body

    @property
    def ref_active_global(self) -> torch.Tensor:
        """Global per-env reference-active mask. Shape ``(num_envs,)``."""
        self.refresh_tensors()
        return self._cached_ref_active_global

    @property
    def right_wrench_cmd_active(self) -> torch.Tensor:
        """``ref_right > 1e-3`` ("meaningful" support). Shape ``(N, num_bodies, num_wrench_basis)``."""
        self.refresh_tensors()
        return self._cached_right_wrench_cmd_active

    @property
    def left_wrench_cmd_active(self) -> torch.Tensor:
        """``ref_left > 1e-3`` ("meaningful" support). Shape ``(N, num_bodies, num_wrench_basis)``."""
        self.refresh_tensors()
        return self._cached_left_wrench_cmd_active

    @property
    def right_wrench_cur_active(self) -> torch.Tensor:
        """``right_contact_wrench_supports > 1e-3``. Shape ``(N, num_bodies, num_wrench_basis)``."""
        self.refresh_tensors()
        return self._cached_right_wrench_cur_active

    @property
    def left_wrench_cur_active(self) -> torch.Tensor:
        """``left_contact_wrench_supports > 1e-3``. Shape ``(N, num_bodies, num_wrench_basis)``."""
        self.refresh_tensors()
        return self._cached_left_wrench_cur_active

    @property
    def right_wrench_cmd_active_per_body(self) -> torch.Tensor:
        """``right_wrench_cmd_active.any(-1)``. Shape ``(N, num_bodies)``."""
        self.refresh_tensors()
        return self._cached_right_wrench_cmd_active_per_body

    @property
    def left_wrench_cmd_active_per_body(self) -> torch.Tensor:
        """``left_wrench_cmd_active.any(-1)``. Shape ``(N, num_bodies)``."""
        self.refresh_tensors()
        return self._cached_left_wrench_cmd_active_per_body

    @property
    def right_wrench_cur_active_per_body(self) -> torch.Tensor:
        """``right_wrench_cur_active.any(-1)``. Shape ``(N, num_bodies)``."""
        self.refresh_tensors()
        return self._cached_right_wrench_cur_active_per_body

    @property
    def left_wrench_cur_active_per_body(self) -> torch.Tensor:
        """``left_wrench_cur_active.any(-1)``. Shape ``(N, num_bodies)``."""
        self.refresh_tensors()
        return self._cached_left_wrench_cur_active_per_body

    @property
    def object_position_e_sq(self) -> torch.Tensor:
        """Object position in env frame with body dim squeezed. Shape ``(num_envs, 3)``."""
        self.refresh_tensors()
        return self._cached_object_position_e_sq

    @property
    def object_wxyz_e_sq(self) -> torch.Tensor:
        """Object orientation in env frame with body dim squeezed. Shape ``(num_envs, 4)``."""
        self.refresh_tensors()
        return self._cached_object_wxyz_e_sq

    ######################################################################
    # Refresh machinery.
    ######################################################################

    def refresh_tensors(self) -> None:
        """Recompute the shared tensors consumed by rewards / observations.

        Lazy: no-op unless ``self._tensors_dirty`` is True. Dirty is raised in
        ``__init__``, at the end of ``_update_command`` (after ``timestep_counter``
        increments), and at the end of ``_resample_command``. Reward and observation
        phases each trigger one refresh; within a phase repeated reads are free.
        """
        if not self._tensors_dirty:
            return
        self._tensors_dirty = False

        # Wrench supports: fill existing self.{right,left}_contact_wrench_supports
        # in place (Python-side, contains a per-body loop so it is not JIT-compiled).
        self._compute_contact_wrench_supports("right")
        self._compute_contact_wrench_supports("left")

        # Pure-tensor derivations fused by refresh_jit.
        (
            self._cached_right_force_sq_per_link,
            self._cached_left_force_sq_per_link,
            self._cached_right_link_in_contact,
            self._cached_left_link_in_contact,
            self._cached_right_in_contact,
            self._cached_left_in_contact,
            self._cached_in_contact,
            self._cached_ref_L,
            self._cached_ref_R,
            self._cached_mask_L,
            self._cached_mask_R,
            self._cached_ref_active_per_cell,
            self._cached_ref_active_per_body,
            self._cached_ref_active_global,
            self._cached_right_wrench_cmd_active,
            self._cached_left_wrench_cmd_active,
            self._cached_right_wrench_cur_active,
            self._cached_left_wrench_cur_active,
            self._cached_right_wrench_cmd_active_per_body,
            self._cached_left_wrench_cmd_active_per_body,
            self._cached_right_wrench_cur_active_per_body,
            self._cached_left_wrench_cur_active_per_body,
            self._cached_object_position_e_sq,
            self._cached_object_wxyz_e_sq,
        ) = refresh_jit(
            right_forces_w=self.right_hand_object_contact_forces_w,
            left_forces_w=self.left_hand_object_contact_forces_w,
            retargeted_left_contact_wrench_supports=self.retargeted_left_contact_wrench_supports,
            retargeted_right_contact_wrench_supports=self.retargeted_right_contact_wrench_supports,
            timestep_counter=self.timestep_counter,
            right_contact_wrench_supports=self.right_contact_wrench_supports,
            left_contact_wrench_supports=self.left_contact_wrench_supports,
            object_position_e=self.object_position_e,
            object_orientation_e=self.object_orientation_e,
        )

    def _compute_contact_wrench_supports(self, side: str) -> None:
        """Fill ``self.{side}_contact_wrench_supports`` in place for one hand.

        Thin Python wrapper over two JIT helpers: ``wrench_preprocess_jit`` fuses
        the contact-position/normal transform into the object COM frame for all
        bodies at once, then ``wrench_support_one_body_jit`` fuses the
        friction-cone edges + wrench-space assembly + support-function reduction
        per body.
        """
        if side == "right":
            contact_positions_w = self.right_hand_object_contact_positions_w
            contact_forces_w = self.right_hand_object_contact_forces_w
            buffer = self.right_contact_wrench_supports
        else:
            contact_positions_w = self.left_hand_object_contact_positions_w
            contact_forces_w = self.left_hand_object_contact_forces_w
            buffer = self.left_contact_wrench_supports

        # Note: Use num_robot_contacts_right for both hands when expanding the
        # object COM state (see legacy left property). Preserving that behavior.
        num_robot_contacts = self.num_robot_contacts_right

        com_state = self.object_com_position_and_wxyz_w  # (N, bodies, 7)
        object_com_position_w = com_state[..., :3].unsqueeze(2)  # (N, bodies, 1, 3)
        object_com_orientation_w = com_state[..., 3:7].unsqueeze(2)  # (N, bodies, 1, 4)

        contact_positions_com, contact_normals_com = wrench_preprocess_jit(
            contact_positions_w=contact_positions_w,
            contact_forces_first_hist_w=contact_forces_w[:, 0],
            object_com_position_w=object_com_position_w,
            object_com_orientation_w=object_com_orientation_w,
            num_envs=self.num_envs,
            num_bodies=self.num_bodies,
            num_robot_contacts=num_robot_contacts,
        )

        friction_coefficients = float(self.cfg.friction_coefficients)
        for body_idx, body_radius in enumerate(self.object_mesh_radius):
            buffer[:, body_idx] = wrench_support_one_body_jit(
                contact_points=contact_positions_com[:, body_idx],
                contact_normals=contact_normals_com[:, body_idx],
                cos_t=self.friction_cone_edge_cosines,
                sin_t=self.friction_cone_edge_sines,
                basis=self.wrench_space_bases[body_idx],
                rc=float(body_radius),
                friction_coefficients=friction_coefficients,
            )

    ######################################################################
    # Specific functions.
    ######################################################################

    def _update_metrics(self) -> None:
        """Update the metrics."""
        # Right hand
        right_hand_wrist_pose_command_e = self.right_hand_wrist_pose_command_e
        self.metrics["right_hand_wrist_position_error"] = torch.norm(
            self.right_hand_wrist_position_e - right_hand_wrist_pose_command_e[:, :3],
            dim=-1,
        )
        self.metrics["right_hand_wrist_wxyz_error"] = math_utils.quat_error_magnitude(
            self.right_hand_wrist_wxyz_e, right_hand_wrist_pose_command_e[:, 3:]
        )
        self.metrics["right_hand_finger_joints_error"] = torch.norm(
            self.right_hand_finger_joint_pos - self.right_hand_finger_joint_pos_command,
            dim=-1,
        )
        # Left hand
        left_hand_wrist_pose_command_e = self.left_hand_wrist_pose_command_e
        self.metrics["left_hand_wrist_position_error"] = torch.norm(
            self.left_hand_wrist_position_e - left_hand_wrist_pose_command_e[:, :3],
            dim=-1,
        )
        self.metrics["left_hand_wrist_wxyz_error"] = math_utils.quat_error_magnitude(
            self.left_hand_wrist_wxyz_e, left_hand_wrist_pose_command_e[:, 3:]
        )
        self.metrics["left_hand_finger_joints_error"] = torch.norm(
            self.left_hand_finger_joint_pos - self.left_hand_finger_joint_pos_command,
            dim=-1,
        )
        # Object
        self.metrics["object_body_position_error"] = torch.norm(
            self.object_position_e - self.object_body_position_command_e,
            dim=-1,
        ).mean(dim=-1)
        self.metrics["object_body_wxyz_error"] = math_utils.quat_error_magnitude(
            self.object_orientation_e,
            self.object_body_wxyz_command_e,
        ).mean(dim=-1)

        # Log the per-env VOC scale actually applied to the object controller.
        self.metrics["virtual_object_controller_scale_factor"] = (
            self.virtual_object_controller_scale_factor_per_env.squeeze(-1)
        )

        _spider_shared = update_spider_metrics(self)
        update_maniptrans_metrics(self, _spider_shared)

        if self._has_multi_object:
            pos_err, rot_err = self.relative_object_pose_error
            self.metrics["relative_object_pos_error"] = pos_err
            self.metrics["relative_object_rot_error"] = rot_err

        if not self.cfg.enable_additional_metrics:
            return

        # DISABLED: voc_decay_fired W&B logging
        # voc_scale_now = float(self.virtual_object_controller_scale_factor)
        # decay_fired = 1.0 if voc_scale_now < self._prev_voc_scale else 0.0
        # self._prev_voc_scale = voc_scale_now
        # self.metrics["voc_decay_fired"] = decay_fired * torch.ones(
        #     self.num_envs, device=self.device
        # )

        # ------------------------------------------------------------------ #
        # DISABLED: per-side contact wrench level metrics (command/current
        # support mean, support ratio). The CV metric below uses
        # contact_wrench_support_reward_jit directly off self.* buffers, so
        # this entire local-variable preamble is dead.
        # ------------------------------------------------------------------ #
        # right_cmd = (
        #     self.right_hand_contact_wrench_supports_command
        # )  # (num_envs, num_bodies, num_basis)
        # right_curr = self.right_contact_wrench_supports  # buffer
        # left_cmd = self.left_hand_contact_wrench_supports_command
        # left_curr = self.left_contact_wrench_supports  # buffer
        #
        # right_cmd_has_contact = right_cmd.amax(dim=-1) > 1e-3  # (num_envs, num_bodies)
        # right_curr_has_contact = right_curr.amax(dim=-1) > 1e-3
        # left_cmd_has_contact = left_cmd.amax(dim=-1) > 1e-3
        # left_curr_has_contact = left_curr.amax(dim=-1) > 1e-3
        #
        # right_cmd_count = right_cmd_has_contact.sum(dim=-1).clamp(
        #     min=1e-3
        # )  # (num_envs,)
        # left_cmd_count = left_cmd_has_contact.sum(dim=-1).clamp(min=1e-3)
        #
        # self.metrics["contact_wrench_command_support_mean_right"] = (
        #     right_cmd.mean(dim=-1) * right_cmd_has_contact
        # ).sum(dim=-1) / right_cmd_count
        # self.metrics["contact_wrench_current_support_mean_right"] = (
        #     right_curr.mean(dim=-1) * right_cmd_has_contact
        # ).sum(dim=-1) / right_cmd_count
        # self.metrics["contact_wrench_command_support_mean_left"] = (
        #     left_cmd.mean(dim=-1) * left_cmd_has_contact
        # ).sum(dim=-1) / left_cmd_count
        # self.metrics["contact_wrench_current_support_mean_left"] = (
        #     left_curr.mean(dim=-1) * left_cmd_has_contact
        # ).sum(dim=-1) / left_cmd_count
        #
        # has_demo_contact_right = right_cmd_has_contact.any(dim=-1)  # (num_envs,)
        # has_demo_contact_left = left_cmd_has_contact.any(dim=-1)
        #
        # def _contact_ratio(
        #     curr: torch.Tensor, cmd: torch.Tensor, cmd_has_contact: torch.Tensor
        # ) -> torch.Tensor:
        #     # (per-basis alignment ratio — see git history for full body)
        #     basis_active = (cmd > 1e-3) & cmd_has_contact.unsqueeze(-1)
        #     per_basis = curr / (cmd + 1e-6)
        #     n_active = basis_active.float().sum(dim=(-2, -1)).clamp(min=1.0)
        #     per_env_ratio = (per_basis * basis_active.float()).sum(
        #         dim=(-2, -1)
        #     ) / n_active
        #     has_demo = cmd_has_contact.any(dim=-1)
        #     n_envs = has_demo.float().sum().clamp(min=1.0)
        #     val = (per_env_ratio * has_demo.float()).sum() / n_envs
        #     return val.expand(curr.shape[0])
        #
        # self.metrics["contact_wrench_support_ratio_right"] = _contact_ratio(
        #     right_curr,
        #     right_cmd,
        #     right_cmd_has_contact,
        # )
        # self.metrics["contact_wrench_support_ratio_left"] = _contact_ratio(
        #     left_curr,
        #     left_cmd,
        #     left_cmd_has_contact,
        # )

        # DISABLED: contact_bodies_coverage_frac_{right,left} W&B logging
        # n_demo_right = has_demo_contact_right.float().sum().clamp(min=1.0)
        # n_demo_left = has_demo_contact_left.float().sum().clamp(min=1.0)
        # coverage_right = (right_cmd_has_contact & right_curr_has_contact).float().sum(
        #     dim=-1
        # ) / right_cmd_count
        # coverage_left = (left_cmd_has_contact & left_curr_has_contact).float().sum(
        #     dim=-1
        # ) / left_cmd_count
        # self.metrics["contact_bodies_coverage_frac_right"] = (
        #     (coverage_right * has_demo_contact_right.float()).sum() / n_demo_right
        # ).expand(self.num_envs)
        # self.metrics["contact_bodies_coverage_frac_left"] = (
        #     (coverage_left * has_demo_contact_left.float()).sum() / n_demo_left
        # ).expand(self.num_envs)

        # DISABLED: object_has_{right,left,any}_contact_frac W&B logging
        # has_curr_right = right_curr_has_contact.any(dim=-1)  # (num_envs,)
        # has_curr_left = left_curr_has_contact.any(dim=-1)
        # has_demo_contact_any = has_demo_contact_right | has_demo_contact_left
        # n_demo_any = has_demo_contact_any.float().sum().clamp(min=1.0)
        # self.metrics["object_has_right_contact_frac"] = (
        #     (has_curr_right.float() * has_demo_contact_right.float()).sum()
        #     / n_demo_right
        # ).expand(self.num_envs)
        # self.metrics["object_has_left_contact_frac"] = (
        #     (has_curr_left.float() * has_demo_contact_left.float()).sum() / n_demo_left
        # ).expand(self.num_envs)
        # self.metrics["object_has_any_contact_frac"] = (
        #     (
        #         (has_curr_right | has_curr_left).float() * has_demo_contact_any.float()
        #     ).sum()
        #     / n_demo_any
        # ).expand(self.num_envs)

        # ------------------------------------------------------------------ #
        # Bounding-box corner tracking error
        # BBOX_CORNER_VECS: (num_envs, num_bodies, 8, 3) — corners (±hx, ±hy, ±hz)
        # scaled by per-body AABB half-extents computed at init from USD geometry.
        #
        # Conditioned on envs that are past the per-episode VOC warmup phase.
        # During the warmup (steps_since_last_reset < virtual_object_control_decay_steps),
        # the timestep_counter is frozen at the reset frame and VOC ≈ 1.0 — so
        # error is trivially near 0 regardless of policy quality and would dilute
        # the mean downward.
        # ------------------------------------------------------------------ #
        past_reset_phase = (
            self.steps_since_last_reset >= self.cfg.virtual_object_control_decay_steps
        )  # (num_envs,)
        n_past_reset = past_reset_phase.float().sum().clamp(min=1.0)

        object_position_exp = self.object_position_e.unsqueeze(2).expand(-1, -1, 8, -1)
        object_wxyz_exp = self.object_orientation_e.unsqueeze(2).expand(-1, -1, 8, -1)
        cmd_position_exp = self.object_body_position_command_e.unsqueeze(2).expand(
            -1, -1, 8, -1
        )
        cmd_wxyz_exp = self.object_body_wxyz_command_e.unsqueeze(2).expand(
            -1, -1, 8, -1
        )

        current_corners, _ = math_utils.combine_frame_transforms(
            object_position_exp, object_wxyz_exp, self.BBOX_CORNER_VECS
        )  # (num_envs, num_bodies, 8, 3)
        command_corners, _ = math_utils.combine_frame_transforms(
            cmd_position_exp, cmd_wxyz_exp, self.BBOX_CORNER_VECS
        )
        corner_error_per_env = torch.norm(
            current_corners - command_corners, dim=-1
        ).mean(
            dim=(-2, -1)
        )  # (num_envs,), meters
        self.metrics["object_bbox_corner_error"] = (
            (corner_error_per_env * past_reset_phase.float()).sum() / n_past_reset
        ).expand(self.num_envs)

        # ------------------------------------------------------------------ #
        # Paper-exact DexMachina ADD metric + AUC.
        # Vertex set: 500 uniform vertex samples per body (paper formula,
        # dexmachina/envs/object.py:305-315, sample_mesh_vertices). Per-frame
        # ADD = mean per-vertex L2 distance between policy pose and demo pose,
        # per body. AUC: flat sliding pool of per-env per-frame per-body ADDs,
        # then `compute_dexmachina_auc` (paper-exact) over the full pool.
        # ------------------------------------------------------------------ #
        if getattr(self, "_dexmachina_metric_enabled", False):
            M = self._dexmachina_num_verts
            dex_pos_exp = self.object_position_e.unsqueeze(2).expand(-1, -1, M, -1)
            dex_wxyz_exp = self.object_orientation_e.unsqueeze(2).expand(-1, -1, M, -1)
            dex_cmd_pos_exp = self.object_body_position_command_e.unsqueeze(2).expand(
                -1, -1, M, -1
            )
            dex_cmd_wxyz_exp = self.object_body_wxyz_command_e.unsqueeze(2).expand(
                -1, -1, M, -1
            )
            dex_current_verts, _ = math_utils.combine_frame_transforms(
                dex_pos_exp, dex_wxyz_exp, self.DEXMACHINA_SAMPLED_VERTS
            )  # (num_envs, num_bodies, M, 3)
            dex_command_verts, _ = math_utils.combine_frame_transforms(
                dex_cmd_pos_exp, dex_cmd_wxyz_exp, self.DEXMACHINA_SAMPLED_VERTS
            )
            # Per-body per-env per-frame mean-vertex ADD (paper's get_all_add).
            dex_add_per_body_env = torch.norm(
                dex_current_verts - dex_command_verts, dim=-1
            ).mean(
                dim=-1
            )  # (num_envs, num_bodies)

            # Fast spatial-mean progress signal (averaged over bodies). Kept as
            # dexmachina_bbox so existing dashboards continue to work.
            dex_vert_err_per_env = dex_add_per_body_env.mean(dim=-1)  # (num_envs,)
            dex_bbox_val = (
                dex_vert_err_per_env * past_reset_phase.float()
            ).sum() / n_past_reset
            self.metrics["dexmachina_bbox"] = dex_bbox_val.expand(self.num_envs)

            # Flat sliding pool: append every per-env entry for every body.
            # `deque(maxlen=...)` drops old samples in FIFO order, so the AUC
            # is always over the most recent ~50 sim-steps × num_envs samples
            # per body. .tolist() pays one host transfer per step but avoids
            # holding a GPU buffer of inf-padded trajectories.
            from robotic_grounding.tasks.v2p.mdp.commands.dexmachina_eval import (  # noqa: PLC0415
                DEXMACHINA_THRESHOLDS_M,
                compute_dexmachina_auc,
            )

            # Gate the pool extension by `past_reset_phase` so VOC-warmup
            # frames (where timestep_counter is frozen and the policy isn't
            # free-running) don't contaminate the AUC. The standalone eval
            # path forces VOC=0 and has no warmup; the train-time metric must
            # match that semantic to be comparable to the paper.
            _past_mask = past_reset_phase.detach().to("cpu").bool().tolist()
            # (num_bodies, num_envs) host-side — one transfer.
            _add_host = dex_add_per_body_env.detach().to("cpu").transpose(0, 1).tolist()
            for b in range(self._dexmachina_num_bodies):
                self._dexmachina_pool[b].extend(
                    v for v, ok in zip(_add_host[b], _past_mask, strict=True) if ok
                )

            # Compute AUC across the full flat pool, per body, then mean.
            # Skip the first few steps where the pool is still tiny (an AUC
            # over <num_envs samples is dominated by reset-frame noise).
            min_pool_samples = max(self.num_envs, 1)
            if all(len(pool) >= min_pool_samples for pool in self._dexmachina_pool):
                _pools_t = torch.tensor(
                    [list(pool) for pool in self._dexmachina_pool],
                    dtype=torch.float32,
                    device=self.device,
                )  # (num_bodies, N)
                _auc = compute_dexmachina_auc(_pools_t, DEXMACHINA_THRESHOLDS_M)
                self.metrics["dexmachina_AUC"] = torch.full(
                    (self.num_envs,),
                    float(_auc.item()),
                    device=self.device,
                    dtype=torch.float32,
                )

        # Z-only tracking error — detects "object stays on table" failure mode
        z_error_per_env = torch.abs(
            self.object_position_e[..., 2] - self.object_body_position_command_e[..., 2]
        ).mean(
            dim=-1
        )  # (num_envs,), meters
        self.metrics["object_body_z_error"] = (
            (z_error_per_env * past_reset_phase.float()).sum() / n_past_reset
        ).expand(self.num_envs)

        # DISABLED: hand_tracking_good_frac W&B logging
        # _WRIST_GOOD_THRESHOLD = 0.05  # 5 cm
        # self.metrics["hand_tracking_good_frac"] = (
        #     (self.metrics["right_hand_wrist_position_error"] < _WRIST_GOOD_THRESHOLD)
        #     & (self.metrics["left_hand_wrist_position_error"] < _WRIST_GOOD_THRESHOLD)
        # ).float()

        # ------------------------------------------------------------------ #
        # Contact wrench support reward CV (W=200 steps)
        # Scale-invariant plateau indicator: CV = std/mean over the rolling
        # buffer. Low CV (≲0.07) means the reward has plateaued; high CV means
        # it is still actively changing. Used as a curriculum decay gate.
        # ------------------------------------------------------------------ #
        cws_step_val = (
            contact_wrench_support_reward_jit(
                right_cmd_active=self.right_wrench_cmd_active,
                right_cur_active=self.right_wrench_cur_active,
                left_cmd_active=self.left_wrench_cmd_active,
                left_cur_active=self.left_wrench_cur_active,
                right_cmd_active_per_body=self.right_wrench_cmd_active_per_body,
                left_cmd_active_per_body=self.left_wrench_cmd_active_per_body,
                right_cmd_supports=self.right_hand_contact_wrench_supports_command,
                right_cur_supports=self.right_hand_contact_wrench_supports,
                left_cmd_supports=self.left_hand_contact_wrench_supports_command,
                left_cur_supports=self.left_hand_contact_wrench_supports,
                tolerance=0.1,
                var=0.1,
            )
            .mean()
            .item()
        )
        self._cws_reward_step_buf.append(cws_step_val)
        if len(self._cws_reward_step_buf) >= 10:
            buf = torch.tensor(
                list(self._cws_reward_step_buf),
                dtype=torch.float32,
                device=self.device,
            )
            mean_abs = buf.mean().abs().clamp(min=1e-6)
            cv = buf.std() / mean_abs
            self.metrics["contact_wrench_support_reward_cv"] = cv.expand(self.num_envs)

        # ------------------------------------------------------------------ #
        # Contact wrench support reward CV (W=200 steps)
        # Scale-invariant plateau indicator: CV = std/mean over the rolling
        # buffer. Low CV (≲0.07) means the reward has plateaued; high CV means
        # it is still actively changing. Used as a curriculum decay gate.
        # ------------------------------------------------------------------ #
        cws_step_val = (
            contact_wrench_support_reward_jit(
                right_cmd_active=self.right_wrench_cmd_active,
                right_cur_active=self.right_wrench_cur_active,
                left_cmd_active=self.left_wrench_cmd_active,
                left_cur_active=self.left_wrench_cur_active,
                right_cmd_active_per_body=self.right_wrench_cmd_active_per_body,
                left_cmd_active_per_body=self.left_wrench_cmd_active_per_body,
                right_cmd_supports=self.right_hand_contact_wrench_supports_command,
                right_cur_supports=self.right_hand_contact_wrench_supports,
                left_cmd_supports=self.left_hand_contact_wrench_supports_command,
                left_cur_supports=self.left_hand_contact_wrench_supports,
                tolerance=0.1,
                var=0.1,
            )
            .mean()
            .item()
        )
        self._cws_reward_step_buf.append(cws_step_val)
        if len(self._cws_reward_step_buf) >= 10:
            buf = torch.tensor(
                list(self._cws_reward_step_buf),
                dtype=torch.float32,
                device=self.device,
            )
            mean_abs = buf.mean().abs().clamp(min=1e-6)
            cv = buf.std() / mean_abs
            self.metrics["contact_wrench_support_reward_cv"] = cv.expand(self.num_envs)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample the command."""
        n = len(env_ids)

        # Reset SPIDER metric running buffers for the envs that just ended, so
        # the new episode starts with a fresh running sim_pos_mean.
        if hasattr(self, "_spider_sim_pos_sum"):
            self._spider_sim_pos_sum[env_ids] = 0.0
            self._spider_step_count[env_ids] = 0.0

        # Reset to a random frame from the original retargeted motion data
        self.timestep_counter[env_ids] = torch.randint(
            low=0,
            high=self.retargeted_horizon - 1,
            size=(n,),
            device=self.device,
            dtype=self.timestep_counter.dtype,
        )

        if self.cfg.always_reset_to_first_frame:
            self.timestep_counter[env_ids] = 0
        elif (
            self.cfg.reset_to_first_frame_prob > 0.0
            and float(self.virtual_object_controller_scale_factor) < 0.1
        ):
            # Close the train/eval gap: once the VOC curriculum has nearly decayed
            # (scale < 0.1), randomly reset some envs to tc=0 with the configured
            # probability.  Eval always starts from tc=0; without this the policy
            # never sees trajectory-start episodes during training and can reward-hack
            # by learning to hold-in-place from mid-trajectory positions.
            first_frame_mask = (
                torch.rand(n, device=self.device) < self.cfg.reset_to_first_frame_prob
            )
            first_frame_local_ids = first_frame_mask.nonzero(as_tuple=False).squeeze(-1)
            if first_frame_local_ids.numel() > 0:
                env_ids_t = (
                    env_ids
                    if isinstance(env_ids, torch.Tensor)
                    else torch.tensor(env_ids, device=self.device)
                )
                self.timestep_counter[env_ids_t[first_frame_local_ids]] = 0

        # Cache the per-env timestep indices and env-origin offsets once.
        tc = self.timestep_counter[env_ids]
        env_origins_sel = self._env.scene.env_origins[env_ids]

        # Update the tracking length, reset curriculum + reset-step counters.
        self.tracking_lengths[env_ids] = (self.retargeted_horizon - tc).clamp(min=1)
        if not self._spider_eval_freeze_voc:
            self.virtual_object_controller_scale_factor_per_env[env_ids] = 1.0
        self.steps_since_last_reset[env_ids] = 0

        # ── JIT-compiled pure-tensor derivations (indexing, cat, rand, clamp) ──
        (
            object_pose,
            object_velocity,
            right_hand_wrist_position_e,
            right_hand_wrist_wxyz,
            left_hand_wrist_position_e,
            left_hand_wrist_wxyz,
            right_hand_wrist_pose,
            left_hand_wrist_pose,
            wrist_zero_velocity,
            right_hand_finger_joint_pos,
            left_hand_finger_joint_pos,
            finger_zero_velocity,
        ) = resample_compute_tensors_jit(
            tc=tc,
            env_origins_sel=env_origins_sel,
            retargeted_object_body_position=self.retargeted_object_body_position,
            retargeted_object_body_wxyz=self.retargeted_object_body_wxyz,
            retargeted_right_wrist_position=self.retargeted_right_wrist_position,
            retargeted_right_wrist_wxyz=self.retargeted_right_wrist_wxyz,
            retargeted_left_wrist_position=self.retargeted_left_wrist_position,
            retargeted_left_wrist_wxyz=self.retargeted_left_wrist_wxyz,
            retargeted_right_finger_joints=self.retargeted_right_finger_joints,
            retargeted_left_finger_joints=self.retargeted_left_finger_joints,
            right_soft_joint_pos_limits_sel=self.right_robot.data.soft_joint_pos_limits[
                env_ids, :
            ],
            left_soft_joint_pos_limits_sel=self.left_robot.data.soft_joint_pos_limits[
                env_ids, :
            ],
            reset_finger_openness=float(self.cfg.reset_finger_openness),
            n=n,
        )

        # Store reset wrist poses (env frame, no env_origins) for action terms.
        self.reset_right_wrist_position_e[env_ids] = right_hand_wrist_position_e
        self.reset_right_wrist_wxyz[env_ids] = right_hand_wrist_wxyz
        self.reset_left_wrist_position_e[env_ids] = left_hand_wrist_position_e
        self.reset_left_wrist_wxyz[env_ids] = left_hand_wrist_wxyz

        ##########################################################
        # Reset the object
        ##########################################################

        # Articulation joint state is kept in Python because it's conditional on
        # the object type and handles both (horizon,) and (horizon, num_joints)
        # source shapes with a dim-check the scripter can't cleanly express.
        has_object_articulation = self.retargeted_object_articulation.numel() > 0
        object_joint_pos: torch.Tensor | None = None
        if has_object_articulation:
            object_joint_pos = self.retargeted_object_articulation[tc]
            if object_joint_pos.dim() == 1:
                object_joint_pos = object_joint_pos.unsqueeze(-1)

        for object_idx, object in enumerate(self.objects):
            object.write_root_pose_to_sim(object_pose[:, object_idx], env_ids=env_ids)
            object.write_root_velocity_to_sim(
                object_velocity[:, object_idx], env_ids=env_ids
            )
            if isinstance(object, Articulation) and object_joint_pos is not None:
                object.write_joint_state_to_sim(
                    object_joint_pos,
                    torch.zeros_like(object_joint_pos),
                    env_ids=env_ids,
                )

        ##########################################################
        # Reset the robots (wrists + finger joints)
        ##########################################################

        self.right_robot.write_root_pose_to_sim(right_hand_wrist_pose, env_ids=env_ids)
        self.right_robot.write_root_velocity_to_sim(
            wrist_zero_velocity, env_ids=env_ids
        )
        self.left_robot.write_root_pose_to_sim(left_hand_wrist_pose, env_ids=env_ids)
        self.left_robot.write_root_velocity_to_sim(wrist_zero_velocity, env_ids=env_ids)

        self.right_robot.write_joint_state_to_sim(
            right_hand_finger_joint_pos, finger_zero_velocity, env_ids=env_ids
        )
        self.left_robot.write_joint_state_to_sim(
            left_hand_finger_joint_pos, finger_zero_velocity, env_ids=env_ids
        )

        ##########################################################
        # Refresh the simulation
        ##########################################################

        # Force a kinematic/data refresh after reset to synchronize states.
        self._env.sim.forward()
        self._env.scene.update(dt=self._env.physics_dt)

        # Reset invalidates all cached tensors.
        self._tensors_dirty = True

    def _update_command(self) -> None:
        """Update the command."""
        self.steps_since_last_reset += 1

        # Decay virtual object control scale factor toward curriculum scale
        if self._spider_eval_freeze_voc:
            # Eval passes that need VOC=0 throughout: skip the schedule writer
            # entirely so per_env stays at whatever the eval set it to.
            pass
        elif self.cfg.virtual_object_control_decay_mode == "linear":
            progress = self.steps_since_last_reset.float() / max(
                self.cfg.virtual_object_control_decay_steps, 1
            )
            self.virtual_object_controller_scale_factor_per_env[:] = (
                (
                    1.0
                    + (self.virtual_object_controller_scale_factor - 1.0)
                    * progress.clamp(max=1.0)
                )
                .clamp(min=0.0)
                .view(self.num_envs, 1)
            )
        elif self.cfg.virtual_object_control_decay_mode in (
            "step",
            "fixed_schedule",
            "custom_schedule",
        ):
            self.virtual_object_controller_scale_factor_per_env[
                self.steps_since_last_reset
                >= self.cfg.virtual_object_control_decay_steps
            ] = self.virtual_object_controller_scale_factor
        else:
            raise ValueError(
                f"Unknown virtual_object_control_decay_mode: {self.cfg.virtual_object_control_decay_mode!r}. "
                "Expected one of: 'linear', 'step', 'fixed_schedule', 'custom_schedule'."
            )

        # If still in the reset phase, don't step the timestep counter
        # Note: This will make each episode length vary, which might cause the episode rewards to be variant.
        not_in_reset_phase_env_ids = (
            self.steps_since_last_reset >= self.cfg.virtual_object_control_decay_steps
        )
        self.timestep_counter[not_in_reset_phase_env_ids] += 1

        # Mark cached tensors stale; next call to refresh_tensors() (first obs of
        # this step, or first reward/obs of step N+1) will repopulate them.
        self._tensors_dirty = True

    def _get_retargeted_contact_counts(self) -> None:
        """Get the number of retargeted contacts for each hand from retargeted contact data."""
        self.num_retargeted_contacts_left = (
            self.retargeted_left_object_contact_positions_o.shape[1]
        )
        self.num_retargeted_contacts_right = (
            self.retargeted_right_object_contact_positions_o.shape[1]
        )
        self.total_num_retargeted_contacts = (
            self.num_retargeted_contacts_left + self.num_retargeted_contacts_right
        )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set the debug visibility."""
        if debug_vis:
            if not hasattr(self, "right_hand_pose_visualizer"):
                self.right_hand_pose_visualizer = VisualizationMarkers(
                    self.cfg.right_hand_pose_visualizer_cfg
                )
            self.right_hand_pose_visualizer.set_visibility(True)
            if not hasattr(self, "left_hand_pose_visualizer"):
                self.left_hand_pose_visualizer = VisualizationMarkers(
                    self.cfg.left_hand_pose_visualizer_cfg
                )
            self.left_hand_pose_visualizer.set_visibility(True)

            # Command visualizers
            if not hasattr(self, "right_hand_goal_pose_visualizer"):
                self.right_hand_goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.right_hand_goal_pose_visualizer_cfg
                )
            self.right_hand_goal_pose_visualizer.set_visibility(True)
            if not hasattr(self, "left_hand_goal_pose_visualizer"):
                self.left_hand_goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.left_hand_goal_pose_visualizer_cfg
                )
            self.left_hand_goal_pose_visualizer.set_visibility(True)

        elif hasattr(self, "goal_pose_visualizer"):
            self.right_hand_pose_visualizer.set_visibility(False)
            self.left_hand_pose_visualizer.set_visibility(False)
            self.object_goal_pose_visualizer.set_visibility(False)
            self.right_hand_goal_pose_visualizer.set_visibility(False)
            self.left_hand_goal_pose_visualizer.set_visibility(False)
            for visualizer in getattr(self, "object_pose_visualizers", []):
                visualizer.set_visibility(False)
            for visualizer in getattr(self, "object_com_visualizers", []):
                visualizer.set_visibility(False)
            for visualizer in getattr(self, "contact_marker_visualizers", []):
                visualizer.set_visibility(False)
            for visualizer in getattr(self, "robot_contact_marker_visualizers", []):
                visualizer.set_visibility(False)

    def _set_contact_vis_impl(self, debug_vis: bool) -> None:
        """Set the contact visibility."""
        if debug_vis:
            # Current object pose visualizers
            self.object_pose_visualizers = []
            self.object_com_visualizers = []
            self.object_goal_pose_visualizers = []
            for body_idx in range(self.num_bodies):
                object_pose_visualizer_cfg = (
                    self.cfg.object_pose_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/object_marker_{body_idx}"
                    )
                )
                object_pose_visualizer = VisualizationMarkers(
                    object_pose_visualizer_cfg
                )
                object_pose_visualizer.set_visibility(True)
                self.object_pose_visualizers.append(object_pose_visualizer)

                object_com_visualizer_cfg = (
                    self.cfg.object_com_pose_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/object_com_marker_{body_idx}"
                    )
                )
                object_com_visualizer = VisualizationMarkers(object_com_visualizer_cfg)
                object_com_visualizer.set_visibility(True)
                self.object_com_visualizers.append(object_com_visualizer)

                object_goal_pose_visualizer_cfg = (
                    self.cfg.object_goal_pose_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/object_goal_marker_{body_idx}"
                    )
                )
                object_goal_pose_visualizer = VisualizationMarkers(
                    object_goal_pose_visualizer_cfg
                )
                object_goal_pose_visualizer.set_visibility(True)
                self.object_goal_pose_visualizers.append(object_goal_pose_visualizer)

            # Contact marker visualizers
            self.command_contact_marker_visualizers = []
            for i in range(self.total_num_retargeted_contacts):
                command_contact_vis_cfg = (
                    self.cfg.target_contact_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/TargetContact_{i}"
                    )
                )
                command_contact_marker_visualizer = VisualizationMarkers(
                    command_contact_vis_cfg
                )
                command_contact_marker_visualizer.set_visibility(True)
                self.command_contact_marker_visualizers.append(
                    command_contact_marker_visualizer
                )

            # Robot current contact marker visualizers
            self.robot_contact_marker_visualizers = []
            for i in range(self.total_num_robot_contacts * self.num_bodies):
                robot_contact_vis_cfg = self.cfg.current_contact_visualizer_cfg.replace(
                    prim_path=f"/Visuals/Command/CurrentContact_{i}"
                )
                robot_contact_marker_visualizer = VisualizationMarkers(
                    robot_contact_vis_cfg
                )
                robot_contact_marker_visualizer.set_visibility(True)
                self.robot_contact_marker_visualizers.append(
                    robot_contact_marker_visualizer
                )

            self.command_fingertip_marker_visualizers = []
            self.robot_fingertip_marker_visualizers = []

            for i in range(
                len(self.right_fingertip_body_ids) + len(self.left_fingertip_body_ids)
            ):
                command_fingertip_vis_cfg = (
                    self.cfg.target_fingertip_position_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/TargetFingertip_{i}"
                    )
                )
                command_fingertip_marker_visualizer = VisualizationMarkers(
                    command_fingertip_vis_cfg
                )
                command_fingertip_marker_visualizer.set_visibility(True)
                self.command_fingertip_marker_visualizers.append(
                    command_fingertip_marker_visualizer
                )

                robot_fingertip_vis_cfg = (
                    self.cfg.current_fingertip_position_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/CurrentFingertip_{i}"
                    )
                )
                robot_fingertip_marker_visualizer = VisualizationMarkers(
                    robot_fingertip_vis_cfg
                )
                robot_fingertip_marker_visualizer.set_visibility(True)
                self.robot_fingertip_marker_visualizers.append(
                    robot_fingertip_marker_visualizer
                )

            if omni_debug_draw is not None:
                self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    def _debug_vis_callback(self, event: Any) -> None:
        """Visualize the goal marker."""
        del event  # unused
        if not hasattr(self._env, "scene"):
            return
        if hasattr(self, "draw_interface"):
            self.draw_interface.clear_lines()

        # Current state visualizers
        for body_idx, object_pose_visualizer in enumerate(self.object_pose_visualizers):
            object_pose_visualizer.visualize(
                translations=self.object_position_w[:, body_idx],
                orientations=self.object_orientation_e[:, body_idx],
            )
        object_com_state_w = self.object_com_position_and_wxyz_w
        for body_idx, object_com_visualizer in enumerate(self.object_com_visualizers):
            object_com_visualizer.visualize(
                translations=object_com_state_w[:, body_idx, :3],
                orientations=object_com_state_w[:, body_idx, 3:7],
            )
        self.right_hand_pose_visualizer.visualize(
            translations=self.right_robot.data.root_link_pos_w,
            orientations=self.right_robot.data.root_link_quat_w,
        )
        self.left_hand_pose_visualizer.visualize(
            translations=self.left_robot.data.root_link_pos_w,
            orientations=self.left_robot.data.root_link_quat_w,
        )
        # Command visualizers
        for body_idx, object_goal_pose_visualizer in enumerate(
            self.object_goal_pose_visualizers
        ):
            object_goal_pose_visualizer.visualize(
                translations=self.object_body_position_command_e[:, body_idx]
                + self._env.scene.env_origins,
                orientations=self.object_body_wxyz_command_e[:, body_idx],
            )
        right_hand_wrist_pose_command_e = self.right_hand_wrist_pose_command_e
        self.right_hand_goal_pose_visualizer.visualize(
            translations=right_hand_wrist_pose_command_e[:, :3]
            + self._env.scene.env_origins,
            orientations=right_hand_wrist_pose_command_e[:, 3:],
        )
        left_hand_wrist_pose_command_e = self.left_hand_wrist_pose_command_e
        self.left_hand_goal_pose_visualizer.visualize(
            translations=left_hand_wrist_pose_command_e[:, :3]
            + self._env.scene.env_origins,
            orientations=left_hand_wrist_pose_command_e[:, 3:],
        )

        # Visualize target contact location on object
        left_hand_contact_command_positions_and_normals_e = (
            self.left_hand_object_contact_command_positions_and_normals_e
        )
        right_hand_contact_command_positions_and_normals_e = (
            self.right_hand_object_contact_command_positions_and_normals_e
        )
        for i in range(self.total_num_retargeted_contacts):
            # Select left or right hand contacts
            if i < self.num_retargeted_contacts_left:
                # Left contacts
                contact_command_positions_e = (
                    left_hand_contact_command_positions_and_normals_e[:, i, :3]
                )
                contact_command_normals_e = (
                    left_hand_contact_command_positions_and_normals_e[:, i, 3:]
                )
            else:
                # Right contacts
                contact_command_positions_e = (
                    right_hand_contact_command_positions_and_normals_e[
                        :, i - self.num_retargeted_contacts_left, :3
                    ]
                )
                contact_command_normals_e = (
                    right_hand_contact_command_positions_and_normals_e[
                        :, i - self.num_retargeted_contacts_left, 3:
                    ]
                )

            self.command_contact_marker_visualizers[i].visualize(
                translations=contact_command_positions_e + self._env.scene.env_origins,
                orientations=self.QUAT_UNIT_VEC,
            )
            if hasattr(self, "draw_interface"):
                self.draw_interface.draw_lines(
                    (
                        contact_command_positions_e + self._env.scene.env_origins
                    ).tolist(),
                    (
                        contact_command_positions_e
                        + self._env.scene.env_origins
                        + contact_command_normals_e * 0.05
                    ).tolist(),
                    [[0.0, 0.4, 1.0, 1.0]] * len(contact_command_positions_e),
                    [3.0] * len(contact_command_positions_e),
                )

        # Visualize current contacts
        right_hand_contact_positions_w = (
            self.right_hand_object_contact_positions_w.view(self.num_envs, -1, 3)
        )
        right_hand_contact_forces_w = self.right_hand_object_contact_forces_w[
            :, 0
        ].view(self.num_envs, -1, 3)
        for i in range(self.num_bodies * self.num_robot_contacts_right):
            self.robot_contact_marker_visualizers[i].visualize(
                translations=right_hand_contact_positions_w[:, i],
                orientations=self.QUAT_UNIT_VEC,
            )
            if hasattr(self, "draw_interface"):
                self.draw_interface.draw_lines(
                    (right_hand_contact_positions_w[:, i]).tolist(),
                    (
                        right_hand_contact_positions_w[:, i]
                        + right_hand_contact_forces_w[:, i] * 0.1
                    ).tolist(),
                    [[0.4, 0.0, 1.0, 1.0]] * len(right_hand_contact_positions_w),
                    [3.0] * len(right_hand_contact_positions_w),
                )

        left_hand_contact_positions_w = self.left_hand_object_contact_positions_w.view(
            self.num_envs, -1, 3
        )
        left_hand_contact_forces_w = self.left_hand_object_contact_forces_w[:, 0].view(
            self.num_envs, -1, 3
        )
        for i in range(self.num_bodies * self.num_robot_contacts_left):
            self.robot_contact_marker_visualizers[
                i + self.num_robot_contacts_right
            ].visualize(
                translations=left_hand_contact_positions_w[:, i],
                orientations=self.QUAT_UNIT_VEC,
            )
            if hasattr(self, "draw_interface"):
                self.draw_interface.draw_lines(
                    (left_hand_contact_positions_w[:, i]).tolist(),
                    (
                        left_hand_contact_positions_w[:, i]
                        + left_hand_contact_forces_w[:, i] * 0.1
                    ).tolist(),
                    [[0.4, 0.0, 1.0, 1.0]] * len(left_hand_contact_positions_w),
                    [3.0] * len(left_hand_contact_positions_w),
                )

        # Visualize fingertip positions
        for i in range(
            len(self.right_fingertip_body_ids) + len(self.left_fingertip_body_ids)
        ):
            if i < len(self.right_fingertip_body_ids):
                self.command_fingertip_marker_visualizers[i].visualize(
                    translations=self.right_hand_fingertip_position_command_e[:, i]
                    + self._env.scene.env_origins,
                    orientations=self.QUAT_UNIT_VEC,
                )
                self.robot_fingertip_marker_visualizers[i].visualize(
                    translations=self.right_hand_fingertip_position_e[:, i]
                    + self._env.scene.env_origins,
                    orientations=self.QUAT_UNIT_VEC,
                )
            else:
                self.command_fingertip_marker_visualizers[i].visualize(
                    translations=self.left_hand_fingertip_position_command_e[
                        :, i - len(self.right_fingertip_body_ids)
                    ]
                    + self._env.scene.env_origins,
                    orientations=self.QUAT_UNIT_VEC,
                )
                self.robot_fingertip_marker_visualizers[i].visualize(
                    translations=self.left_hand_fingertip_position_e[
                        :, i - len(self.right_fingertip_body_ids)
                    ]
                    + self._env.scene.env_origins,
                    orientations=self.QUAT_UNIT_VEC,
                )
