"""SimToolReal DirectRLEnv port (IIWA14 + left Sharpa, claw_hammer / swing_down).

Reward / observation / action logic ported from the Isaac Gym `SimToolReal` task
(isaacgymenvs/tasks/simtoolreal/env.py). Sim plumbing rewritten against Isaac Lab's
DirectRLEnv. Quaternions are Isaac Lab native (wxyz) throughout — trained from scratch,
so internal consistency (not byte-matching the original xyzw layout) is what matters.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform

from .simtoolreal_env_cfg import (
    FINGERTIP_BODIES,
    JOINT_NAMES_ISAACGYM,
    PALM_BODY,
    SimToolRealEnvCfg,
)

# Palm/fingertip local offsets and keypoint corners (observation_action_utils_sharpa.py).
PALM_OFFSET = (-0.0, -0.02, 0.16)
FINGERTIP_OFFSET = (0.02, 0.002, 0.0)
KEYPOINT_CORNERS = [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
FIXED_SIZE = (0.141, 0.03025, 0.0271)  # fixedSize keypoint extents

# Exact joint-limit constants (JOINT_NAMES_ISAACGYM / canonical order) from
# observation_action_utils_sharpa.py:Q_{LOWER,UPPER}_LIMITS_np. Used VERBATIM by the
# pretrained Isaac Gym policy to unscale joint_pos (obs) and scale hand-action targets.
# In pretrained_compat mode the env uses these instead of the USD-derived limits so the
# obs/action math matches the original/deployment bit-for-bit.
Q_LOWER_LIMITS = (
    -2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543,  # arm 0:7
    -0.1745, -0.3491, -0.5236, -0.3491, 0.0,                        # thumb 7:12
    -0.1745, -0.0349, 0.0, 0.0,                                     # index 12:16
    -0.1745, -0.0349, 0.0, 0.0,                                     # middle 16:20
    -0.1745, -0.0349, 0.0, 0.0,                                     # ring 20:24
    0.0, -0.1745, -0.0349, 0.0, 0.0,                                # pinky 24:29
)
Q_UPPER_LIMITS = (
    2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543,         # arm 0:7
    1.9199, 0.1309, 1.3963, 0.3491, 1.7453,                         # thumb 7:12
    1.5708, 0.0349, 1.7453, 1.3963,                                 # index 12:16
    1.5708, 0.0349, 1.7453, 1.3963,                                 # middle 16:20
    1.5708, 0.0349, 1.7453, 1.3963,                                 # ring 20:24
    0.2618, 1.5708, 0.0349, 1.7453, 1.3963,                         # pinky 24:29
)


class SimToolRealEnv(DirectRLEnv):
    cfg: SimToolRealEnvCfg

    def __init__(self, cfg: SimToolRealEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # canonical(JOINT_NAMES_ISAACGYM) -> sim joint index permutation
        self.canonical_dof_ids = [self.robot.joint_names.index(n) for n in JOINT_NAMES_ISAACGYM]
        self.canonical_dof_ids_t = torch.tensor(self.canonical_dof_ids, device=self.device, dtype=torch.long)
        self.num_dofs = len(self.canonical_dof_ids)  # 29
        self.arm_slice = slice(0, 7)
        self.hand_slice = slice(7, self.num_dofs)

        # body indices for palm + fingertips
        self.palm_body_id = self.robot.body_names.index(PALM_BODY)
        self.fingertip_body_ids = [self.robot.body_names.index(b) for b in FINGERTIP_BODIES]
        self.num_fingertips = len(self.fingertip_body_ids)

        # joint limits in canonical order, from the articulation
        lim = self.robot.root_physx_view.get_dof_limits().to(self.device)  # (N, J, 2)
        self.dof_lower = lim[..., 0][:, self.canonical_dof_ids_t]  # (N, 29)
        self.dof_upper = lim[..., 1][:, self.canonical_dof_ids_t]

        # control buffers (canonical order)
        z = lambda: torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.prev_targets = z()
        self.cur_targets = z()
        self.actions = z()

        # constant offset tensors
        self.palm_offset = torch.tensor(PALM_OFFSET, device=self.device).repeat(self.num_envs, 1)
        self.fingertip_offset = torch.tensor(FINGERTIP_OFFSET, device=self.device).repeat(
            self.num_envs, self.num_fingertips, 1
        )
        # keypoint offsets: corners * base_size * keypoint_scale / 2 * fixed_size
        corners = torch.tensor(KEYPOINT_CORNERS, device=self.device, dtype=torch.float)  # (4,3)
        half = torch.tensor(FIXED_SIZE, device=self.device) * self.cfg.object_base_size * self.cfg.keypoint_scale / 2.0
        self.keypoint_offsets = (corners * half).repeat(self.num_envs, 1, 1)  # (N,4,3)
        self.num_keypoints = corners.shape[0]
        # fixed-size keypoints used for the SUCCESS distance when demo_mode (original
        # fixedSizeKeypointReward): metric extents = fixedSize * keypoint_scale / 2 (NO base_size).
        half_fs = torch.tensor(FIXED_SIZE, device=self.device) * self.cfg.keypoint_scale / 2.0
        self.keypoint_offsets_fixed = (corners * half_fs).repeat(self.num_envs, 1, 1)  # (N,4,3)
        self.keypoints_max_dist_fixed = torch.zeros(self.num_envs, device=self.device)

        # reward / success state buffers
        f = lambda: torch.zeros(self.num_envs, device=self.device)
        # -1 sentinel: lazily initialized to the actual distance on the first frame of each
        # episode/goal in _compute_intermediate_values (mirrors the Isaac Gym env). Using a
        # large finite value instead would create a spurious first-step delta-reward spike.
        self.closest_fingertip_dist = torch.full((self.num_envs, self.num_fingertips), -1.0, device=self.device)
        self.furthest_hand_dist = torch.full((self.num_envs,), -1.0, device=self.device)
        self.closest_keypoint_max_dist = torch.full((self.num_envs,), -1.0, device=self.device)
        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.successes = f()
        self.near_goal_steps = f()
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # goal pose (env-local frame): pos (N,3) + quat wxyz (N,4)
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_quat[:, 0] = 1.0

        # object init pose (env-local) captured at reset, for lifting reference
        self.object_init_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # object scales obs (fixed-size for this single-object slice)
        self.fixed_size_t = torch.tensor(FIXED_SIZE, device=self.device)
        self.object_scale_noise = torch.ones((self.num_envs, 1), device=self.device)  # per-env keypoint/scale multiplier (DR)
        self.object_scales = self.fixed_size_t.repeat(self.num_envs, 1)

        # --- pretrained Isaac Gym checkpoint compatibility (zero-shot deploy) -----------------
        # The original policy expects: (a) object_scales = the dextoolbench "scale given to
        # policy" (NOT metric) = (0.10,0.0225,0.015)/0.04 for claw_hammer, with keypoint
        # half-extent = 0.03*scale; (b) palm_rot/object_rot in XYZW; (c) joint_pos unscale and
        # hand-action scale using the exact Q_{LOWER,UPPER} constants. Toggle obs/action to that
        # convention here so obs[:140] matches the checkpoint's running_mean_std bit-for-bit.
        if self.cfg.pretrained_compat:
            os_ = torch.tensor(self.cfg.pretrained_object_scale, device=self.device, dtype=torch.float)
            self.fixed_size_t = os_  # so _reset_idx keeps the right scale (object_scales = fixed_size * noise)
            self.object_scales = os_.repeat(self.num_envs, 1)
            corners = torch.tensor(KEYPOINT_CORNERS, device=self.device, dtype=torch.float)  # (4,3)
            half = os_ * self.cfg.object_base_size * self.cfg.keypoint_scale / 2.0  # 0.03 * scale
            self.keypoint_offsets = (corners * half).repeat(self.num_envs, 1, 1)  # (N,4,3)
            ql = torch.tensor(Q_LOWER_LIMITS, device=self.device, dtype=torch.float)
            qu = torch.tensor(Q_UPPER_LIMITS, device=self.device, dtype=torch.float)
            self.dof_lower = ql.repeat(self.num_envs, 1)  # (N,29) canonical order
            self.dof_upper = qu.repeat(self.num_envs, 1)

        # curriculum state: annealed tolerance + per-env last-episode successes (gate metric, like original)
        self.success_tol = float(self.cfg.success_tolerance)
        self._curr_step = 0
        self._last_curriculum_update = 0
        self.prev_episode_successes = torch.zeros(self.num_envs, device=self.device)

        # optional fixed-goal trajectory (eval)
        self.trajectory_goals = None
        self.demo_start_pose = None  # (7,) xyz_xyzw world-frame, set in demo_mode
        if self.cfg.use_fixed_goal_trajectory:
            with open(self.cfg.trajectory_path) as fp:
                traj = json.load(fp)
            self.trajectory_goals = torch.tensor(traj["goals"], device=self.device)  # (T,7) world xyz_xyzw
            # demo_mode: fixed object init from the trajectory's recorded start_pose (+ z_offset),
            # exactly like eval_interactive.py (traj_data["start_pose"][2] += Z_OFFSET).
            if self.cfg.demo_mode and "start_pose" in traj:
                sp = list(traj["start_pose"])  # xyz_xyzw, world frame
                sp[2] += self.cfg.demo_z_offset
                self.demo_start_pose = torch.tensor(sp, device=self.device)  # (7,)

        # friction (original modifyAssetFrictions) — needed for reliable grasping with the
        # pretrained policy (high fingertip grip). Applied once after the sim views exist.
        if self.cfg.pretrained_compat and self.cfg.apply_compat_friction:
            self._apply_compat_frictions()

    def _apply_compat_frictions(self):
        """Set per-shape friction to match the original: the 5 *_DP fingertip links get high
        friction (grip), arm/object/table get low friction (Isaac Gym modifyAssetFrictions)."""
        idx = torch.arange(self.num_envs)  # CPU indices for set_material_properties

        def set_uniform(asset, fric):
            m = asset.root_physx_view.get_material_properties().clone()
            m[..., 0:2] = fric  # static, dynamic friction
            asset.root_physx_view.set_material_properties(m, idx)

        set_uniform(self.object, self.cfg.object_friction)
        set_uniform(self.table, self.cfg.table_friction)

        # robot: all links -> robot_friction, then override the DP fingertip links -> high friction
        m = self.robot.root_physx_view.get_material_properties().clone()
        m[..., 0:2] = self.cfg.robot_friction
        nspb = [
            self.robot._physics_sim_view.create_rigid_body_view(p).max_shapes
            for p in self.robot.root_physx_view.link_paths[0]
        ]  # shapes per body, physx order (== body_names order)
        for b in self.fingertip_body_ids:
            s = sum(nspb[:b])
            m[:, s : s + nspb[b], 0:2] = self.cfg.fingertip_friction
        self.robot.root_physx_view.set_material_properties(m, idx)
        print(
            f"[friction] total robot shapes={self.robot.root_physx_view.max_shapes} "
            f"DP fingertip bodies={self.fingertip_body_ids} -> {self.cfg.fingertip_friction}; "
            f"arm={self.cfg.robot_friction} object={self.cfg.object_friction} table={self.cfg.table_friction}"
        )

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ action
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # Compute joint position targets ONCE per control step (held across the `decimation`
        # physics substeps). Matches the original controlFrequencyInv=1 @ 60 Hz (dt=1/60, substeps=2):
        # the action EMA/integration must advance once per control step, not once per substep.
        # Hand (7:29): scale action to limits, EMA. Arm (0:7): integrate at control_dt, clamp, EMA.
        cur = self.prev_targets.clone()
        h = self.hand_slice
        scaled = 0.5 * (self.actions[:, h] + 1.0) * (self.dof_upper[:, h] - self.dof_lower[:, h]) + self.dof_lower[:, h]
        cur[:, h] = self.cfg.hand_moving_average * scaled + (1.0 - self.cfg.hand_moving_average) * self.prev_targets[:, h]
        cur[:, h] = torch.clamp(cur[:, h], self.dof_lower[:, h], self.dof_upper[:, h])

        a = self.arm_slice
        arm = self.prev_targets[:, a] + self.cfg.dof_speed_scale * self.cfg.control_dt * self.actions[:, a]
        arm = torch.clamp(arm, self.dof_lower[:, a], self.dof_upper[:, a])
        cur[:, a] = self.cfg.arm_moving_average * arm + (1.0 - self.cfg.arm_moving_average) * self.prev_targets[:, a]

        self.cur_targets = cur
        self.prev_targets = cur

        # DR: intermittent random force/torque kicks on the object once it's lifted (forceOnlyWhenLifted)
        if self.cfg.domain_randomization and self.cfg.perturb_force_scale > 0.0:
            kick = (torch.rand(self.num_envs, device=self.device) < self.cfg.perturb_prob) & self.lifted_object
            mask = kick.float().view(self.num_envs, 1, 1)
            forces = torch.randn((self.num_envs, 1, 3), device=self.device) * self.cfg.perturb_force_scale * mask
            torques = torch.randn((self.num_envs, 1, 3), device=self.device) * self.cfg.perturb_torque_scale * mask
            self.object.set_external_force_and_torque(forces, torques)

    def _apply_action(self) -> None:
        # target computed once per control step in _pre_physics_step; re-applied each physics substep
        self.robot.set_joint_position_target(self.cur_targets, joint_ids=self.canonical_dof_ids)

    # ------------------------------------------------------------------ obs
    def _compute_intermediate_values(self):
        eo = self.scene.env_origins
        # joints (canonical order)
        self.joint_pos = self.robot.data.joint_pos[:, self.canonical_dof_ids_t]
        self.joint_vel = self.robot.data.joint_vel[:, self.canonical_dof_ids_t]
        # palm
        palm_pos = self.robot.data.body_pos_w[:, self.palm_body_id] - eo
        self.palm_quat = self.robot.data.body_quat_w[:, self.palm_body_id]
        self.palm_center = palm_pos + quat_apply(self.palm_quat, self.palm_offset)
        # fingertips (N, F, 3)
        ft_pos = self.robot.data.body_pos_w[:, self.fingertip_body_ids] - eo.unsqueeze(1)
        ft_quat = self.robot.data.body_quat_w[:, self.fingertip_body_ids]
        self.fingertip_pos = ft_pos + quat_apply(ft_quat, self.fingertip_offset)
        # object
        self.object_pos = self.object.data.root_pos_w - eo
        self.object_quat = self.object.data.root_quat_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w
        # keypoints
        self.object_keypoints = self._keypoints(self.object_pos, self.object_quat)
        self.goal_keypoints = self._keypoints(self.goal_pos, self.goal_quat)
        # distances
        self.curr_fingertip_distances = torch.norm(
            self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1
        )  # (N,F)
        self.keypoints_max_dist = torch.norm(self.object_keypoints - self.goal_keypoints, dim=-1).max(dim=-1).values
        # demo_mode success uses the FIXED-SIZE keypoints (original fixedSizeKeypointReward) so the
        # success tolerance is object-scale-invariant (matches eval_interactive.py).
        if self.cfg.demo_mode:
            of = self.keypoint_offsets_fixed
            qo = self.object_quat.unsqueeze(1).expand(-1, self.num_keypoints, -1)
            qg = self.goal_quat.unsqueeze(1).expand(-1, self.num_keypoints, -1)
            obj_fixed = self.object_pos.unsqueeze(1) + quat_apply(qo, of)
            goal_fixed = self.goal_pos.unsqueeze(1) + quat_apply(qg, of)
            self.keypoints_max_dist_fixed = torch.norm(obj_fixed - goal_fixed, dim=-1).max(dim=-1).values

        # lazy-init the "closest/furthest" trackers on the first frame of an episode/goal
        # (sentinel < 0 -> set to current). Removes the first-step delta-reward artifact.
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0, self.curr_fingertip_distances, self.closest_fingertip_dist
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0, self.curr_fingertip_distances[:, 0], self.furthest_hand_dist
        )
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0, self.keypoints_max_dist, self.closest_keypoint_max_dist
        )

    def _keypoints(self, pos, quat):
        # pos (N,3), quat (N,4 wxyz) -> (N,K,3). Offsets scaled by per-env DR multiplier.
        offsets = self.keypoint_offsets * self.object_scale_noise.unsqueeze(-1)  # (N,1,1) broadcast
        kp = pos.unsqueeze(1) + quat_apply(quat.unsqueeze(1).expand(-1, self.num_keypoints, -1), offsets)
        return kp

    def _get_observations(self) -> dict:
        ko = self.cfg.clamp_abs_observations
        unscaled_q = (2.0 * self.joint_pos - self.dof_upper - self.dof_lower) / (self.dof_upper - self.dof_lower)
        # The pretrained Isaac Gym policy expects palm_rot/object_rot in XYZW (scipy/Isaac Gym
        # convention); Isaac Lab is WXYZ. Reorder [w,x,y,z] -> [x,y,z,w] for compat.
        if self.cfg.pretrained_compat:
            palm_rot = self.palm_quat[:, [1, 2, 3, 0]]
            object_rot = self.object_quat[:, [1, 2, 3, 0]]
        else:
            palm_rot, object_rot = self.palm_quat, self.object_quat
        obs = torch.cat(
            (
                unscaled_q,                                                  # 29
                self.joint_vel,                                             # 29
                self.prev_targets,                                          # 29
                self.palm_center,                                           # 3
                palm_rot,                                                   # 4
                object_rot,                                                 # 4
                (self.fingertip_pos - self.palm_center.unsqueeze(1)).reshape(self.num_envs, -1),  # 15
                (self.object_keypoints - self.palm_center.unsqueeze(1)).reshape(self.num_envs, -1),  # 12
                (self.object_keypoints - self.goal_keypoints).reshape(self.num_envs, -1),  # 12
                self.object_scales,                                         # 3
            ),
            dim=-1,
        )
        # DR: Gaussian observation noise (disabled at eval via domain_randomization=False)
        if self.cfg.domain_randomization and self.cfg.obs_noise_std > 0.0:
            obs = obs + torch.randn_like(obs) * self.cfg.obs_noise_std
        obs = obs.clamp(-ko, ko)
        # SAPG eval: append exploit exploration-coefficient at index 140 (observation_space
        # stays 140 -> player sets coef_id_idx=140; norm_obs normalizes only [:140]).
        if self.cfg.eval_append_expl_coef:
            coef = torch.full((self.num_envs, 1), float(self.cfg.expl_exploit_coef), device=self.device)
            obs = torch.cat([obs, coef], dim=-1)
        out = {"policy": obs}
        # asymmetric critic (privileged state) for SAPG central_value
        if self.cfg.state_space and self.cfg.state_space > 0:
            state = torch.cat(
                (
                    obs,                                              # 140
                    self.object_linvel,                              # 3
                    self.object_angvel,                              # 3
                    self.lifted_object.float().unsqueeze(-1),        # 1
                    self.keypoints_max_dist.unsqueeze(-1),           # 1
                    self.closest_keypoint_max_dist.unsqueeze(-1),    # 1
                    self.closest_fingertip_dist,                     # 5
                    self.successes.unsqueeze(-1),                    # 1
                ),
                dim=-1,
            ).clamp(-ko, ko)
            out["critic"] = state
        return out

    # ------------------------------------------------------------------ reward
    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        # lifting
        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_pos[:, 2]
        lifting_rew = torch.clip(z_lift, 0, 0.5)
        lifted = (z_lift > c.lifting_bonus_threshold) | self.lifted_object
        just_lifted = lifted & ~self.lifted_object
        lift_bonus = c.lifting_bonus * just_lifted.float()
        lifting_rew = lifting_rew * (~lifted).float()
        self.lifted_object = lifted

        # fingertip approach (only before lifted)
        ft_delta = torch.clip(self.closest_fingertip_dist - self.curr_fingertip_distances, 0, 10)
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, self.curr_fingertip_distances)
        fingertip_rew = ft_delta.sum(dim=-1) * (~lifted).float()

        # keypoint reward (only after lifted)
        kp_delta = torch.clip(self.closest_keypoint_max_dist - self.keypoints_max_dist, 0, 100)
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist)
        keypoint_rew = kp_delta * lifted.float()

        # success bookkeeping (keypoint within the curriculum-annealed tolerance)
        kp_tol = self.success_tol * c.keypoint_scale
        # demo_mode: success measured on fixed-size keypoints (fixedSizeKeypointReward); else object-scale.
        success_dist = self.keypoints_max_dist_fixed if c.demo_mode else self.keypoints_max_dist
        near_goal = success_dist <= kp_tol
        if c.force_consecutive_near_goal_steps:
            self.near_goal_steps = (self.near_goal_steps + near_goal.float()) * near_goal.float()
        else:
            self.near_goal_steps = self.near_goal_steps + near_goal.float()
        is_success = self.near_goal_steps >= c.success_steps
        self.successes += is_success.float()
        self.reset_goal_buf = is_success

        # action penalties
        kuka_pen = -self.joint_vel[:, self.arm_slice].abs().sum(-1) * c.kuka_actions_penalty_scale
        hand_pen = -self.joint_vel[:, self.hand_slice].abs().sum(-1) * c.hand_actions_penalty_scale

        if c.force_consecutive_near_goal_steps:
            bonus = is_success.float() * c.reach_goal_bonus
        else:
            bonus = near_goal.float() * (c.reach_goal_bonus / c.success_steps)

        reward = (
            fingertip_rew * c.distance_delta_rew_scale
            + lifting_rew * c.lifting_rew_scale
            + lift_bonus
            + keypoint_rew * c.keypoint_rew_scale
            + kuka_pen
            + hand_pen
            + bonus
        )

        # advance goal on success
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._resample_goals(goal_env_ids, base="goal")
            self.near_goal_steps[goal_env_ids] = 0
            self.closest_keypoint_max_dist[goal_env_ids] = -1.0  # re-init to new goal's distance next frame

        # tolerance curriculum (matches original utils.tolerance_curriculum): once `interval` control
        # steps have elapsed since the last update, tighten the tolerance iff mean successes/episode
        # >= threshold (3.0). If the gate fails, the timer is NOT reset (re-checks each step), as in the original.
        if c.use_tolerance_curriculum:
            self._curr_step += 1
            if self._curr_step - self._last_curriculum_update >= c.tolerance_curriculum_interval:
                if self.prev_episode_successes.mean().item() >= c.curriculum_success_threshold:
                    self.success_tol = min(float(c.success_tolerance), self.success_tol * c.tolerance_curriculum_increment)
                    self.success_tol = max(c.target_success_tolerance, self.success_tol)
                    self._last_curriculum_update = self._curr_step

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["successes"] = self.successes.mean()
        self.extras["log"]["success_tolerance"] = self.success_tol
        return reward * c.reward_shaper_scale

    # ------------------------------------------------------------------ dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        object_fell = self.object_pos[:, 2] < 0.1
        hand_far = self.curr_fingertip_distances.max(dim=-1).values > 1.5
        terminated = object_fell | hand_far
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_successes > 0:
            time_out = time_out | (self.successes >= self.cfg.max_consecutive_successes)
        return terminated, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        eo = self.scene.env_origins[env_ids]

        # object: default state + position noise (env-local -> world)
        obj_state = self.object.data.default_root_state[env_ids].clone()
        noise = sample_uniform(-1.0, 1.0, (n, 3), self.device)
        noise = noise * torch.tensor(
            [self.cfg.reset_position_noise_x, self.cfg.reset_position_noise_y, self.cfg.reset_position_noise_z],
            device=self.device,
        )
        # default_root_state pos is env-LOCAL; add env_origins to get world (as IsaacLab templates do).
        obj_state[:, 0:3] = obj_state[:, 0:3] + noise + eo
        # demo_mode: deterministic fixed init from the trajectory start_pose (no noise/yaw),
        # exactly like eval_interactive.py (useFixedInitObjectPose + objectStartPose).
        if self.cfg.demo_mode and self.demo_start_pose is not None:
            obj_state[:, 0:3] = self.demo_start_pose[0:3].unsqueeze(0) + eo  # world-frame pos
            obj_state[:, 3:7] = self.demo_start_pose[[6, 3, 4, 5]].unsqueeze(0)  # xyzw -> wxyz
        # DR: random object yaw at reset
        elif self.cfg.domain_randomization and self.cfg.randomize_object_yaw:
            yaw = sample_uniform(-torch.pi, torch.pi, (n,), self.device)
            obj_state[:, 3] = torch.cos(yaw / 2)  # quat wxyz: w
            obj_state[:, 4] = 0.0
            obj_state[:, 5] = 0.0
            obj_state[:, 6] = torch.sin(yaw / 2)  # z
        obj_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(obj_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)
        self.object_init_pos[env_ids] = obj_state[:, 0:3] - eo  # back to env-local for lifting ref

        # DR: per-env object scale multiplier (affects keypoints + object_scales obs)
        if self.cfg.domain_randomization:
            lo_s, hi_s = self.cfg.object_scale_noise_range
            self.object_scale_noise[env_ids] = lo_s + (hi_s - lo_s) * torch.rand((n, 1), device=self.device)
        self.object_scales[env_ids] = self.fixed_size_t * self.object_scale_noise[env_ids]

        # robot: default joint pos + noise
        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_noise = sample_uniform(-1.0, 1.0, dof_pos.shape, self.device)
        arm_ids = self.canonical_dof_ids_t[self.arm_slice]
        hand_ids = self.canonical_dof_ids_t[self.hand_slice]
        dof_pos[:, arm_ids] += self.cfg.reset_dof_pos_noise_arm * dof_noise[:, arm_ids]
        dof_pos[:, hand_ids] += self.cfg.reset_dof_pos_noise_fingers * dof_noise[:, hand_ids]
        dof_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(dof_pos, env_ids=env_ids)

        # DR: per-env actuator gain jitter (+/- pd_gain_noise on the default stiffness/damping)
        if self.cfg.domain_randomization and self.cfg.randomize_pd_gains:
            stiff = self.robot.data.default_joint_stiffness[env_ids].clone()
            damp = self.robot.data.default_joint_damping[env_ids].clone()
            stiff = stiff * (1.0 + self.cfg.pd_gain_noise * sample_uniform(-1.0, 1.0, stiff.shape, self.device))
            damp = damp * (1.0 + self.cfg.pd_gain_noise * sample_uniform(-1.0, 1.0, damp.shape, self.device))
            self.robot.write_joint_stiffness_to_sim(stiff, env_ids=env_ids)
            self.robot.write_joint_damping_to_sim(damp, env_ids=env_ids)

        # reset control buffers (canonical order)
        self.prev_targets[env_ids] = dof_pos[:, self.canonical_dof_ids_t]
        self.cur_targets[env_ids] = dof_pos[:, self.canonical_dof_ids_t]

        # capture last-episode successes for the tolerance-curriculum gate (before zeroing)
        if n > 0:
            self.prev_episode_successes[env_ids] = self.successes[env_ids]

        # reset reward/success state
        self.closest_fingertip_dist[env_ids] = -1.0
        self.furthest_hand_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.lifted_object[env_ids] = False
        self.successes[env_ids] = 0.0
        self.near_goal_steps[env_ids] = 0.0

        # initial goal: delta from object init pose
        self._compute_intermediate_values()
        self._resample_goals(env_ids, base="object_init")

    def _resample_goals(self, env_ids, base: str):
        n = len(env_ids)
        if self.cfg.use_fixed_goal_trajectory and self.trajectory_goals is not None:
            idx = (self.successes[env_ids].long()) % self.trajectory_goals.shape[0]
            g = self.trajectory_goals[idx]
            self.goal_pos[env_ids] = g[:, 0:3]
            self.goal_quat[env_ids] = g[:, [6, 3, 4, 5]]  # xyzw -> wxyz
            return
        lo = torch.tensor(self.cfg.target_volume_min, device=self.device)
        hi = torch.tensor(self.cfg.target_volume_max, device=self.device)
        d = self.cfg.delta_goal_distance
        delta_rad = torch.pi * self.cfg.delta_rotation_degrees / 180.0
        if base == "object_init":
            # FIRST goal: uniform-random absolute pose in the elevated target volume + uniform-random
            # orientation, then floor z to require lifting (original is_first_goal branch + _clip_goal_z).
            self.goal_pos[env_ids] = lo + (hi - lo) * torch.rand((n, 3), device=self.device)
            min_z = self.object_init_pos[env_ids, 2] - 0.05 + self.cfg.lifting_bonus_threshold
            self.goal_pos[env_ids, 2] = torch.maximum(self.goal_pos[env_ids, 2], min_z)
            self.goal_quat[env_ids] = torch.nn.functional.normalize(
                torch.randn((n, 4), device=self.device), dim=-1
            )  # uniform random orientation
            return
        # subsequent goals (on success): per-axis uniform delta in [-d, d] clamped to the volume, plus a
        # body-frame rotation of random axis & signed magnitude in [-delta_rad, delta_rad]
        # (matches original _sample_delta_goal / sample_delta_quat_xyzw).
        base_pos, base_quat = self.goal_pos[env_ids], self.goal_quat[env_ids]
        new_pos = torch.clamp(base_pos + sample_uniform(-d, d, (n, 3), self.device), lo, hi)
        angle = sample_uniform(-delta_rad, delta_rad, (n,), self.device)
        axis = torch.nn.functional.normalize(sample_uniform(-1.0, 1.0, (n, 3), self.device), dim=-1)
        dq = quat_from_angle_axis(angle, axis)
        self.goal_pos[env_ids] = new_pos
        self.goal_quat[env_ids] = quat_mul(base_quat, dq)  # base ∘ delta (body-frame), matches original
