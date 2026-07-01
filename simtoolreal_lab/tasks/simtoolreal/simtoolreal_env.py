"""SimToolReal DirectRLEnv port (IIWA14 + left Sharpa, claw_hammer / swing_down).

Reward / observation / action logic ported from the Isaac Gym `SimToolReal` task
(isaacgymenvs/tasks/simtoolreal/env.py). Sim plumbing rewritten against Isaac Lab's
DirectRLEnv. Quaternions are Isaac Lab native (wxyz) throughout — trained from scratch,
so internal consistency (not byte-matching the original xyzw layout) is what matters.
"""

from __future__ import annotations

import json
import math
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
SCREW_KEYPOINT_HALF = (0.008, 0.008, 0.015)  # screw/nail keypoint half-extents (m); matches keypoint_utils.SCREW_HALF

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

        # Robot identifiers: cfg overrides (robot-swap tasks, e.g. Vega) else the module-level
        # IIWA14 + left-Sharpa constants. Defaults keep the original tasks byte-identical.
        joint_names = getattr(self.cfg, "joint_names", None) or JOINT_NAMES_ISAACGYM
        palm_body = getattr(self.cfg, "palm_body", None) or PALM_BODY
        fingertip_bodies = getattr(self.cfg, "fingertip_bodies", None) or FINGERTIP_BODIES

        # canonical(joint_names) -> sim joint index permutation
        self.canonical_dof_ids = [self.robot.joint_names.index(n) for n in joint_names]
        self.canonical_dof_ids_t = torch.tensor(self.canonical_dof_ids, device=self.device, dtype=torch.long)
        self.num_dofs = len(self.canonical_dof_ids)  # 29
        self.arm_slice = slice(0, 7)
        self.hand_slice = slice(7, self.num_dofs)

        # body indices for palm + fingertips
        self.palm_body_id = self.robot.body_names.index(palm_body)
        self.fingertip_body_ids = [self.robot.body_names.index(b) for b in fingertip_bodies]
        self.num_fingertips = len(self.fingertip_body_ids)

        # joint limits in canonical order, from the articulation
        lim = self.robot.root_physx_view.get_dof_limits().to(self.device)  # (N, J, 2)
        self.dof_lower = lim[..., 0][:, self.canonical_dof_ids_t]  # (N, 29)
        self.dof_upper = lim[..., 1][:, self.canonical_dof_ids_t]
        # ROBOT-order limits (for the reset-noise lerp toward a full-range random pose, matching the
        # original Isaac Gym reset; see _reset_idx).
        self._dof_lower_robot = lim[..., 0]  # (N, J) robot order
        self._dof_upper_robot = lim[..., 1]

        # control buffers (canonical order)
        z = lambda: torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.prev_targets = z()
        self.cur_targets = z()
        # the EXPERT's intended joint targets this step (clean copy, BEFORE any joint-teleport override of
        # cur_targets). The BC collector records the action from THIS so a joint teleport (which overrides
        # cur_targets to hold the teleported config) never corrupts the recorded expert action.
        self.expert_targets = z()
        self.actions = z()

        # constant offset tensors (cfg overrides for a robot-swap; else the IIWA+Sharpa defaults)
        palm_off = getattr(self.cfg, "palm_offset", None) or PALM_OFFSET
        ft_off = getattr(self.cfg, "fingertip_offset", None) or FINGERTIP_OFFSET
        self.palm_offset = torch.tensor(palm_off, device=self.device).repeat(self.num_envs, 1)
        self.fingertip_offset = torch.tensor(ft_off, device=self.device).repeat(
            self.num_envs, self.num_fingertips, 1
        )
        # keypoint offsets: corners * object_NORMALIZED_scale * base_size * keypoint_scale / 2.
        # The object scale here MUST be the NORMALIZED bbox (metric_bbox / object_base_size) so the
        # object_base_size factor cancels to a metric half-extent -- exactly like the compat branch
        # below and the original/authors' Isaac Sim port. For this single-object slice that normalized
        # scale IS cfg.pretrained_object_scale (e.g. claw_hammer (2.5,0.5625,0.375) = (0.10,0.0225,0.015)
        # / 0.04). [BUGFIX: this previously used the *metric* FIXED_SIZE, double-applying object_base_size
        # and shrinking the native (pretrained_compat=False / training) keypoints ~25x -> degenerate
        # keypoints_rel_palm/keypoints_rel_goal obs + ~25x-weak keypoint reward -> from-scratch stall.]
        corners = torch.tensor(KEYPOINT_CORNERS, device=self.device, dtype=torch.float)  # (4,3)
        obj_scale = torch.tensor(self.cfg.pretrained_object_scale, device=self.device, dtype=torch.float)
        half = obj_scale * self.cfg.object_base_size * self.cfg.keypoint_scale / 2.0
        self.keypoint_offsets = (corners * half).repeat(self.num_envs, 1, 1)  # (N,4,3)
        self.num_keypoints = corners.shape[0]
        # fixed-size keypoints used for the SUCCESS distance when demo_mode (original
        # fixedSizeKeypointReward): metric extents = fixedSize * keypoint_scale / 2 (NO base_size).
        half_fs = torch.tensor(FIXED_SIZE, device=self.device) * self.cfg.keypoint_scale / 2.0
        self.keypoint_offsets_fixed = (corners * half_fs).repeat(self.num_envs, 1, 1)  # (N,4,3)
        self.keypoints_max_dist_fixed = torch.zeros(self.num_envs, device=self.device)
        # screw/nail keypoints for the goal-free teacher actor: 4 corners at the DYNAMIC screw pose
        self.screw_kp_offsets = corners * torch.tensor(SCREW_KEYPOINT_HALF, device=self.device)  # (4,3)
        self.screw_body_idx = None  # lazily resolved from screw_asm body_names
        self.screw_keypoints = torch.zeros((self.num_envs, self.num_keypoints, 3), device=self.device)

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

        # object_scales obs = the object's NORMALIZED bbox scale (metric/object_base_size), matching the
        # original native obs (NOT the metric FIXED_SIZE). For this single-object slice that is
        # cfg.pretrained_object_scale. [BUGFIX: was the metric FIXED_SIZE -> wrong magnitude; rl_games
        # normalizes a constant away so it was benign for a single object, but now it is correct.]
        self.fixed_size_t = torch.tensor(self.cfg.pretrained_object_scale, device=self.device)
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
        # curriculum progress 0(loose)->1(tight); exposed for pluggable reward modules to anneal their
        # own tolerances in lockstep. Updated in _get_rewards as success_tol anneals.
        self.curriculum_progress = 0.0
        # optional pluggable reward augmentation (default off): module.augment_reward(env)
        import importlib
        self._reward_mod = (importlib.import_module(self.cfg.reward_module)
                            if getattr(self.cfg, "reward_module", None) else None)
        # tip-tolerance curriculum (used by a reward module that gates on env.tip_tol): start loose +
        # anneal to the tight target on its own success gate, or fix at the target if curriculum is off.
        # Independent of the keypoint-tolerance curriculum.
        if self._reward_mod is not None:
            self._tip_tol_target = float(getattr(self._reward_mod, "TIP_TOL_TARGET", 0.004))
            _tip_start = float(getattr(self._reward_mod, "TIP_TOL_START", 0.02))
            self.tip_tol = _tip_start if getattr(self.cfg, "tip_tol_curriculum", False) else self._tip_tol_target
        else:
            self.tip_tol = self._tip_tol_target = 0.0
        self._tip_last_update = 0

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
        if self.cfg.apply_compat_friction and (self.cfg.pretrained_compat or getattr(self.cfg, "force_grasp_friction", False)):
            self._apply_compat_frictions()

        # --- random force/torque perturbations on the object (original forceScale/torqueScale DR) ---
        # Per-env object mass (single-body tool) for mass-scaled kicks: force = randn*mass*scale, so the
        # induced acceleration is randn*scale regardless of mass (faithful to the Isaac Gym env, which
        # the previous port omitted — making light tools ~mass^-1 over-perturbed). Trigger probabilities
        # are sampled log-uniformly in perturb_prob_range, independently for force vs torque, resampled
        # each reset (random_force_prob / random_torque_prob).
        self.object_mass = self.object.root_physx_view.get_masses().to(self.device)[:, 0].view(self.num_envs, 1, 1)
        self.random_force_prob = self._sample_log_uniform(self.cfg.perturb_prob_range, self.num_envs)
        self.random_torque_prob = self._sample_log_uniform(self.cfg.perturb_prob_range, self.num_envs)
        # tool-displacement perturbation state: per-env cooldown (control steps until the next teleport
        # may fire) + a per-episode event counter (for diagnostics / verifying it actually fires).
        self._displace_cooldown = torch.zeros(self.num_envs, device=self.device)
        self.tool_displace_events = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # joint-displacement perturbation state (independent of the tool teleport): own cooldown + counter.
        self._joint_displace_cooldown = torch.zeros(self.num_envs, device=self.device)
        self.joint_displace_events = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # random-action-burst state: control steps remaining in the current burst (>0 == executing random
        # actions this step) + a per-episode counter of bursts STARTED (diagnostics).
        self._burst_steps_left = torch.zeros(self.num_envs, device=self.device)
        self.random_action_events = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # per-step flag: is THIS control step a random-action burst step? (set in _pre_physics_step). The BC
        # collector DROPS these steps from the dataset (the robot is executing random, not expert, actions).
        self._random_action_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # per-step flag: did a teleport fire THIS control step (set in _pre_physics_step). Recorded by
        # the BC collector so the trainer can MASK the action-chunk loss past a teleport (the post-
        # teleport actions are unpredictable from the pre-teleport observation).
        self._teleported_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # control steps since the last teleport (large = none recently) -> used to reject a task success
        # that lands within tool_displace_success_block_steps of a teleport (likely a teleport-to-goal fluke).
        self._steps_since_displace = torch.full((self.num_envs,), 1e9, device=self.device)

    def _sample_log_uniform(self, rng, n: int) -> torch.Tensor:
        """Sample n values log-uniformly in [rng[0], rng[1]] (both > 0); shape (n,) on device."""
        lo, hi = math.log(rng[0]), math.log(rng[1])
        return torch.exp(torch.rand(n, device=self.device) * (hi - lo) + lo)

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

    def _apply_table_dist(self):
        """Move the table's spawn FURTHER from the robot (cfg.table_dist m along -y; robot is at +y),
        WITHOUT moving the objects (they stay where the robot reaches). Capped so >=10cm of table stays
        under the objects at y~0 (else they fall off the receding near edge). Idempotent; call before
        the table is spawned in _setup_scene."""
        if getattr(self, "_table_dist_applied", False):
            return
        self._table_dist_applied = True
        td = float(getattr(self.cfg, "table_dist", 0.0))
        if td <= 0.0:
            return
        safe = max(0.0, self.cfg.table_cfg.spawn.size[1] / 2.0 - 0.10)
        td_c = min(td, safe)
        if td_c < td:
            print(f"[env] table_dist {td:.3f} capped to {td_c:.3f}m (keep the objects on the table)", flush=True)
        x, y, z = self.cfg.table_cfg.init_state.pos
        self.cfg.table_cfg.init_state.pos = (x, y - td_c, z)
        print(f"[env] work-table moved {td_c:.3f}m further from the robot (table_dist={td:.3f})", flush=True)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._apply_table_dist()
        self.table = RigidObject(self.cfg.table_cfg)
        self._build_per_env_camera()  # created before clone so it replicates per env
        self._build_wrist_camera()    # ditto -- mounted on the palm link
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        self._register_per_env_camera()
        self._register_wrist_camera()
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ optional per-env camera
    def _build_per_env_camera(self):
        """Create one TiledCamera view per sub-env (if cfg.per_env_camera). Call BEFORE
        clone_environments so the camera prim is replicated to every env."""
        self.per_env_cam = None
        if not getattr(self.cfg, "per_env_camera", False):
            return
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        from isaaclab.utils.math import quat_from_matrix, create_rotation_matrix_from_view
        # Bake the look-at into the OffsetCfg (applied per-env relative to each env prim, so it is
        # correct for every env and survives TiledCamera's per-step re-application of the offset --
        # unlike set_world_poses_from_view, which the camera overwrites each update). The view matrix
        # is OpenGL convention (-Z fwd, +Y up); env prims only translate, so the world look-at
        # rotation == the per-env-local offset rotation.
        eye = torch.tensor([self.cfg.cam_eye], device=self.device, dtype=torch.float32)
        tgt = torch.tensor([self.cfg.cam_lookat], device=self.device, dtype=torch.float32)
        rot = quat_from_matrix(create_rotation_matrix_from_view(eye, tgt, up_axis="Z", device=str(self.device)))[0]
        cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/PerEnvCam",
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(float(v) for v in self.cfg.cam_eye),
                                            rot=tuple(float(v) for v in rot.tolist()), convention="opengl"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=float(getattr(self.cfg, "cam_focal", 24.0)),
                                             clipping_range=(0.005, float(getattr(self.cfg, "cam_z_far", 50.0)))),
            width=int(self.cfg.cam_width), height=int(self.cfg.cam_height),
        )
        self.per_env_cam = TiledCamera(cfg)

    def _register_per_env_camera(self):
        if getattr(self, "per_env_cam", None) is not None:
            self.scene.sensors["per_env_cam"] = self.per_env_cam

    def _build_wrist_camera(self):
        """Create one TiledCamera mounted on PALM_BODY (iiwa14_link_7) per sub-env, looking at the
        palm/fingers (if cfg.wrist_camera). The eye/lookat are LINK-LOCAL: the camera is a child of
        the link prim, so the look-at rotation in the link frame keeps it pointed at the palm as the
        hand moves. Built before clone_environments so it replicates to every env."""
        self.wrist_cam = None
        if not getattr(self.cfg, "wrist_camera", False):
            return
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        from isaaclab.utils.math import quat_from_matrix, create_rotation_matrix_from_view
        eye = torch.tensor([self.cfg.wrist_cam_eye], device=self.device, dtype=torch.float32)
        tgt = torch.tensor([self.cfg.wrist_cam_lookat], device=self.device, dtype=torch.float32)
        rot = quat_from_matrix(create_rotation_matrix_from_view(
            eye, tgt, up_axis=getattr(self.cfg, "wrist_cam_up", "Y"), device=str(self.device)))[0]
        palm_body = getattr(self.cfg, "palm_body", None) or PALM_BODY
        cfg = TiledCameraCfg(
            prim_path=f"/World/envs/env_.*/Robot/{palm_body}/WristCam",
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(float(v) for v in self.cfg.wrist_cam_eye),
                                            rot=tuple(float(v) for v in rot.tolist()), convention="opengl"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=float(getattr(self.cfg, "wrist_cam_focal", 12.0)),
                                             clipping_range=(0.002, 5.0)),
            width=int(self.cfg.wrist_cam_width), height=int(self.cfg.wrist_cam_height),
        )
        self.wrist_cam = TiledCamera(cfg)

    def _register_wrist_camera(self):
        if getattr(self, "wrist_cam", None) is not None:
            self.scene.sensors["wrist_cam"] = self.wrist_cam

    # ------------------------------------------------------------------ action
    def _compute_targets(self, actions: torch.Tensor, prev_targets: torch.Tensor) -> torch.Tensor:
        """PURE action->joint-target map (no side effects): given a [-1,1] action and the previous
        targets, return the new joint position targets. Hand (7:29): scale to limits + EMA. Arm (0:7):
        integrate at control_dt + clamp + EMA. Used by `_pre_physics_step` AND by DAgger to compute the
        expert's would-be target at a learner-visited state without disturbing the rollout."""
        actions = actions.clamp(-1.0, 1.0)
        cur = prev_targets.clone()
        h = self.hand_slice
        scaled = 0.5 * (actions[:, h] + 1.0) * (self.dof_upper[:, h] - self.dof_lower[:, h]) + self.dof_lower[:, h]
        cur[:, h] = self.cfg.hand_moving_average * scaled + (1.0 - self.cfg.hand_moving_average) * prev_targets[:, h]
        cur[:, h] = torch.clamp(cur[:, h], self.dof_lower[:, h], self.dof_upper[:, h])

        a = self.arm_slice
        arm = prev_targets[:, a] + self.cfg.dof_speed_scale * self.cfg.control_dt * actions[:, a]
        arm = torch.clamp(arm, self.dof_lower[:, a], self.dof_upper[:, a])
        cur[:, a] = self.cfg.arm_moving_average * arm + (1.0 - self.cfg.arm_moving_average) * prev_targets[:, a]
        return cur

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # Compute joint position targets ONCE per control step (held across the `decimation`
        # physics substeps). Matches the original controlFrequencyInv=1 @ 60 Hz (dt=1/60, substeps=2):
        # the action EMA/integration must advance once per control step, not once per substep.
        prev0 = self.prev_targets                        # previous step's EXECUTED target (the action history)
        self.cur_targets = self._compute_targets(self.actions, prev0)
        self.prev_targets = self.cur_targets
        self.expert_targets = self.cur_targets.clone()   # clean expert target (perturbations won't corrupt this)

        # Intermittent random force/torque kicks on the object once it's lifted (original forceScale/
        # torqueScale DR, forceOnlyWhenLifted/torqueOnlyWhenLifted). Enabled by DR (training) OR the
        # standalone force_perturbation flag (eval / BC collection). Each kick lasts one control step
        # (set every step, zeros on non-kick steps == the original forceDecay=0 one-shot push). Force
        # and torque trigger INDEPENDENTLY at the per-env log-uniform rates; both are mass-scaled.
        if (self.cfg.force_perturbation or self.cfg.domain_randomization) and self.cfg.perturb_force_scale > 0.0:
            lifted = self.lifted_object.view(self.num_envs, 1, 1).float()
            fkick = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).view(self.num_envs, 1, 1).float()
            tkick = (torch.rand(self.num_envs, device=self.device) < self.random_torque_prob).view(self.num_envs, 1, 1).float()
            forces = torch.randn((self.num_envs, 1, 3), device=self.device) * self.object_mass * self.cfg.perturb_force_scale * fkick * lifted
            torques = torch.randn((self.num_envs, 1, 3), device=self.device) * self.object_mass * self.cfg.perturb_torque_scale * tkick * lifted
            self.object.set_external_force_and_torque(forces, torques)

        # Tool-displacement perturbation: with low probability (once lifted, respecting a per-env
        # cooldown), TELEPORT the tool by a random delta pose + zero its velocity -> it slips/falls out
        # of the grasp. Simulates failure cases (tool drops, grasp slips); the expert then has to recover
        # (re-grasp / re-position), and only the recovered episodes survive success-filtering. Applied
        # here (pre-physics) so the displaced tool then falls/interacts during this step's substeps.
        # --- teleport perturbations (tool and/or joint): shared per-step bookkeeping ---
        # _teleported_this_step (the chunk-loss-masking flag) and _steps_since_displace (the within-1s
        # success block) are driven by ANY teleport, so a joint teleport masks/blocks just like a tool one.
        if self.cfg.tool_displacement or self.cfg.joint_displacement or self.cfg.random_action:
            self._teleported_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._steps_since_displace = self._steps_since_displace + 1.0   # time since the last teleport/burst

        if self.cfg.tool_displacement:
            self._displace_cooldown = torch.clamp(self._displace_cooldown - 1.0, min=0.0)
            # gate: post-grasp only (lifted) by default, or any time if tool_displace_pregrasp
            gate = torch.ones_like(self.lifted_object) if self.cfg.tool_displace_pregrasp else self.lifted_object
            fire = ((torch.rand(self.num_envs, device=self.device) < self.cfg.tool_displace_prob)
                    & gate & (self._displace_cooldown <= 0))
            ids = torch.nonzero(fire).flatten()
            if ids.numel() > 0:
                m = ids.numel()
                c = self.cfg
                pos = self.object.data.root_pos_w[ids].clone()           # world
                quat = self.object.data.root_quat_w[ids].clone()         # wxyz
                # position: magnitude HALF-NORMAL over [min,max] = min + |N(0,σ)| (σ=(max-min)/2), clamped to
                # max -> mode at min, decaying -> MORE small slips than big drops. Random direction.
                pmag = (c.tool_displace_pos_min + (c.tool_displace_pos - c.tool_displace_pos_min) * 0.5
                        * torch.randn(m, device=self.device).abs()).clamp(max=c.tool_displace_pos)
                pdir = torch.randn(m, 3, device=self.device)
                pdir = pdir / (pdir.norm(dim=-1, keepdim=True) + 1e-9)
                dpos = pdir * pmag.unsqueeze(-1)
                # rotation: angle HALF-NORMAL over [min,max] (more small than big) about a random axis
                rmag = (c.tool_displace_rot_min + (c.tool_displace_rot - c.tool_displace_rot_min) * 0.5
                        * torch.randn(m, device=self.device).abs()).clamp(max=c.tool_displace_rot)
                ax = torch.randn(m, 3, device=self.device)
                ax = ax / (ax.norm(dim=-1, keepdim=True) + 1e-9)
                dq = quat_from_angle_axis(rmag, ax)                      # wxyz
                new_pose = torch.cat([pos + dpos, quat_mul(dq, quat)], dim=-1)
                self.object.write_root_pose_to_sim(new_pose, ids)
                self.object.write_root_velocity_to_sim(torch.zeros((m, 6), device=self.device), ids)
                self._displace_cooldown[ids] = float(self.cfg.tool_displace_cooldown)
                self.tool_displace_events[ids] += 1
            self._teleported_this_step |= fire
            self._steps_since_displace = torch.where(fire, torch.zeros_like(self._steps_since_displace), self._steps_since_displace)

        # Joint-displacement: TELEPORT the robot's 29 arm+hand joint positions by a random per-joint delta
        # (+ zero joint velocity), with INDEPENDENT sampling (own prob/cooldown/scale). cur/prev TARGETS are
        # left unchanged so the recorded action stays the clean expert command and the PD + expert recover.
        if self.cfg.joint_displacement:
            c = self.cfg
            self._joint_displace_cooldown = torch.clamp(self._joint_displace_cooldown - 1.0, min=0.0)
            jfire = ((torch.rand(self.num_envs, device=self.device) < c.joint_displace_prob)
                     & (self._joint_displace_cooldown <= 0))
            ids = torch.nonzero(jfire).flatten()
            if ids.numel() > 0:
                m = ids.numel()
                # SEPARATE arm vs hand magnitudes (arm moves the EE ~10x more per rad -> smaller delta). Each
                # is a per-event HALF-NORMAL scale over its [min,max] (= min+|N(0,(max-min)/2)| clamped -> more
                # small jolts than big); per-joint delta = randn * scale (Gaussian per joint).
                s_arm = (c.joint_displace_arm_scale_min + (c.joint_displace_arm_scale - c.joint_displace_arm_scale_min)
                         * 0.5 * torch.randn(m, 1, device=self.device).abs()).clamp(max=c.joint_displace_arm_scale)
                s_hand = (c.joint_displace_hand_scale_min + (c.joint_displace_hand_scale - c.joint_displace_hand_scale_min)
                          * 0.5 * torch.randn(m, 1, device=self.device).abs()).clamp(max=c.joint_displace_hand_scale)
                delta = torch.randn(m, self.num_dofs, device=self.device)   # canonical: arm 0:7, hand 7:29
                delta[:, self.arm_slice] *= s_arm
                delta[:, self.hand_slice] *= s_hand
                cur_jp = self.robot.data.joint_pos[ids][:, self.canonical_dof_ids_t]      # (m,29) canonical
                new_jp = torch.clamp(cur_jp + delta, self.dof_lower[ids], self.dof_upper[ids])
                self.robot.write_joint_state_to_sim(new_jp, torch.zeros_like(new_jp),
                                                    joint_ids=self.canonical_dof_ids_t, env_ids=ids)
                # HOLD the teleported config: set the PD targets to new_jp so the controller does NOT
                # snap the joints back to the stale pre-teleport targets (the "bounce-back"). The closed-
                # loop expert then re-predicts from the teleported joint_pos + object state next step and
                # recovers. expert_targets is left clean -> the recorded action is the expert's command.
                self.cur_targets[ids] = new_jp
                self.prev_targets[ids] = new_jp
                self._joint_displace_cooldown[ids] = float(c.joint_displace_cooldown)
                self.joint_displace_events[ids] += 1
            self._teleported_this_step |= jfire
            self._steps_since_displace = torch.where(jfire, torch.zeros_like(self._steps_since_displace), self._steps_since_displace)

        # Random-action burst: with random_action_prob/step, START a burst of N control steps (N =
        # round(|N(0, random_action_steps_std^2)|), so MORE short bursts than long; NO cooldown -- a new
        # burst may re-arm the step after one ends) during which the robot EXECUTES a random 29-dim delta
        # action (each step: a ~ N(0, random_action_std^2) mapped through the SAME action->target path),
        # driving it off-distribution (flail / drop the tool); the closed-loop expert then recovers. The
        # burst steps are flagged in `_random_action_this_step` -> the BC collector DROPS them from the
        # dataset (only the EXPERT-controlled steps, incl. the post-burst recovery, are recorded; the
        # collector also marks the gap so action chunks don't span it). Fires any time (no lifted gate);
        # independent per-env trigger; no overlapping bursts (re-arm only at burst end).
        if self.cfg.random_action:
            start = ((self._burst_steps_left <= 0)
                     & (torch.rand(self.num_envs, device=self.device) < self.cfg.random_action_prob))
            if start.any():
                n = (torch.randn(self.num_envs, device=self.device).abs()
                     * self.cfg.random_action_steps_std).round().clamp(min=1.0)
                self._burst_steps_left = torch.where(start, n, self._burst_steps_left)
                self.random_action_events += start.long()              # count bursts STARTED (diagnostics)
            bursting = self._burst_steps_left > 0
            self._random_action_this_step = bursting.clone()           # burst steps -> EXCLUDED from the dataset
            ids = torch.nonzero(bursting).flatten()
            if ids.numel() > 0:
                rand_act = torch.randn(self.num_envs, self.num_dofs, device=self.device) * self.cfg.random_action_std
                rand_tgt = self._compute_targets(rand_act, prev0)      # random action via the normal map, from the action history
                self.cur_targets[ids] = rand_tgt[ids]                  # EXECUTE random (alias -> prev_targets[ids])
                self._burst_steps_left[ids] -= 1.0
            self._steps_since_displace = torch.where(bursting, torch.zeros_like(self._steps_since_displace), self._steps_since_displace)

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
        if self.cfg.actor_infer_goal_from_screw:
            self._update_screw_keypoints()
        # distances
        self.curr_fingertip_distances = torch.norm(
            self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1
        )  # (N,F)
        self.keypoints_max_dist = torch.norm(self.object_keypoints - self.goal_keypoints, dim=-1).max(dim=-1).values
        # FIXED-SIZE keypoints for success (original fixedSizeKeypointReward) so the success tolerance is
        # object-scale-invariant. Used at eval (demo_mode) AND, via fixed_size_success, in training
        # (matches the original, which used fixed-size in both).
        if self.cfg.demo_mode or self.cfg.fixed_size_success:
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

    def _update_screw_keypoints(self):
        """4 keypoints at the ACTUAL (dynamic) screw/nail pose, env-local. Mirrors keypoint_utils:
        prefer the physical screw_asm body (rotates/sinks), then the kinematic screw, then the nominal
        pose; if there is no screw (base claw_hammer) fall back to the goal keypoints."""
        eo = self.scene.env_origins
        asm = getattr(self, "screw_asm", None)
        if asm is not None:
            if self.screw_body_idx is None:
                names = list(asm.body_names)
                cand = [i for i, n in enumerate(names) if "screw" in n.lower() or "nail" in n.lower()]
                self.screw_body_idx = cand[-1] if cand else len(names) - 1
            spos = asm.data.body_pos_w[:, self.screw_body_idx] - eo
            squat = asm.data.body_quat_w[:, self.screw_body_idx]
        elif getattr(self, "screw", None) is not None:
            spos = self.screw.data.root_pos_w - eo
            squat = self.screw.data.root_quat_w
        elif getattr(self, "screw_nom_pos", None) is not None:
            spos = self.screw_nom_pos - eo
            squat = self.screw_nom_quat
        else:
            self.screw_keypoints = self.goal_keypoints  # no screw -> degenerate to the goal box
            return
        off = self.screw_kp_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)            # (N,4,3)
        q = squat.unsqueeze(1).expand(-1, self.num_keypoints, -1)
        self.screw_keypoints = spos.unsqueeze(1) + quat_apply(q, off)                     # (N,4,3)

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
                # actor: keypoints_rel_goal (default) OR dynamic SCREW keypoints rel-palm (goal-free teacher
                # -> the actor must infer the goal from the screw layout; it sees only what the BC student can)
                (((self.screw_keypoints - self.palm_center.unsqueeze(1)) if self.cfg.actor_infer_goal_from_screw
                  else (self.object_keypoints - self.goal_keypoints)).reshape(self.num_envs, -1)),  # 12
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
            terms = [obs]                                             # 140 (actor obs; goal-free if flag)
            if self.cfg.actor_infer_goal_from_screw:                 # critic KEEPS the goal the actor dropped
                terms.append((self.object_keypoints - self.goal_keypoints).reshape(self.num_envs, -1))  # +12
            terms += [
                self.object_linvel,                              # 3
                self.object_angvel,                              # 3
                self.lifted_object.float().unsqueeze(-1),        # 1
                self.keypoints_max_dist.unsqueeze(-1),           # 1
                self.closest_keypoint_max_dist.unsqueeze(-1),    # 1
                self.closest_fingertip_dist,                     # 5
                self.successes.unsqueeze(-1),                    # 1
            ]
            state = torch.cat(terms, dim=-1).clamp(-ko, ko)
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
        success_dist = self.keypoints_max_dist_fixed if (c.demo_mode or c.fixed_size_success) else self.keypoints_max_dist
        near_goal = success_dist <= kp_tol
        # pluggable reward augmentation (e.g. the cross-slot tip gate + proximity bonus); off if no module
        if self._reward_mod is not None:
            reward_add, success_gate = self._reward_mod.augment_reward(self)
            near_goal = near_goal & success_gate
        else:
            reward_add = 0.0
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
            + reward_add            # pluggable reward-module augmentation (0.0 if none)
        )

        # advance goal on success
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._resample_goals(goal_env_ids, base="goal")
            self.near_goal_steps[goal_env_ids] = 0
            self.closest_keypoint_max_dist[goal_env_ids] = -1.0  # re-init to new goal's distance next frame

        # curriculum step counter (drives BOTH the keypoint and the tip-tolerance curricula)
        self._curr_step += 1
        gate = self.prev_episode_successes.mean().item() >= c.curriculum_success_threshold
        # keypoint-tolerance curriculum (matches original utils.tolerance_curriculum): once `interval`
        # steps elapse since the last update, tighten iff mean successes/episode >= threshold. If the
        # gate fails, the timer is NOT reset (re-checks each step), as in the original.
        if c.use_tolerance_curriculum:
            if self._curr_step - self._last_curriculum_update >= c.tolerance_curriculum_interval and gate:
                self.success_tol = min(float(c.success_tolerance), self.success_tol * c.tolerance_curriculum_increment)
                self.success_tol = max(c.target_success_tolerance, self.success_tol)
                self._last_curriculum_update = self._curr_step
        # tip-tolerance curriculum: same gate/interval/increment, but anneals env.tip_tol (read by the
        # reward module) toward the tight target -- INDEPENDENT of the (often fixed) keypoint tolerance.
        if c.tip_tol_curriculum and self._reward_mod is not None:
            if self._curr_step - self._tip_last_update >= c.tolerance_curriculum_interval and gate:
                self.tip_tol = max(self._tip_tol_target, self.tip_tol * c.tolerance_curriculum_increment)
                self._tip_last_update = self._curr_step
        # progress 0(loose)->1(tight) for any reward module still keying off it
        _span = float(c.success_tolerance) - float(c.target_success_tolerance)
        self.curriculum_progress = 0.0 if _span <= 0 else max(0.0, min(1.0, (float(c.success_tolerance) - self.success_tol) / _span))

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

        # robot: default joint pos randomized per the ORIGINAL Isaac Gym reset -- lerp `noise_coef` of
        # the way toward a uniform-random FULL-RANGE joint pose (per-joint, asymmetric):
        #   dof = lerp(default, U[lower, upper], noise_coef)  == default + noise_coef*(U[lower,upper]-default)
        # This matches env.py reset_idx (and the authors' reset_utils lerp); the previous symmetric
        # `default += coef*U(-1,1)` was much NARROWER on wide arm joints (+/-0.1 rad vs up to ~0.6 rad),
        # starving the start-state diversity the original relies on for grasp exploration. Uses the SAME
        # single uniform draw (shape = dof_pos), so deploy/eval (noise_coef=0 -> lerp gives default) is
        # bit-identical. Non-controlled joints (coef 0) stay at default.
        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        arm_ids = self.canonical_dof_ids_t[self.arm_slice]
        hand_ids = self.canonical_dof_ids_t[self.hand_slice]
        u01 = sample_uniform(0.0, 1.0, dof_pos.shape, self.device)
        sampled = self._dof_lower_robot[env_ids] + (self._dof_upper_robot[env_ids] - self._dof_lower_robot[env_ids]) * u01
        coef = torch.zeros_like(dof_pos)
        coef[:, arm_ids] = self.cfg.reset_dof_pos_noise_arm
        coef[:, hand_ids] = self.cfg.reset_dof_pos_noise_fingers
        dof_pos = torch.lerp(dof_pos, sampled, coef)
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
        # resample per-env perturbation trigger probabilities (log-uniform, like random_force_prob)
        self.random_force_prob[env_ids] = self._sample_log_uniform(self.cfg.perturb_prob_range, n)
        self.random_torque_prob[env_ids] = self._sample_log_uniform(self.cfg.perturb_prob_range, n)
        self._displace_cooldown[env_ids] = 0.0
        self._steps_since_displace[env_ids] = 1e9
        self.tool_displace_events[env_ids] = 0
        self._joint_displace_cooldown[env_ids] = 0.0
        self.joint_displace_events[env_ids] = 0
        self._burst_steps_left[env_ids] = 0.0
        self.random_action_events[env_ids] = 0

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
