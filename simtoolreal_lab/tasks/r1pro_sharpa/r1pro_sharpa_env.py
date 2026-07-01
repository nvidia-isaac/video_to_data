"""R1 Pro + Sharpa env for GR00T N1.7 (REAL_R1_PRO_SHARPA) inference.

Stage 2a (this file): fixed-base robot + table + object + the 3 GR00T cameras, with the GR00T
observation assembled in `get_gr00t_obs()` (3 cams + left/right wrist EEF + left/right 22-DOF hand
joints + task prompt). The action application uses differential IK on each arm to track the GR00T
relative-EEF action, plus absolute Sharpa hand-joint targets.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    combine_frame_transforms, matrix_from_quat, quat_from_matrix, subtract_frame_transforms,
)

from .r1pro_sharpa_env_cfg import (
    EEF_BODIES, LEFT_ARM_JOINTS, LEFT_HAND_JOINTS, RIGHT_ARM_JOINTS, RIGHT_HAND_JOINTS,
    R1ProSharpaEnvCfg,
)


def _quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """wxyz quat -> 6D rotation (first two columns of R, flattened). (N,4)->(N,6)."""
    R = matrix_from_quat(quat)            # (N,3,3)
    return R[..., :, :2].reshape(*quat.shape[:-1], 6)


def _rot6d_to_quat(r6: torch.Tensor) -> torch.Tensor:
    """6D rotation -> wxyz quat (Gram-Schmidt). (N,6)->(N,4)."""
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / (a2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)
    return quat_from_matrix(R)


class R1ProSharpaEnv(DirectRLEnv):
    cfg: R1ProSharpaEnvCfg

    def __init__(self, cfg: R1ProSharpaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        jn = self.robot.joint_names
        self._larm_ids = [jn.index(n) for n in LEFT_ARM_JOINTS]
        self._rarm_ids = [jn.index(n) for n in RIGHT_ARM_JOINTS]
        self._lhand_ids = [jn.index(n) for n in LEFT_HAND_JOINTS]
        self._rhand_ids = [jn.index(n) for n in RIGHT_HAND_JOINTS]
        self._larm_t = torch.tensor(self._larm_ids, device=self.device)
        self._rarm_t = torch.tensor(self._rarm_ids, device=self.device)
        self._lhand_t = torch.tensor(self._lhand_ids, device=self.device)
        self._rhand_t = torch.tensor(self._rhand_ids, device=self.device)
        self._leef = self.robot.body_names.index(EEF_BODIES["left"])
        self._reef = self.robot.body_names.index(EEF_BODIES["right"])
        # one differential-IK controller per arm (pose target, applied to the 7 arm joints)
        ikc = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self._lik = DifferentialIKController(ikc, num_envs=self.num_envs, device=self.device)
        self._rik = DifferentialIKController(ikc, num_envs=self.num_envs, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._hold = True   # stage 2a: hold the init pose; set False once GR00T drives it

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.ego_cam = Camera(self.cfg.ego_cam_cfg)
        self.left_wrist_cam = Camera(self.cfg.left_wrist_cam_cfg)
        self.right_wrist_cam = Camera(self.cfg.right_wrist_cam_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["ego_cam"] = self.ego_cam
        self.scene.sensors["left_wrist_cam"] = self.left_wrist_cam
        self.scene.sensors["right_wrist_cam"] = self.right_wrist_cam
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ action
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        if self._hold:
            # stage 2a: hold arms+hands at their default pose (no GR00T action yet)
            self.robot.set_joint_position_target(
                self.robot.data.default_joint_pos[:, self._larm_t], joint_ids=self._larm_ids)
            self.robot.set_joint_position_target(
                self.robot.data.default_joint_pos[:, self._rarm_t], joint_ids=self._rarm_ids)
            return
        a = self.actions
        leef_d, reef_d = a[:, 0:9], a[:, 9:18]
        lhand, rhand = a[:, 18:40], a[:, 40:62]
        self._apply_eef(self._lik, self._leef, self._larm_ids, self._larm_t, leef_d)
        self._apply_eef(self._rik, self._reef, self._rarm_ids, self._rarm_t, reef_d)
        self.robot.set_joint_position_target(lhand, joint_ids=self._lhand_ids)
        self.robot.set_joint_position_target(rhand, joint_ids=self._rhand_ids)

    def _apply_eef(self, ik, eef_idx, arm_ids, arm_t, delta9):
        """Apply a RELATIVE EEF delta (xyz + rot6d) to one arm via differential IK."""
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        ee_pos_w = self.robot.data.body_pos_w[:, eef_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, eef_idx]
        # current EEF in the robot base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos, root_quat, ee_pos_w, ee_quat_w)
        tgt_pos_b = ee_pos_b + delta9[:, 0:3]
        tgt_quat_b = _rot6d_to_quat(delta9[:, 3:9])          # delta orientation in base frame (abs target)
        ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))
        jac = self.robot.root_physx_view.get_jacobians()[:, eef_idx - 1, :, arm_t]
        q = self.robot.data.joint_pos[:, arm_t]
        q_des = ik.compute(ee_pos_b, ee_quat_b, jac, q)
        self.robot.set_joint_position_target(q_des, joint_ids=arm_ids)

    # ------------------------------------------------------------------ GR00T observation
    def get_gr00t_obs(self) -> dict:
        """Assemble the GR00T REAL_R1_PRO_SHARPA observation (nested dict, batch=num_envs)."""
        def rgb(cam):
            return cam.data.output["rgb"][..., :3].to(torch.uint8)   # (N,H,W,3)
        root_pos, root_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w

        def eef9(idx):
            pos_b, quat_b = subtract_frame_transforms(root_pos, root_quat,
                self.robot.data.body_pos_w[:, idx], self.robot.data.body_quat_w[:, idx])
            return torch.cat([pos_b, _quat_to_rot6d(quat_b)], dim=-1)   # (N,9)

        N = self.num_envs
        return {
            "video": {
                "ego_view_res320x240_freq20": rgb(self.ego_cam).unsqueeze(1),          # (N,1,H,W,3)
                "left_wrist_view_res320x240_freq20": rgb(self.left_wrist_cam).unsqueeze(1),
                "right_wrist_view_res320x240_freq20": rgb(self.right_wrist_cam).unsqueeze(1),
            },
            "state": {
                "left_wrist_eef": eef9(self._leef).unsqueeze(1),                        # (N,1,9)
                "right_wrist_eef": eef9(self._reef).unsqueeze(1),
                "left_hand_joints": self.robot.data.joint_pos[:, self._lhand_t].unsqueeze(1),   # (N,1,22)
                "right_hand_joints": self.robot.data.joint_pos[:, self._rhand_t].unsqueeze(1),
            },
            "annotation.human.coarse_action": [[self.cfg.task_prompt]] * N,
        }

    # ------------------------------------------------------------------ DirectRLEnv hooks
    def _get_observations(self) -> dict:
        # flat zero placeholder for the DirectRLEnv interface; GR00T inference uses get_gr00t_obs()
        return {"policy": torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        eo = self.scene.env_origins[env_ids]
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        for obj in (self.table, self.object):
            st = obj.data.default_root_state[env_ids].clone()
            st[:, 0:3] = st[:, 0:3] + eo
            obj.write_root_pose_to_sim(st[:, :7], env_ids)
            obj.write_root_velocity_to_sim(st[:, 7:], env_ids)
        self._lik.reset(env_ids)
        self._rik.reset(env_ids)
