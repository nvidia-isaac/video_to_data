"""GR1-Screwdriver SCENE/SCAFFOLD env (DirectRLEnv).

Stands up a fixed-base Fourier GR1T2 + the screwdriver-task objects on a table. The action drives
the right arm (7) + right hand (11) joint position targets; obs = those joints' pos/vel + the
screwdriver/screw/right-hand poses. Reward is a placeholder (0). This is a starting point -- there
is no trained controller (the pretrained SimToolReal policy is for the IIWA+Sharpa, not the GR-1).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_mul  # noqa: F401  (kept for downstream task work)

from .gr1_screwdriver_env_cfg import CONTROLLED_JOINTS, RIGHT_HAND_LINK, GR1ScrewdriverEnvCfg


class GR1ScrewdriverEnv(DirectRLEnv):
    cfg: GR1ScrewdriverEnvCfg

    def __init__(self, cfg: GR1ScrewdriverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # indices of the 18 controlled joints (right arm + right hand), in articulation order
        self._ctrl_ids = [self.robot.joint_names.index(n) for n in CONTROLLED_JOINTS]
        self._ctrl_ids_t = torch.tensor(self._ctrl_ids, device=self.device, dtype=torch.long)
        self._hand_body_id = self.robot.body_names.index(RIGHT_HAND_LINK)
        # default pose + joint limits for the controlled joints (for the action mapping)
        self._ctrl_default = self.robot.data.default_joint_pos[:, self._ctrl_ids_t].clone()
        lim = self.robot.root_physx_view.get_dof_limits().to(self.device)   # (N, J, 2)
        self._ctrl_lower = lim[..., 0][:, self._ctrl_ids_t]                 # (N, 18)
        self._ctrl_upper = lim[..., 1][:, self._ctrl_ids_t]
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.screwdriver = RigidObject(self.cfg.screwdriver_cfg)
        self.screw = RigidObject(self.cfg.screw_cfg)
        self.thread_test = RigidObject(self.cfg.thread_test_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["screwdriver"] = self.screwdriver
        self.scene.rigid_objects["screw"] = self.screw
        self.scene.rigid_objects["thread_test"] = self.thread_test
        self.scene.rigid_objects["table"] = self.table
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        # joint-position target = default pose + scaled action, clamped to the joint limits
        target = self._ctrl_default + self.actions * self.cfg.action_scale
        target = torch.clamp(target, self._ctrl_lower, self._ctrl_upper)
        self.robot.set_joint_position_target(target, joint_ids=self._ctrl_ids)

    # ------------------------------------------------------------------ obs / reward / done
    def _get_observations(self) -> dict:
        eo = self.scene.env_origins
        jp = self.robot.data.joint_pos[:, self._ctrl_ids_t]                  # (N,18)
        jv = self.robot.data.joint_vel[:, self._ctrl_ids_t]                  # (N,18)
        sd_pos = self.screwdriver.data.root_pos_w - eo
        sd_quat = self.screwdriver.data.root_quat_w
        sc_pos = self.screw.data.root_pos_w - eo
        sc_quat = self.screw.data.root_quat_w
        hand_pos = self.robot.data.body_pos_w[:, self._hand_body_id] - eo
        hand_quat = self.robot.data.body_quat_w[:, self._hand_body_id]
        obs = torch.cat([jp, jv, sd_pos, sd_quat, sc_pos, sc_quat, hand_pos, hand_quat], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # scaffold: no task reward yet
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        eo = self.scene.env_origins[env_ids]

        # robot back to the default standing/manipulation pose
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = self.robot.data.default_joint_vel[env_ids].clone()
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, 0:3] = root[:, 0:3] + eo
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        self.actions[env_ids] = 0.0

        # objects back to their default poses on the table
        for obj in (self.screwdriver, self.screw, self.thread_test, self.table):
            st = obj.data.default_root_state[env_ids].clone()
            st[:, 0:3] = st[:, 0:3] + eo
            obj.write_root_pose_to_sim(st[:, :7], env_ids)
            obj.write_root_velocity_to_sim(st[:, 7:], env_ids)
