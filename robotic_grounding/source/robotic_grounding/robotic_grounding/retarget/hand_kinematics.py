# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Any, Literal, Optional

import mujoco
import numpy as np
import pink
import pinocchio as pin
import torch
import viser
from judo.visualizers.model import ViserMjModel
from loop_rate_limiters import RateLimiter
from pink import solve_ik
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, RelativeFrameTask
from scipy.spatial.transform import Rotation as R

from robotic_grounding.retarget.params import (
    MANO_JOINTS_ORDER,
    SHARPA_RELATIVE_FRAMES,
    SHARPA_TO_MANO_MAPPING,
)
from robotic_grounding.retarget.utils import subtract_frame_transforms


class HandKinematics:
    """Hand kinematics class."""

    def __init__(
        self,
        side: Literal["right", "left"],
        robot_xml: str,
        use_relative_frames: bool = False,
        solver: str = "daqp",
        max_iter: int = 200,
        frequency: float = 200.0,
        frame_tasks_converged_threshold: float = 1e-6,
    ) -> None:
        """Initialize the Sharpa hand kinematics."""
        self.side = side
        self.robot_xml = robot_xml
        self.use_relative_frames = use_relative_frames
        self.solver = solver
        self.max_iter = max_iter
        self.frame_tasks_converged_threshold = frame_tasks_converged_threshold

        # Load robot model
        self.robot = pin.RobotWrapper.BuildFromMJCF(
            filename=self.robot_xml,
        )

        # Record robot info
        self.robot_joint_names = {}
        for i in range(1, self.robot.model.nq + 1):
            joint_name = self.robot.model.names[i]
            self.robot_joint_names[i] = joint_name

        self.robot_frame_names = {}
        # include body, joint, and site frames
        for i in range(1, len(self.robot.model.frames)):
            frame_name = self.robot.model.frames[i].name
            self.robot_frame_names[i] = frame_name

        # Setup pink solver
        self.configuration = pink.Configuration(
            self.robot.model,
            self.robot.data,
            self.robot.q0,
        )

        rate = RateLimiter(frequency=frequency, warn=False)
        self.dt = rate.period

        self.configuration_limits = [
            ConfigurationLimit(self.robot.model),
            VelocityLimit(self.robot.model),
        ]

        self.sharpa_to_mano_mapping = {
            k.replace(".*", self.side): v for k, v in SHARPA_TO_MANO_MAPPING.items()
        }
        self.sharpa_relative_frames = [
            (
                entry[0].replace(".*", self.side),
                entry[1].replace(".*", self.side),
                entry[2],
                entry[3],
            )
            for entry in SHARPA_RELATIVE_FRAMES
        ]

        # Setup IK frame tasks
        self.frame_tasks = {}
        for robot_site_name, (
            _,
            position_cost,
            orientation_cost,
        ) in self.sharpa_to_mano_mapping.items():
            self.frame_tasks[robot_site_name] = FrameTask(
                robot_site_name,
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=1.0,
            )
            self.frame_tasks[robot_site_name].set_target_from_configuration(
                self.configuration
            )

        # Setup relative frame tasks
        if self.use_relative_frames:
            self.relative_frame_tasks = {}
            for (
                robot_target_site_name,
                robot_root_site_name,
                position_cost,
                orientation_cost,
            ) in self.sharpa_relative_frames:
                task_name = (
                    f"relative_{robot_target_site_name}_to_{robot_root_site_name}"
                )
                self.relative_frame_tasks[task_name] = RelativeFrameTask(
                    frame=robot_target_site_name,
                    root=robot_root_site_name,
                    position_cost=position_cost,
                    orientation_cost=orientation_cost,
                    lm_damping=1.0,
                )

    def visualize(
        self,
        viser_server: viser.ViserServer,
        qpos: np.ndarray,
        visualize_sites: bool = True,
    ) -> None:
        """Visualize the robot."""
        if not hasattr(self, "robot_viser_model"):
            self._mujoco_robot_model = mujoco.MjModel.from_xml_path(self.robot_xml)
            self._mujoco_robot_data = mujoco.MjData(self._mujoco_robot_model)
            self._mujoco_robot_spec = mujoco.MjSpec.from_file(self.robot_xml)

            self.robot_viser_model = ViserMjModel(
                viser_server,
                self._mujoco_robot_spec,
                geom_exclude_substring="collision",
            )
            self.robot_viser_model.remove_traces()
            mujoco.mj_forward(self._mujoco_robot_model, self._mujoco_robot_data)
            self.robot_viser_model.set_data(self._mujoco_robot_data)

        if visualize_sites and not hasattr(self, "viser_robot_sites_handles"):
            self.viser_robot_sites_handles = {}
            for _, robot_site_name in enumerate(self.sharpa_to_mano_mapping.keys()):
                site_pos = self.configuration.get_transform_frame_to_world(
                    robot_site_name
                ).translation
                site_quat = R.from_matrix(
                    self.configuration.get_transform_frame_to_world(
                        robot_site_name
                    ).rotation
                ).as_quat(scalar_first=True)
                self.viser_robot_sites_handles[robot_site_name] = (
                    viser_server.scene.add_frame(
                        f"/robot/sites/{self.side}/{robot_site_name}",
                        position=site_pos,
                        wxyz=site_quat,
                        axes_length=0.018,
                        axes_radius=0.0008,
                    )
                )

        # Update robot data and viser model
        self._mujoco_robot_data.qpos = qpos.copy()
        mujoco.mj_forward(self._mujoco_robot_model, self._mujoco_robot_data)
        self.robot_viser_model.set_data(self._mujoco_robot_data)
        pin.framesForwardKinematics(self.robot.model, self.robot.data, qpos)

        if visualize_sites:
            vis_configuration = pink.Configuration(
                self.robot.model, self.robot.data, qpos.copy()
            )
            for _, robot_site_name in enumerate(self.sharpa_to_mano_mapping.keys()):
                site_pos = vis_configuration.get_transform_frame_to_world(
                    robot_site_name
                ).translation
                site_quat = R.from_matrix(
                    vis_configuration.get_transform_frame_to_world(
                        robot_site_name
                    ).rotation
                ).as_quat(scalar_first=True)
                self.viser_robot_sites_handles[robot_site_name].position = site_pos
                self.viser_robot_sites_handles[robot_site_name].wxyz = site_quat

    def set_frame_tasks_target(
        self,
        mano_joints: np.ndarray,
        mano_joints_wxyz: np.ndarray,
        mano_to_robot_scale: float = 1.0,
    ) -> None:
        """Set the target for the IK frame tasks."""
        wrist_pos = mano_joints[MANO_JOINTS_ORDER.index("wrist")]
        for robot_site_name, (
            mano_joint_name,
            _,
            _,
        ) in self.sharpa_to_mano_mapping.items():
            # 1.1 Compute the target position with scale factor
            mano_joint_idx = MANO_JOINTS_ORDER.index(mano_joint_name)
            target_pos = mano_joints[mano_joint_idx]
            target_pos = wrist_pos + (target_pos - wrist_pos) * mano_to_robot_scale
            target_wxyz = mano_joints_wxyz[mano_joint_idx]
            target_rot = R.from_quat(target_wxyz, scalar_first=True).as_matrix()
            # 1.2 Set the target for the IK frame task
            self.frame_tasks[robot_site_name].transform_target_to_world.translation = (
                target_pos.copy()
            )
            self.frame_tasks[robot_site_name].transform_target_to_world.rotation = (
                target_rot.copy()
            )

    def set_relative_frame_tasks_target(
        self,
        mano_joints: np.ndarray,
        mano_joints_wxyz: np.ndarray,
        mano_to_robot_scale: float = 1.0,
    ) -> None:
        """Set the target for the IK relative frame tasks."""
        for (
            robot_target_site_name,
            robot_root_site_name,
            _,
            _,
        ) in self.sharpa_relative_frames:
            # 2.1 Extract the target position
            task_name = f"relative_{robot_target_site_name}_to_{robot_root_site_name}"
            mano_target_joint_name = self.sharpa_to_mano_mapping[
                robot_target_site_name
            ][0]
            mano_target_joint_idx = MANO_JOINTS_ORDER.index(mano_target_joint_name)
            mano_target_joint_pos = mano_joints[mano_target_joint_idx]
            mano_target_joint_wxyz = mano_joints_wxyz[mano_target_joint_idx]
            mano_root_joint_name = self.sharpa_to_mano_mapping[robot_root_site_name][0]
            mano_root_joint_idx = MANO_JOINTS_ORDER.index(mano_root_joint_name)
            mano_root_joint_pos = mano_joints[mano_root_joint_idx]
            mano_root_joint_wxyz = mano_joints_wxyz[mano_root_joint_idx]
            # 2.2 Compute the target position with scale factor
            mano_root_p_target, mano_root_q_target = subtract_frame_transforms(
                mano_root_joint_pos,
                mano_root_joint_wxyz,
                mano_target_joint_pos,
                mano_target_joint_wxyz,
            )
            # TODO: Implement relative frame tasks
            self.relative_frame_tasks[
                task_name
            ].transform_target_to_world.translation = mano_root_p_target.copy()
            self.relative_frame_tasks[task_name].transform_target_to_world.rotation = (
                mano_root_q_target.copy()
            )
            raise NotImplementedError("Relative frame tasks are not debugged yet.")

    def compute(
        self,
        mano_joints: torch.Tensor | np.ndarray,
        mano_joints_wxyz: torch.Tensor | np.ndarray,
        mano_to_robot_scale: float = 1.0,
        qpos: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Compute the hand kinematics."""
        # 0. Move to numpy
        if isinstance(mano_joints, torch.Tensor):
            mano_joints = mano_joints.detach().cpu().numpy()
        if isinstance(mano_joints_wxyz, torch.Tensor):
            mano_joints_wxyz = mano_joints_wxyz.detach().cpu().numpy()

        # 1. Set the target for the IK frame tasks
        self.set_frame_tasks_target(
            mano_joints=mano_joints,
            mano_joints_wxyz=mano_joints_wxyz,
            mano_to_robot_scale=mano_to_robot_scale,
        )
        tasks = [*self.frame_tasks.values()]

        # 2. Set the target for the relative frame tasks
        if self.use_relative_frames:
            self.set_relative_frame_tasks_target(
                mano_joints, mano_joints_wxyz, mano_to_robot_scale
            )
            tasks = [*tasks, *self.relative_frame_tasks.values()]

        # 3. Solve the IK tasks
        self.configuration.q = self.robot.q0.copy() if qpos is None else qpos.copy()

        frame_tasks_pos_error = {
            task_name: float(np.inf) for task_name in self.frame_tasks.keys()
        }
        frame_tasks_converged = {
            task_name: False for task_name in self.frame_tasks.keys()
        }

        num_optimization_iterations = 0
        for _ in range(self.max_iter):
            vel = solve_ik(
                configuration=self.configuration,
                tasks=tasks,
                dt=self.dt,
                solver=self.solver,
                safety_break=False,
                limits=self.configuration_limits,
            )
            self.configuration.integrate_inplace(vel, self.dt)
            num_optimization_iterations += 1

            # Check if the solution is converged, terminate if it is
            for task_name, task in self.frame_tasks.items():
                pos_error = float(
                    np.linalg.norm(task.compute_error(self.configuration)[:3])
                )
                last_pos_error = frame_tasks_pos_error[task_name]
                if (
                    abs(pos_error - last_pos_error)
                    < self.frame_tasks_converged_threshold
                ):
                    frame_tasks_converged[task_name] = True
                frame_tasks_pos_error[task_name] = pos_error

            if all(frame_tasks_converged.values()):
                break

        # 4. Compute frame poses
        frame_pose = []
        for _, frame_name in self.robot_frame_names.items():
            pos = self.configuration.get_transform_frame_to_world(
                frame_name
            ).translation
            rot_matrix = self.configuration.get_transform_frame_to_world(
                frame_name
            ).rotation
            wxyz = R.from_matrix(rot_matrix).as_quat(scalar_first=True)
            frame_pose.append(np.hstack([pos, wxyz]).tolist())
        frame_pose = np.asarray(frame_pose)

        return {
            "q": self.configuration.q.copy(),
            "frame_pose": frame_pose,
            "frame_task_errors": [
                frame_tasks_pos_error[k] for k in self.frame_tasks.keys()
            ],
            "num_optimization_iterations": num_optimization_iterations,
        }
