# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Any, Literal, Optional

import numpy as np
import pink
import pinocchio as pin
import torch
import viser
from loop_rate_limiters import RateLimiter
from pink import solve_ik
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask
from scipy.spatial.transform import Rotation as R

from robotic_grounding.retarget.params import (
    MANO_JOINTS_ORDER,
    SHARPA_TO_MANO_MAPPING,
    SHARPA_TO_MANO_ROTATION_OFFSET,
)
from robotic_grounding.retarget.pinocchio_viser_visualizer import ViserVisualizer


class HandKinematics:
    """Hand kinematics class."""

    def __init__(
        self,
        side: Literal["right", "left"],
        robot_xml: str,
        solver: str = "daqp",
        max_iter: int = 200,
        frequency: float = 200.0,
        frame_tasks_converged_threshold: float = 1e-6,
    ) -> None:
        """Initialize the Sharpa hand kinematics."""
        self.side = side
        self.robot_xml = robot_xml
        self.solver = solver
        self.max_iter = max_iter
        self.frame_tasks_converged_threshold = frame_tasks_converged_threshold

        # Load robot model
        self.robot = pin.RobotWrapper.BuildFromMJCF(
            filename=self.robot_xml,
            root_joint=pin.JointModelFreeFlyer(),
        )

        # Record robot info, free layer creates 7 joints
        self.robot_finger_joint_names = {}
        # Record the rest of the joints
        for i in range(2, self.robot.model.nq - 7 + 2):
            joint_name = self.robot.model.names[i]
            self.robot_finger_joint_names[i - 2] = joint_name

        self.robot_frame_names = {}
        # include body, joint, and site frames
        for i in range(len(self.robot.model.frames)):
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
        self.sharpa_to_mano_rotation_offset = {
            k.replace(".*", self.side): v
            for k, v in SHARPA_TO_MANO_ROTATION_OFFSET.items()
        }

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

    def visualize(
        self,
        viser_server: viser.ViserServer,
        qpos: np.ndarray,
    ) -> None:
        """Visualize the robot."""
        if not hasattr(self, "robot_viser_model"):
            self.robot_viser_model = ViserVisualizer(
                viser_server=viser_server,
                model=self.robot.model,
                visual_model=self.robot.visual_model,
                collision_model=self.robot.collision_model,
            )

        self.robot_viser_model.display(qpos)

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
            target_rot = R.from_quat(target_wxyz, scalar_first=True)
            # 1.2 Apply rotation offset
            if robot_site_name in self.sharpa_to_mano_rotation_offset:
                target_rot = (
                    target_rot
                    * R.from_quat(
                        self.sharpa_to_mano_rotation_offset[robot_site_name],
                        scalar_first=True,
                    ).inv()
                )
            target_rot = target_rot.as_matrix()

            # 1.3 Set the target for the IK frame task
            self.frame_tasks[robot_site_name].transform_target_to_world.translation = (
                target_pos.copy()
            )
            self.frame_tasks[robot_site_name].transform_target_to_world.rotation = (
                target_rot.copy()
            )

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

        # 2. Solve the IK tasks
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

        # 3. Compute frame poses
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
