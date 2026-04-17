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
    G1_WHOLEBODY_TO_NVHUMAN_MAPPING,
    NVHUMAN_JOINTS_ORDER,
    R_NVHUMAN_TO_ROBOT,
    R_PALM_CORRECTION_LEFT,
    R_PALM_CORRECTION_RIGHT,
)
from robotic_grounding.retarget.pinocchio_viser_visualizer import ViserVisualizer


class WholeBodyKinematics:
    """Base class for whole body kinematics retargeting.

    Follows the same patterns as HandKinematics but operates on the full body:
    pelvis, torso, arms/hands, and legs/feet.
    """

    def __init__(
        self,
        robot_asset_path: str,
        source_model: Literal["nvhuman", "smplh"],
        package_dirs: Optional[list[str]] = None,
        solver: str = "daqp",
        max_iter: int = 200,
        frequency: float = 200.0,
        frame_tasks_converged_threshold: float = 1e-6,
    ) -> None:
        """Initialize the whole body kinematics.

        Args:
            robot_asset_path: Path to the robot URDF/MJCF file.
            source_model: Source motion model ("nvhuman" or "smplh").
            package_dirs: Directories to search for package:// mesh URLs.
            solver: IK solver to use.
            max_iter: Maximum IK iterations.
            frequency: IK solve frequency.
            frame_tasks_converged_threshold: Convergence threshold.
        """
        self.robot_asset_path = robot_asset_path
        self.source_model = source_model
        self.package_dirs = package_dirs or []
        self.solver = solver
        self.max_iter = max_iter
        self.frequency = frequency
        self.frame_tasks_converged_threshold = frame_tasks_converged_threshold

        self.robot = self.load_robot_model()

        self.robot_joint_names = {
            i: str(self.robot.model.names[i])
            for i in range(1, self.robot.model.njoints)
        }

        self.robot_frame_names = {
            i: str(self.robot.model.frames[i].name)
            for i in range(len(self.robot.model.frames))
        }

        self.configuration_limits = [
            ConfigurationLimit(self.robot.model),
            VelocityLimit(self.robot.model),
        ]

        self.configuration = pink.Configuration(
            self.robot.model,
            self.robot.data,
            self.robot.q0,
        )

        rate = RateLimiter(frequency=frequency, warn=False)
        self.dt = rate.period

        self.source_joint_order = self.get_source_joint_order()
        self.target_to_source = self.get_target_to_source_mapping()
        self.frame_tasks = self.setup_frame_tasks()

    def load_robot_model(self) -> pin.RobotWrapper:
        """Load the robot model. Must be implemented by subclass."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement this method."
        )

    def get_source_joint_order(self) -> list[str]:
        """Get the source joint order based on source model."""
        if self.source_model == "nvhuman":
            return NVHUMAN_JOINTS_ORDER
        elif self.source_model == "smplh":
            raise NotImplementedError("SMPL-H source model is not yet implemented.")
        else:
            raise ValueError(f"Unknown source model: {self.source_model}")

    def get_target_to_source_mapping(self) -> dict[str, tuple[str, float, float]]:
        """Get the target-to-source mapping.

        Returns:
            Dictionary of {robot_frame: (source_joint, position_cost, orientation_cost)}.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement this method."
        )

    def get_base_source_joint(self) -> str:
        """Get the base source joint name for scaling."""
        if self.source_model == "nvhuman":
            return "Hips"
        elif self.source_model == "smplh":
            raise NotImplementedError("SMPL-H source model is not yet implemented.")
        else:
            raise ValueError(f"Unknown source model: {self.source_model}")

    def transform_source_position(self, position: np.ndarray) -> np.ndarray:
        """Transform source position to robot convention.

        Override in subclass if coordinate system transformation is needed.
        """
        return position

    def transform_source_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Transform a source body-local rotation matrix to robot convention.

        Uses a similarity transform ``R_src_to_robot @ M @ R_src_to_robot.T``
        so the rotation remains expressed in the robot's basis while acting on
        vectors from the body's local frame. Override in subclass if a
        coordinate system transformation is needed.
        """
        return rotation

    def transform_world_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Transform a source world-frame rotation matrix to robot convention.

        For world-frame transforms (e.g. an object's global pose, not a
        body-local joint rotation), only a left-multiply by
        ``R_src_to_robot`` is needed. Subclasses that rotate source
        coordinates should override.
        """
        return rotation

    def get_frame_rotation_correction(self, frame_name: str) -> np.ndarray:
        """Get rotation correction for a specific robot frame.

        Override in subclass if per-frame rotation corrections are needed.
        """
        return np.eye(3)

    def visualize(
        self,
        viser_server: viser.ViserServer,
        qpos: np.ndarray,
    ) -> None:
        """Visualize the robot in viser."""
        if not hasattr(self, "robot_viser_model"):
            self.robot_viser_model = ViserVisualizer(
                viser_server=viser_server,
                model=self.robot.model,
                visual_model=self.robot.visual_model,
                collision_model=self.robot.collision_model,
            )
        self.robot_viser_model.display(qpos)

    def get_scale_anchor_position(
        self,
        source_joints: np.ndarray,
    ) -> np.ndarray:
        """Get the anchor position for scaling (lowest point in vertical axis).

        Scaling is anchored at the lowest joint (feet) so ground contact is
        preserved. XY is centered on the base joint (hips).

        Args:
            source_joints: Array of shape (num_joints, 3) with joint positions.

        Returns:
            Anchor position of shape (3,) in robot coordinates.
        """
        all_positions = np.array(
            [
                self.transform_source_position(source_joints[i])
                for i in range(len(source_joints))
            ]
        )
        lowest_idx = np.argmin(all_positions[:, 2])
        anchor = all_positions[lowest_idx].copy()
        anchor[:2] = self.transform_source_position(
            source_joints[self.source_joint_order.index(self.get_base_source_joint())]
        )[:2]
        return anchor

    def set_frame_tasks_target(
        self,
        source_joints: np.ndarray,
        source_joints_wxyz: np.ndarray,
        source_to_robot_scale: float = 1.0,
    ) -> None:
        """Set the target for the IK frame tasks.

        Scaling is applied relative to the lowest point (feet) so that
        ground contact is preserved when scaling down.

        Args:
            source_joints: Array of shape (num_joints, 3) with joint positions.
            source_joints_wxyz: Array of shape (num_joints, 4) with joint orientations as wxyz quaternions.
            source_to_robot_scale: Scale factor for positions relative to ground anchor.
        """
        anchor_pos = self.get_scale_anchor_position(source_joints)

        for robot_frame_name, (
            source_joint_name,
            _,
            _,
        ) in self.target_to_source.items():
            source_joint_idx = self.source_joint_order.index(source_joint_name)
            target_pos = self.transform_source_position(source_joints[source_joint_idx])
            target_pos = anchor_pos + (target_pos - anchor_pos) * source_to_robot_scale

            target_wxyz = source_joints_wxyz[source_joint_idx]
            target_rot = R.from_quat(target_wxyz, scalar_first=True).as_matrix()
            target_rot = self.transform_source_rotation(target_rot)

            frame_correction = self.get_frame_rotation_correction(robot_frame_name)
            target_rot = target_rot @ frame_correction

            self.frame_tasks[robot_frame_name].transform_target_to_world.translation = (
                target_pos.copy()
            )
            self.frame_tasks[robot_frame_name].transform_target_to_world.rotation = (
                target_rot.copy()
            )

    def setup_frame_tasks(self) -> dict[str, FrameTask]:
        """Setup the IK frame tasks."""
        frame_tasks = {}
        for robot_frame_name, (
            _,
            position_cost,
            orientation_cost,
        ) in self.target_to_source.items():
            frame_tasks[robot_frame_name] = FrameTask(
                robot_frame_name,
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=1.0,
            )
            frame_tasks[robot_frame_name].set_target_from_configuration(
                self.configuration
            )
        return frame_tasks

    def compute(
        self,
        source_joints: torch.Tensor | np.ndarray,
        source_joints_wxyz: torch.Tensor | np.ndarray,
        source_to_robot_scale: float = 1.0,
        qpos: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Compute whole body IK.

        Args:
            source_joints: Joint positions from source model, shape (num_joints, 3).
            source_joints_wxyz: Joint orientations as wxyz quaternions, shape (num_joints, 4).
            source_to_robot_scale: Scale factor for positions relative to ground anchor.
            qpos: Initial joint configuration. Uses q0 if None.

        Returns:
            Dictionary with keys: q, frame_pose, frame_task_errors, num_optimization_iterations.
        """
        if isinstance(source_joints, torch.Tensor):
            source_joints = source_joints.detach().cpu().numpy()
        if isinstance(source_joints_wxyz, torch.Tensor):
            source_joints_wxyz = source_joints_wxyz.detach().cpu().numpy()

        self.set_frame_tasks_target(
            source_joints=source_joints,
            source_joints_wxyz=source_joints_wxyz,
            source_to_robot_scale=source_to_robot_scale,
        )
        tasks = list(self.frame_tasks.values())

        self.configuration.q = self.robot.q0.copy() if qpos is None else qpos.copy()

        frame_tasks_pos_error = {
            task_name: float(np.inf) for task_name in self.frame_tasks
        }
        frame_tasks_converged = {task_name: False for task_name in self.frame_tasks}

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
            "frame_task_errors": [frame_tasks_pos_error[k] for k in self.frame_tasks],
            "num_optimization_iterations": num_optimization_iterations,
        }


class G1WholeBodyKinematics(WholeBodyKinematics):
    """G1 robot whole body kinematics for NVHuman retargeting."""

    def __init__(
        self,
        robot_asset_path: str,
        package_dirs: Optional[list[str]] = None,
        source_model: Literal["nvhuman", "smplh"] = "nvhuman",
        solver: str = "daqp",
        max_iter: int = 200,
        frequency: float = 200.0,
        frame_tasks_converged_threshold: float = 1e-6,
    ) -> None:
        """Initialize G1 whole body kinematics."""
        self._R_nvhuman_to_robot = np.array(R_NVHUMAN_TO_ROBOT, dtype=np.float64)
        self._left_palm_correction = np.array(R_PALM_CORRECTION_LEFT, dtype=np.float64)
        self._right_palm_correction = np.array(
            R_PALM_CORRECTION_RIGHT, dtype=np.float64
        )

        super().__init__(
            robot_asset_path=robot_asset_path,
            source_model=source_model,
            package_dirs=package_dirs,
            solver=solver,
            max_iter=max_iter,
            frequency=frequency,
            frame_tasks_converged_threshold=frame_tasks_converged_threshold,
        )

    def load_robot_model(self) -> pin.RobotWrapper:
        """Load the G1 robot model from URDF with a free-flyer root joint."""
        return pin.RobotWrapper.BuildFromURDF(
            filename=self.robot_asset_path,
            package_dirs=self.package_dirs,
            root_joint=pin.JointModelFreeFlyer(),
        )

    def get_target_to_source_mapping(self) -> dict[str, tuple[str, float, float]]:
        """Get the G1 to NVHuman body mapping."""
        if self.source_model == "nvhuman":
            return dict(G1_WHOLEBODY_TO_NVHUMAN_MAPPING)
        elif self.source_model == "smplh":
            raise NotImplementedError(
                "SMPL-H source model is not yet implemented for G1."
            )
        else:
            raise ValueError(f"Unknown source model: {self.source_model}")

    def get_frame_rotation_correction(self, frame_name: str) -> np.ndarray:
        """Get rotation correction for palm frames to align with human hand."""
        if "left" in frame_name and "palm" in frame_name:
            return self._left_palm_correction
        if "right" in frame_name and "palm" in frame_name:
            return self._right_palm_correction
        return np.eye(3)

    def transform_source_position(self, position: np.ndarray) -> np.ndarray:
        """Transform position from NVHuman to robot convention."""
        return position @ self._R_nvhuman_to_robot.T

    def transform_source_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Transform a body-local rotation from NVHuman to robot convention."""
        return self._R_nvhuman_to_robot @ rotation @ self._R_nvhuman_to_robot.T

    def transform_world_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Transform a world-frame rotation from NVHuman to robot convention."""
        return self._R_nvhuman_to_robot @ rotation
