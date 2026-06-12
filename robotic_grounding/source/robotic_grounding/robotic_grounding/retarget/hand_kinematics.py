# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Literal, Optional

import numpy as np
import pink
import pinocchio as pin
import torch
import viser
from loop_rate_limiters import RateLimiter
from pink import solve_ik
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, RelativeFrameTask
from scipy.spatial.transform import Rotation as R

from robotic_grounding.retarget.params import (
    DEX3_TO_MANO_MAPPING,
    MANO_JOINTS_ORDER,
    SHARPA_RELATIVE_FRAMES,
    SHARPA_TO_MANO_MAPPING,
    SHARPA_TO_MANO_ROTATION_OFFSET,
)
from robotic_grounding.retarget.pinocchio_viser_visualizer import ViserVisualizer
from robotic_grounding.retarget.utils import subtract_frame_transforms


class HandKinematics:
    """Base class for hand kinematics."""

    def __init__(
        self,
        side: Literal["right", "left"],
        robot_asset_path: str,
        source_model: Literal["mano"],
        use_relative_frames: bool = False,
        solver: str = "daqp",
        max_iter: int = 200,
        frequency: float = 200.0,
        frame_tasks_converged_threshold: float = 1e-6,
    ) -> None:
        """
        Initialize the hand kinematics.

        Args:
            side: Hand side ("left" or "right").
            robot_asset_path: Path to the robot URDF file.
            source_model: Source motion model. Only "mano" is supported.
            use_relative_frames: Whether to use relative frame tasks.
            solver: IK solver to use.
            max_iter: Maximum IK iterations.
            frequency: IK solve frequency.
            frame_tasks_converged_threshold: Convergence threshold.
        """
        self.side = side
        self.robot_asset_path = robot_asset_path
        self.source_model = source_model
        self.use_relative_frames = use_relative_frames
        self.solver = solver
        self.max_iter = max_iter
        self.frequency = frequency
        self.frame_tasks_converged_threshold = frame_tasks_converged_threshold

        # Load robot model
        self.robot = self.load_robot_model()

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

        self.configuration_limits = [
            ConfigurationLimit(self.robot.model),
            VelocityLimit(self.robot.model),
        ]

        # Setup pink solver
        self.configuration = pink.Configuration(
            self.robot.model,
            self.robot.data,
            self.robot.q0,
        )

        # Setup rate limiter
        rate = RateLimiter(frequency=frequency, warn=False)
        self.dt = rate.period

        # Get source joint order based on source model
        self.source_joint_order = self.get_source_joint_order()

        # Get source to target mapping
        self.target_to_source = self.get_target_to_source_mapping()
        self.target_to_source_rel = self.get_target_to_source_rel()

        # Setup IK frame tasks
        self.frame_tasks = self.setup_frame_tasks()

        # Setup relative frame tasks
        if self.use_relative_frames and self.target_to_source_rel is not None:
            self.relative_frame_tasks = self.setup_relative_frame_tasks()

    def load_robot_model(self) -> pin.RobotWrapper:
        """Load the robot model.

        Returns:
            pin.RobotWrapper: The robot model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement this method."
        )

    def get_source_joint_order(self) -> list[str]:
        """Get the source joint order. ``source_model`` is MANO-only."""
        return MANO_JOINTS_ORDER

    def get_target_to_source_mapping(self) -> dict[str, tuple[str, float, float]]:
        """Get the source to target mapping.

        Expected to return a dictionary of the form {target_site_name: (source_joint_name, position_cost, orientation_cost)}.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement this method."
        )

    def get_target_to_source_rel(self) -> list[tuple[str, str, float, float]]:
        """Get the source to target relative mapping.

        Returns:
            List of tuples of the form (target_site_name, root_site_name, position_cost, orientation_cost).
        """
        return []

    def get_base_source_joint(self) -> str:
        """Base source joint name for scaling. ``source_model`` is MANO-only."""
        return "wrist"

    def transform_source_position(self, position: np.ndarray) -> np.ndarray:
        """Transform source position to robot convention.

        Override in subclass if coordinate system transformation is needed.
        Default implementation returns the position unchanged.

        Args:
            position: Position array of shape (3,)

        Returns:
            Transformed position with same shape as input.
        """
        return position

    def transform_source_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Transform source rotation matrix to robot convention.

        Override in subclass if coordinate system transformation is needed.
        Default implementation returns the rotation unchanged.

        Args:
            rotation: Rotation matrix of shape (3, 3).

        Returns:
            Transformed rotation matrix of shape (3, 3).
        """
        return rotation

    def get_frame_rotation_correction(self, frame_name: str) -> np.ndarray:
        """Get rotation correction for a specific robot frame.

        Override in subclass if per-frame rotation corrections are needed
        Default implementation returns identity (no correction).

        Args:
            frame_name: Name of the robot frame.

        Returns:
            Rotation correction matrix of shape (3, 3).
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

    def set_frame_tasks_target(
        self,
        source_joints: np.ndarray,
        source_joints_wxyz: np.ndarray,
        base_source_joint: Optional[str] = None,
        source_to_robot_scale: float = 1.0,
    ) -> None:
        """Set the target for the IK frame tasks.

        Args:
            source_joints: Array of shape (num_joints, 3) with joint positions.
            source_joints_wxyz: Array of shape (num_joints, 4) with joint orientations as wxyz quaternions.
            base_source_joint: Name of the base joint for scaling. If None, uses get_base_source_joint().
            source_to_robot_scale: Scale factor for positions relative to base joint.
        """
        if base_source_joint is None:
            base_source_joint = self.get_base_source_joint()

        # Get base position, apply coordinate transform if needed
        base_joint_idx = self.source_joint_order.index(base_source_joint)
        base_pos = self.transform_source_position(source_joints[base_joint_idx])

        for robot_frame_name, (
            source_joint_name,
            _,
            _,
        ) in self.target_to_source.items():
            # 1.1 Get source joint position, apply coordinate transform if needed
            source_joint_idx = self.source_joint_order.index(source_joint_name)
            target_pos = self.transform_source_position(source_joints[source_joint_idx])

            # 1.2 Apply scale factor relative to base joint
            target_pos = base_pos + (target_pos - base_pos) * source_to_robot_scale

            # 1.3 Get source joint rotation, apply coordinate transform if needed
            target_wxyz = source_joints_wxyz[source_joint_idx]
            target_rot = R.from_quat(target_wxyz, scalar_first=True).as_matrix()
            target_rot = self.transform_source_rotation(target_rot)

            # 1.4 Apply per-frame rotation corrections
            frame_correction = self.get_frame_rotation_correction(robot_frame_name)
            target_rot = target_rot @ frame_correction

            # 1.5 Set the target for the IK frame task
            self.frame_tasks[robot_frame_name].transform_target_to_world.translation = (
                target_pos.copy()
            )
            self.frame_tasks[robot_frame_name].transform_target_to_world.rotation = (
                target_rot.copy()
            )

    def set_relative_frame_tasks_target(
        self,
        source_joints: np.ndarray,
        source_joints_wxyz: np.ndarray,
        source_to_robot_scale: float = 1.0,
    ) -> None:
        """Set the target for the IK relative frame tasks."""
        for (
            robot_target_site_name,
            robot_root_site_name,
            _,
            _,
        ) in self.target_to_source_rel:
            # 2.1 Extract the target position
            task_name = f"relative_{robot_target_site_name}_to_{robot_root_site_name}"
            source_target_joint_name = self.target_to_source[robot_target_site_name][0]
            source_target_joint_idx = self.source_joint_order.index(
                source_target_joint_name
            )
            source_target_joint_pos = source_joints[source_target_joint_idx]
            source_target_joint_wxyz = source_joints_wxyz[source_target_joint_idx]
            source_root_joint_name = self.target_to_source[robot_root_site_name][0]
            source_root_joint_idx = self.source_joint_order.index(
                source_root_joint_name
            )
            source_root_joint_pos = source_joints[source_root_joint_idx]
            source_root_joint_wxyz = source_joints_wxyz[source_root_joint_idx]
            # 2.2 Compute the target position with scale factor
            source_root_p_target, source_root_q_target = subtract_frame_transforms(
                source_root_joint_pos,
                source_root_joint_wxyz,
                source_target_joint_pos,
                source_target_joint_wxyz,
            )
            # TODO: Implement relative frame tasks
            self.relative_frame_tasks[
                task_name
            ].transform_target_to_world.translation = source_root_p_target.copy()
            self.relative_frame_tasks[task_name].transform_target_to_world.rotation = (
                source_root_q_target.copy()
            )
            raise NotImplementedError("Relative frame tasks are not debugged yet.")

    def setup_frame_tasks(self) -> dict[str, FrameTask]:
        """Setup the IK frame tasks."""
        frame_tasks = {}
        for robot_site_name, (
            _,
            position_cost,
            orientation_cost,
        ) in self.target_to_source.items():
            frame_tasks[robot_site_name] = FrameTask(
                robot_site_name,
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=1.0,
            )
            frame_tasks[robot_site_name].set_target_from_configuration(
                self.configuration
            )
        return frame_tasks

    def setup_relative_frame_tasks(self) -> dict[str, RelativeFrameTask]:
        """Setup the relative frame tasks."""
        relative_frame_tasks = {}
        for (
            robot_target_site_name,
            robot_root_site_name,
            position_cost,
            orientation_cost,
        ) in self.target_to_source_rel:
            task_name = f"relative_{robot_target_site_name}_to_{robot_root_site_name}"
            relative_frame_tasks[task_name] = RelativeFrameTask(
                frame=robot_target_site_name,
                root=robot_root_site_name,
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=1.0,
            )
        return relative_frame_tasks

    def compute(
        self,
        source_joints: torch.Tensor | np.ndarray,
        source_joints_wxyz: torch.Tensor | np.ndarray,
        source_to_robot_scale: float = 1.0,
        qpos: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Compute the hand kinematics."""
        # 0. Move to numpy
        if isinstance(source_joints, torch.Tensor):
            source_joints = source_joints.detach().cpu().numpy()
        if isinstance(source_joints_wxyz, torch.Tensor):
            source_joints_wxyz = source_joints_wxyz.detach().cpu().numpy()

        # 1. Set the target for the IK frame tasks
        self.set_frame_tasks_target(
            source_joints=source_joints,
            source_joints_wxyz=source_joints_wxyz,
            source_to_robot_scale=source_to_robot_scale,
        )
        tasks = [*self.frame_tasks.values()]

        # 2. Set the target for the relative frame tasks
        if self.use_relative_frames:
            self.set_relative_frame_tasks_target(
                source_joints, source_joints_wxyz, source_to_robot_scale
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
                # See ``WholeBodyKinematics.compute``: cast through
                # ``np.asarray`` so we always slice a plain ndarray, not an
                # Eigen-backed view from pinocchio whose ``__getitem__`` can
                # fail on slice objects on some eigenpy builds.
                err_vec = np.asarray(task.compute_error(self.configuration))
                pos_error = float(np.linalg.norm(err_vec[:3]))
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


class SharpaHandKinematics(HandKinematics):
    """Sharpa hand kinematics class."""

    def __init__(
        self,
        side: Literal["right", "left"],
        robot_asset_path: str,
        source_model: Literal["mano"],
        use_relative_frames: bool = False,
        solver: str = "daqp",
        max_iter: int = 200,
        frequency: float = 200.0,
        frame_tasks_converged_threshold: float = 1e-6,
    ) -> None:
        """Initialize the Sharpa hand kinematics."""
        super().__init__(
            side=side,
            robot_asset_path=robot_asset_path,
            source_model=source_model,
            use_relative_frames=use_relative_frames,
            solver=solver,
            max_iter=max_iter,
            frequency=frequency,
            frame_tasks_converged_threshold=frame_tasks_converged_threshold,
        )
        self._sharpa_rotation_corrections: dict[str, np.ndarray] = {}
        for frame_pattern, offset_wxyz in SHARPA_TO_MANO_ROTATION_OFFSET.items():
            frame_name = frame_pattern.replace(".*", self.side)
            self._sharpa_rotation_corrections[frame_name] = (
                R.from_quat(offset_wxyz, scalar_first=True).inv().as_matrix()
            )

    def load_robot_model(self) -> pin.RobotWrapper:
        """Load the robot model."""
        return pin.RobotWrapper.BuildFromMJCF(
            filename=self.robot_asset_path,
            root_joint=pin.JointModelFreeFlyer(),
        )

    def get_target_to_source_mapping(self) -> dict[str, tuple[str, float, float]]:
        """Source-to-target mapping. ``source_model`` is MANO-only."""
        return {
            k.replace(".*", self.side): v for k, v in SHARPA_TO_MANO_MAPPING.items()
        }

    def get_target_to_source_rel(self) -> list[tuple[str, str, float, float]]:
        """Source-to-target relative mapping. ``source_model`` is MANO-only."""
        return [
            (
                entry[0].replace(".*", self.side),
                entry[1].replace(".*", self.side),
                entry[2],
                entry[3],
            )
            for entry in SHARPA_RELATIVE_FRAMES
        ]

    def get_frame_rotation_correction(self, frame_name: str) -> np.ndarray:
        """Apply SHARPA_TO_MANO_ROTATION_OFFSET for wrist frame alignment."""
        if frame_name in self._sharpa_rotation_corrections:
            return self._sharpa_rotation_corrections[frame_name]
        return np.eye(3)


class Dex3HandKinematics(HandKinematics):
    """Dex3 hand kinematics class for MANO to Dex3 retargeting."""

    def __init__(
        self,
        side: Literal["right", "left"],
        robot_asset_path: str,
        package_dirs: Optional[list[str]] = None,
        source_model: Literal["mano"] = "mano",
        use_relative_frames: bool = False,
        solver: str = "daqp",
        max_iter: int = 100,
        frequency: float = 100.0,
        frame_tasks_converged_threshold: float = 1e-6,
    ) -> None:
        """Initialize the Dex3 hand kinematics."""
        self.package_dirs = package_dirs or []

        super().__init__(
            side=side,
            robot_asset_path=robot_asset_path,
            source_model=source_model,
            use_relative_frames=use_relative_frames,
            solver=solver,
            max_iter=max_iter,
            frequency=frequency,
            frame_tasks_converged_threshold=frame_tasks_converged_threshold,
        )

    def load_robot_model(self) -> pin.RobotWrapper:
        """Load the robot model from URDF with free-flyer root joint."""
        return pin.RobotWrapper.BuildFromURDF(
            filename=self.robot_asset_path,
            package_dirs=self.package_dirs,
            root_joint=pin.JointModelFreeFlyer(),
        )

    def get_target_to_source_mapping(self) -> dict[str, tuple[str, float, float]]:
        """Dex3 source-to-target mapping. ``source_model`` is MANO-only."""
        return {k.replace(".*", self.side): v for k, v in DEX3_TO_MANO_MAPPING.items()}

    # No transform_source_position / transform_source_rotation overrides:
    # MANO is in the same convention as the Dex3 robot, so the base-class
    # identity defaults (HandKinematics.transform_source_*) are correct.

    # MANO→Dex3 palm frame corrections (180° rotations)
    _R_MANO_PALM_RIGHT = np.array(
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64
    )  # 180° about Z
    _R_MANO_PALM_LEFT = np.array(
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64
    )  # 180° about Y

    def get_frame_rotation_correction(self, frame_name: str) -> np.ndarray:
        """Get rotation correction for palm frames to align with human hand."""
        if "palm" in frame_name:
            if self.side == "right":
                return self._R_MANO_PALM_RIGHT
            return self._R_MANO_PALM_LEFT
        return np.eye(3)
