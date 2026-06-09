# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Any, Literal, Optional, cast

import numpy as np
import pink
import pinocchio as pin
import torch
import viser
from loop_rate_limiters import RateLimiter
from pink import solve_ik
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, PostureTask, Task
from scipy.spatial.transform import Rotation as R

from robotic_grounding.retarget.params import (
    SOMA_JOINTS_ORDER,
)
from robotic_grounding.retarget.pinocchio_viser_visualizer import ViserVisualizer
from robotic_grounding.retarget.robot_config import RobotRetargetConfig


class WholeBodyKinematics:
    """Base class for whole body kinematics retargeting.

    Follows the same patterns as HandKinematics but operates on the full body:
    pelvis, torso, arms/hands, and legs/feet.
    """

    def __init__(
        self,
        robot_asset_path: str,
        source_model: Literal["smplh", "soma"],
        package_dirs: Optional[list[str]] = None,
        solver: str = "daqp",
        max_iter: int = 200,
        frequency: float = 200.0,
        frame_tasks_converged_threshold: float = 1e-6,
    ) -> None:
        """Initialize the whole body kinematics.

        Args:
            robot_asset_path: Path to the robot URDF/MJCF file.
            source_model: Source motion model ("soma" is the canonical
                whole-body source; "smplh" is reserved for future SMPL-H
                support and currently raises NotImplementedError).
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
        if self.source_model == "soma":
            return SOMA_JOINTS_ORDER
        if self.source_model == "smplh":
            raise NotImplementedError("SMPL-H source model is not yet implemented.")
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
        if self.source_model == "soma":
            return "Hips"
        if self.source_model == "smplh":
            raise NotImplementedError("SMPL-H source model is not yet implemented.")
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

    def get_frame_translation_offset(self, frame_name: str) -> np.ndarray:
        """Get translation offset for a specific robot frame.

        Returned vector is expressed in the corrected robot-link local
        frame and applied at runtime as ``target_pos += target_rot @
        t_offset`` (mirrors soma-retargeter's
        ``wp.quat_rotate(q, offset_tx.p)`` term in
        ``wp_compute_scaled_effectors``). Default is zero; override in
        subclasses that read offsets from a config.
        """
        return np.zeros(3, dtype=np.float64)

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
            # ``source_joints_wxyz`` carries world-frame joint orientations.
            # SOMA: world rotations come from FK over ``t_pose_world`` and
            # need a single left-multiply by ``R_src_to_robot`` to land in
            # the robot world frame. The similarity-transform variant
            # (``transform_source_rotation``) is intentionally NOT used
            # here: when ``R_src_to_robot`` is not symmetric, the
            # similarity form silently degrades alignment.
            # ``source_model`` is narrowed to ``Literal["smplh", "soma"]``
            # and the smplh path raises NotImplementedError earlier in
            # init, so SOMA is the only live source at this point.
            target_rot = self.transform_world_rotation(target_rot)

            frame_correction = self.get_frame_rotation_correction(robot_frame_name)
            target_rot = target_rot @ frame_correction

            # Apply per-bone translation offset in the corrected effector
            # frame, mirroring soma-retargeter's
            # ``t = ... + wp.quat_rotate(q, offset_tx.p)``. ``target_rot``
            # at this point is already the final
            # ``R_world @ source_rot @ correction`` that Pink consumes,
            # so right-multiplying ``t_offset`` by it expresses the
            # offset in robot world. Returns zeros for frames without a
            # configured offset.
            t_offset = self.get_frame_translation_offset(robot_frame_name)
            if np.any(t_offset):
                target_pos = target_pos + target_rot @ t_offset

            self.frame_tasks[robot_frame_name].transform_target_to_world.translation = (
                target_pos.copy()
            )
            self.frame_tasks[robot_frame_name].transform_target_to_world.rotation = (
                target_rot.copy()
            )

    def _rebuild_configuration(self) -> None:
        """Rebuild ``self.configuration`` and ``self.configuration_limits``.

        Works around an eigenpy attribute-cache flake on the IsaacSim-bundled
        pinocchio build: the same ``pink.Configuration`` object, reused for
        many ``solve_ik`` calls, sporadically mis-resolves ``self.model``
        mid-sequence. Building fresh objects (including the limits, which
        hold the same ``Model`` reference) clears the per-instance cache.

        Preserves the current ``q`` so the IK warm-start is unaffected; the
        frame-task targets are owned separately and also carry over.
        """
        current_q = self.configuration.q.copy()
        self.configuration = pink.Configuration(
            self.robot.model,
            self.robot.data,
            current_q,
        )
        self.configuration_limits = [
            ConfigurationLimit(self.robot.model),
            VelocityLimit(self.robot.model),
        ]

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

    def _extra_tasks(self, qpos: Optional[np.ndarray]) -> list[Task]:
        """Extra Pink tasks appended to the QP each ``compute()`` call.

        Subclasses can return additional tasks (e.g. posture
        regularization) that should be solved alongside the frame
        tasks. The default returns an empty list, preserving the
        legacy behaviour of solving only frame tasks.

        ``qpos`` is the warm-start configuration the caller passed in
        (``None`` if defaulted to ``q0``). It is forwarded so subclasses
        can refresh per-frame targets (e.g. a "track previous q"
        posture task) before each solve.
        """
        del qpos
        return []

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
        # Subclass-supplied extras (e.g. posture regularization). Note
        # we resolve these BEFORE seeding ``self.configuration.q`` from
        # ``qpos`` so subclasses can refresh per-frame targets using the
        # caller's warm-start without depending on solver-side state.
        extra_tasks = list(self._extra_tasks(qpos))
        tasks = list(self.frame_tasks.values()) + extra_tasks
        has_extra_tasks = bool(extra_tasks)

        self.configuration.q = self.robot.q0.copy() if qpos is None else qpos.copy()

        frame_tasks_pos_error = {
            task_name: float(np.inf) for task_name in self.frame_tasks
        }
        frame_tasks_converged = {task_name: False for task_name in self.frame_tasks}

        num_optimization_iterations = 0
        # Bound the number of times we'll rebuild `self.configuration` in
        # response to the binding flake described below; 2 is enough in
        # practice (seen once in a ~1200-frame sequence) and keeps us from
        # masking an actual runaway bug.
        max_rebuilds_this_frame = 2
        rebuilds_this_frame = 0
        for _ in range(self.max_iter):
            # Defensive solve. Pink's ``Configuration.get_transform_frame_to_world``
            # resolves the frame id via ``self.model.getFrameId(frame)`` on a
            # pinocchio ``Model``. On the IsaacSim-bundled pinocchio / eigenpy
            # build, repeated use of the same ``Configuration`` object can
            # cause eigenpy's attribute cache to mis-resolve ``self.model``
            # and surface as either ``TypeError: 'Model' object is not
            # callable`` or ``AttributeError: 'Model' object has no attribute
            # 'model'`` — random, mid-sequence, non-deterministic. See
            # https://github.com/stack-of-tasks/eigenpy for similar reports.
            # Rebuilding ``Configuration`` from scratch drops the offending
            # per-instance attribute cache; we retry the current iteration
            # once (per frame) before accepting the last good ``q``.
            try:
                vel = solve_ik(
                    configuration=self.configuration,
                    tasks=tasks,
                    dt=self.dt,
                    solver=self.solver,
                    safety_break=False,
                    limits=self.configuration_limits,
                )
            except (AttributeError, TypeError) as exc:
                if rebuilds_this_frame >= max_rebuilds_this_frame:
                    print(
                        f"[whole_body_kinematics] giving up on frame after "
                        f"{rebuilds_this_frame} Pink/eigenpy rebuilds: {exc}"
                    )
                    break
                rebuilds_this_frame += 1
                self._rebuild_configuration()
                continue
            self.configuration.integrate_inplace(vel, self.dt)
            num_optimization_iterations += 1

            # Convergence check.
            #
            # Even when ``solve_ik`` succeeded, the subsequent per-task
            # ``compute_error`` path can still trip the same flake (it goes
            # through the same ``get_transform_frame_to_world``). Use the
            # solver's velocity norm as a fallback and as a secondary exit
            # condition — equivalent equilibrium signal that avoids the
            # flaky binding path entirely.
            vel_norm = float(np.linalg.norm(vel))
            for task_name, task in self.frame_tasks.items():
                try:
                    err_vec = np.asarray(task.compute_error(self.configuration))
                    pos_error = float(np.linalg.norm(err_vec[:3]))
                except (AttributeError, TypeError):
                    pos_error = vel_norm
                last_pos_error = frame_tasks_pos_error[task_name]
                if (
                    abs(pos_error - last_pos_error)
                    < self.frame_tasks_converged_threshold
                ):
                    frame_tasks_converged[task_name] = True
                frame_tasks_pos_error[task_name] = pos_error

            # When extra (e.g. posture) tasks are active, the
            # frame-task plateau check is not safe as an exit
            # condition: frame-task errors can be stationary while a
            # posture task is still pulling the configuration. In that
            # case fall back to the full-QP equilibrium signal
            # (``vel_norm``), which sees every active task. With no
            # extra tasks, behavior is bit-equivalent to before.
            if not has_extra_tasks and all(frame_tasks_converged.values()):
                break
            if vel_norm < self.frame_tasks_converged_threshold:
                break

        # Read each frame's world pose exactly ONCE. Access the pinocchio
        # model through ``self.robot.model`` (the RobotWrapper's stable
        # handle) rather than ``self.configuration.model`` to sidestep the
        # eigenpy attribute-cache flake that can mis-resolve attributes on
        # the ``Configuration`` object. Same data object throughout.
        model = self.robot.model
        data = self.configuration.data
        try:
            pin.forwardKinematics(model, data, self.configuration.q)
            pin.updateFramePlacements(model, data)
        except (AttributeError, TypeError) as exc:
            print(
                f"[whole_body_kinematics] forwardKinematics post-solve flake "
                f"({exc}); rebuilding configuration and retrying once."
            )
            self._rebuild_configuration()
            data = self.configuration.data
            pin.forwardKinematics(model, data, self.configuration.q)
            pin.updateFramePlacements(model, data)
        frame_pose = []
        for frame_id, _frame_name in self.robot_frame_names.items():
            oMf = data.oMf[frame_id]
            pos = oMf.translation
            wxyz = R.from_matrix(oMf.rotation).as_quat(scalar_first=True)
            frame_pose.append(np.hstack([pos, wxyz]).tolist())
        frame_pose = np.asarray(frame_pose)

        return {
            "q": self.configuration.q.copy(),
            "frame_pose": frame_pose,
            "frame_task_errors": [frame_tasks_pos_error[k] for k in self.frame_tasks],
            "num_optimization_iterations": num_optimization_iterations,
        }


class ConfigDrivenWholeBodyKinematics(WholeBodyKinematics):
    """Whole-body IK driven by a :class:`RobotRetargetConfig`.

    This class is the SOMA-to-robot core. All robot-specific data (URDF
    path, ik_map, world axis swap, per-bone offsets, per-link tweaks)
    comes from the config; the IK runtime
    (:meth:`WholeBodyKinematics.compute`,
    :meth:`WholeBodyKinematics.set_frame_tasks_target`) is unchanged.

    Numerical contract: for the verified G1 SOMA setup, this class
    produces frame task targets and IK output that match the legacy
    G1 SOMA implementation within ``atol=1e-8`` when the config
    matches the legacy constants. See
    ``tests/test_soma_g1_config_regression.py``.
    """

    def __init__(
        self,
        config: RobotRetargetConfig,
        *,
        solver: str = "daqp",
        max_iter: int = 200,
        frequency: float = 200.0,
        frame_tasks_converged_threshold: float = 1e-6,
    ) -> None:
        """Initialize from a fully materialized :class:`RobotRetargetConfig`."""
        self._config = config
        self._R_source_to_robot = np.asarray(config.r_world, dtype=np.float64)
        self._frame_corrections = {
            frame_name: np.asarray(matrix, dtype=np.float64)
            for frame_name, matrix in config.r_per_link.items()
        }
        self._frame_translations = {
            frame_name: np.asarray(vec, dtype=np.float64)
            for frame_name, vec in config.t_per_link.items()
        }

        # Only "soma" is fully implemented; "smplh" raises
        # NotImplementedError in every parent-class branch and would
        # only surface a confusing late error. Fail fast at init.
        if config.source_model != "soma":
            raise ValueError(
                f"ConfigDrivenWholeBodyKinematics: source_model="
                f"{config.source_model!r} is not implemented (only 'soma' "
                f"is supported). Update the robot config or implement the "
                f"smplh path before instantiating."
            )
        super().__init__(
            robot_asset_path=str(config.urdf_path),
            source_model=cast(Literal["soma"], config.source_model),
            package_dirs=[str(p) for p in config.package_dirs],
            solver=solver,
            max_iter=max_iter,
            frequency=frequency,
            frame_tasks_converged_threshold=frame_tasks_converged_threshold,
        )

        # Posture-regularization tasks. Built only when their cost is
        # strictly positive so the default-zero config path stays
        # bit-equivalent to the legacy SOMA IK (no extra task object,
        # no extra QP block, ``_extra_tasks`` returns ``[]``). The
        # ``q0`` task target never changes; the ``q_prev`` target is
        # refreshed per frame from ``compute()``'s ``qpos`` argument.
        pt = config.posture_task
        self._posture_q0_task: Optional[PostureTask] = (
            self._build_posture_task(pt.q0_cost, pt.lm_damping, pt.gain)
            if pt.q0_cost > 0.0
            else None
        )
        self._posture_qprev_task: Optional[PostureTask] = (
            self._build_posture_task(pt.q_prev_cost, pt.lm_damping, pt.gain)
            if pt.q_prev_cost > 0.0
            else None
        )

    @property
    def config(self) -> RobotRetargetConfig:
        """Return the underlying config bundle (read-only handle)."""
        return self._config

    def load_robot_model(self) -> pin.RobotWrapper:
        """Load the robot URDF with a free-flyer root joint."""
        return pin.RobotWrapper.BuildFromURDF(
            filename=self.robot_asset_path,
            package_dirs=self.package_dirs,
            root_joint=pin.JointModelFreeFlyer(),
        )

    def get_target_to_source_mapping(self) -> dict[str, tuple[str, float, float]]:
        """Build the IK frame->source mapping straight from the config."""
        return {
            frame_name: (entry.soma_joint, entry.position_cost, entry.orientation_cost)
            for frame_name, entry in self._config.ik_map.items()
        }

    def get_base_source_joint(self) -> str:
        """Return the SOMA joint used as the scaling anchor."""
        return self._config.base_source_joint

    def transform_source_position(self, position: np.ndarray) -> np.ndarray:
        """Transform a SOMA-world position into robot world (right-multiply by ``R.T``)."""
        return position @ self._R_source_to_robot.T

    def transform_world_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Transform a SOMA-world rotation into robot world (left-multiply by ``R``)."""
        return self._R_source_to_robot @ rotation

    def transform_source_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Similarity transform for body-local source rotations.

        Not used by the SOMA path (which goes through
        :meth:`transform_world_rotation`), but retained so future
        non-SOMA source models can reuse this class for similarity-
        transformed body-local rotations.
        """
        R_src = self._R_source_to_robot
        return R_src @ rotation @ R_src.T

    def get_frame_rotation_correction(self, frame_name: str) -> np.ndarray:
        """Return the per-link right-multiply correction for ``frame_name``.

        Falls back to identity for any frame the config does not
        explicitly correct, so adding a position-only IK target is a
        single JSON edit (no Python).
        """
        correction = self._frame_corrections.get(frame_name)
        if correction is None:
            return np.eye(3)
        return correction

    def get_frame_translation_offset(self, frame_name: str) -> np.ndarray:
        """Return the per-link translation offset for ``frame_name``.

        Falls back to zeros for any frame the config does not provide,
        which keeps frames without a configured ``t_offset`` consistent
        with the legacy ``target_pos`` behavior.
        """
        offset = self._frame_translations.get(frame_name)
        if offset is None:
            return np.zeros(3, dtype=np.float64)
        return offset

    def _build_posture_task(
        self, cost: float, lm_damping: float, gain: float
    ) -> PostureTask:
        """Build a Pink ``PostureTask`` and seed its target at ``q0``.

        The "regularize toward q0" task keeps this initial target for
        its lifetime; the "track previous q" task overwrites it on
        every ``compute()`` call (see :meth:`_extra_tasks`). Seeding
        both at q0 here avoids a separate "first call" branch in
        :meth:`_extra_tasks`.
        """
        task = PostureTask(cost=cost, lm_damping=lm_damping, gain=gain)
        task.set_target(self.robot.q0.copy())
        return task

    def _extra_tasks(self, qpos: Optional[np.ndarray]) -> list[Task]:
        """Return the configured posture tasks for this frame's solve.

        Refreshes the "track previous q" task's target from the
        caller-provided warm-start ``qpos`` (the script passes the
        previous frame's IK solution; for frame 0 ``qpos`` is ``q0`` so
        the term collapses to a pull toward q0).
        """
        extras: list[Task] = []
        q0_task = self._posture_q0_task
        if q0_task is not None:
            extras.append(q0_task)
        qprev_task = self._posture_qprev_task
        if qprev_task is not None:
            target = self.robot.q0.copy() if qpos is None else qpos.copy()
            qprev_task.set_target(target)
            extras.append(qprev_task)
        return extras
