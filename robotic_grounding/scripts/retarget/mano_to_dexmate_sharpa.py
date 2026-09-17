# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-base inverse kinematics for named Dexmate-Sharpa frame targets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pink
import pinocchio as pin
from pink import solve_ik
from pink.barriers import BodySphericalBarrier
from pink.limits import ConfigurationLimit, Limit
from pink.tasks import FrameTask, PostureTask, Task
from robotic_grounding.retarget import ASSETS_DIR
from robotic_grounding.retarget.joint_limits import clamp_position_limits
from scipy.spatial.transform import Rotation as R

VEGA_SHARPA_URDF = ASSETS_DIR / "urdfs" / "vega_sharpa" / "vega_sharpa_reduced.urdf"
IK_JOINT_LIMIT_MARGIN_RAD = 1e-6


@dataclass(frozen=True)
class FrameTarget:
    """A Cartesian frame target expressed as position and wxyz orientation."""

    position: np.ndarray
    wxyz: np.ndarray

    def __post_init__(self) -> None:
        """Copy, validate, and normalize the target pose values."""
        position = np.asarray(self.position, dtype=np.float64).copy()
        wxyz = np.asarray(self.wxyz, dtype=np.float64).copy()

        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        if not np.all(np.isfinite(position)):
            raise ValueError("position must be finite")
        if wxyz.shape != (4,):
            raise ValueError(f"wxyz must have shape (4,), got {wxyz.shape}")
        if not np.all(np.isfinite(wxyz)):
            raise ValueError("wxyz must be finite")

        quaternion_scale = np.max(np.abs(wxyz))
        if quaternion_scale == 0.0:
            raise ValueError("wxyz norm must be positive")
        scaled_wxyz = wxyz / quaternion_scale
        scaled_norm = np.linalg.norm(scaled_wxyz)
        if quaternion_scale <= np.finfo(np.float64).eps / scaled_norm:
            raise ValueError("wxyz norm must be positive")

        normalized_wxyz = scaled_wxyz / scaled_norm
        position.setflags(write=False)
        normalized_wxyz.setflags(write=False)

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "wxyz", normalized_wxyz)


@dataclass(frozen=True)
class FrameTaskSpec:
    """Costs used to construct a Cartesian frame task."""

    position_cost: float
    orientation_cost: float


HAND_TASK_SUFFIX_SPECS = (
    ("hand_C_MC", FrameTaskSpec(0.2, 0.2)),
    ("thumb_MCP_VL", FrameTaskSpec(0.1, 0.0)),
    ("thumb_fingertip", FrameTaskSpec(1.0, 0.05)),
    ("index_MP", FrameTaskSpec(0.1, 0.0)),
    ("index_fingertip", FrameTaskSpec(1.0, 0.1)),
    ("middle_MP", FrameTaskSpec(0.1, 0.0)),
    ("middle_fingertip", FrameTaskSpec(1.0, 0.1)),
    ("ring_MP", FrameTaskSpec(0.1, 0.0)),
    ("ring_fingertip", FrameTaskSpec(1.0, 0.1)),
    ("pinky_MP", FrameTaskSpec(0.1, 0.0)),
    ("pinky_fingertip", FrameTaskSpec(0.5, 0.1)),
)
DEFAULT_FRAME_TASK_SPECS = {
    f"{side}_{suffix}": spec
    for side in ("left", "right")
    for suffix, spec in HAND_TASK_SUFFIX_SPECS
}
HAND_FRAMES_BY_WRIST = {
    f"{side}_hand_C_MC": tuple(
        f"{side}_{suffix}" for suffix, _spec in HAND_TASK_SUFFIX_SPECS
    )
    for side in ("left", "right")
}
DEFAULT_REFERENCE_Q = {
    "L_arm_j1": np.pi / 2.0,
    "L_arm_j2": np.pi / 6.0,
    "L_arm_j4": -np.pi / 2.0,
    "R_arm_j1": -np.pi / 2.0,
    "R_arm_j2": -np.pi / 6.0,
    "R_arm_j4": -np.pi / 2.0,
}


@dataclass(frozen=True)
class IKResult:
    """Result of an inverse-kinematics solve and its per-frame errors."""

    q: np.ndarray
    converged: bool
    iterations: int
    frame_errors: dict[str, np.ndarray]


def _validated_runtime_float(value: Any, name: str, *, allow_zero: bool) -> float:
    """Validate and normalize one finite real-valued runtime control."""
    requirement = "finite and nonnegative" if allow_zero else "finite and positive"
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be {requirement}, got {value!r}")
    normalized = float(value)
    if not np.isfinite(normalized) or (
        normalized < 0.0 if allow_zero else normalized <= 0.0
    ):
        raise ValueError(f"{name} must be {requirement}, got {value!r}")
    return normalized


def _validated_max_iters(value: Any) -> int:
    """Validate and normalize the positive integer iteration budget."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"max_iters must be a positive integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class CollisionBarrierSpec:
    """Configuration for one frame-origin spherical collision barrier."""

    frames: tuple[str, str]
    d_min: float
    gain: float = 1.0
    safe_displacement_gain: float = 0.0

    def __post_init__(self) -> None:
        """Validate the frame pair and normalize the Pink barrier controls."""
        if not isinstance(self.frames, tuple):
            raise ValueError("frames must be a tuple")
        if len(self.frames) != 2:
            raise ValueError("frames must contain exactly two names")
        if any(not isinstance(frame, str) or not frame for frame in self.frames):
            raise ValueError("frames must be nonempty strings")
        if self.frames[0] == self.frames[1]:
            raise ValueError("frames must be distinct")

        object.__setattr__(
            self,
            "d_min",
            _validated_runtime_float(self.d_min, "d_min", allow_zero=False),
        )
        object.__setattr__(
            self,
            "gain",
            _validated_runtime_float(self.gain, "gain", allow_zero=False),
        )
        object.__setattr__(
            self,
            "safe_displacement_gain",
            _validated_runtime_float(
                self.safe_displacement_gain,
                "safe_displacement_gain",
                allow_zero=True,
            ),
        )


DEFAULT_ARM_TORSO_COLLISION_SPECS = tuple(
    CollisionBarrierSpec(
        frames=("torso_l3", f"{side}_arm_l{link_index}"),
        d_min=0.3,
    )
    for side in ("L", "R")
    for link_index in range(2, 8)
)


def _target_to_matrix(target: FrameTarget) -> np.ndarray:
    """Convert a frame target to a homogeneous transform."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_quat(target.wxyz, scalar_first=True).as_matrix()
    transform[:3, 3] = target.position
    return transform


def _target_from_matrix(transform: np.ndarray) -> FrameTarget:
    """Convert a homogeneous transform to a frame target."""
    return FrameTarget(
        position=transform[:3, 3],
        wxyz=R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True),
    )


def build_neutral_hand_targets(
    reference_targets: Mapping[str, FrameTarget],
    desired_wrist_targets: Mapping[str, FrameTarget],
    wrist_to_hand_frames: Mapping[str, Sequence[str]],
) -> dict[str, FrameTarget]:
    """Rigidly carry reference hand frames with desired wrist poses."""
    expected = set(wrist_to_hand_frames)
    provided = set(desired_wrist_targets)
    if provided != expected:
        raise ValueError(
            f"wrist targets mismatch: missing={sorted(expected - provided)}, "
            f"extra={sorted(provided - expected)}"
        )

    result: dict[str, FrameTarget] = {}
    for wrist_name, frame_names in wrist_to_hand_frames.items():
        if wrist_name not in reference_targets:
            raise ValueError(f"reference targets missing wrist frame {wrist_name!r}")
        reference_wrist = _target_to_matrix(reference_targets[wrist_name])
        desired_wrist = _target_to_matrix(desired_wrist_targets[wrist_name])
        reference_to_desired = desired_wrist @ np.linalg.inv(reference_wrist)
        for frame_name in frame_names:
            if frame_name not in reference_targets:
                raise ValueError(f"reference targets missing hand frame {frame_name!r}")
            result[frame_name] = _target_from_matrix(
                reference_to_desired @ _target_to_matrix(reference_targets[frame_name])
            )
    return result


class WorldPositionTask(Task):
    """Position-only frame task whose residual is true world xyz."""

    def __init__(
        self,
        frame: str,
        position_cost: float,
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        """Configure the target frame and scalar position cost."""
        if not isinstance(frame, str) or not frame:
            raise ValueError(f"frame must be a nonempty string, got {frame!r}")
        cost = _validated_runtime_float(
            position_cost, "position_cost", allow_zero=False
        )
        super().__init__(cost=np.full(3, cost), gain=gain, lm_damping=lm_damping)
        self.frame = frame
        self._target_position: np.ndarray | None = None

    def set_target(self, position: np.ndarray) -> None:
        """Store a validated world-frame target position."""
        value = np.asarray(position, dtype=np.float64).copy()
        if value.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("position must be finite")
        value.setflags(write=False)
        self._target_position = value

    def compute_error_requires_target_guard(self) -> np.ndarray:
        """Return the stored target or fail closed when it is unset."""
        if self._target_position is None:
            raise ValueError("target position is unset; call set_target first")
        return self._target_position

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        """Return ``target_position - current_position`` in world frame."""
        target = self.compute_error_requires_target_guard()
        transform = configuration.get_transform_frame_to_world(self.frame)
        return target - np.asarray(transform.translation, dtype=np.float64)

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        """Return minus the LOCAL_WORLD_ALIGNED linear frame Jacobian."""
        self.compute_error_requires_target_guard()
        transform = configuration.get_transform_frame_to_world(self.frame)
        local_jacobian = np.asarray(
            configuration.get_frame_jacobian(self.frame), dtype=np.float64
        )
        rotation = np.asarray(transform.rotation, dtype=np.float64)
        return -(rotation @ local_jacobian[:3, :])

    def __repr__(self) -> str:
        """Describe the task frame and cost for solver diagnostics."""
        return f"WorldPositionTask(frame={self.frame!r}, cost={self.cost!r})"


class FixedAnchorBoxLimit(Limit):
    """Hard absolute configuration box recomputed at every inner iterate."""

    def __init__(self, model: pin.Model, lower: np.ndarray, upper: np.ndarray) -> None:
        """Validate the joint representation and the fixed box bounds."""
        if model.nq != model.nv:
            raise ValueError(
                f"FixedAnchorBoxLimit requires nq == nv one-DoF joints, got nq={model.nq}, nv={model.nv}"
            )
        lower_array = np.asarray(lower, dtype=np.float64).copy()
        upper_array = np.asarray(upper, dtype=np.float64).copy()
        expected = (model.nq,)
        if lower_array.shape != expected or upper_array.shape != expected:
            raise ValueError(
                f"bounds must have shape {expected}, got {lower_array.shape} and {upper_array.shape}"
            )
        if not (np.all(np.isfinite(lower_array)) and np.all(np.isfinite(upper_array))):
            raise ValueError("bounds must be finite")
        if not np.all(lower_array <= upper_array):
            raise ValueError("lower must be elementwise at or below upper")
        lower_array.setflags(write=False)
        upper_array.setflags(write=False)
        self.lower = lower_array
        self.upper = upper_array
        self._nv = int(model.nv)

    def compute_qp_inequalities(
        self, configuration: pink.Configuration, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bound the tangent step so the iterate cannot leave the box."""
        q = np.asarray(configuration.q, dtype=np.float64)
        matrix = np.vstack([np.eye(self._nv), -np.eye(self._nv)])
        bound = np.concatenate([self.upper - q, q - self.lower])
        return matrix, bound


class WristAngularGateLimit(Limit):
    """Linearized per-frame geodesic orientation gate for static projection."""

    _ANGLE_EPSILON = 1e-9

    def __init__(
        self,
        model: pin.Model,
        frame_gates: Mapping[str, tuple[np.ndarray, float]],
    ) -> None:
        """Validate the gated frames, target rotations, and gate angles."""
        if model.nq != model.nv:
            raise ValueError(
                f"WristAngularGateLimit requires nq == nv one-DoF joints, got nq={model.nq}, nv={model.nv}"
            )
        if not frame_gates:
            raise ValueError("frame_gates must contain at least one frame")
        gates: dict[str, tuple[np.ndarray, float]] = {}
        for frame, (target_wxyz, gate_rad) in dict(frame_gates).items():
            if not isinstance(frame, str) or not frame:
                raise ValueError(f"frame names must be nonempty strings, got {frame!r}")
            target = FrameTarget(position=np.zeros(3), wxyz=target_wxyz)
            gate = _validated_runtime_float(gate_rad, "gate_rad", allow_zero=False)
            gates[frame] = (target.wxyz, gate)
        self._frame_gates = gates
        self._nv = int(model.nv)

    def compute_qp_inequalities(
        self, configuration: pink.Configuration, dt: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Linearize every active angular gate as one inequality row."""
        rows: list[np.ndarray] = []
        bounds: list[float] = []
        for frame, (target_wxyz, gate) in self._frame_gates.items():
            transform = configuration.get_transform_frame_to_world(frame)
            rotation = np.asarray(transform.rotation, dtype=np.float64)
            target_rotation = R.from_quat(target_wxyz, scalar_first=True).as_matrix()
            error_rotvec = R.from_matrix(rotation.T @ target_rotation).as_rotvec()
            angle = float(np.linalg.norm(error_rotvec))
            if angle < self._ANGLE_EPSILON:
                continue
            axis_local = error_rotvec / angle
            local_jacobian = np.asarray(
                configuration.get_frame_jacobian(frame), dtype=np.float64
            )
            gradient = -(axis_local @ local_jacobian[3:6, :])
            rows.append(gradient)
            bounds.append(gate - angle)
        if not rows:
            return None
        return np.vstack(rows), np.asarray(bounds, dtype=np.float64)


class DexmateSharpaIK:
    """Source-agnostic fixed-base IK for named Dexmate-Sharpa frames."""

    def __init__(
        self,
        urdf_path: Path = VEGA_SHARPA_URDF,
        frame_task_specs: Mapping[str, FrameTaskSpec] | None = None,
        *,
        collision_barrier_specs: Sequence[CollisionBarrierSpec] | None = None,
        posture_cost: float = 1e-2,
        lm_damping: float = 1e-3,
        position_tolerance: float = 5e-4,
        orientation_tolerance: float = 2e-3,
        dt: float = 1e-2,
        max_iters: int = 100,
        solver: str = "daqp",
    ) -> None:
        """Build the fixed-base model, natural posture, and Pink tasks."""
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Dexmate-Sharpa URDF not found: {self.urdf_path}")
        self.position_tolerance = _validated_runtime_float(
            position_tolerance, "position_tolerance", allow_zero=True
        )
        self.orientation_tolerance = _validated_runtime_float(
            orientation_tolerance, "orientation_tolerance", allow_zero=True
        )
        self.dt = _validated_runtime_float(dt, "dt", allow_zero=False)
        self.max_iters = _validated_max_iters(max_iters)
        self.solver = solver
        self._frame_task_specs = dict(
            DEFAULT_FRAME_TASK_SPECS if frame_task_specs is None else frame_task_specs
        )
        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(self.urdf_path),
            package_dirs=[str(self.urdf_path.parent)],
            root_joint=None,
        )
        self._robot_joint_names = tuple(
            str(name) for name in self.robot.model.names[1:]
        )
        model_frames = {str(frame.name) for frame in self.robot.model.frames}
        missing_frames = sorted(set(self._frame_task_specs) - model_frames)
        if missing_frames:
            raise ValueError(f"IK task frames not found in URDF: {missing_frames}")
        selected_collision_specs = tuple(
            DEFAULT_ARM_TORSO_COLLISION_SPECS
            if collision_barrier_specs is None
            else collision_barrier_specs
        )
        if any(
            not isinstance(spec, CollisionBarrierSpec)
            for spec in selected_collision_specs
        ):
            raise ValueError(
                "collision_barrier_specs must contain CollisionBarrierSpec values"
            )
        missing_collision_frames = sorted(
            {
                frame
                for spec in selected_collision_specs
                for frame in spec.frames
                if frame not in model_frames
            }
        )
        if missing_collision_frames:
            raise ValueError(
                "Collision barrier frames not found in URDF: "
                f"{missing_collision_frames}"
            )
        seen_collision_pairs: set[frozenset[str]] = set()
        for spec in selected_collision_specs:
            pair_key = frozenset(spec.frames)
            if pair_key in seen_collision_pairs:
                raise ValueError(f"Duplicate collision barrier pair: {spec.frames}")
            seen_collision_pairs.add(pair_key)
        self._collision_barrier_specs = selected_collision_specs
        self._q_reference = self.robot.q0.copy()
        missing_joints = sorted(set(DEFAULT_REFERENCE_Q) - set(self._robot_joint_names))
        if missing_joints:
            raise ValueError(f"Reference joints not found in URDF: {missing_joints}")
        for joint_name, value in DEFAULT_REFERENCE_Q.items():
            joint_id = self.robot.model.getJointId(joint_name)
            joint = self.robot.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(
                    f"Reference joint {joint_name!r} has nq={joint.nq}, expected 1"
                )
            self._q_reference[joint.idx_q] = value
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self._q_reference.copy()
        )
        self.configuration_limits = [ConfigurationLimit(self.robot.model)]
        self._collision_barriers = tuple(
            BodySphericalBarrier(
                spec.frames,
                d_min=spec.d_min,
                gain=spec.gain,
                safe_displacement_gain=spec.safe_displacement_gain,
            )
            for spec in self._collision_barrier_specs
        )
        self.frame_tasks = {
            name: FrameTask(
                name,
                position_cost=spec.position_cost,
                orientation_cost=spec.orientation_cost,
                lm_damping=lm_damping,
            )
            for name, spec in self._frame_task_specs.items()
        }
        for task in self.frame_tasks.values():
            task.set_target_from_configuration(self.configuration)
        self.posture_task = PostureTask(cost=posture_cost, lm_damping=lm_damping)
        self.posture_task.set_target(self._q_reference.copy())

    @property
    def q_reference(self) -> np.ndarray:
        """Return a defensive copy of the natural reference configuration."""
        return self._q_reference.copy()

    @property
    def robot_joint_names(self) -> tuple[str, ...]:
        """Return movable joint names in Pinocchio q order."""
        return self._robot_joint_names

    @property
    def task_frame_names(self) -> tuple[str, ...]:
        """Return configured frame-task names in deterministic solve order."""
        return tuple(self._frame_task_specs)

    @property
    def collision_barrier_specs(self) -> tuple[CollisionBarrierSpec, ...]:
        """Return the immutable collision-barrier configuration."""
        return self._collision_barrier_specs

    @property
    def collision_barriers(self) -> tuple[BodySphericalBarrier, ...]:
        """Return the Pink collision barriers used by every solve."""
        return self._collision_barriers

    def _validate_q(self, q: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(q, dtype=np.float64).copy()
        expected = (self.robot.model.nq,)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
        return value

    def _clamp_q_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Project one solver configuration strictly inside the URDF limits."""
        return clamp_position_limits(
            q,
            self.robot.model.lowerPositionLimit,
            self.robot.model.upperPositionLimit,
            IK_JOINT_LIMIT_MARGIN_RAD,
        )

    def fk(
        self, q: np.ndarray, frame_names: Iterable[str] | None = None
    ) -> dict[str, FrameTarget]:
        """Compute normalized world targets for selected robot frames."""
        value = self._validate_q(q, "q")
        names = self.task_frame_names if frame_names is None else tuple(frame_names)
        known = {str(frame.name) for frame in self.robot.model.frames}
        unknown = sorted(set(names) - known)
        if unknown:
            raise ValueError(f"FK frames not found in URDF: {unknown}")
        pin.forwardKinematics(self.robot.model, self.robot.data, value)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        return {
            name: FrameTarget(
                self.robot.data.oMf[self.robot.model.getFrameId(name)].translation,
                R.from_matrix(
                    self.robot.data.oMf[self.robot.model.getFrameId(name)].rotation
                ).as_quat(scalar_first=True),
            )
            for name in names
        }

    def _validated_targets(
        self, frame_targets: Mapping[str, FrameTarget]
    ) -> dict[str, FrameTarget]:
        """Validate and copy a complete configured-frame target mapping."""
        expected = set(self.task_frame_names)
        provided = set(frame_targets)
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        if missing or extra:
            raise ValueError(
                f"frame targets mismatch: missing={missing}, extra={extra}"
            )
        return {
            name: FrameTarget(
                frame_targets[name].position,
                frame_targets[name].wxyz,
            )
            for name in self.task_frame_names
        }

    def _frame_errors(self) -> dict[str, np.ndarray]:
        """Return validated, copied task errors in configured frame order."""
        errors: dict[str, np.ndarray] = {}
        for name in self.task_frame_names:
            error = np.asarray(
                self.frame_tasks[name].compute_error(self.configuration),
                dtype=np.float64,
            ).copy()
            if error.shape != (6,):
                raise ValueError(
                    f"frame error must have shape (6,) for {name!r}, got {error.shape}"
                )
            if not np.all(np.isfinite(error)):
                raise FloatingPointError(f"frame error must be finite for {name!r}")
            errors[name] = error
        return errors

    def _converged(self, errors: Mapping[str, np.ndarray]) -> bool:
        """Check active task components and all collision barriers."""
        for name, error in errors.items():
            if not np.all(np.isfinite(error)):
                return False
            spec = self._frame_task_specs[name]
            if (
                spec.position_cost > 0.0
                and np.linalg.norm(error[:3]) > self.position_tolerance
            ):
                return False
            if (
                spec.orientation_cost > 0.0
                and np.linalg.norm(error[3:]) > self.orientation_tolerance
            ):
                return False
        for index, barrier in enumerate(self._collision_barriers):
            value = np.asarray(
                barrier.compute_barrier(self.configuration), dtype=np.float64
            ).copy()
            if value.shape != (1,):
                raise ValueError(
                    "collision barrier value must have shape (1,) for barrier "
                    f"{index}, got {value.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"collision barrier value must be finite for barrier {index}"
                )
            if value[0] < 0.0:
                return False
        return True

    def solve(
        self,
        frame_targets: Mapping[str, FrameTarget],
        q_init: np.ndarray | None = None,
    ) -> IKResult:
        """Solve all configured robot frame targets from one warm start."""
        targets = self._validated_targets(frame_targets)
        initial_q = (
            self._q_reference.copy()
            if q_init is None
            else self._validate_q(q_init, "q_init")
        )
        initial_q = self._clamp_q_to_limits(initial_q)
        self.configuration.update(initial_q)
        for name, target in targets.items():
            task = self.frame_tasks[name]
            task.transform_target_to_world = pin.SE3(
                R.from_quat(target.wxyz, scalar_first=True).as_matrix(),
                target.position.copy(),
            )

        tasks = [
            *(self.frame_tasks[name] for name in self.task_frame_names),
            self.posture_task,
        ]
        errors: dict[str, np.ndarray] = {}
        converged = False
        iterations = 0
        for _ in range(self.max_iters):
            iterations += 1
            velocity = np.asarray(
                solve_ik(
                    configuration=self.configuration,
                    tasks=tasks,
                    dt=self.dt,
                    solver=self.solver,
                    safety_break=False,
                    limits=self.configuration_limits,
                    barriers=self._collision_barriers or None,
                ),
                dtype=np.float64,
            ).copy()
            expected_velocity_shape = (self.robot.model.nv,)
            if velocity.shape != expected_velocity_shape:
                raise ValueError(
                    "solver velocity must have shape "
                    f"{expected_velocity_shape}, got {velocity.shape}"
                )
            if not np.all(np.isfinite(velocity)):
                raise FloatingPointError("solver velocity must be finite")
            integrated_q = pin.integrate(
                self.robot.model,
                self.configuration.q,
                velocity * self.dt,
            )
            self.configuration.update(self._clamp_q_to_limits(integrated_q))
            errors = self._frame_errors()
            converged = self._converged(errors)
            if converged:
                break

        return IKResult(
            q=self.configuration.q.copy(),
            converged=converged,
            iterations=iterations,
            frame_errors=errors,
        )
