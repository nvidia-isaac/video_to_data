"""Franka Panda kinematics used by the MECKA bimanual renderer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ARM_JOINTS = [f"joint{index}" for index in range(1, 8)]
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
DEFAULT_MAX_JOINT_STEP_RAD = 0.3
_JOINT_LIMIT_TOLERANCE = 1e-8


class IKCandidateError(RuntimeError):
    """An IK candidate failed a hard acceptance gate."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class IKSolveResult:
    """Accepted IK result and the route used to obtain it."""

    residual_m: float
    backend: str
    joint_step_rad: float
    ssik_status: str


def validate_arm_candidate(
    candidate: np.ndarray,
    *,
    previous_q: np.ndarray | None,
    joint_ranges: np.ndarray,
    max_joint_step_rad: float = DEFAULT_MAX_JOINT_STEP_RAD,
) -> float:
    """Validate one candidate before it may become solver state."""
    q = np.asarray(candidate, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise IKCandidateError("invalid", "IK candidate must be finite with shape (7,)")
    if ranges.shape != (7, 2) or not np.isfinite(ranges).all():
        raise ValueError("joint_ranges must be finite with shape (7,2)")
    if not np.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0.0:
        raise ValueError("max_joint_step_rad must be positive and finite")
    if np.any(q < ranges[:, 0] - _JOINT_LIMIT_TOLERANCE) or np.any(
        q > ranges[:, 1] + _JOINT_LIMIT_TOLERANCE
    ):
        raise IKCandidateError("joint_limits", "IK candidate exceeds joint limits")
    if previous_q is None:
        return 0.0
    previous = np.asarray(previous_q, dtype=np.float64)
    if previous.shape != (7,) or not np.isfinite(previous).all():
        raise ValueError("previous_q must be finite with shape (7,)")
    joint_step = float(np.max(np.abs(q - previous)))
    if joint_step > max_joint_step_rad + _JOINT_LIMIT_TOLERANCE:
        raise IKCandidateError(
            "continuity",
            (
                f"IK candidate joint step {joint_step:.6f} rad exceeds "
                f"{max_joint_step_rad:.6f} rad"
            ),
        )
    return joint_step


def build_panda_model(
    panda_dir: str | Path, fovy_deg: float, width: int, height: int
) -> mujoco.MjModel:
    """Build the overlay model in memory without modifying the asset tree."""
    directory = Path(panda_dir).expanduser().resolve()
    source = directory / "panda.xml"
    if not source.is_file():
        raise FileNotFoundError(source)
    xml = source.read_text(encoding="utf-8")
    xml = xml.replace(
        'meshdir="assets"',
        f'meshdir="{directory / "assets"}"',
        1,
    )
    xml = xml.replace(
        '<mujoco model="panda">',
        (
            '<mujoco model="panda">\n'
            f'  <visual><global offwidth="{width}" offheight="{height}"/></visual>'
        ),
        1,
    )
    xml = xml.replace(
        '<body name="link0" childclass="panda">',
        '<body name="link0" childclass="panda">\n      <freejoint name="base"/>',
        1,
    )
    xml = xml.replace(
        "<worldbody>",
        (
            "<worldbody>\n"
            '    <light name="l1" pos="0 0 -0.6" dir="0 0 1" '
            'diffuse="0.8 0.8 0.8"/>\n'
            '    <light name="l2" pos="0 -0.6 0.3" dir="0 1 0.4" '
            'diffuse="0.5 0.5 0.5"/>\n'
            f'    <camera name="ego" pos="0 0 0" xyaxes="1 0 0 0 -1 0" '
            f'fovy="{fovy_deg:.8f}"/>\n'
        ),
        1,
    )
    return mujoco.MjModel.from_xml_string(xml)


def _frame_from(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    first = np.asarray(primary, dtype=np.float64)
    first /= np.linalg.norm(first) + 1e-12
    second = np.asarray(secondary, dtype=np.float64)
    second -= np.dot(second, first) * first
    if np.linalg.norm(second) < 1e-6:
        reference = (
            np.array([1.0, 0.0, 0.0])
            if abs(first[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        second = reference - np.dot(reference, first) * first
    second /= np.linalg.norm(second) + 1e-12
    return np.column_stack([first, second, np.cross(first, second)])


def gravity_axes(camera_to_world_xyzw: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return gravity-aligned down/back/right axes in camera coordinates."""
    rotation = Rotation.from_quat(camera_to_world_xyzw)
    down = rotation.apply([0.0, 0.0, -1.0], inverse=True)
    down /= np.linalg.norm(down) + 1e-12
    image_down = np.array([0.0, 1.0, 0.0])
    back = image_down - np.dot(image_down, down) * down
    back /= np.linalg.norm(back) + 1e-12
    right = np.cross(back, down)
    right /= np.linalg.norm(right) + 1e-12
    if right[0] < 0.0:
        right = -right
    return down, back, right


class PandaIK:
    """One free-base Panda arm with DLS and optional SSIK solving."""

    def __init__(
        self,
        model: mujoco.MjModel,
        base_position: np.ndarray | None = None,
        aim_position: np.ndarray | None = None,
        up_world: tuple[float, float, float] = (0.0, -1.0, 0.0),
    ) -> None:
        self.model = model
        self.m = model
        self.data = mujoco.MjData(model)
        self.d = self.data
        self.hand_id = model.body("hand").id
        self.left_finger_id = model.body("left_finger").id
        self.right_finger_id = model.body("right_finger").id
        self.elbow_id = model.body("link4").id
        self.base_qadr = model.joint("base").qposadr[0]
        self.arm_qadr = np.asarray(
            [model.joint(name).qposadr[0] for name in ARM_JOINTS]
        )
        self.arm_dof = np.asarray([model.joint(name).dofadr[0] for name in ARM_JOINTS])
        self.finger_qadr = [
            model.joint("finger_joint1").qposadr[0],
            model.joint("finger_joint2").qposadr[0],
        ]
        self.ranges = np.asarray([model.joint(name).range for name in ARM_JOINTS])
        self.position_jacobian = np.zeros((3, model.nv))
        self.rotation_jacobian = np.zeros((3, model.nv))
        self.elbow_jacobian = np.zeros((3, model.nv))
        self._ssik = None
        self._ssik_link8_to_hand: np.ndarray | None = None
        self._ssik_seed = HOME_QPOS.copy()

        self.data.qpos[self.base_qadr : self.base_qadr + 3] = 0.0
        self.data.qpos[self.base_qadr + 3 : self.base_qadr + 7] = [1, 0, 0, 0]
        self.reset_arm()
        hand = self.data.body(self.hand_id)
        reach = hand.xpos.copy()
        self.reach_local = reach / (np.linalg.norm(reach) + 1e-12)
        self.up_local = np.array([0.0, 0.0, 1.0])
        left = self.data.body(self.left_finger_id)
        right = self.data.body(self.right_finger_id)
        opening_world = (
            left.xmat.reshape(3, 3) @ model.jnt_axis[model.joint("finger_joint1").id]
        )
        self.semantic_to_hand = _frame_from(
            np.array([0.0, 0.0, 1.0]),
            hand.xmat.reshape(3, 3).T @ opening_world,
        )
        self.tip_local = np.array([0.0, 0.0, 0.05])
        left_tip = left.xpos + left.xmat.reshape(3, 3) @ self.tip_local
        right_tip = right.xpos + right.xmat.reshape(3, 3) @ self.tip_local
        self.tip_offset_hand = hand.xmat.reshape(3, 3).T @ (
            0.5 * (left_tip + right_tip) - hand.xpos
        )
        if base_position is not None and aim_position is not None:
            self.set_base(
                base_position,
                aim_position,
                up_camera=np.asarray(up_world),
                reset_arm=True,
            )

    @staticmethod
    def _transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        return transform

    def reset_arm(self) -> None:
        self.data.qpos[self.arm_qadr] = HOME_QPOS
        self._ssik_seed = HOME_QPOS.copy()
        mujoco.mj_forward(self.model, self.data)

    def set_base(
        self,
        base_position: np.ndarray,
        aim_position: np.ndarray,
        *,
        up_camera: np.ndarray | None = None,
        up_world: np.ndarray | None = None,
        reset_arm: bool = True,
    ) -> None:
        if up_camera is None:
            up_camera = (
                np.asarray(up_world)
                if up_world is not None
                else np.array([0.0, -1.0, 0.0])
            )
        target = _frame_from(
            np.asarray(aim_position) - np.asarray(base_position), up_camera
        )
        source = _frame_from(self.reach_local, self.up_local)
        quaternion = Rotation.from_matrix(target @ source.T).as_quat()
        self.data.qpos[self.base_qadr : self.base_qadr + 3] = base_position
        self.data.qpos[self.base_qadr + 3 : self.base_qadr + 7] = [
            quaternion[3],
            quaternion[0],
            quaternion[1],
            quaternion[2],
        ]
        if reset_arm:
            self.reset_arm()
        else:
            mujoco.mj_forward(self.model, self.data)

    def target_rotation(self, semantic_rotation: np.ndarray) -> np.ndarray:
        return np.asarray(semantic_rotation) @ self.semantic_to_hand.T

    def fingertip_center(self) -> np.ndarray:
        hand = self.data.body(self.hand_id)
        return hand.xpos + hand.xmat.reshape(3, 3) @ self.tip_offset_hand

    def _init_ssik(self) -> None:
        if self._ssik is not None:
            return
        try:
            from ssik.prebuilt import franka_panda_ik
        except ImportError as exc:
            raise RuntimeError("SSIK is not installed") from exc
        mujoco.mj_forward(self.model, self.data)
        base = self.data.body("link0")
        hand = self.data.body(self.hand_id)
        world_base = self._transform(base.xpos, base.xmat.reshape(3, 3))
        world_hand = self._transform(hand.xpos, hand.xmat.reshape(3, 3))
        base_hand = np.linalg.inv(world_base) @ world_hand
        base_link8 = franka_panda_ik.fk(self.data.qpos[self.arm_qadr])
        self._ssik_link8_to_hand = np.linalg.inv(base_link8) @ base_hand
        self._ssik = franka_panda_ik

    def _ssik_candidate(
        self,
        position: np.ndarray,
        semantic_rotation: np.ndarray,
        *,
        q_seed: np.ndarray,
    ) -> np.ndarray | None:
        """Generate an SSIK candidate without mutating accepted arm state."""
        self._init_ssik()
        assert self._ssik_link8_to_hand is not None
        base = self.data.body("link0")
        world_base = self._transform(base.xpos, base.xmat.reshape(3, 3))
        target_rotation = self.target_rotation(semantic_rotation)
        hand_position = position - target_rotation @ self.tip_offset_hand
        world_hand = self._transform(hand_position, target_rotation)
        base_link8 = (
            np.linalg.inv(world_base)
            @ world_hand
            @ np.linalg.inv(self._ssik_link8_to_hand)
        )
        solutions = self._ssik.solve(
            base_link8,
            q_seed=np.asarray(q_seed, dtype=np.float64).copy(),
            max_solutions=1,
        )
        if not solutions:
            return None
        return np.asarray(solutions[0].q, dtype=np.float64).copy()

    def _commit_candidate(
        self,
        candidate: np.ndarray,
        *,
        position: np.ndarray,
        aperture: float,
    ) -> float:
        self.data.qpos[self.arm_qadr] = candidate
        self.data.qpos[self.finger_qadr] = float(np.clip(aperture / 2.0, 0.0, 0.04))
        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(position - self.fingertip_center()))

    def _restore_state(
        self,
        arm_q: np.ndarray,
        finger_q: np.ndarray,
        ssik_seed: np.ndarray,
    ) -> None:
        """Restore the last accepted solver state after a rejected attempt."""
        self.data.qpos[self.arm_qadr] = arm_q
        self.data.qpos[self.finger_qadr] = finger_q
        self._ssik_seed = ssik_seed.copy()
        mujoco.mj_forward(self.model, self.data)

    def solve_ssik(
        self,
        position: np.ndarray,
        semantic_rotation_or_approach: np.ndarray,
        aperture_or_opening: float | np.ndarray,
        aperture: float | None = None,
        *,
        previous_q: np.ndarray | None = None,
        max_joint_step_rad: float = DEFAULT_MAX_JOINT_STEP_RAD,
    ) -> float | None:
        """Compatibility SSIK solve with the same hard gate as production."""
        if aperture is None:
            semantic_rotation = np.asarray(semantic_rotation_or_approach)
            width = float(aperture_or_opening)
        else:
            semantic_rotation = _frame_from(
                np.asarray(semantic_rotation_or_approach),
                np.asarray(aperture_or_opening),
            )
            width = aperture
        accepted_arm = self.data.qpos[self.arm_qadr].copy()
        accepted_fingers = self.data.qpos[self.finger_qadr].copy()
        accepted_seed = self._ssik_seed.copy()
        reference = accepted_arm if previous_q is None else np.asarray(previous_q)
        candidate = self._ssik_candidate(
            position,
            semantic_rotation,
            q_seed=reference,
        )
        if candidate is None:
            return None
        try:
            validate_arm_candidate(
                candidate,
                previous_q=reference,
                joint_ranges=self.ranges,
                max_joint_step_rad=max_joint_step_rad,
            )
        except IKCandidateError:
            self._restore_state(accepted_arm, accepted_fingers, accepted_seed)
            return None
        try:
            residual = self._commit_candidate(
                candidate,
                position=position,
                aperture=width,
            )
        except Exception:
            self._restore_state(accepted_arm, accepted_fingers, accepted_seed)
            raise
        self._ssik_seed = candidate.copy()
        return residual

    def solve_target(
        self,
        position: np.ndarray,
        semantic_rotation: np.ndarray,
        aperture: float,
        *,
        previous_q: np.ndarray | None,
        elbow_outward: np.ndarray,
        backend: str = "dls",
        orientation_weight: float = 0.5,
        max_joint_step_rad: float = DEFAULT_MAX_JOINT_STEP_RAD,
        iterations: int = 160,
        damping: float = 0.2,
    ) -> IKSolveResult:
        """Solve one target through an acceptance gate shared by all backends."""
        if backend not in {"dls", "hybrid"}:
            raise ValueError("backend must be 'dls' or 'hybrid'")
        # Validate the configured limit even on the first frame, where there is
        # no previous_q against which to measure a transition.
        validate_arm_candidate(
            self.data.qpos[self.arm_qadr],
            previous_q=None,
            joint_ranges=self.ranges,
            max_joint_step_rad=max_joint_step_rad,
        )
        accepted_arm = self.data.qpos[self.arm_qadr].copy()
        accepted_fingers = self.data.qpos[self.finger_qadr].copy()
        accepted_seed = self._ssik_seed.copy()
        ssik_status = "not_attempted"

        if backend == "hybrid":
            try:
                seed = accepted_arm if previous_q is None else previous_q
                candidate = self._ssik_candidate(
                    position,
                    semantic_rotation,
                    q_seed=seed,
                )
            except RuntimeError as exc:
                candidate = None
                ssik_status = (
                    "unavailable" if str(exc) == "SSIK is not installed" else "error"
                )
            if candidate is None and ssik_status == "not_attempted":
                ssik_status = "no_solution"
            if candidate is not None:
                try:
                    joint_step = validate_arm_candidate(
                        candidate,
                        previous_q=previous_q,
                        joint_ranges=self.ranges,
                        max_joint_step_rad=max_joint_step_rad,
                    )
                except IKCandidateError as exc:
                    ssik_status = f"rejected_{exc.reason}"
                else:
                    try:
                        residual = self._commit_candidate(
                            candidate,
                            position=position,
                            aperture=aperture,
                        )
                    except Exception:
                        self._restore_state(
                            accepted_arm,
                            accepted_fingers,
                            accepted_seed,
                        )
                        raise
                    self._ssik_seed = candidate.copy()
                    return IKSolveResult(
                        residual_m=residual,
                        backend="ssik",
                        joint_step_rad=joint_step,
                        ssik_status="accepted",
                    )

            # Candidate generation and validation are side-effect free, but
            # restore both states explicitly so future refactors cannot leak a
            # rejected SSIK proposal into the DLS fallback.
            self._restore_state(accepted_arm, accepted_fingers, accepted_seed)

        try:
            residual = self.solve_dls(
                position,
                semantic_rotation,
                aperture,
                previous_q=previous_q,
                elbow_outward=elbow_outward,
                orientation_weight=orientation_weight,
                iterations=iterations,
                damping=damping,
                max_joint_step_rad=max_joint_step_rad,
            )
            candidate = self.data.qpos[self.arm_qadr].copy()
            joint_step = validate_arm_candidate(
                candidate,
                previous_q=previous_q,
                joint_ranges=self.ranges,
                max_joint_step_rad=max_joint_step_rad,
            )
        except Exception:
            self._restore_state(accepted_arm, accepted_fingers, accepted_seed)
            raise
        return IKSolveResult(
            residual_m=residual,
            backend="dls",
            joint_step_rad=joint_step,
            ssik_status=ssik_status,
        )

    def solve(
        self,
        position: np.ndarray,
        approach: np.ndarray,
        opening: np.ndarray,
        width: float,
        **kwargs: object,
    ) -> float:
        """Compatibility wrapper for the original debug viewer API."""
        return self.solve_dls(
            position,
            _frame_from(approach, opening),
            width,
            previous_q=kwargs.pop("q_ref", None),
            elbow_outward=np.asarray(
                kwargs.pop("elbow_outward", np.zeros(3)), dtype=np.float64
            ),
            orientation_weight=float(kwargs.pop("w_ori", 0.5)),
            iterations=int(kwargs.pop("iters", 160)),
            damping=float(kwargs.pop("lam", 0.2)),
            max_joint_step_rad=float(
                kwargs.pop("max_joint_step_rad", DEFAULT_MAX_JOINT_STEP_RAD)
            ),
        )

    def solve_from_home(
        self,
        position: np.ndarray,
        approach: np.ndarray,
        opening: np.ndarray,
        width: float,
        **kwargs: object,
    ) -> float:
        """Reset before the legacy feasibility solve."""
        self.reset_arm()
        return self.solve(position, approach, opening, width, **kwargs)

    def solve_dls(
        self,
        position: np.ndarray,
        semantic_rotation: np.ndarray,
        aperture: float,
        *,
        previous_q: np.ndarray | None,
        elbow_outward: np.ndarray,
        orientation_weight: float = 0.5,
        iterations: int = 160,
        damping: float = 0.2,
        max_joint_step_rad: float = DEFAULT_MAX_JOINT_STEP_RAD,
    ) -> float:
        """Solve the 6-DoF target with damped least squares."""
        if not np.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be positive and finite")
        target_rotation = self.target_rotation(semantic_rotation)
        self.data.qpos[self.finger_qadr] = float(np.clip(aperture / 2.0, 0.0, 0.04))
        for _ in range(iterations):
            mujoco.mj_forward(self.model, self.data)
            center = self.fingertip_center()
            current_rotation = self.data.body(self.hand_id).xmat.reshape(3, 3)
            position_error = position - center
            orientation_error = (
                orientation_weight
                * Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()
            )
            if (
                np.linalg.norm(position_error) < 1.5e-3
                and np.linalg.norm(orientation_error) < 0.02
            ):
                break
            mujoco.mj_jac(
                self.model,
                self.data,
                self.position_jacobian,
                self.rotation_jacobian,
                center,
                self.hand_id,
            )
            jacobian = np.vstack(
                [
                    self.position_jacobian[:, self.arm_dof],
                    orientation_weight * self.rotation_jacobian[:, self.arm_dof],
                ]
            )
            inverse = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), np.eye(6)
            )
            delta = inverse @ np.concatenate([position_error, orientation_error])
            secondary = np.zeros(7)
            if previous_q is not None:
                secondary += 0.5 * (previous_q - self.data.qpos[self.arm_qadr])
            mujoco.mj_jacBody(
                self.model,
                self.data,
                self.elbow_jacobian,
                None,
                self.elbow_id,
            )
            secondary += self.elbow_jacobian[:, self.arm_dof].T @ elbow_outward
            delta += 0.08 * ((np.eye(7) - inverse @ jacobian) @ secondary)
            lower, upper = self.ranges[:, 0], self.ranges[:, 1]
            if previous_q is not None:
                lower = np.maximum(lower, previous_q - max_joint_step_rad)
                upper = np.minimum(upper, previous_q + max_joint_step_rad)
            self.data.qpos[self.arm_qadr] = np.clip(
                self.data.qpos[self.arm_qadr] + delta, lower, upper
            )
        mujoco.mj_forward(self.model, self.data)
        validate_arm_candidate(
            self.data.qpos[self.arm_qadr],
            previous_q=previous_q,
            joint_ranges=self.ranges,
            max_joint_step_rad=max_joint_step_rad,
        )
        return float(np.linalg.norm(position - self.fingertip_center()))
