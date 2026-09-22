# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-resident SONIC prior with policy residuals for G1+Dex3."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import newton
import numpy as np
import warp as wp

from flash_chord.assets import ASSETS_DIR
from flash_chord.embodiments.g1_dex3 import G1_BODY_JOINT_NAMES, G1_DEX3_JOINT_NAMES
from flash_chord.runtime.delay import TargetDelayBuffer, TargetDelayConfig

if TYPE_CHECKING:
    from flash_chord.scene.builder import Scene


_POLICY_DIR = ASSETS_DIR / "policies" / "sonic"
_ENCODER_SHA256 = "6071c943492cb6c91e99b392200a288e829f5fea9491b530b34451fcf9e8d02a"
_DECODER_SHA256 = "de629511bc3a110a26827a54389e82d35a722ab9b7cc4232aab865b8451d2a18"

SONIC_ENCODER_DIM = 1762
SONIC_LATENT_DIM = 64
SONIC_JOINT_DIM = 29
SONIC_HISTORY_LENGTH = 10
SONIC_CURRENT_OBSERVATION_DIM = 93
SONIC_DECODER_HISTORY_DIM = 930
SONIC_DECODER_DIM = 994
RECON_BODY_ACTION_HISTORY_LENGTH = 3

# IsaacLab's ``Articulation.find_joints`` resolves the configured SONIC-name set
# in asset order. This is therefore the checkpoint tensor order, even though the
# selection constant in the upstream G1 config is grouped by limb.
SONIC_POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

SONIC_DIRECT_JOINT_NAMES = (
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
)

# IsaacLab creates one DelayedImplicitActuator per group below. Each actuator
# samples its own 0--2 physics-step lag on reset; Dex3 finger joints are not
# delayed. Group membership follows the upstream G1 asset configuration.
SONIC_ACTUATOR_DELAY_GROUPS = (
    (
        "left_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "right_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
    ),
    (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ),
    ("waist_roll_joint", "waist_pitch_joint"),
    ("waist_yaw_joint",),
    (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
)


@dataclass(frozen=True)
class SonicJointResidualActionConfig:
    """Pinned GEAR-SONIC prior plus ReconBody's body and Dex3 residual seam."""

    policy_dir: Path = _POLICY_DIR
    residual_joint_names: tuple[str, ...] = SONIC_POLICY_JOINT_NAMES
    residual_scale: float = 0.15
    finger_residual_scale: float = 0.15
    use_tanh: bool = False
    input_mode: Literal["raw"] = "raw"
    normalized_mapping: Literal["linear"] = "linear"
    body_delay: TargetDelayConfig = field(default_factory=lambda: TargetDelayConfig(min_steps=0, max_steps=2))
    enable_cuda_graph: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_dir", Path(self.policy_dir))
        object.__setattr__(self, "residual_joint_names", tuple(self.residual_joint_names))
        expected = SONIC_POLICY_JOINT_NAMES
        if self.residual_joint_names != expected:
            raise ValueError(
                "the pinned ReconBody contract requires all SONIC body residual joints "
                f"{expected}, got {self.residual_joint_names}"
            )
        for name in ("residual_scale", "finger_residual_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if self.input_mode != "raw" or self.normalized_mapping != "linear":
            raise ValueError("SONIC residuals require raw identity input with linear metadata")

    def build(self, scene: Scene, device=None) -> SonicJointResidualAction:
        return SonicJointResidualAction.build(scene, config=self, device=device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized Hamilton product for xyzw quaternions."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def _quat_to_six_dimensional_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return the first two rotation-matrix columns in upstream row-major flatten order."""
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    matrix = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))
    return matrix[..., :2].reshape(quaternion.shape[:-1] + (6,))


def build_sonic_encoder_reference(scene: Scene) -> np.ndarray:
    """Precompute the immutable 1762-float tokenizer input for every reference frame."""
    reference = scene.robot_reference
    scalar = scene.layout.scalar_joints
    if scalar is None:
        raise ValueError("SONIC requires an embodiment with named scalar joints")
    name_to_q = dict(zip(scalar.names, scalar.q_ids, strict=True))
    missing = tuple(name for name in SONIC_POLICY_JOINT_NAMES if name not in name_to_q)
    if missing:
        raise ValueError(f"SONIC reference is missing G1 body joints: {missing}")
    sonic_q_ids = np.asarray([name_to_q[name] for name in SONIC_POLICY_JOINT_NAMES], dtype=np.int64)

    step_float = 0.1 * reference.fps
    future_step = round(step_float)
    if future_step <= 0 or not np.isclose(step_float, future_step, atol=1.0e-6):
        raise ValueError(f"SONIC's 0.1 s future spacing is not integral at reference fps {reference.fps:g}")
    offsets = np.arange(SONIC_HISTORY_LENGTH, dtype=np.int64) * future_step
    frames = np.arange(reference.num_frames, dtype=np.int64)[:, None]
    future = np.minimum(frames + offsets[None, :], reference.num_frames - 1)

    joint_position = np.asarray(reference.joint_q[:, sonic_q_ids], dtype=np.float32)
    joint_velocity = np.zeros_like(joint_position)
    joint_velocity[:-1] = (joint_position[1:] - joint_position[:-1]) * np.float32(reference.fps)
    future_position = joint_position[future].reshape(reference.num_frames, -1)
    future_velocity = joint_velocity[future].reshape(reference.num_frames, -1)

    root_quaternion = np.asarray(reference.joint_q[:, 3:7], dtype=np.float64)
    root_quaternion /= np.linalg.norm(root_quaternion, axis=1, keepdims=True)
    current_inverse = root_quaternion.copy()
    current_inverse[:, :3] *= -1.0
    relative = _quat_mul_xyzw(root_quaternion[future], current_inverse[:, None, :])
    relative /= np.linalg.norm(relative, axis=-1, keepdims=True)
    future_orientation = _quat_to_six_dimensional_xyzw(relative).reshape(reference.num_frames, -1)

    encoder = np.zeros((reference.num_frames, SONIC_ENCODER_DIM), dtype=np.float32)
    cursor = 4
    encoder[:, cursor : cursor + 290] = future_position
    cursor += 290
    encoder[:, cursor : cursor + 290] = future_velocity
    cursor += 290 + 17
    encoder[:, cursor : cursor + 60] = future_orientation
    return np.ascontiguousarray(encoder)


@wp.kernel
def gather_sonic_encoder_input(
    timestep: wp.array(dtype=wp.int32),
    reference: wp.array(dtype=wp.float32, ndim=2),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
    sonic_q_ids: wp.array(dtype=wp.int32),
    num_frames: int,
    q_per_world: int,
    encoder_input: wp.array(dtype=wp.float32, ndim=2),
) -> None:
    world, column = wp.tid()
    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    value = reference[frame, column]
    if column >= 4 and column < 4 + SONIC_HISTORY_LENGTH * SONIC_JOINT_DIM:
        joint = (column - 4) % SONIC_JOINT_DIM
        value += reference_joint_q_offset[world * q_per_world + sonic_q_ids[joint]]
    encoder_input[world, column] = value


@wp.kernel
def compute_sonic_current_observation(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    bodies_per_world: int,
    q_per_world: int,
    dof_per_world: int,
    root_body_id: int,
    sonic_q_ids: wp.array(dtype=wp.int32),
    sonic_dof_ids: wp.array(dtype=wp.int32),
    default_joint_position: wp.array(dtype=wp.float32),
    last_sonic_action: wp.array(dtype=wp.float32, ndim=2),
    current_observation: wp.array(dtype=wp.float32, ndim=2),
) -> None:
    world = wp.tid()
    root_xf = body_q[world * bodies_per_world + root_body_id]
    root_inverse = wp.transform_inverse(root_xf)
    root_velocity = body_qd[world * bodies_per_world + root_body_id]
    angular_velocity_b = wp.transform_vector(root_inverse, wp.spatial_bottom(root_velocity))
    gravity_b = wp.transform_vector(root_inverse, wp.vec3(0.0, 0.0, -1.0))
    for axis in range(3):
        current_observation[world, axis] = angular_velocity_b[axis]
        current_observation[world, 90 + axis] = gravity_b[axis]

    q_base = world * q_per_world
    dof_base = world * dof_per_world
    for joint in range(SONIC_JOINT_DIM):
        current_observation[world, 3 + joint] = joint_q[q_base + sonic_q_ids[joint]] - default_joint_position[joint]
        current_observation[world, 32 + joint] = joint_qd[dof_base + sonic_dof_ids[joint]]
        current_observation[world, 61 + joint] = last_sonic_action[world, joint]


@wp.kernel
def append_sonic_history(
    current_observation: wp.array(dtype=wp.float32, ndim=2),
    history: wp.array(dtype=wp.float32, ndim=3),
    history_head: wp.array(dtype=wp.int32),
    history_initialized: wp.array(dtype=wp.int32),
) -> None:
    world = wp.tid()
    if history_initialized[world] == 0:
        for slot in range(SONIC_HISTORY_LENGTH):
            for column in range(SONIC_CURRENT_OBSERVATION_DIM):
                history[world, slot, column] = current_observation[world, column]
        history_head[world] = SONIC_HISTORY_LENGTH - 1
        history_initialized[world] = 1
        return

    head = (history_head[world] + 1) % SONIC_HISTORY_LENGTH
    for column in range(SONIC_CURRENT_OBSERVATION_DIM):
        history[world, head, column] = current_observation[world, column]
    history_head[world] = head


@wp.kernel
def pack_sonic_decoder_input(
    latent: wp.array(dtype=wp.float32, ndim=2),
    history: wp.array(dtype=wp.float32, ndim=3),
    history_head: wp.array(dtype=wp.int32),
    decoder_input: wp.array(dtype=wp.float32, ndim=2),
) -> None:
    world = wp.tid()
    for column in range(SONIC_LATENT_DIM):
        decoder_input[world, column] = latent[world, column]

    oldest = (history_head[world] + 1) % SONIC_HISTORY_LENGTH
    output = SONIC_LATENT_DIM
    for age in range(SONIC_HISTORY_LENGTH):
        slot = (oldest + age) % SONIC_HISTORY_LENGTH
        for axis in range(3):
            decoder_input[world, output + age * 3 + axis] = history[world, slot, axis]
    output += 30
    for age in range(SONIC_HISTORY_LENGTH):
        slot = (oldest + age) % SONIC_HISTORY_LENGTH
        for joint in range(SONIC_JOINT_DIM):
            decoder_input[world, output + age * SONIC_JOINT_DIM + joint] = history[world, slot, 3 + joint]
    output += 290
    for age in range(SONIC_HISTORY_LENGTH):
        slot = (oldest + age) % SONIC_HISTORY_LENGTH
        for joint in range(SONIC_JOINT_DIM):
            decoder_input[world, output + age * SONIC_JOINT_DIM + joint] = history[world, slot, 32 + joint]
    output += 290
    for age in range(SONIC_HISTORY_LENGTH):
        slot = (oldest + age) % SONIC_HISTORY_LENGTH
        for joint in range(SONIC_JOINT_DIM):
            decoder_input[world, output + age * SONIC_JOINT_DIM + joint] = history[world, slot, 61 + joint]
    output += 290
    for age in range(SONIC_HISTORY_LENGTH):
        slot = (oldest + age) % SONIC_HISTORY_LENGTH
        for axis in range(3):
            decoder_input[world, output + age * 3 + axis] = history[world, slot, 90 + axis]


@wp.kernel
def process_sonic_joint_targets(
    action: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    decoder_action: wp.array(dtype=wp.float32, ndim=2),
    reference_joint_target: wp.array(dtype=wp.float32, ndim=2),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
    reference_num_frames: int,
    dof_per_world: int,
    q_per_world: int,
    action_dim: int,
    processed_dim: int,
    num_body_residuals: int,
    num_fingers: int,
    sonic_dof_ids: wp.array(dtype=wp.int32),
    sonic_scalar_slots: wp.array(dtype=wp.int32),
    residual_slot_by_sonic: wp.array(dtype=wp.int32),
    finger_q_ids: wp.array(dtype=wp.int32),
    finger_dof_ids: wp.array(dtype=wp.int32),
    finger_scalar_slots: wp.array(dtype=wp.int32),
    sonic_scale: wp.array(dtype=wp.float32),
    default_sonic_position: wp.array(dtype=wp.float32),
    residual_scale: float,
    finger_residual_scale: float,
    use_tanh: bool,
    raw_action: wp.array(dtype=wp.float32),
    processed_target: wp.array(dtype=wp.float32),
    last_sonic_action: wp.array(dtype=wp.float32, ndim=2),
    joint_target: wp.array(dtype=wp.float32),
    action_l2: wp.array(dtype=wp.float32),
    action_rate_l2: wp.array(dtype=wp.float32),
    actor_action_history: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    action_base = world * action_dim
    processed_base = world * processed_dim
    dof_base = world * dof_per_world
    l2 = float(0.0)  # noqa: UP018 -- explicit Warp scalar type
    rate_l2 = float(0.0)  # noqa: UP018 -- explicit Warp scalar type
    for index in range(action_dim):
        value = action[action_base + index]
        previous = raw_action[action_base + index]
        raw_action[action_base + index] = value
        l2 += value * value
        delta = value - previous
        rate_l2 += delta * delta

    for sonic_joint in range(SONIC_JOINT_DIM):
        decoded = decoder_action[world, sonic_joint]
        residual_slot = residual_slot_by_sonic[sonic_joint]
        residual = float(0.0)  # noqa: UP018 -- explicit Warp scalar type
        if residual_slot >= 0:
            residual = action[action_base + residual_slot]
            if use_tanh:
                residual = wp.tanh(residual)
        target = (decoded + residual_scale * residual) * sonic_scale[sonic_joint]
        target += default_sonic_position[sonic_joint]
        joint_target[dof_base + sonic_dof_ids[sonic_joint]] = target
        processed_target[processed_base + sonic_scalar_slots[sonic_joint]] = target
        last_sonic_action[world, sonic_joint] = decoded

    frame = wp.clamp(timestep[world], 0, reference_num_frames - 1)
    for finger in range(num_fingers):
        residual = action[action_base + num_body_residuals + finger]
        if use_tanh:
            residual = wp.tanh(residual)
        dof_id = finger_dof_ids[finger]
        target = reference_joint_target[frame, dof_id]
        target += reference_joint_q_offset[world * q_per_world + finger_q_ids[finger]]
        target += finger_residual_scale * residual
        joint_target[dof_base + dof_id] = target
        processed_target[processed_base + finger_scalar_slots[finger]] = target
    action_l2[world] = l2
    action_rate_l2[world] = rate_l2
    history_base = world * RECON_BODY_ACTION_HISTORY_LENGTH * processed_dim
    for slot in range(RECON_BODY_ACTION_HISTORY_LENGTH - 1):
        for joint in range(processed_dim):
            actor_action_history[history_base + slot * processed_dim + joint] = actor_action_history[
                history_base + (slot + 1) * processed_dim + joint
            ]
    newest = history_base + (RECON_BODY_ACTION_HISTORY_LENGTH - 1) * processed_dim
    for joint in range(processed_dim):
        actor_action_history[newest + joint] = processed_target[processed_base + joint]


@wp.kernel
def reset_sonic_action(
    reset_mask: wp.array(dtype=wp.int32),
    action_dim: int,
    processed_dim: int,
    dof_per_world: int,
    num_fingers: int,
    sonic_dof_ids: wp.array(dtype=wp.int32),
    sonic_scalar_slots: wp.array(dtype=wp.int32),
    finger_dof_ids: wp.array(dtype=wp.int32),
    finger_scalar_slots: wp.array(dtype=wp.int32),
    default_sonic_position: wp.array(dtype=wp.float32),
    default_finger_position: wp.array(dtype=wp.float32),
    raw_action: wp.array(dtype=wp.float32),
    processed_target: wp.array(dtype=wp.float32),
    last_sonic_action: wp.array(dtype=wp.float32, ndim=2),
    history_head: wp.array(dtype=wp.int32),
    history_initialized: wp.array(dtype=wp.int32),
    joint_target: wp.array(dtype=wp.float32),
    action_l2: wp.array(dtype=wp.float32),
    action_rate_l2: wp.array(dtype=wp.float32),
    actor_action_history: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    action_base = world * action_dim
    processed_base = world * processed_dim
    dof_base = world * dof_per_world
    for index in range(action_dim):
        raw_action[action_base + index] = 0.0
    for joint in range(SONIC_JOINT_DIM):
        value = default_sonic_position[joint]
        last_sonic_action[world, joint] = value
        joint_target[dof_base + sonic_dof_ids[joint]] = value
        processed_target[processed_base + sonic_scalar_slots[joint]] = value
    for finger in range(num_fingers):
        value = default_finger_position[finger]
        joint_target[dof_base + finger_dof_ids[finger]] = value
        processed_target[processed_base + finger_scalar_slots[finger]] = value
    history_head[world] = SONIC_HISTORY_LENGTH - 1
    history_initialized[world] = 0
    action_l2[world] = 0.0
    action_rate_l2[world] = 0.0
    history_base = world * RECON_BODY_ACTION_HISTORY_LENGTH * processed_dim
    for index in range(RECON_BODY_ACTION_HISTORY_LENGTH * processed_dim):
        actor_action_history[history_base + index] = 0.0


@dataclass
class _OrtBoundSession:
    session: object
    io_binding: object
    provider: str

    @classmethod
    def build(
        cls,
        path: Path,
        input_array: wp.array,
        output_array: wp.array,
        *,
        expected_input_name: str,
        expected_output_name: str,
        device,
        enable_cuda_graph: bool,
    ) -> _OrtBoundSession:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("SONIC requires the pinned onnxruntime-gpu dependency") from error

        if device.is_cuda:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("SONIC was requested on CUDA, but ONNX Runtime has no CUDA execution provider")
            stream = wp.get_stream(device)
            provider_options = {
                "device_id": str(device.ordinal),
                "user_compute_stream": str(stream.cuda_stream),
                "use_ep_level_unified_stream": "1",
                "enable_cuda_graph": "1" if enable_cuda_graph else "0",
            }
            providers = [("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"]
            memory_device = "cuda"
            memory_device_id = device.ordinal
        else:
            providers = ["CPUExecutionProvider"]
            memory_device = "cpu"
            memory_device_id = 0

        session = ort.InferenceSession(str(path), providers=providers)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].name != expected_input_name
            or tuple(inputs[0].shape[1:]) != input_array.shape[1:]
        ):
            raise ValueError(
                f"unexpected SONIC input contract for {path.name}: "
                f"{[(value.name, value.shape, value.type) for value in inputs]}"
            )
        if (
            len(outputs) != 1
            or outputs[0].name != expected_output_name
            or tuple(outputs[0].shape[1:]) != output_array.shape[1:]
        ):
            raise ValueError(
                f"unexpected SONIC output contract for {path.name}: "
                f"{[(value.name, value.shape, value.type) for value in outputs]}"
            )
        binding = session.io_binding()
        binding.bind_input(
            expected_input_name,
            memory_device,
            memory_device_id,
            np.float32,
            input_array.shape,
            int(input_array.ptr),
        )
        binding.bind_output(
            expected_output_name,
            memory_device,
            memory_device_id,
            np.float32,
            output_array.shape,
            int(output_array.ptr),
        )
        bound = cls(session=session, io_binding=binding, provider=session.get_providers()[0])
        bound.run()
        if device.is_cuda and enable_cuda_graph:
            bound.run()
        return bound

    def run(self) -> None:
        self.session.run_with_iobinding(self.io_binding)


@dataclass
class SonicJointResidualAction:
    """Zero-copy ONNX Runtime SONIC inference with Warp observation/action kernels."""

    world_count: int
    q_per_world: int
    dof_per_world: int
    bodies_per_world: int
    root_body_id: int
    reference_num_frames: int
    residual_joint_names: tuple[str, ...]
    finger_joint_names: tuple[str, ...]
    sonic_q_ids: wp.array
    sonic_dof_ids: wp.array
    sonic_scalar_slots: wp.array
    residual_slot_by_sonic: wp.array
    finger_q_ids: wp.array
    finger_dof_ids: wp.array
    finger_scalar_slots: wp.array
    sonic_scale: wp.array
    default_sonic_position: wp.array
    default_finger_position: wp.array
    residual_scale: float
    finger_residual_scale: float
    use_tanh: bool
    encoder_reference: wp.array
    reference_joint_target: wp.array
    reference_joint_q_offset: wp.array
    encoder_input: wp.array
    latent: wp.array
    current_observation: wp.array
    history: wp.array
    history_head: wp.array
    history_initialized: wp.array
    decoder_input: wp.array
    decoder_action: wp.array
    raw_action: wp.array
    processed_target: wp.array
    last_sonic_action: wp.array
    joint_target: wp.array
    action_l2: wp.array
    action_rate_l2: wp.array
    actor_action_history: wp.array
    delays: tuple[TargetDelayBuffer, ...]
    encoder_session: _OrtBoundSession
    decoder_session: _OrtBoundSession

    @classmethod
    def build(
        cls,
        scene: Scene,
        config: SonicJointResidualActionConfig | None = None,
        device=None,
    ) -> SonicJointResidualAction:
        config = config or SonicJointResidualActionConfig()
        device = wp.get_device(device)
        scalar = scene.layout.scalar_joints
        if scalar is None:
            raise ValueError("SONIC requires the G1 named scalar-joint layout")
        if set(scalar.names) != set(G1_BODY_JOINT_NAMES + G1_DEX3_JOINT_NAMES):
            raise ValueError("SONIC requires the exact 29-body/14-Dex3 G1 joint contract")
        scalar_slot = {name: index for index, name in enumerate(scalar.names)}
        q_by_name = dict(zip(scalar.names, scalar.q_ids, strict=True))
        dof_by_name = dict(zip(scalar.names, scalar.dof_ids, strict=True))
        sonic_q_ids = tuple(q_by_name[name] for name in SONIC_POLICY_JOINT_NAMES)
        sonic_dof_ids = tuple(dof_by_name[name] for name in SONIC_POLICY_JOINT_NAMES)
        sonic_scalar_slots = tuple(scalar_slot[name] for name in SONIC_POLICY_JOINT_NAMES)
        finger_dof_ids = tuple(dof_by_name[name] for name in SONIC_DIRECT_JOINT_NAMES)
        finger_q_ids = tuple(q_by_name[name] for name in SONIC_DIRECT_JOINT_NAMES)
        finger_scalar_slots = tuple(scalar_slot[name] for name in SONIC_DIRECT_JOINT_NAMES)
        residual_slot = {name: index for index, name in enumerate(config.residual_joint_names)}
        residual_slot_by_sonic = tuple(residual_slot.get(name, -1) for name in SONIC_POLICY_JOINT_NAMES)

        q_per_world = scene.model.joint_coord_count // scene.world_count
        dof_per_world = scene.model.joint_dof_count // scene.world_count
        bodies_per_world = scene.model.body_count // scene.world_count
        if q_per_world < scene.layout.num_joint_q or dof_per_world < scene.layout.num_joint_dof:
            raise ValueError("scene per-world joint dimensions are smaller than the G1 layout")
        pelvis = tuple(frame for frame in scene.layout.semantic_frames if frame.name == "pelvis")
        if len(pelvis) != 1 or pelvis[0].body_to_frame_pos != (0.0, 0.0, 0.0):
            raise ValueError("SONIC requires one identity pelvis semantic frame")

        target_mode = np.asarray(scene.model.joint_target_mode.numpy(), dtype=np.int32)[:dof_per_world]
        allowed_modes = {int(newton.JointTargetMode.POSITION), int(newton.JointTargetMode.POSITION_VELOCITY)}
        invalid = tuple(name for name in scalar.names if int(target_mode[dof_by_name[name]]) not in allowed_modes)
        if invalid:
            raise ValueError(f"SONIC requires position-capable scalar joint drives, got {invalid}")

        model_default = np.asarray(scene.model.joint_target_pos.numpy(), dtype=np.float32)[:dof_per_world]
        model_stiffness = np.asarray(scene.model.joint_target_ke.numpy(), dtype=np.float32)[:dof_per_world]
        model_effort = np.asarray(scene.model.joint_effort_limit.numpy(), dtype=np.float32)[:dof_per_world]
        sonic_stiffness = model_stiffness[list(sonic_dof_ids)]
        if np.any(~np.isfinite(sonic_stiffness)) or np.any(sonic_stiffness <= 0.0):
            raise ValueError("SONIC body joints require positive finite stiffness")
        sonic_scale = 0.25 * model_effort[list(sonic_dof_ids)] / sonic_stiffness
        if np.any(~np.isfinite(sonic_scale)) or np.any(sonic_scale <= 0.0):
            raise ValueError("SONIC body action scales must be positive and finite")
        default_sonic = model_default[list(sonic_dof_ids)]
        default_finger = model_default[list(finger_dof_ids)]

        encoder_path = config.policy_dir / "encoder_batched.onnx"
        decoder_path = config.policy_dir / "decoder_batched.onnx"
        expected_files = ((encoder_path, _ENCODER_SHA256), (decoder_path, _DECODER_SHA256))
        for path, expected_sha256 in expected_files:
            if not path.is_file():
                raise FileNotFoundError(f"SONIC policy file does not exist: {path}")
            actual_sha256 = _sha256(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"SONIC policy SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual_sha256}"
                )

        world_count = scene.world_count
        action_dim = len(config.residual_joint_names) + len(SONIC_DIRECT_JOINT_NAMES)
        processed_dim = len(scalar.names)
        encoder_input = wp.zeros((world_count, SONIC_ENCODER_DIM), dtype=wp.float32, device=device)
        latent = wp.zeros((world_count, SONIC_LATENT_DIM), dtype=wp.float32, device=device)
        decoder_input = wp.zeros((world_count, SONIC_DECODER_DIM), dtype=wp.float32, device=device)
        decoder_action = wp.zeros((world_count, SONIC_JOINT_DIM), dtype=wp.float32, device=device)
        encoder_session = _OrtBoundSession.build(
            encoder_path,
            encoder_input,
            latent,
            expected_input_name="obs_dict",
            expected_output_name="encoded_tokens",
            device=device,
            enable_cuda_graph=config.enable_cuda_graph,
        )
        decoder_session = _OrtBoundSession.build(
            decoder_path,
            decoder_input,
            decoder_action,
            expected_input_name="obs_dict",
            expected_output_name="action",
            device=device,
            enable_cuda_graph=config.enable_cuda_graph,
        )

        delays = ()
        if config.body_delay.max_steps > 0:
            delays = tuple(
                TargetDelayBuffer.build(
                    world_count=world_count,
                    num_joint_dof=dof_per_world,
                    delayed_dof_ids=tuple(dof_by_name[name] for name in group),
                    config=replace(config.body_delay, seed=config.body_delay.seed + group_index),
                    device=device,
                )
                for group_index, group in enumerate(SONIC_ACTUATOR_DELAY_GROUPS)
            )
        return cls(
            world_count=world_count,
            q_per_world=q_per_world,
            dof_per_world=dof_per_world,
            bodies_per_world=bodies_per_world,
            root_body_id=pelvis[0].body_id,
            reference_num_frames=scene.robot_reference.num_frames,
            residual_joint_names=config.residual_joint_names,
            finger_joint_names=SONIC_DIRECT_JOINT_NAMES,
            sonic_q_ids=wp.array(sonic_q_ids, dtype=wp.int32, device=device),
            sonic_dof_ids=wp.array(sonic_dof_ids, dtype=wp.int32, device=device),
            sonic_scalar_slots=wp.array(sonic_scalar_slots, dtype=wp.int32, device=device),
            residual_slot_by_sonic=wp.array(residual_slot_by_sonic, dtype=wp.int32, device=device),
            finger_q_ids=wp.array(finger_q_ids, dtype=wp.int32, device=device),
            finger_dof_ids=wp.array(finger_dof_ids, dtype=wp.int32, device=device),
            finger_scalar_slots=wp.array(finger_scalar_slots, dtype=wp.int32, device=device),
            sonic_scale=wp.array(sonic_scale, dtype=wp.float32, device=device),
            default_sonic_position=wp.array(default_sonic, dtype=wp.float32, device=device),
            default_finger_position=wp.array(default_finger, dtype=wp.float32, device=device),
            residual_scale=config.residual_scale,
            finger_residual_scale=config.finger_residual_scale,
            use_tanh=config.use_tanh,
            encoder_reference=wp.array(build_sonic_encoder_reference(scene), dtype=wp.float32, device=device),
            reference_joint_target=wp.array(scene.robot_reference.joint_target, dtype=wp.float32, device=device),
            reference_joint_q_offset=wp.zeros(
                world_count * q_per_world,
                dtype=wp.float32,
                device=device,
            ),
            encoder_input=encoder_input,
            latent=latent,
            current_observation=wp.zeros((world_count, SONIC_CURRENT_OBSERVATION_DIM), dtype=wp.float32, device=device),
            history=wp.zeros(
                (world_count, SONIC_HISTORY_LENGTH, SONIC_CURRENT_OBSERVATION_DIM),
                dtype=wp.float32,
                device=device,
            ),
            history_head=wp.full(world_count, SONIC_HISTORY_LENGTH - 1, dtype=wp.int32, device=device),
            history_initialized=wp.zeros(world_count, dtype=wp.int32, device=device),
            decoder_input=decoder_input,
            decoder_action=decoder_action,
            raw_action=wp.zeros(world_count * action_dim, dtype=wp.float32, device=device),
            processed_target=wp.zeros(world_count * processed_dim, dtype=wp.float32, device=device),
            last_sonic_action=wp.zeros((world_count, SONIC_JOINT_DIM), dtype=wp.float32, device=device),
            joint_target=wp.zeros(world_count * dof_per_world, dtype=wp.float32, device=device),
            action_l2=wp.zeros(world_count, dtype=wp.float32, device=device),
            action_rate_l2=wp.zeros(world_count, dtype=wp.float32, device=device),
            actor_action_history=wp.zeros(
                world_count * RECON_BODY_ACTION_HISTORY_LENGTH * processed_dim,
                dtype=wp.float32,
                device=device,
            ),
            delays=delays,
            encoder_session=encoder_session,
            decoder_session=decoder_session,
        )

    @property
    def action_dim(self) -> int:
        return len(self.residual_joint_names) + len(self.finger_joint_names)

    @property
    def processed_dim(self) -> int:
        return SONIC_JOINT_DIM + len(self.finger_joint_names)

    @property
    def sides(self) -> tuple[str, ...]:
        return ("left", "right")

    @property
    def input_mode(self) -> Literal["raw"]:
        return "raw"

    @property
    def normalized_mapping(self) -> Literal["linear"]:
        return "linear"

    @property
    def input_mapping(self) -> Literal["identity"]:
        return "identity"

    @property
    def input_scale_values(self) -> tuple[float, ...]:
        return (1.0,) * self.action_dim

    @property
    def block_names(self) -> tuple[str, ...]:
        return (
            f"sonic_body_joint_residual[{','.join(self.residual_joint_names)}]",
            f"dex3_direct_joint_residual[{','.join(self.finger_joint_names)}]",
        )

    @property
    def block_ranges(self) -> tuple[tuple[int, int], ...]:
        body = len(self.residual_joint_names)
        return ((0, body), (body, body + len(self.finger_joint_names)))

    def process(self, action: wp.array, timestep: wp.array, state=None) -> None:
        if state is None:
            raise ValueError("SONIC action processing requires the current Newton state")
        expected = self.world_count * self.action_dim
        if action.shape != (expected,):
            raise ValueError(f"SONIC action has shape {action.shape}; expected ({expected},)")
        wp.launch(
            gather_sonic_encoder_input,
            dim=(self.world_count, SONIC_ENCODER_DIM),
            inputs=[
                timestep,
                self.encoder_reference,
                self.reference_joint_q_offset,
                self.sonic_q_ids,
                self.reference_num_frames,
                self.q_per_world,
            ],
            outputs=[self.encoder_input],
        )
        self.encoder_session.run()
        wp.launch(
            compute_sonic_current_observation,
            dim=self.world_count,
            inputs=[
                state.body_q,
                state.body_qd,
                state.joint_q,
                state.joint_qd,
                self.bodies_per_world,
                self.q_per_world,
                self.dof_per_world,
                self.root_body_id,
                self.sonic_q_ids,
                self.sonic_dof_ids,
                self.default_sonic_position,
                self.last_sonic_action,
            ],
            outputs=[self.current_observation],
        )
        wp.launch(
            append_sonic_history,
            dim=self.world_count,
            inputs=[self.current_observation],
            outputs=[self.history, self.history_head, self.history_initialized],
        )
        wp.launch(
            pack_sonic_decoder_input,
            dim=self.world_count,
            inputs=[self.latent, self.history, self.history_head],
            outputs=[self.decoder_input],
        )
        self.decoder_session.run()
        wp.launch(
            process_sonic_joint_targets,
            dim=self.world_count,
            inputs=[
                action,
                timestep,
                self.decoder_action,
                self.reference_joint_target,
                self.reference_joint_q_offset,
                self.reference_num_frames,
                self.dof_per_world,
                self.q_per_world,
                self.action_dim,
                self.processed_dim,
                len(self.residual_joint_names),
                len(self.finger_joint_names),
                self.sonic_dof_ids,
                self.sonic_scalar_slots,
                self.residual_slot_by_sonic,
                self.finger_q_ids,
                self.finger_dof_ids,
                self.finger_scalar_slots,
                self.sonic_scale,
                self.default_sonic_position,
                self.residual_scale,
                self.finger_residual_scale,
                self.use_tanh,
            ],
            outputs=[
                self.raw_action,
                self.processed_target,
                self.last_sonic_action,
                self.joint_target,
                self.action_l2,
                self.action_rate_l2,
                self.actor_action_history,
            ],
        )

    def bind_reference_joint_q_offset(self, offset: wp.array) -> None:
        """Share the environment's per-world reference offset with SONIC conditioning."""
        expected = (self.world_count * self.q_per_world,)
        if offset.shape != expected or offset.dtype != wp.float32 or offset.device != self.joint_target.device:
            raise ValueError(
                f"SONIC reference offset must be float32 {expected} on {self.joint_target.device}, "
                f"got shape={offset.shape}, dtype={offset.dtype}, device={offset.device}"
            )
        self.reference_joint_q_offset = offset

    def prepare_control(self, control) -> None:
        wp.copy(control.joint_target_pos, self.joint_target)

    def apply_control(self, state, control) -> None:
        for delay in self.delays:
            delay.advance_into(self.joint_target, control.joint_target_pos)

    def reset(self, reset_mask: wp.array) -> None:
        wp.launch(
            reset_sonic_action,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.action_dim,
                self.processed_dim,
                self.dof_per_world,
                len(self.finger_joint_names),
                self.sonic_dof_ids,
                self.sonic_scalar_slots,
                self.finger_dof_ids,
                self.finger_scalar_slots,
                self.default_sonic_position,
                self.default_finger_position,
            ],
            outputs=[
                self.raw_action,
                self.processed_target,
                self.last_sonic_action,
                self.history_head,
                self.history_initialized,
                self.joint_target,
                self.action_l2,
                self.action_rate_l2,
                self.actor_action_history,
            ],
        )
        for delay in self.delays:
            delay.reset(reset_mask)
