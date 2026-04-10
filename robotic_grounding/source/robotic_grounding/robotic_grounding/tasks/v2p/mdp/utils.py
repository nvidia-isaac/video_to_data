# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Utility functions for the V2P environment."""

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from robotic_grounding.retarget.data_logger import ManoSharpaData

##########################################################
# Interpolation
##########################################################


def interpolate_robot_motion_data(
    motion_data: ManoSharpaData,
    target_num_frames: float,
) -> ManoSharpaData:
    """Interpolate the robot motion data to the target FPS.

    Now only supports interpolation of the following fields:
    - object_body_position -> linear interpolation
    - object_body_wxyz -> Slerp
    - object_articulation -> linear interpolation
    - robot_right_wrist_position -> linear interpolation
    - robot_right_wrist_wxyz -> Slerp interpolation
    - robot_right_finger_joints -> linear interpolation
    - robot_left_wrist_position -> linear interpolation
    - robot_left_wrist_wxyz -> Slerp interpolation
    - robot_left_finger_joints -> linear interpolation
    - robot_right_frames -> linear interpolation for position, Slerp for orientation
    - robot_left_frames -> linear interpolation for position, Slerp for orientation
    - mano_*_link_contact_positions / mano_*_object_contact_positions -> linear
      between non-zero values only; frames between 0 and non-zero are set to 0
    TODO: we may need to improve this function for more general cases.

    Args:
        motion_data: The motion data to interpolate.
        target_num_frames: The target number of frames to interpolate to.

    Returns:
        The interpolated motion data.
    """

    # --- Helper Functions ---
    def get_timestamps() -> tuple[np.ndarray, np.ndarray]:
        """Get the timestamps for the motion data."""
        n_frames = len(motion_data.robot_right_wrist_position)
        src = np.arange(n_frames) / motion_data.fps
        tgt = np.linspace(0, src[-1], int(src[-1] * target_num_frames))
        return src, tgt

    def interp_linear(data: list[float]) -> list[float]:
        """Vectorized linear interpolation for any shape (T, ...)."""
        data_arr = np.asarray(data)
        return interp1d(src_times, data_arr, kind="linear", axis=0)(tgt_times).tolist()

    def interp_contact_linear(
        data: list[list[float]], part_ids: list[int]
    ) -> list[float]:
        """Linear interpolation for contact positions (n_timesteps, n_links, xyz_id).

        For each link, x, y, z, and id are interpolated across time. Each component
        is only interpolated when both bracketing source values are non-zero;
        if either is zero, the result is set to zero.
        """
        H, N = len(data), len(data[0])
        data_arr = np.concatenate(
            [np.asarray(data), np.asarray(part_ids).reshape(H, N, 1)], axis=-1
        )
        # data_arr: (n_timesteps, n_links, 4) for x, y, z, id
        interp_result = interp1d(src_times, data_arr, kind="linear", axis=0)(tgt_times)
        tol = 1e-8
        # Per-component non-zero: (n_timesteps, n_links, 4)
        nonzero_src = np.abs(data_arr) > tol
        idx_lo = np.searchsorted(src_times, tgt_times, side="right") - 1
        idx_lo = np.clip(idx_lo, 0, len(src_times) - 1)
        idx_hi = np.minimum(idx_lo + 1, len(src_times) - 1)
        # For each target time and each (link, xyz_id), require both endpoints non-zero
        mask_lo = nonzero_src[idx_lo]  # (n_tgt, n_links, 4)
        mask_hi = nonzero_src[idx_hi]
        mask_both = mask_lo & mask_hi
        interp_result = np.where(mask_both, interp_result, 0.0)
        return interp_result.tolist()

    def interp_slerp(quat_data: list[list[float]]) -> list[list[float]]:
        """Slerp for a single time sequence of quaternions (T, 4)."""
        quats = np.asarray(quat_data)
        rotations = Rotation.from_quat(quats, scalar_first=True)
        slerp = Slerp(src_times, rotations)
        return slerp(tgt_times).as_quat(scalar_first=True).tolist()

    def interp_slerp_batch(
        quat_data: list[list[list[float]]],
    ) -> list[list[list[float]]]:
        """Slerp for batched quaternions (T, N, 4). Loops over N."""
        data_arr = np.asarray(quat_data)
        # Shape is (Time, N, 4) -> Iterate over N
        n_items = data_arr.shape[1]
        results = []
        for i in range(n_items):
            # Extract (Time, 4) for the i-th item
            results.append(interp_slerp(data_arr[:, i, :]))
        # Stack back to (Time, N, 4) and convert to list
        return np.array(results).transpose(1, 0, 2).tolist()

    def interp_frames(frame_data: list[list[list[float]]]) -> list[list[list[float]]]:
        """Handles (T, N, 7) arrays: Split Pos (linear) and Rot (Slerp)."""
        arr = np.asarray(frame_data)
        # 1. Linear interp on position (first 3 cols) - Vectorized
        pos_interp = np.array(interp_linear(arr[:, :, :3]))
        # 2. Slerp on rotation (last 4 cols) - Batched
        rot_interp = np.array(interp_slerp_batch(arr[:, :, 3:]))
        # 3. Combine
        return np.concatenate([pos_interp, rot_interp], axis=2).tolist()

    # --- Execution ---

    # 0. Setup Times
    src_times, tgt_times = get_timestamps()

    # 1. Simple Linear Fields
    motion_data.object_articulation = interp_linear(motion_data.object_articulation)
    motion_data.robot_right_finger_joints = interp_linear(
        motion_data.robot_right_finger_joints
    )
    motion_data.robot_left_finger_joints = interp_linear(
        motion_data.robot_left_finger_joints
    )
    motion_data.robot_right_wrist_position = interp_linear(
        motion_data.robot_right_wrist_position
    )
    motion_data.robot_left_wrist_position = interp_linear(
        motion_data.robot_left_wrist_position
    )

    # 2. Simple Rotation Fields (Slerp)
    motion_data.robot_right_wrist_wxyz = interp_slerp(
        motion_data.robot_right_wrist_wxyz
    )
    motion_data.robot_left_wrist_wxyz = interp_slerp(motion_data.robot_left_wrist_wxyz)

    # 3. Object Bodies (Batched Position & Rotation)
    # Note: interp_linear handles (Time, N, 3) automatically without loops
    motion_data.object_body_position = interp_linear(motion_data.object_body_position)
    motion_data.object_body_wxyz = interp_slerp_batch(motion_data.object_body_wxyz)

    # 4. Frames (Batched Mixed Data)
    motion_data.robot_right_frames = interp_frames(motion_data.robot_right_frames)
    motion_data.robot_left_frames = interp_frames(motion_data.robot_left_frames)

    # 5. Contact links (T, num_links, 4) — interpolate only between non-zero
    #    values; samples between 0 and non-zero are set to zero
    motion_data.mano_right_link_contact_positions = interp_contact_linear(
        getattr(motion_data, "mano_right_link_contact_positions", []),
        getattr(motion_data, "mano_right_object_contact_part_ids", []),
    )
    motion_data.mano_right_link_contact_normals = interp_contact_linear(
        getattr(motion_data, "mano_right_link_contact_normals", []),
        getattr(motion_data, "mano_right_object_contact_part_ids", []),
    )
    motion_data.mano_right_object_contact_positions = interp_contact_linear(
        getattr(motion_data, "mano_right_object_contact_positions", []),
        getattr(motion_data, "mano_right_object_contact_part_ids", []),
    )
    motion_data.mano_right_object_contact_normals = interp_contact_linear(
        getattr(motion_data, "mano_right_object_contact_normals", []),
        getattr(motion_data, "mano_right_object_contact_part_ids", []),
    )

    motion_data.mano_left_link_contact_positions = interp_contact_linear(
        getattr(motion_data, "mano_left_link_contact_positions", []),
        getattr(motion_data, "mano_left_object_contact_part_ids", []),
    )
    motion_data.mano_left_link_contact_normals = interp_contact_linear(
        getattr(motion_data, "mano_left_link_contact_normals", []),
        getattr(motion_data, "mano_left_object_contact_part_ids", []),
    )
    motion_data.mano_left_object_contact_positions = interp_contact_linear(
        getattr(motion_data, "mano_left_object_contact_positions", []),
        getattr(motion_data, "mano_left_object_contact_part_ids", []),
    )
    motion_data.mano_left_object_contact_normals = interp_contact_linear(
        getattr(motion_data, "mano_left_object_contact_normals", []),
        getattr(motion_data, "mano_left_object_contact_part_ids", []),
    )

    return motion_data


##########################################################
# Tensor Deque
##########################################################


class TensorDeque:
    """A deque for storing tensors."""

    def __init__(self, capacity: int, feature_shape: int, device: str) -> None:
        """Initialize the deque."""
        self.capacity = capacity
        self.buffer: torch.Tensor = torch.zeros(
            (capacity, feature_shape), device=device
        )  # (capacity, feature_shape)
        self.head: int = 0  # index of the first element
        self.tail: int = 0  # index of the last element
        self.size: int = 0  # number of elements in the deque

    def append(self, x: torch.Tensor) -> None:
        """Append a tensor to the deque."""
        self.buffer[self.tail] = x  # (feature_shape,)
        self.tail = (self.tail + 1) % self.capacity  # (1,)
        if self.size < self.capacity:
            self.size += 1  # (1,)
        else:
            # overwrite oldest
            self.head = (self.head + 1) % self.capacity  # (1,)

    def append_batch(self, x: torch.Tensor) -> None:
        """Append batch with shape = (K, feature_shape)."""
        K = len(x)

        # If batch larger than capacity → keep only last capacity elements
        if K >= self.capacity:
            x = x[-self.capacity :]
            K = self.capacity
            self.buffer[:] = x
            self.head = 0
            self.tail = 0
            self.size = self.capacity
            return

        space_left = self.capacity - self.tail

        if K <= space_left:
            # No wrap
            self.buffer[self.tail : self.tail + K] = x
        else:
            # Wrap around
            self.buffer[self.tail :] = x[:space_left]
            self.buffer[: K - space_left] = x[space_left:]

        # Update tail
        self.tail = (self.tail + K) % self.capacity

        # Update size/head
        if self.size + K <= self.capacity:
            self.size += K
        else:
            overflow = self.size + K - self.capacity
            self.size = self.capacity
            self.head = (self.head + overflow) % self.capacity

    def popleft(self) -> torch.Tensor:
        """Pop the oldest tensor from the deque."""
        if self.size == 0:
            raise IndexError("pop from empty deque")
        x = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return x

    def is_full(self) -> bool:
        """Check if the deque is full."""
        return self.size == self.capacity

    def __len__(self) -> int:
        """Get the number of elements in the deque."""
        return self.size  # (1,)

    def get_all(self) -> torch.Tensor:
        """Get all the tensors in the deque."""
        if self.size == 0:
            return torch.empty(0, *self.buffer.shape[1:], device=self.buffer.device)
        if self.head < self.tail:
            return self.buffer[self.head : self.tail]
        else:
            return torch.cat(
                [self.buffer[self.head :], self.buffer[: self.tail]], dim=0
            )

    def clear(self) -> None:
        """Reset deque without reallocating memory. Does NOT zero memory for performance reasons."""
        self.head = 0
        self.tail = 0
        self.size = 0


##########################################################
# Contact Reward
##########################################################


def chamfer_distance(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    pts1_valid: torch.Tensor,
    pts2_valid: torch.Tensor,
    max_dist: float = 100.0,
) -> torch.Tensor:
    """Symmetric chamfer distance ignoring invalid points.

    Args:
        pts1: (B, N1, 3).
        pts2: (B, N2, 3).
        pts1_valid: (B, N1).
        pts2_valid: (B, N2).
        max_dist: Value used for invalid min-distances.

    Returns:
        (B,) symmetric chamfer distance per batch.
    """
    # Compute the distance between all points in pts1 and pts2
    dist1 = torch.cdist(pts1, pts2, p=2.0)  # (B, N1, N2)
    # Fill invalid points in pts2 with max distance
    dist1 = dist1.masked_fill((~pts2_valid).unsqueeze(1), max_dist)
    min_dists1 = torch.min(dist1, dim=-1).values  # (B, N1)
    num_valids1 = pts1_valid.sum(dim=-1)  # (B,)

    dist2 = torch.cdist(pts2, pts1, p=2.0)  # (B, N2, N1)
    # Fill invalid points in pts1 with max distance
    dist2 = dist2.masked_fill((~pts1_valid).unsqueeze(1), max_dist)
    min_dists2 = torch.min(dist2, dim=-1).values  # (B, N2)
    num_valids2 = pts2_valid.sum(dim=-1)

    both_zero = (num_valids1 == 0) & (num_valids2 == 0)
    one_zero = (num_valids1 == 0) ^ (num_valids2 == 0)
    both_has_valid = (num_valids1 > 0) & (num_valids2 > 0)

    chamfer_forward = torch.zeros(pts1.shape[0], device=pts1.device)
    chamfer_backward = torch.zeros(pts2.shape[0], device=pts2.device)
    chamfer_forward[both_zero] = 0.0
    chamfer_backward[both_zero] = 0.0
    chamfer_forward[one_zero] = max_dist
    chamfer_backward[one_zero] = max_dist
    if both_has_valid.sum() > 0:
        valid_sums1 = torch.sum(min_dists1 * pts1_valid, dim=-1)
        valid_sums2 = torch.sum(min_dists2 * pts2_valid, dim=-1)
        chamfer_forward[both_has_valid] = (valid_sums1 / num_valids1.clamp(min=1e-5))[
            both_has_valid
        ]
        chamfer_backward[both_has_valid] = (valid_sums2 / num_valids2.clamp(min=1e-5))[
            both_has_valid
        ]
    return (chamfer_forward + chamfer_backward) / 2.0


def sample_wrench_space_basis_scaled(
    num_samples: int,
    rc: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sample and scale wrench space basis directions uniformly on the surface of a 6D unit sphere.

    Args:
        num_samples: number of samples to draw
        rc: scale of the torque, typically the radius of the object's bounding ball
        device: device for the output tensor
        dtype: dtype for the output tensor
        eps: numerical stability

    Returns:
        (num_samples, dim)
    """
    basis = torch.randn(num_samples, 6, device=device, dtype=dtype)
    basis[:, 3:] = basis[:, 3:] / rc
    return basis / basis.norm(dim=-1, keepdim=True).clamp_min(eps)


def compute_tangent_basis(
    n: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute orthonormal tangent basis from normal vectors (Frisvad 2012).

    Args:
        n: (..., 3) normal vectors (need not be unit length)
        eps: numerical stability

    Returns:
        t1: (..., 3)
        t2: (..., 3)
    """
    nx, ny, nz = n.unbind(dim=-1)

    sign = torch.where(nz >= 0, 1.0, -1.0)
    den = sign + nz
    den = torch.where(den.abs() < eps, sign * eps, den)

    a = -1.0 / den
    b = nx * ny * a

    t1 = torch.stack(
        (1.0 + sign * nx * nx * a, sign * b, -sign * nx),
        dim=-1,
    )
    t2 = torch.stack(
        (b, sign + ny * ny * a, -ny),
        dim=-1,
    )

    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(eps)
    t2 = t2 / t2.norm(dim=-1, keepdim=True).clamp_min(eps)

    return t1, t2


def compute_friction_cone_edges(
    normals: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    friction_coefficients: float = 0.5,
    eps: float = 1e-6,
    append_normal: bool = True,
) -> torch.Tensor:
    """Build the edges of the friction cone based on contact normals and friction coefficients.

    Args:
        normals: (batch_size, num_contacts, 3)
        cos_t: the cosine of the friction cone edges phase angles (1, num_friction_cone_edges, 1)
        sin_t: the sine of the friction cone edges phase angles (1, num_friction_cone_edges, 1)
        friction_coefficients: float
        eps: float
        append_normal: bool whether to append the contact normal to the friction cone edges

    Returns:
        (batch_size, num_contacts, num_friction_cone_edges, 3)
    """
    batch_size, num_contacts, _ = normals.shape

    # Ensure unit normals before tangent construction.
    normals_flat = normals.reshape(-1, 3)
    t1, t2 = compute_tangent_basis(normals_flat, eps=eps)  # (B*C, 2, 3)

    n_exp = normals_flat.unsqueeze(1)  # (B*C, 1, 3)
    t1_exp = t1.unsqueeze(1)  # (B*C, 1, 3)
    t2_exp = t2.unsqueeze(1)  # (B*C, 1, 3)

    # Polyhedral approximation rays: n + mu * (cos(theta) * t1 + sin(theta) * t2)
    edges = n_exp + friction_coefficients * (
        cos_t * t1_exp + sin_t * t2_exp
    )  # (B*C, K, 3)
    edges = edges / edges.norm(dim=-1, keepdim=True).clamp_min(eps)

    if append_normal:
        edges = torch.cat([edges, normals_flat.unsqueeze(1)], dim=1)  # (B*C, K+1, 3)

    num_edges = cos_t.shape[1] + int(append_normal)
    return edges.view(batch_size, num_contacts, num_edges, 3)


def compute_wrench_space(
    contact_points: torch.Tensor,
    contact_normals: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    rc: float,
    friction_coefficients: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the wrench space from contact points and contact normals.

    Args:
        contact_points: in the object frame (batch_size, num_contacts, 3)
        contact_normals: pointing into the object (batch_size, num_contacts, 3)
        cos_t: the cosine of the friction cone edges phase angles (1, num_friction_cone_edges, 1)
        sin_t: the sine of the friction cone edges phase angles (1, num_friction_cone_edges, 1)
        rc: scale of the torque, typically the radius of the object's bounding ball
        friction_coefficients: float
        eps: float

    Returns:
        (batch_size, 6, num_contacts * num_friction_cone_edges,)
    """
    batch_size = len(contact_points)

    contact_normals = contact_normals / contact_normals.norm(
        dim=-1, keepdim=True
    ).clamp_min(eps)

    contact_is_active = contact_normals.norm(dim=-1) > 1e-3

    forces = compute_friction_cone_edges(
        normals=contact_normals,
        cos_t=cos_t,
        sin_t=sin_t,
        friction_coefficients=friction_coefficients,
        eps=eps,
    )  # (batch_size, num_contacts, num_friction_cone_edges, 3)

    torques = torch.cross(
        contact_points.unsqueeze(2).expand_as(forces), forces, dim=-1
    )  # (batch_size, num_contacts, num_friction_cone_edges, 3)

    wrench_space = torch.cat(
        (forces, torques / rc), dim=-1
    )  # (batch_size, num_contacts, num_friction_cone_edges, 6)
    wrench_space *= contact_is_active.view(batch_size, -1, 1, 1)
    wrench_space = wrench_space.view(batch_size, -1, 6)

    return wrench_space.transpose(1, 2).contiguous()


def compute_wrench_space_support_function(
    wrench_space: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the wrench space support function at basis directions.

    The support function of wrench space W at direction d is $max_{w in W} d^T w$.
    The support function is clamped to be non-negative in wrench space.

    Args:
        wrench_space: (batch_size, 6, num_contacts)
        basis: (num_basis, 6)

    Returns:
        (batch_size, num_basis)
    """
    return torch.clamp(
        torch.matmul(basis.unsqueeze(0), wrench_space).amax(dim=-1),
        min=0.0,
    )
