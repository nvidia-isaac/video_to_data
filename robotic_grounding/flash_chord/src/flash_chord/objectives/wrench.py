# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grasp wrench-space support function (the contact-objective fingerprint).

Per object body, ``num_basis`` directions are sampled on the 6-D wrench sphere (torque dims scaled by
``1/rc``, ``rc`` = body bounding radius). Each active contact (point + normal, object frame) contributes
a polyhedral friction cone of ``num_edges`` rays + the normal; each ray's wrench is ``[f ; (p×f)/rc]``.
The support along a direction ``d`` is ``clamp(max_w d·w, 0)`` → a ``(num_basis,)`` non-negative vector.
The basis is sampled once and shared by the reference and the sim.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from flash_chord.utils.quat import wxyz_to_xyzw


def sample_wrench_basis(num_basis: int, rc: float, seed: int = 0) -> np.ndarray:
    """``(num_basis, 6)`` directions on the 6-D unit sphere with torque dims pre-scaled by ``1/rc``."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((num_basis, 6))
    basis[:, 3:] /= rc
    basis /= np.linalg.norm(basis, axis=-1, keepdims=True).clip(min=1e-8)
    return basis.astype(np.float32)


def friction_cone_angles(num_edges: int) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin of ``num_edges`` evenly spaced phase angles in ``[0, 2π)``."""
    theta = np.linspace(0.0, 2.0 * np.pi, num_edges + 1)[:-1]
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


@wp.func
def _wrench_support_range(
    contact_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_enabled: wp.array(dtype=wp.int32),
    contact_start: int,
    enabled_start: int,
    num_contacts: int,
    d: wp.spatial_vector,
    cos_t: wp.array(dtype=wp.float32),  # [num_edges]
    sin_t: wp.array(dtype=wp.float32),  # [num_edges]
    mu: float,
    rc: float,
    num_edges: int,
) -> float:
    """Support of contacts in one contiguous range along direction ``d``."""
    df = wp.spatial_top(d)  # force part
    dt = wp.spatial_bottom(d)  # torque part
    best = float(0.0)  # support clamped to be non-negative
    for c in range(num_contacts):
        n = contact_normal[contact_start + c]
        if contact_enabled[enabled_start + c] != 0 and wp.length(n) > 1.0e-3:
            nn = wp.normalize(n)
            p = contact_pos[contact_start + c]
            # Frisvad 2012 tangent basis
            sign = wp.where(nn[2] >= 0.0, 1.0, -1.0)
            a = -1.0 / (sign + nn[2])
            bb = nn[0] * nn[1] * a
            t1 = wp.normalize(wp.vec3(1.0 + sign * nn[0] * nn[0] * a, sign * bb, -sign * nn[0]))
            t2 = wp.normalize(wp.vec3(bb, sign + nn[1] * nn[1] * a, -nn[1]))
            for e in range(num_edges + 1):
                if e < num_edges:
                    f = wp.normalize(nn + mu * (cos_t[e] * t1 + sin_t[e] * t2))
                else:
                    f = nn  # appended normal
                torque = wp.cross(p, f) / rc
                val = wp.dot(df, f) + wp.dot(dt, torque)
                best = wp.max(best, val)
    return best


@wp.kernel
def wrench_support(
    contact_pos: wp.array(dtype=wp.vec3),  # [num_bodies * num_contacts] object-frame contact points
    contact_normal: wp.array(dtype=wp.vec3),  # [num_bodies * num_contacts] (zero/near-zero = inactive)
    contact_enabled: wp.array(dtype=wp.int32),  # [num_contacts] semantic contact inclusion mask
    basis: wp.array(dtype=wp.spatial_vector),  # [num_bodies * num_basis] per-body sampled directions [f(3), t(3)]
    cos_t: wp.array(dtype=wp.float32),  # [num_edges]
    sin_t: wp.array(dtype=wp.float32),  # [num_edges]
    mu: float,
    rc: wp.array(dtype=wp.float32),  # [num_bodies]
    num_contacts: int,
    num_edges: int,
    num_basis: int,
    support: wp.array(dtype=wp.float32),  # [num_bodies * num_basis] out, non-negative
) -> None:
    """Per (body, basis-direction) support = clamp(max over contacts×(edges+normal) of d·wrench, 0)."""
    tid = wp.tid()
    body = tid // num_basis
    support[tid] = _wrench_support_range(
        contact_pos,
        contact_normal,
        contact_enabled,
        body * num_contacts,
        0,
        num_contacts,
        basis[tid],
        cos_t,
        sin_t,
        mu,
        rc[body],
        num_edges,
    )


@wp.kernel
def batched_wrench_support(
    contact_pos_o: wp.array(dtype=wp.vec3),
    contact_direction_o: wp.array(dtype=wp.vec3),
    contact_enabled: wp.array(dtype=wp.int32),  # [slots_per_world] semantic link mask
    hand_slot_starts: wp.array(dtype=wp.int32),
    link_counts: wp.array(dtype=wp.int32),
    slots_per_world: int,
    num_hands: int,
    num_bodies: int,
    basis: wp.array(dtype=wp.spatial_vector),  # [num_bodies * num_basis]
    cos_t: wp.array(dtype=wp.float32),
    sin_t: wp.array(dtype=wp.float32),
    mu: float,
    rc: wp.array(dtype=wp.float32),
    num_edges: int,
    num_basis: int,
    support: wp.array(dtype=wp.float32),  # [world * hand * body * basis]
) -> None:
    """Compute live support for every world/hand/object-body cell."""
    tid = wp.tid()
    basis_id = tid % num_basis
    body = (tid // num_basis) % num_bodies
    hand = (tid // (num_basis * num_bodies)) % num_hands
    world = tid // (num_basis * num_bodies * num_hands)
    enabled_start = hand_slot_starts[hand] + body * link_counts[hand]
    contact_start = world * slots_per_world + enabled_start
    support[tid] = _wrench_support_range(
        contact_pos_o,
        contact_direction_o,
        contact_enabled,
        contact_start,
        enabled_start,
        link_counts[hand],
        basis[body * num_basis + basis_id],
        cos_t,
        sin_t,
        mu,
        rc[world * num_bodies + body],
        num_edges,
    )


@wp.kernel
def merge_wrench_support(
    included_support: wp.array(dtype=wp.float32),
    support: wp.array(dtype=wp.float32),
) -> None:
    """Union two disjoint contact sets by taking their support-function maximum."""
    index = wp.tid()
    support[index] = wp.max(support[index], included_support[index])


def _to_object_frame(points_w, normals_w, obj_pos, obj_quat_wxyz):
    """Rotate/translate world contacts into the per-frame object-COM frame (host numpy).

    points_w/normals_w: (T, K, 3); obj_pos: (T, 3); obj_quat_wxyz: (T, 4). Returns (T, K, 3) each."""
    from scipy.spatial.transform import Rotation

    r_inv = Rotation.from_quat(wxyz_to_xyzw(np.asarray(obj_quat_wxyz))).inv()  # (T,)
    p_o = np.stack([r_inv[t].apply(points_w[t] - obj_pos[t]) for t in range(points_w.shape[0])])
    n_o = np.stack([r_inv[t].apply(normals_w[t]) for t in range(normals_w.shape[0])])
    return p_o.astype(np.float32), n_o.astype(np.float32)


def compute_reference_supports(
    reference,
    num_basis: int = 512,
    num_edges: int = 8,
    mu: float = 0.1,
    seed: int = 0,
    sides: tuple[str, ...] | None = None,
    device=None,
) -> np.ndarray:
    """Demo (command) wrench supports from the reference's mano contacts: ``(T, H, B, num_basis)``
    (H = sides, B = object bodies). Contacts are transformed world→object-COM frame and bucketed by
    ``part_id``; the per-body basis (rc = ``object_mesh_radius``) is shared with the sim-side supports."""
    obj_pos = np.asarray(reference.object_body_pos_w(), dtype=np.float64)  # (T, B, 3)
    obj_quat = np.asarray(reference.object_body_quat_w(), dtype=np.float64)  # (T, B, 4) wxyz
    rc = np.asarray(reference.object_mesh_radius(), dtype=np.float64)  # (B,)
    sides = reference.sides if sides is None else sides
    T, B, H = obj_pos.shape[0], obj_pos.shape[1], len(sides)
    cos_t, sin_t = friction_cone_angles(num_edges)
    basis_per_body = [sample_wrench_basis(num_basis, 1.0, seed + b) for b in range(B)]  # B × (N,6)

    out = np.zeros((T, H, B, num_basis), dtype=np.float32)
    cos_d = wp.array(cos_t, dtype=wp.float32, device=device)
    sin_d = wp.array(sin_t, dtype=wp.float32, device=device)
    for h, side in enumerate(sides):
        cp_w = np.asarray(reference.contact_pos_w(side), dtype=np.float64)  # (T, K, 3)
        cn_w = np.asarray(reference.contact_normal_w(side), dtype=np.float64)
        pid = np.asarray(reference.contact_part_ids(side))  # (T, K) 1-indexed body
        K = cp_w.shape[1]
        if K == 0:
            continue
        enabled_d = wp.ones(K, dtype=wp.int32, device=device)
        for b in range(B):
            p_o, n_o = _to_object_frame(cp_w, cn_w, obj_pos[:, b], obj_quat[:, b])  # (T, K, 3)
            n_o = n_o * (pid == (b + 1))[..., None]  # zero normals of contacts not on this body -> inactive
            with wp.ScopedDevice(device):
                cp = wp.array(p_o.reshape(-1, 3), dtype=wp.vec3)
                cn = wp.array(n_o.reshape(-1, 3), dtype=wp.vec3)
                basis = wp.array(np.tile(basis_per_body[b], (T, 1)), dtype=wp.spatial_vector)  # (T*N, 6)
                rc_arr = wp.array(np.full(T, rc[b], dtype=np.float32), dtype=wp.float32)
                support = wp.zeros(T * num_basis, dtype=wp.float32)
                wp.launch(
                    wrench_support,
                    dim=T * num_basis,
                    inputs=[
                        cp,
                        cn,
                        enabled_d,
                        basis,
                        cos_d,
                        sin_d,
                        float(mu),
                        rc_arr,
                        K,
                        num_edges,
                        num_basis,
                    ],
                    outputs=[support],
                )
                out[:, h, b] = support.numpy().reshape(T, num_basis)
    return out
