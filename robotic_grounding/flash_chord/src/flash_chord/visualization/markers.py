# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Debug-visualization markers for reference and simulated task geometry.

One function per marker type, each drawing BOTH the reference and the simulated state:
  i.   :func:`log_wrist_axes`              (``log_gizmo``)
  ii.  :func:`log_object_axes`             (``log_gizmo``)
  iii. :func:`log_hand_keypoint_spheres`   (``log_points`` — wrist + fingertips)
  iv.  :func:`log_contact_position_spheres`(``log_points``)
  v.   :func:`log_contact_normals`         (``log_arrows`` — from contact point along the normal)
:func:`log_debug_markers` composes all five. Reference markers are green/yellow; simulated are blue/magenta.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from flash_chord.runtime.contact import read_contacts_w
from flash_chord.utils.quat import wxyz_to_xyzw

_AXIS_RGB = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)  # X red, Y green, Z blue

_REF_KP = (0.1, 0.9, 0.2)  # reference hand-keypoint spheres (green)
_SIM_KP = (0.2, 0.45, 1.0)  # simulated hand-keypoint spheres (blue)
_REF_CONTACT = (1.0, 0.85, 0.1)  # reference contact spheres/normals (yellow)
_SIM_CONTACT = (1.0, 0.25, 0.9)  # simulated contact spheres/normals (magenta)

_MIN_NORMAL = 1e-3  # contact-normal magnitude below this is a degenerate / stale slot


# --- small drawing helpers ---------------------------------------------------
def _axes_lines(viewer, name, poses, axis_len, device) -> None:
    """Draw a coordinate frame per ``(pos, quat_xyzw)`` as 3 colored segments (X red, Y green, Z blue)."""
    if not poses:
        viewer.log_lines(name, None, None, None)
        return
    starts, ends, colors = [], [], []
    for pos, quat_xyzw in poses:
        rot = Rotation.from_quat(np.asarray(quat_xyzw)).as_matrix()  # columns = rotated axes
        for i in range(3):
            starts.append(pos)
            ends.append(np.asarray(pos) + rot[:, i] * axis_len)
            colors.append(_AXIS_RGB[i])
    s = wp.array(np.asarray(starts, dtype=np.float32), dtype=wp.vec3, device=device)
    e = wp.array(np.asarray(ends, dtype=np.float32), dtype=wp.vec3, device=device)
    c = wp.array(np.asarray(colors, dtype=np.float32), dtype=wp.vec3, device=device)
    viewer.log_lines(name, s, e, c)


def _points(viewer, name, pts, color, radius, device) -> None:
    arr = None if len(pts) == 0 else wp.array(np.asarray(pts, dtype=np.float32), dtype=wp.vec3, device=device)
    viewer.log_points(name, arr, radii=radius, colors=color)


def _normal_arrows(viewer, name, pts, normals, color, length, device) -> None:
    if len(pts) == 0:
        viewer.log_arrows(name, None, None, None)
        return
    pts = np.asarray(pts, dtype=np.float32)
    n = np.asarray(normals, dtype=np.float32)
    norms = np.linalg.norm(n, axis=-1, keepdims=True)
    dirs = np.zeros_like(n)
    valid = (norms > 1e-6).squeeze(-1)  # degenerate normals -> zero-length arrow (no runaway endpoints)
    dirs[valid] = n[valid] / norms[valid] * length
    s = wp.array(pts, dtype=wp.vec3, device=device)
    e = wp.array(pts + dirs, dtype=wp.vec3, device=device)
    viewer.log_arrows(name, s, e, color)


# --- object-body + contact helpers ------------------------------------------
def _obj_sim_body(scene, b: int) -> int:
    """Resolve one reference object body through the scene's explicit body bindings."""
    matches = [
        body.body_id
        for binding in scene.objects
        for body in binding.bodies
        if body.reference_body_id == b
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"reference object body {b} is mapped more than once: {matches}")
    raise IndexError(f"no simulated object body for reference body {b}")


def _semantic_frame_pose(body_q: np.ndarray, frame) -> tuple[np.ndarray, np.ndarray]:
    """Compose one retained-body xyzw pose with a semantic body-local frame."""
    body_pose = body_q[frame.body_id]
    body_rotation = Rotation.from_quat(body_pose[3:7])
    position = body_pose[:3] + body_rotation.apply(frame.body_to_frame_pos)
    orientation = (body_rotation * Rotation.from_quat(frame.body_to_frame_quat_xyzw)).as_quat()
    return position, orientation


def _ref_contacts(reference, frame_id):
    """Active demo contact points + normals (both hands concatenated)."""
    pts, normals = [], []
    for side in reference.sides:
        cn = np.asarray(reference.contact_normal_w(side)[frame_id])
        if cn.shape[0] == 0:
            continue
        cp = np.asarray(reference.contact_pos_w(side)[frame_id])
        active = np.linalg.norm(cn, axis=-1) > _MIN_NORMAL
        pts.extend(cp[active])
        normals.extend(cn[active])
    return pts, normals


def _sim_object_contacts(scene, state, contacts, model, n_obj_bodies):
    """Live hand↔object contact points + world-frame normals (from the collision readout)."""
    obj_bodies = {_obj_sim_body(scene, b) for b in range(n_obj_bodies)}
    rd = read_contacts_w(model, state, contacts)
    pts, normals = [], []
    for i in range(len(rd)):
        if int(rd.body0_ids[i]) in obj_bodies or int(rd.body1_ids[i]) in obj_bodies:
            if np.linalg.norm(rd.contact_normal_w[i]) > _MIN_NORMAL:  # skip degenerate/stale contact slots
                pts.append(rd.contact_pos_w[i])
                normals.append(rd.contact_normal_w[i])
    return pts, normals


# --- the five marker types (each draws reference + simulated) -----------------
def log_wrist_axes(viewer, scene, reference, frame_id, state, device=None, axis_len=0.05) -> None:
    """i. Wrist coordinate axes (RGB segments), per hand, reference + simulated."""
    body_q = state.body_q.numpy()
    ref_poses, sim_poses = [], []
    for hand in scene.layout.hands:
        bound = scene.robot_reference.hand(hand.side)
        ref_poses.append((bound.wrist_pos_w[frame_id], wxyz_to_xyzw(bound.wrist_quat_w[frame_id])))
        sim_poses.append(_semantic_frame_pose(body_q, hand.palm_frame))
    _axes_lines(viewer, "/ref/wrist_axes", ref_poses, axis_len, device)
    _axes_lines(viewer, "/sim/wrist_axes", sim_poses, axis_len, device)


def log_object_axes(viewer, scene, reference, frame_id, state, device=None, axis_len=0.08) -> None:
    """ii. Object coordinate axes (RGB segments), per body, reference + simulated."""
    body_q = state.body_q.numpy()
    obj_pos = np.asarray(reference.object_body_pos_w()[frame_id])
    obj_quat = np.asarray(reference.object_body_quat_w()[frame_id])  # wxyz
    ref_poses, sim_poses = [], []
    for b in range(obj_pos.shape[0]):
        ref_poses.append((obj_pos[b], wxyz_to_xyzw(obj_quat[b])))
        bq = body_q[_obj_sim_body(scene, b)]
        sim_poses.append((bq[:3], bq[3:7]))
    _axes_lines(viewer, "/ref/object_axes", ref_poses, axis_len, device)
    _axes_lines(viewer, "/sim/object_axes", sim_poses, axis_len, device)


def log_hand_keypoint_spheres(
    viewer,
    scene,
    reference,
    frame_id,
    state,
    device=None,
    radius=0.01,
    keypoint_frame="dp",
) -> None:
    """iii. Hand-keypoint spheres (wrist + fingertips), reference + simulated."""
    body_q = state.body_q.numpy()
    ref_kp, sim_kp = [], []
    if keypoint_frame not in ("dp", "fingertip"):
        raise ValueError(f"hand keypoint frame must be 'dp' or 'fingertip', got {keypoint_frame!r}")
    for hand in scene.layout.hands:
        bound = scene.robot_reference.hand(hand.side)
        ref_kp.extend(bound.keypoint_pos_w(keypoint_frame)[frame_id])
        digit_frames = hand.dp_frames if keypoint_frame == "dp" else hand.fingertip_frames
        sim_kp.append(_semantic_frame_pose(body_q, hand.palm_frame)[0])
        sim_kp.extend(_semantic_frame_pose(body_q, frame)[0] for frame in digit_frames)
    _points(viewer, "/ref/hand_keypoints", ref_kp, _REF_KP, radius, device)
    _points(viewer, "/sim/hand_keypoints", sim_kp, _SIM_KP, radius, device)


def log_contact_position_spheres(
    viewer, scene, reference, frame_id, state, contacts, model, device=None, radius=0.01
) -> None:
    """iv. Contact-position spheres, reference (demo) + simulated (live)."""
    ref_pts, _ = _ref_contacts(reference, frame_id)
    sim_pts, _ = _sim_object_contacts(scene, state, contacts, model, np.asarray(reference.object_body_pos_w()).shape[1])
    _points(viewer, "/ref/contact_positions", ref_pts, _REF_CONTACT, radius, device)
    _points(viewer, "/sim/contact_positions", sim_pts, _SIM_CONTACT, radius, device)


def log_contact_normals(
    viewer, scene, reference, frame_id, state, contacts, model, device=None, normal_len=0.03
) -> None:
    """v. Contact-normal arrows, reference (demo) + simulated (live)."""
    ref_pts, ref_n = _ref_contacts(reference, frame_id)
    sim_pts, sim_n = _sim_object_contacts(
        scene, state, contacts, model, np.asarray(reference.object_body_pos_w()).shape[1]
    )
    _normal_arrows(viewer, "/ref/contact_normals", ref_pts, ref_n, _REF_CONTACT, normal_len, device)
    _normal_arrows(viewer, "/sim/contact_normals", sim_pts, sim_n, _SIM_CONTACT, normal_len, device)


def log_debug_markers(
    viewer,
    scene,
    reference,
    frame_id,
    state,
    contacts,
    model,
    device=None,
    radius=0.01,
    normal_len=0.03,
    keypoint_frame="dp",
) -> None:
    """Draw all five ref+sim debug markers (System Design Verification 2.f)."""
    log_wrist_axes(viewer, scene, reference, frame_id, state, device)
    log_object_axes(viewer, scene, reference, frame_id, state, device)
    log_hand_keypoint_spheres(
        viewer,
        scene,
        reference,
        frame_id,
        state,
        device,
        radius,
        keypoint_frame,
    )
    log_contact_position_spheres(viewer, scene, reference, frame_id, state, contacts, model, device, radius)
    log_contact_normals(viewer, scene, reference, frame_id, state, contacts, model, device, normal_len)


def log_configured_markers(viewer, config, scene, reference, frame_id, state, contacts, model, device=None) -> None:
    """Draw the marker subsets selected by a replay or evaluation config."""
    if config.enabled:
        if config.axes:
            log_wrist_axes(viewer, scene, reference, frame_id, state, device)
            log_object_axes(viewer, scene, reference, frame_id, state, device)
        if config.keypoints:
            log_hand_keypoint_spheres(
                viewer,
                scene,
                reference,
                frame_id,
                state,
                device,
                keypoint_frame=config.keypoint_frame,
            )
        if config.contacts:
            log_contact_position_spheres(viewer, scene, reference, frame_id, state, contacts, model, device)
            log_contact_normals(viewer, scene, reference, frame_id, state, contacts, model, device)
    if config.raw_contacts:
        viewer.log_contacts(contacts, state)
