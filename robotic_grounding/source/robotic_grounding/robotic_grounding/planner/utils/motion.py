"""V2P retargeted-motion manipulation: input resampling + output assembly.

`interpolate_robot_motion_data` is the input side: takes a
`ManoDex3Data` / `ManoSharpaData` dataclass and resamples every
time-series field to the planner's target FPS (linear for positions /
joint angles / object articulation, SLERP for quats, contact-aware
masked interp for contact fields).

`assemble_object_fields` and `assemble_hand_contact_fields` are the
output side: read the same motion dataclass plus the planner-frame
object body arrays, and build the per-field dicts that
`save_planner_parquet` assembles into a `MotionData` row. They apply
the per-frame V2P→planner rigid transform to hand keypoints and
contacts so every field of the output parquet lives in one coherent
frame.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from robotic_grounding.planner.utils.transforms import (
    quat_conj,
    quat_mul,
    transform_contact_dir_by_part,
    transform_contact_pos_by_part,
    transform_primary_pos,
    transform_primary_quat,
)


def interpolate_robot_motion_data(motion_data: Any, target_fps: float) -> Any:
    """Resample a V2P retargeted motion's time-series fields to ``target_fps``.

    Linear interpolation for positions / joint angles / object articulation,
    SLERP for wrist + per-body object quaternions, and contact-aware linear
    interpolation that zeroes intervals where either neighbour is inactive.
    """
    n_frames = len(motion_data.robot_right_wrist_position)
    src_times = np.arange(n_frames) / motion_data.fps
    tgt_times = np.linspace(0, src_times[-1], int(src_times[-1] * target_fps))

    def _linear(data: Any) -> Any:
        return interp1d(src_times, np.asarray(data), kind="linear", axis=0)(
            tgt_times
        ).tolist()

    def _slerp(quat_data: Any) -> Any:
        quats = np.asarray(quat_data)
        return (
            Slerp(src_times, Rotation.from_quat(quats, scalar_first=True))(tgt_times)
            .as_quat(scalar_first=True)
            .tolist()
        )

    def _slerp_batch(quat_data: Any) -> Any:
        arr = np.asarray(quat_data)
        results = [_slerp(arr[:, i, :]) for i in range(arr.shape[1])]
        return np.array(results).transpose(1, 0, 2).tolist()

    def _frames(frame_data: Any) -> Any:
        arr = np.asarray(frame_data)
        pos = np.array(_linear(arr[:, :, :3]))
        rot = np.array(_slerp_batch(arr[:, :, 3:]))
        return np.concatenate([pos, rot], axis=2).tolist()

    def _contact_linear(data: Any, part_ids: Any) -> Any:
        if not data:
            return []
        H, N = len(data), len(data[0])
        arr = np.concatenate(
            [np.asarray(data), np.asarray(part_ids).reshape(H, N, 1)], axis=-1
        )
        interp_result = interp1d(src_times, arr, kind="linear", axis=0)(tgt_times)
        nonzero_src = np.abs(arr) > 1e-8
        idx_lo = np.clip(
            np.searchsorted(src_times, tgt_times, side="right") - 1,
            0,
            len(src_times) - 1,
        )
        idx_hi = np.minimum(idx_lo + 1, len(src_times) - 1)
        mask = nonzero_src[idx_lo] & nonzero_src[idx_hi]
        return np.where(mask, interp_result, 0.0).tolist()

    motion_data.object_articulation = _linear(motion_data.object_articulation)
    motion_data.robot_right_finger_joints = _linear(
        motion_data.robot_right_finger_joints
    )
    motion_data.robot_left_finger_joints = _linear(motion_data.robot_left_finger_joints)
    motion_data.robot_right_wrist_position = _linear(
        motion_data.robot_right_wrist_position
    )
    motion_data.robot_left_wrist_position = _linear(
        motion_data.robot_left_wrist_position
    )
    motion_data.robot_right_wrist_wxyz = _slerp(motion_data.robot_right_wrist_wxyz)
    motion_data.robot_left_wrist_wxyz = _slerp(motion_data.robot_left_wrist_wxyz)
    motion_data.object_body_position = _linear(motion_data.object_body_position)
    motion_data.object_body_wxyz = _slerp_batch(motion_data.object_body_wxyz)
    motion_data.robot_right_frames = _frames(motion_data.robot_right_frames)
    motion_data.robot_left_frames = _frames(motion_data.robot_left_frames)

    for side in ("right", "left"):
        part_ids = getattr(motion_data, f"mano_{side}_object_contact_part_ids", [])
        for field in (
            "link_contact_positions",
            "object_contact_positions",
            "object_contact_normals",
        ):
            attr = f"mano_{side}_{field}"
            val = getattr(motion_data, attr, [])
            if val:
                setattr(motion_data, attr, _contact_linear(val, part_ids))

    return motion_data


def assemble_object_fields(
    motion: Any | None,
    obj_body_pos_arr: np.ndarray,
    obj_body_wxyz_arr: np.ndarray,
    T_use: int,
    fallback_object_name: str,
    resolve_asset_path: Callable[[str | None], str | None],
    warn_missing_deps: Callable[[list[str]], None],
) -> dict[str, Any]:
    """Build object-side fields of the planner's output parquet.

    ``obj_body_pos_arr`` / ``obj_body_wxyz_arr`` are the planner-frame
    object body arrays sliced to ``T_use`` (caller has already pulled
    them from ref_data and converted). ``motion`` is the upstream
    ManoSharpaData/ManoDex3Data dataclass; metadata fields are carried
    through, missing ones fall back to sensible defaults. Asset paths
    are remapped via ``resolve_asset_path``, and URDF mesh dependencies
    surfaced via ``warn_missing_deps``.

    ``object_root_*`` always derives from body 0 of the planner-frame
    pose so the env's articulated scene init lands where the trajectory
    starts, regardless of whether the upstream motion carried a
    separately-resampled root field.
    """
    object_root_position: list = obj_body_pos_arr[:, 0, :].astype(np.float32).tolist()
    object_root_axis_angle: list = (
        Rotation.from_quat(obj_body_wxyz_arr[:, 0, :], scalar_first=True)
        .as_rotvec()
        .astype(np.float32)
        .tolist()
    )

    object_name = str(fallback_object_name)
    object_body_names: list[str] = ["object"]
    safe_object_body_names: list[str] = ["object"]
    object_mesh_paths: list[str] = []
    object_urdf_paths: list[str] = []
    object_mesh_radius: list[float] | None = None
    object_articulation: list[float] = [0.0] * T_use
    safe_object_name = object_name

    if motion is not None:
        if getattr(motion, "object_body_names", None):
            object_body_names = list(motion.object_body_names)
        if getattr(motion, "safe_object_body_names", None):
            safe_object_body_names = list(motion.safe_object_body_names)
        if getattr(motion, "object_mesh_paths", None):
            object_mesh_paths = [
                str(resolved)
                for p in motion.object_mesh_paths
                if (resolved := resolve_asset_path(p))
            ]
        if getattr(motion, "object_urdf_paths", None):
            object_urdf_paths = [
                str(resolved)
                for p in motion.object_urdf_paths
                if (resolved := resolve_asset_path(p))
            ]
            # The URDF can resolve locally while its <mesh filename=> visual
            # or collision dependencies don't. Surface those gaps now so the
            # user fixes the workspace before training hits the import crash.
            warn_missing_deps(object_urdf_paths)
        if getattr(motion, "object_mesh_radius", None):
            object_mesh_radius = [float(r) for r in motion.object_mesh_radius]
        if getattr(motion, "safe_object_name", None):
            safe_object_name = motion.safe_object_name
        obj_art = getattr(motion, "object_articulation", None)
        if obj_art is not None:
            object_articulation = np.asarray(obj_art, dtype=np.float32)[:T_use].tolist()

    return {
        "object_name": object_name,
        "safe_object_name": safe_object_name,
        "object_body_names": object_body_names,
        "safe_object_body_names": safe_object_body_names,
        "object_mesh_paths": object_mesh_paths,
        "object_urdf_paths": object_urdf_paths,
        "object_mesh_radius": object_mesh_radius,
        "object_articulation": object_articulation,
        "object_root_position": object_root_position,
        "object_root_axis_angle": object_root_axis_angle,
    }


def assemble_hand_contact_fields(
    motion: Any | None,
    obj_body_pos_arr: np.ndarray,
    obj_body_wxyz_arr: np.ndarray,
    T_use: int,
) -> dict[str, Any]:
    """Build per-side hand-frame + contact fields in the planner frame.

    Builds the per-frame V2P→planner rigid transform anchored on the
    primary object body using ``motion.object_body_*`` as the raw source
    and the supplied planner-frame arrays as the destination, then
    applies it to ``robot_*_frames`` (primary-body transform) and to the
    four contact arrays (per-body transform via part_ids). Re-derives
    ``hand_contact_active`` from object-contact-position magnitudes.

    Returns empty per-side lists when ``motion`` is ``None`` or a side
    has no wrist position; callers should still emit those empty lists
    so the output parquet's per-side fields stay length-2.
    """
    out: dict[str, Any] = {
        "hand_sides": [],
        "hand_frame_names": [],
        "hand_frames_w": [],
        "hand_finger_joint_names": [],
        "hand_finger_joints": [],
        "hand_contact_link_names": [],
        "hand_link_contact_positions": [],
        "hand_link_contact_normals": [],
        "hand_object_contact_positions": [],
        "hand_object_contact_normals": [],
        "hand_object_contact_part_ids": [],
        "hand_contact_active": [],
    }
    if motion is None:
        return out

    raw_obj_pos_all = np.asarray(motion.object_body_position, dtype=np.float32)[:T_use]
    raw_obj_quat_all = np.asarray(motion.object_body_wxyz, dtype=np.float32)[:T_use]
    if raw_obj_pos_all.ndim == 2:
        raw_obj_pos_all = raw_obj_pos_all[:, None]
        raw_obj_quat_all = raw_obj_quat_all[:, None]

    # Length-align in case a per-frame field was skipped during trimming.
    common_T = min(raw_obj_pos_all.shape[0], obj_body_pos_arr.shape[0])
    raw_obj_pos_all = raw_obj_pos_all[:common_T]
    raw_obj_quat_all = raw_obj_quat_all[:common_T]
    dst_obj_pos_all = obj_body_pos_arr[:common_T]
    dst_obj_quat_all = obj_body_wxyz_arr[:common_T]
    primary_r_rel = quat_mul(dst_obj_quat_all[:, 0], quat_conj(raw_obj_quat_all[:, 0]))
    raw_primary_pos = raw_obj_pos_all[:, 0]
    dst_primary_pos = dst_obj_pos_all[:, 0]

    for side in ("left", "right"):
        wrist_pos = getattr(motion, f"robot_{side}_wrist_position", None)
        if wrist_pos is None:
            continue
        out["hand_sides"].append(side)
        frames = getattr(motion, f"robot_{side}_frames", None) or []
        frame_names = getattr(motion, f"{side}_robot_frame_names", None) or []
        finger_joints = getattr(motion, f"robot_{side}_finger_joints", None) or []
        finger_joint_names = (
            getattr(motion, f"{side}_robot_finger_joint_names", None) or []
        )
        link_contacts = (
            getattr(motion, f"mano_{side}_link_contact_positions", None) or []
        )
        link_normals = getattr(motion, f"mano_{side}_link_contact_normals", None) or []
        obj_contacts = (
            getattr(motion, f"mano_{side}_object_contact_positions", None) or []
        )
        obj_normals = getattr(motion, f"mano_{side}_object_contact_normals", None) or []
        part_ids_attr = (
            getattr(motion, f"mano_{side}_object_contact_part_ids", None) or []
        )

        out["hand_frame_names"].append(list(frame_names))
        # Lift hand_frames_w into the planner frame. The consumer in
        # tracking_command._precompute_hand_keypoints_in_object_frame
        # combines these keypoint poses with object_body_position to build
        # the wrist/fingertip targets; passing through V2P-frame keypoints
        # against a planner-frame object pose silently produces targets up
        # to a metre off.
        if frames:
            frames_arr = np.asarray(frames, dtype=np.float32)[:common_T]
            frame_pos = transform_primary_pos(
                frames_arr[..., :3],
                raw_primary_pos,
                dst_primary_pos,
                primary_r_rel,
            )
            frame_quat = transform_primary_quat(frames_arr[..., 3:], primary_r_rel)
            out["hand_frames_w"].append(
                np.concatenate([frame_pos, frame_quat], axis=-1).tolist()
            )
        else:
            out["hand_frames_w"].append([])

        out["hand_finger_joint_names"].append(list(finger_joint_names))
        out["hand_finger_joints"].append(
            np.asarray(finger_joints, dtype=np.float32)[:T_use].tolist()
            if finger_joints
            else []
        )
        out["hand_contact_link_names"].append([])

        # part_ids are 1-indexed object-body indices that drive the per-body
        # contact transform. Some loaders carry them in a dedicated array
        # (often left at source fps), others embed them in the 4th column of
        # the contact-position arrays (already at planner fps). Probe both
        # and nearest-neighbor upsample the dedicated array if it hasn't
        # been interpolated.
        part_ids_arr: np.ndarray | None = None
        if len(part_ids_attr):
            src = np.asarray(part_ids_attr, dtype=np.int64)
            if src.shape[0] >= common_T:
                part_ids_arr = src[:common_T]
            elif src.shape[0] > 0:
                src_t = np.linspace(0.0, 1.0, src.shape[0])
                dst_t = np.linspace(0.0, 1.0, common_T)
                nn_idx = np.clip(
                    np.searchsorted(src_t, dst_t, side="right") - 1,
                    0,
                    src.shape[0] - 1,
                )
                part_ids_arr = src[nn_idx]
        if part_ids_arr is None and obj_contacts:
            oc_probe = np.asarray(obj_contacts, dtype=np.float32)
            if oc_probe.ndim == 3 and oc_probe.shape[-1] >= 4:
                part_ids_arr = np.rint(oc_probe[:common_T, :, 3]).astype(np.int64)
                inactive = np.linalg.norm(oc_probe[:common_T, :, :3], axis=-1) <= 1e-8
                part_ids_arr = np.where(inactive, 0, part_ids_arr)

        # Contacts ride the same per-body rigid transform as the object
        # bodies they reference. Positions translate + rotate, normals
        # rotate only.
        if obj_contacts:
            oc_a = np.asarray(obj_contacts, dtype=np.float32)[:common_T, :, :3]
            oc_transformed = transform_contact_pos_by_part(
                oc_a,
                raw_obj_pos_all,
                dst_obj_pos_all,
                raw_obj_quat_all,
                dst_obj_quat_all,
                part_ids_arr,
            )
            out["hand_object_contact_positions"].append(oc_transformed.tolist())
        else:
            out["hand_object_contact_positions"].append([])

        if obj_normals:
            on_a = np.asarray(obj_normals, dtype=np.float32)[:common_T, :, :3]
            on_transformed = transform_contact_dir_by_part(
                on_a,
                raw_obj_quat_all,
                dst_obj_quat_all,
                part_ids_arr,
            )
            out["hand_object_contact_normals"].append(on_transformed.tolist())
        else:
            out["hand_object_contact_normals"].append([])

        if link_contacts:
            lc_a = np.asarray(link_contacts, dtype=np.float32)[:common_T, :, :3]
            lc_transformed = transform_contact_pos_by_part(
                lc_a,
                raw_obj_pos_all,
                dst_obj_pos_all,
                raw_obj_quat_all,
                dst_obj_quat_all,
                part_ids_arr,
            )
            out["hand_link_contact_positions"].append(lc_transformed.tolist())
        else:
            out["hand_link_contact_positions"].append([])

        if link_normals:
            ln_a = np.asarray(link_normals, dtype=np.float32)[:common_T, :, :3]
            ln_transformed = transform_contact_dir_by_part(
                ln_a,
                raw_obj_quat_all,
                dst_obj_quat_all,
                part_ids_arr,
            )
            out["hand_link_contact_normals"].append(ln_transformed.tolist())
        else:
            out["hand_link_contact_normals"].append([])

        out["hand_object_contact_part_ids"].append(
            part_ids_arr.tolist() if part_ids_arr is not None else []
        )

        # Per-frame contact-active mask: 1 when at least one contact point
        # is recorded against the object, 0 otherwise. Derived from the
        # already-upsampled `obj_contacts` so the mask length matches the
        # other per-frame contact arrays. tracking_command refuses to load
        # motion files where both sides are absent, so always emit a
        # per-side mask (zero-filled in the worst case).
        if obj_contacts:
            cp = np.asarray(obj_contacts, dtype=np.float32)[:common_T, :, :3]
            active = (np.abs(cp).sum(axis=-1) > 1e-5).any(axis=-1).astype(np.float32)
        else:
            active = np.zeros((common_T,), dtype=np.float32)
        out["hand_contact_active"].append(active.tolist())

    return out
