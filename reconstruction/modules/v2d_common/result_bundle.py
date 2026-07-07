# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the portable ``result/`` bundle for hand-object recon pipelines.

The bundle intentionally mirrors the ego-e2e gsplat branch's stable surface:

    result/
      result.npz
      mesh.obj
      material.mtl
      <textures>
      manifest.json

Transforms in the NPZ are time-aligned on the frame index axis. Missing data is
represented by safe defaults plus explicit validity masks.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from v2d.common.datatypes import CameraIntrinsics, Transform3d


_TEXTURE_KEYS = {
    "map_Ka", "map_Kd", "map_Ks", "map_Ke", "map_Ns", "map_d",
    "map_Bump", "map_bump", "bump", "disp", "decal", "norm",
}


def _iter_frame_jsons(directory: str | None) -> list[tuple[int, str]]:
    if not directory or not os.path.isdir(directory):
        return []
    out: list[tuple[int, str]] = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if not stem.isdigit():
            continue
        out.append((int(stem), path))
    return out


def _infer_n_frames(frames_dir: str | None, *pose_dirs: str | None) -> int:
    frame_indices: list[int] = []
    if frames_dir and os.path.isdir(frames_dir):
        for path in (
            glob.glob(os.path.join(frames_dir, "*.png")) +
            glob.glob(os.path.join(frames_dir, "*.jpg")) +
            glob.glob(os.path.join(frames_dir, "*.jpeg"))
        ):
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem.isdigit():
                frame_indices.append(int(stem))
    for directory in pose_dirs:
        frame_indices.extend(fidx for fidx, _ in _iter_frame_jsons(directory))
    if not frame_indices:
        raise FileNotFoundError(
            "Could not infer frame count from frames_dir or pose directories."
        )
    return max(frame_indices) + 1


def _load_transform_matrices(
    poses_dir: str | None,
    n_frames: int,
    *,
    pose_convention: str = "camera_to_world",
) -> tuple[np.ndarray, np.ndarray]:
    """Load per-frame ``Transform3d`` JSONs into ``(N, 4, 4)`` matrices.

    ``pose_convention`` describes the JSON files. The returned matrices follow
    the same convention unless ``pose_convention="world_to_camera"``, in which
    case they are inverted so callers receive camera-to-world matrices.
    """
    if pose_convention not in {"camera_to_world", "world_to_camera", "as_is"}:
        raise ValueError(f"Unsupported pose convention: {pose_convention!r}")
    matrices = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    valid = np.zeros((n_frames,), dtype=bool)
    for fidx, path in _iter_frame_jsons(poses_dir):
        if not (0 <= fidx < n_frames):
            continue
        M = Transform3d.load(path).to_matrix().astype(np.float32)
        if pose_convention == "world_to_camera":
            M = np.linalg.inv(M).astype(np.float32)
        matrices[fidx] = M
        valid[fidx] = True
    return matrices, valid


def _load_object_to_camera(
    poses_dir: str | None,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _load_transform_matrices(poses_dir, n_frames, pose_convention="as_is")


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(axis_angle.astype(np.float64)).as_matrix()


def _load_hand_track(track_dir: str | None, n_frames: int) -> dict[str, np.ndarray]:
    """Load one aligned/refined hand-track directory into stacked arrays."""
    betas = np.zeros((10,), dtype=np.float32)
    wrist_orient = np.zeros((n_frames, 3), dtype=np.float32)
    wrist_trans = np.zeros((n_frames, 3), dtype=np.float32)
    finger_pose = np.zeros((n_frames, 15, 3), dtype=np.float32)
    scale = np.ones((n_frames,), dtype=np.float32)
    is_valid = np.zeros((n_frames,), dtype=bool)
    wrist_to_camera = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    got_betas = False

    for fidx, path in _iter_frame_jsons(track_dir):
        if not (0 <= fidx < n_frames):
            continue
        with open(path) as f:
            rec = json.load(f)
        mano = rec.get("mano", {})
        if not got_betas and "betas" in mano:
            betas = np.asarray(mano["betas"], dtype=np.float32).reshape(10)
            got_betas = True
        orient = np.asarray(mano.get("global_orient", [0.0, 0.0, 0.0]),
                            dtype=np.float32).reshape(3)
        hand_pose = np.asarray(mano.get("hand_pose", np.zeros((15, 3))),
                               dtype=np.float32).reshape(15, 3)
        cam_t = np.asarray(rec.get("cam_t", [0.0, 0.0, 0.0]),
                           dtype=np.float32).reshape(3)

        wrist_orient[fidx] = orient
        finger_pose[fidx] = hand_pose
        wrist_trans[fidx] = cam_t
        scale[fidx] = float(rec.get("hand_scale", 1.0))
        is_valid[fidx] = True

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = _axis_angle_to_matrix(orient).astype(np.float32)
        T[:3, 3] = cam_t
        wrist_to_camera[fidx] = T

    return {
        "betas": betas,
        "wrist_orient": wrist_orient,
        "wrist_trans": wrist_trans,
        "finger_pose": finger_pose,
        "scale": scale,
        "is_valid": is_valid,
        "wrist_to_camera": wrist_to_camera,
    }


def _compose_batched(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.einsum("nij,njk->nik", A, B).astype(np.float32)


_GRAVITY_TARGETS = {
    # Name means "this world axis points up"; gravity points opposite it.
    "z_up":   np.array([0.0, 0.0, -1.0], dtype=np.float64),
    "z_down": np.array([0.0, 0.0,  1.0], dtype=np.float64),
    "y_up":   np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "y_down": np.array([0.0,  1.0, 0.0], dtype=np.float64),
    "x_up":   np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "x_down": np.array([ 1.0, 0.0, 0.0], dtype=np.float64),
}

_GRAVITY_ALIGNMENT_METHOD_Z_UP = "geocalib_yup_to_opencv_yflip_zup_frame"
_GRAVITY_ALIGNMENT_METHOD_VECTOR = "geocalib_yup_to_opencv_yflip_vector_align"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def _proper_rotation(R: np.ndarray) -> np.ndarray:
    """Project a noisy 3x3 block onto SO(3)."""
    U, _, Vt = np.linalg.svd(R.astype(np.float64))
    out = U @ Vt
    if np.linalg.det(out) < 0:
        U[:, -1] *= -1.0
        out = U @ Vt
    return out


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return R such that ``R @ source ~= target`` for unit 3-vectors."""
    a = _normalize(source.astype(np.float64))
    b = _normalize(target.astype(np.float64))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-8:
        basis = np.eye(3, dtype=np.float64)
        axis = basis[int(np.argmin(np.abs(a)))]
        axis = _normalize(np.cross(a, axis))
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + K + K @ K * ((1.0 - c) / (s * s))


def _target_gravity_vector(target: str) -> np.ndarray:
    if target not in _GRAVITY_TARGETS:
        raise ValueError(
            f"Unsupported gravity target {target!r}. "
            f"Choices: {sorted(_GRAVITY_TARGETS)}"
        )
    return _GRAVITY_TARGETS[target].copy()


def _iter_geocalib_gravity_samples(calibration_path: str) -> list[tuple[int, np.ndarray]]:
    with open(calibration_path) as f:
        calibration = json.load(f)

    samples: list[tuple[int, np.ndarray]] = []
    for sample in calibration.get("samples", []):
        gravity = sample.get("gravity", {})
        vec = gravity.get("vector_camera")
        if vec is None:
            continue
        samples.append((int(sample.get("frame_index", 0)), _normalize(np.asarray(vec))))

    if not samples:
        vec = calibration.get("gravity", {}).get("vector_camera")
        if vec is None:
            raise ValueError(f"No GeoCalib gravity vector in {calibration_path}")
        samples.append((0, _normalize(np.asarray(vec))))
    return samples


def _geocalib_gravity_to_opencv_camera(gravity_camera: np.ndarray) -> np.ndarray:
    """Convert GeoCalib's Y-up camera gravity vector to OpenCV camera axes.

    GeoCalib's gravity frame reports image vertical with the opposite sign from
    the OpenCV convention used by the result bundle (+Y down, +Z forward).
    """
    g = np.asarray(gravity_camera, dtype=np.float64).copy()
    g[1] *= -1.0
    return _normalize(g)


def _estimate_world_gravity_and_forward_from_result(
    result_arrays: dict[str, np.ndarray],
    geocalib_calibration_path: str,
) -> tuple[np.ndarray, np.ndarray, list[int], list[list[float]], list[list[float]]]:
    c2w = result_arrays["camera_to_world_transform"]
    camera_valid = result_arrays.get(
        "camera_is_valid",
        np.ones((c2w.shape[0],), dtype=bool),
    ).astype(bool)

    frame_indices: list[int] = []
    gravity_world_samples: list[np.ndarray] = []
    forward_world_samples: list[np.ndarray] = []
    for frame_idx, gravity_camera_raw in _iter_geocalib_gravity_samples(geocalib_calibration_path):
        if not (0 <= frame_idx < c2w.shape[0]) or not bool(camera_valid[frame_idx]):
            continue
        R_c2w = _proper_rotation(c2w[frame_idx, :3, :3])
        gravity_camera = _geocalib_gravity_to_opencv_camera(gravity_camera_raw)
        gravity_world_samples.append(_normalize(R_c2w @ gravity_camera))
        forward_world_samples.append(_normalize(R_c2w @ np.array([0.0, 0.0, 1.0], dtype=np.float64)))
        frame_indices.append(frame_idx)

    if not gravity_world_samples:
        raise RuntimeError(
            "No GeoCalib gravity samples overlapped valid camera poses in result.npz"
        )

    gravity_world = _normalize(np.mean(np.stack(gravity_world_samples, axis=0), axis=0))
    forward_world = _normalize(np.mean(np.stack(forward_world_samples, axis=0), axis=0))
    return (
        gravity_world,
        forward_world,
        frame_indices,
        [g.tolist() for g in gravity_world_samples],
        [f.tolist() for f in forward_world_samples],
    )


def _gravity_alignment_method(target: str) -> str:
    return _GRAVITY_ALIGNMENT_METHOD_Z_UP if target == "z_up" else _GRAVITY_ALIGNMENT_METHOD_VECTOR


def _z_up_alignment_from_gravity_and_forward(
    gravity_world: np.ndarray,
    forward_world: np.ndarray,
) -> tuple[np.ndarray, dict]:
    z_up = _normalize(-gravity_world)
    y_forward = np.asarray(forward_world, dtype=np.float64)
    y_forward = y_forward - float(np.dot(y_forward, z_up)) * z_up
    if np.linalg.norm(y_forward) < 1e-8:
        y_forward = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_forward = y_forward - float(np.dot(y_forward, z_up)) * z_up
    y_forward = _normalize(y_forward)
    x_right = _normalize(np.cross(y_forward, z_up))
    y_forward = _normalize(np.cross(z_up, x_right))

    basis = np.stack([x_right, y_forward, z_up], axis=1)
    R_align = basis.T
    return R_align, {
        "x_right_axis_in_pre_alignment_world": x_right.tolist(),
        "y_forward_axis_in_pre_alignment_world": y_forward.tolist(),
        "z_up_axis_in_pre_alignment_world": z_up.tolist(),
        "basis_pre_alignment_world_columns": basis.tolist(),
    }


def _existing_gravity_alignment_transform(arrays: dict[str, np.ndarray]) -> np.ndarray | None:
    applied = arrays.get("gravity_alignment_applied")
    if applied is None or not bool(np.asarray(applied).reshape(-1)[0]):
        return None
    A = arrays.get("gravity_alignment_transform")
    if A is None:
        return None
    return np.asarray(A, dtype=np.float64).reshape(4, 4)


def _undo_existing_gravity_alignment(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    old_A = _existing_gravity_alignment_transform(arrays)
    if old_A is None:
        return dict(arrays), None

    old_A_inv = np.linalg.inv(old_A)
    out = dict(arrays)
    for key in (
        "camera_to_world_transform",
        "object_to_world_transform",
        "hand_left_wrist_to_world_transform",
        "hand_right_wrist_to_world_transform",
    ):
        if key in out:
            out[key] = _left_multiply_transforms(out[key], old_A_inv.astype(np.float32))
    return out, old_A

def _left_multiply_transforms(T: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk->nik", A, T).astype(np.float32)


def result_bundle_has_gravity_alignment(result_dir: str, target: str | None = None) -> bool:
    manifest_path = os.path.join(result_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    alignment = manifest.get("gravity_alignment", {})
    if not bool(alignment.get("enabled", False)):
        return False

    if target is not None and alignment.get("target") != target:
        return False

    method = alignment.get("method")
    if target is not None:
        return method == _gravity_alignment_method(target)
    return method in {_GRAVITY_ALIGNMENT_METHOD_Z_UP, _GRAVITY_ALIGNMENT_METHOD_VECTOR}


def gravity_align_result_bundle(
    result_dir: str,
    geocalib_calibration_path: str,
    *,
    target: str = "z_up",
) -> str:
    """Rotate world-space arrays in ``result.npz`` so gravity matches target.

    GeoCalib reports gravity in a Y-up camera frame. The result bundle uses
    OpenCV camera axes (+X right, +Y image-down, +Z forward), so GeoCalib's Y
    component is flipped before combining gravity with camera-to-world poses.

    For ``target="z_up"`` the remaining yaw is fixed by projecting the average
    camera-forward direction onto the horizontal plane, yielding a right-handed
    world frame: +Z up, +Y horizontal camera-forward, +X right. Camera-space
    transforms and mesh assets are unchanged.
    """
    expected_method = _gravity_alignment_method(target)
    if result_bundle_has_gravity_alignment(result_dir, target=target):
        print(f"  [skip] result/ already has current gravity_alignment metadata: {result_dir}")
        return os.path.join(result_dir, "result.npz")

    result_npz = os.path.join(result_dir, "result.npz")
    manifest_path = os.path.join(result_dir, "manifest.json")
    if not os.path.exists(result_npz):
        raise FileNotFoundError(f"Missing result bundle NPZ: {result_npz}")
    if not os.path.exists(geocalib_calibration_path):
        raise FileNotFoundError(
            f"Missing GeoCalib calibration JSON: {geocalib_calibration_path}"
        )

    with np.load(result_npz) as npz:
        arrays = {k: npz[k] for k in npz.files}

    pre_arrays, old_alignment = _undo_existing_gravity_alignment(arrays)
    (
        gravity_world,
        forward_world,
        frame_indices,
        gravity_world_samples,
        forward_world_samples,
    ) = _estimate_world_gravity_and_forward_from_result(
        pre_arrays,
        geocalib_calibration_path,
    )
    target_gravity = _target_gravity_vector(target)

    alignment_details: dict = {}
    if target == "z_up":
        R_align, alignment_details = _z_up_alignment_from_gravity_and_forward(
            gravity_world,
            forward_world,
        )
    else:
        R_align = _rotation_between_vectors(gravity_world, target_gravity)

    A = np.eye(4, dtype=np.float32)
    A[:3, :3] = R_align.astype(np.float32)

    arrays_out = dict(pre_arrays)
    for key in (
        "camera_to_world_transform",
        "object_to_world_transform",
        "hand_left_wrist_to_world_transform",
        "hand_right_wrist_to_world_transform",
    ):
        if key in arrays_out:
            arrays_out[key] = _left_multiply_transforms(arrays_out[key], A)

    arrays_out["gravity_alignment_transform"] = A
    arrays_out["gravity_in_world_before_alignment"] = gravity_world.astype(np.float32)
    arrays_out["gravity_target_world"] = target_gravity.astype(np.float32)
    arrays_out["gravity_alignment_applied"] = np.array(True)

    tmp_npz = os.path.join(result_dir, "result.gravity_tmp.npz")
    np.savez(tmp_npz, **arrays_out)
    os.replace(tmp_npz, result_npz)

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    keys = sorted(arrays_out.keys())
    manifest["keys"] = keys
    manifest["gravity_alignment"] = {
        "enabled": True,
        "target": target,
        "target_gravity_world": target_gravity.tolist(),
        "gravity_in_world_before_alignment": gravity_world.tolist(),
        "alignment_transform": A.tolist(),
        "source_calibration_path": geocalib_calibration_path,
        "sample_frame_indices": frame_indices,
        "sample_gravity_vectors_world": gravity_world_samples,
        "sample_camera_forward_vectors_world": forward_world_samples,
        "method": expected_method,
        "geocalib_y_flipped_to_opencv": True,
        "old_alignment_transform": old_alignment.tolist() if old_alignment is not None else None,
        **alignment_details,
    }
    if target == "z_up":
        manifest["world_coordinate_convention"] = {
            "handedness": "right_handed",
            "x_axis": "right",
            "y_axis": "average_horizontal_camera_forward",
            "z_axis": "up",
            "gravity_axis": "negative_z",
        }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        "  Gravity-aligned result bundle "
        f"({target}; method={expected_method}; "
        f"gravity {gravity_world.tolist()} -> {target_gravity.tolist()})"
    )
    return result_npz

def _resolve_source_path(base_dir: str, raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str(Path(base_dir) / path)


def _rewrite_mtl_to_result(mtl_src: str, result_dir: str) -> list[str]:
    copied_textures: list[str] = []
    mtl_src_dir = os.path.dirname(os.path.abspath(mtl_src))
    mtl_dst = os.path.join(result_dir, os.path.basename(mtl_src))
    lines: list[str] = []

    with open(mtl_src) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lines.append(line)
                continue
            parts = stripped.split()
            if parts and parts[0] in _TEXTURE_KEYS and len(parts) >= 2:
                tex_token = parts[-1]
                tex_src = _resolve_source_path(mtl_src_dir, tex_token)
                tex_name = os.path.basename(tex_token)
                if os.path.exists(tex_src):
                    tex_dst = os.path.join(result_dir, tex_name)
                    if os.path.abspath(tex_src) != os.path.abspath(tex_dst):
                        shutil.copy2(tex_src, tex_dst)
                    copied_textures.append(tex_name)
                parts[-1] = tex_name
                lines.append(" ".join(parts) + "\n")
            else:
                lines.append(line)

    with open(mtl_dst, "w") as f:
        f.writelines(lines)
    return copied_textures


def _copy_mesh_assets(mesh_path: str, result_dir: str, dest_name: str = "mesh.obj") -> dict:
    """Copy OBJ, MTL, and texture files into ``result_dir``.

    OBJ ``mtllib`` entries and MTL texture entries are rewritten to local
    basenames so the bundle is portable even when source assets lived in a
    nested directory.
    """
    os.makedirs(result_dir, exist_ok=True)
    mesh_src_dir = os.path.dirname(os.path.abspath(mesh_path))
    mesh_dst = os.path.join(result_dir, dest_name)
    mtl_files: list[str] = []
    obj_lines: list[str] = []

    with open(mesh_path) as f:
        for line in f:
            if line.startswith("mtllib "):
                raw_names = line.strip().split()[1:]
                for raw in raw_names:
                    mtl_src = _resolve_source_path(mesh_src_dir, raw)
                    if os.path.exists(mtl_src):
                        mtl_files.append(mtl_src)
                obj_lines.append(
                    "mtllib " + " ".join(os.path.basename(p) for p in raw_names) + "\n"
                )
            else:
                obj_lines.append(line)

    with open(mesh_dst, "w") as f:
        f.writelines(obj_lines)

    copied_mtls: list[str] = []
    copied_textures: list[str] = []
    for mtl_src in mtl_files:
        copied_mtls.append(os.path.basename(mtl_src))
        copied_textures.extend(_rewrite_mtl_to_result(mtl_src, result_dir))

    return {
        "mesh": dest_name,
        "materials": sorted(set(copied_mtls)),
        "textures": sorted(set(copied_textures)),
    }


def write_result_bundle(
    result_dir: str,
    frames_dir: str,
    intrinsics_path: str,
    mesh_path: str,
    *,
    object_poses_dir: str | None = None,
    object_scale: float = 1.0,
    camera_to_world_dir: str | None = None,
    camera_pose_convention: str = "camera_to_world",
    left_hand_dir: str | None = None,
    right_hand_dir: str | None = None,
    source_manifest: dict | None = None,
) -> str:
    """Write ``result.npz`` + mesh assets and return the NPZ path."""
    n_frames = _infer_n_frames(
        frames_dir,
        camera_to_world_dir,
        object_poses_dir,
        left_hand_dir,
        right_hand_dir,
    )
    intr = CameraIntrinsics.load(intrinsics_path)
    camera_intrinsics = np.array([intr.fx, intr.fy, intr.cx, intr.cy], dtype=np.float32)

    if camera_to_world_dir and os.path.isdir(camera_to_world_dir):
        camera_to_world_transform, camera_is_valid = _load_transform_matrices(
            camera_to_world_dir,
            n_frames,
            pose_convention=camera_pose_convention,
        )
    else:
        camera_to_world_transform = np.tile(
            np.eye(4, dtype=np.float32), (n_frames, 1, 1)
        )
        camera_is_valid = np.ones((n_frames,), dtype=bool)

    object_to_camera_transform, object_is_valid = _load_object_to_camera(
        object_poses_dir, n_frames,
    )
    object_to_world_transform = _compose_batched(
        camera_to_world_transform, object_to_camera_transform,
    )

    left = _load_hand_track(left_hand_dir, n_frames)
    right = _load_hand_track(right_hand_dir, n_frames)
    left_wrist_to_world = _compose_batched(
        camera_to_world_transform, left["wrist_to_camera"],
    )
    right_wrist_to_world = _compose_batched(
        camera_to_world_transform, right["wrist_to_camera"],
    )

    os.makedirs(result_dir, exist_ok=True)
    mesh_manifest = _copy_mesh_assets(mesh_path, result_dir, dest_name="mesh.obj")
    result_npz = os.path.join(result_dir, "result.npz")
    np.savez(
        result_npz,
        camera_to_world_transform=camera_to_world_transform,
        camera_is_valid=camera_is_valid,
        camera_intrinsics=camera_intrinsics,
        object_to_camera_transform=object_to_camera_transform,
        object_to_world_transform=object_to_world_transform,
        object_scale=np.float32(object_scale),
        object_is_valid=object_is_valid,
        hand_left_betas=left["betas"],
        hand_left_wrist_orient_in_camera=left["wrist_orient"],
        hand_left_wrist_trans_in_camera=left["wrist_trans"],
        hand_left_wrist_to_camera_transform=left["wrist_to_camera"],
        hand_left_wrist_to_world_transform=left_wrist_to_world,
        hand_left_finger_pose=left["finger_pose"],
        hand_left_scale=left["scale"],
        hand_left_is_valid=left["is_valid"],
        hand_right_betas=right["betas"],
        hand_right_wrist_orient_in_camera=right["wrist_orient"],
        hand_right_wrist_trans_in_camera=right["wrist_trans"],
        hand_right_wrist_to_camera_transform=right["wrist_to_camera"],
        hand_right_wrist_to_world_transform=right_wrist_to_world,
        hand_right_finger_pose=right["finger_pose"],
        hand_right_scale=right["scale"],
        hand_right_is_valid=right["is_valid"],
    )

    with np.load(result_npz) as npz:
        result_keys = sorted(npz.files)

    manifest = {
        "schema_version": 1,
        "n_frames": n_frames,
        "result_npz": "result.npz",
        "camera_pose_convention": "camera_to_world",
        "mesh": mesh_manifest,
        "sources": {
            "frames_dir": frames_dir,
            "intrinsics_path": intrinsics_path,
            "mesh_path": mesh_path,
            "object_poses_dir": object_poses_dir,
            "object_scale": object_scale,
            "camera_to_world_dir": camera_to_world_dir,
            "input_camera_pose_convention": camera_pose_convention,
            "left_hand_dir": left_hand_dir,
            "right_hand_dir": right_hand_dir,
            **(source_manifest or {}),
        },
        "keys": result_keys,
    }
    with open(os.path.join(result_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return result_npz


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--intrinsics_path", required=True)
    p.add_argument("--mesh_path", required=True)
    p.add_argument("--object_poses_dir", default=None)
    p.add_argument("--object_scale", type=float, default=1.0)
    p.add_argument("--camera_to_world_dir", default=None)
    p.add_argument("--camera_pose_convention", default="camera_to_world",
                   choices=("camera_to_world", "world_to_camera", "as_is"))
    p.add_argument("--left_hand_dir", default=None)
    p.add_argument("--right_hand_dir", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = write_result_bundle(
        result_dir=args.result_dir,
        frames_dir=args.frames_dir,
        intrinsics_path=args.intrinsics_path,
        mesh_path=args.mesh_path,
        object_poses_dir=args.object_poses_dir,
        object_scale=args.object_scale,
        camera_to_world_dir=args.camera_to_world_dir,
        camera_pose_convention=args.camera_pose_convention,
        left_hand_dir=args.left_hand_dir,
        right_hand_dir=args.right_hand_dir,
    )
    print(f"Wrote result bundle -> {out}")


if __name__ == "__main__":
    main()
