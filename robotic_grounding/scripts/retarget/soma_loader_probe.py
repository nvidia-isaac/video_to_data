# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loader-only verification probe for SOMA-to-G1 retargeting.

Runs inside the retarget Docker image and prints the key SOMA schema /
shape / coordinate-frame information that milestone 2 of the SOMA-to-G1
plan requires before the full retargeting script is exercised.

What this probe checks:
- ``soma_params.npz`` contains the expected fields with sane shapes / dtypes
- the loader returns ``joints``, ``joints_wxyz``, ``vertices``, ``num_frames``
- ``keep_root=False`` semantics: the reconstructed Hips world position lies
  near ``transl[t]`` (after the loader's first-frame anchoring is undone)
- coordinate frame: foot Z is below pelvis Z and the ground plane is roughly
  consistent with ``ground_plane.json``

Usage (inside Docker):

    python scripts/retarget/soma_loader_probe.py <data_folder>
    python scripts/retarget/soma_loader_probe.py <data_folder> --visualize

The ``--visualize`` flag opens a viser server (port 8080) that plays the
SOMA body mesh and the same ``object_mesh/output_aligned.glb`` asset consumed
by ``soma_to_g1.py`` in the loader-anchored source frame so you can confirm
the raw SOMA inputs without involving G1 IK.

This script is read-only; it does not write parquet output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser
from robotic_grounding.retarget.params import SOMA_JOINTS_ORDER
from robotic_grounding.retarget.read_soma import SOMA
from robotic_grounding.retarget.robot_config import load_robot_config
from scipy.spatial.transform import Rotation as R


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="SOMA loader probe")
    p.add_argument(
        "data_folder",
        type=str,
        help="Path to a SOMA reconstruction folder containing soma_params.npz.",
    )
    p.add_argument(
        "--identity-model-type",
        type=str,
        default="mhr",
    )
    p.add_argument(
        "--soma-data-root",
        type=str,
        default=None,
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Open a viser server on port 8080 and play the SOMA body mesh + "
            "object mesh in the loader-anchored source frame. Bypasses the G1 "
            "retargeter entirely."
        ),
    )
    p.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser port when --visualize is set.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback frame rate when --visualize is set.",
    )
    p.add_argument(
        "--anchor",
        action="store_true",
        help=(
            "Apply the loader's first-frame anchoring (Hips at origin, "
            "frame-0 root rotation removed). The G1 retargeter does NOT "
            "use this anchoring, so the default is to keep the body in "
            "raw SOMA world frame, matching what the IK consumes."
        ),
    )
    p.add_argument(
        "--robot-frame",
        action="store_true",
        help=(
            "Rotate the visualized body and object trajectory by the loaded "
            "robot config's ``r_world`` so the scene matches the G1 robot "
            "convention (X=forward, Y=left, Z=up). Diagnostics still "
            "print in both source and robot frames regardless of this "
            "flag; this only affects the viser scene."
        ),
    )
    p.add_argument(
        "--robot-name",
        type=str,
        default="g1",
        help=(
            "Robot config folder under "
            "`source/robotic_grounding/robotic_grounding/retarget/configs/`. "
            "Used to resolve the source-to-robot rotation matrix."
        ),
    )
    return p.parse_args()


def _load_object_mesh(folder: Path) -> trimesh.Trimesh | None:
    """Load the retargeter object mesh as a single ``trimesh.Trimesh``.

    Returns None when the GLB is missing so the visualizer still runs body-only.
    """
    glb = folder / "object_mesh" / "output_aligned.glb"
    if not glb.is_file():
        return None
    mesh = trimesh.load(glb, force="scene")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"[probe] WARNING: could not extract a Trimesh from {glb}")
        return None
    return mesh


# Object poses in ``poses.npy`` are in OpenCV camera convention. SOMA
# body lives in a Y-up convention; convert before anchoring so body and
# object share a world frame. Mirrors ``soma_to_g1._convert_object_poses_cv_to_soma``.
_R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0, 1.0])


def _convert_object_poses_cv_to_soma(object_poses_cv: np.ndarray) -> np.ndarray:
    """Apply OpenCV->SOMA world-frame conversion to a (T, 4, 4) object trajectory."""
    return np.einsum("ij,tjk->tik", _R_CV_TO_SOMA, object_poses_cv)


def _apply_first_frame_anchor(
    object_poses_world: np.ndarray,
    transl_first: np.ndarray,
    R_first_inv: np.ndarray,
) -> np.ndarray:
    """Mirror of ``soma_to_g1._apply_first_frame_anchor_to_objects``.

    Inlined here so the probe stays a single-file dependency-light script.
    """
    norm_transform = np.eye(4)
    norm_transform[:3, :3] = R_first_inv
    norm_transform[:3, 3] = -R_first_inv @ np.asarray(transl_first, dtype=np.float64)
    return np.array([norm_transform @ pose for pose in object_poses_world])


def _rotate_points_to_robot(
    points: np.ndarray, R_src_to_robot: np.ndarray
) -> np.ndarray:
    """Rotate ``(..., 3)`` points from SOMA frame to robot frame (Z-up, X-fwd).

    Mirrors ``WholeBodyKinematics.transform_source_position``: ``p @ R.T``.
    """
    return np.asarray(points, dtype=np.float64) @ np.asarray(R_src_to_robot).T


def _rotate_object_poses_to_robot(
    object_poses: np.ndarray,
    R_src_to_robot: np.ndarray,
) -> np.ndarray:
    """Rotate (T, 4, 4) homogeneous object poses from SOMA frame to robot frame.

    Applied as a left-multiply by the homogeneous form of ``R_SOMA_TO_ROBOT``,
    so both translation and orientation columns end up in the robot frame.
    """
    R_homog = np.eye(4)
    R_homog[:3, :3] = np.asarray(R_src_to_robot, dtype=np.float64)
    return np.einsum("ij,tjk->tik", R_homog, np.asarray(object_poses, dtype=np.float64))


def _visualize_raw_soma(
    *,
    server: viser.ViserServer,
    motion: dict,
    soma_faces: np.ndarray | None,
    object_mesh: trimesh.Trimesh | None,
    object_poses: np.ndarray,
    fps: float,
) -> None:
    """Drive a viser scene with the loader-anchored body + object trajectory.

    Adds a slider so individual frames can be inspected manually. Frame 0
    is shown initially, then a self-advancing loop steps through frames at
    ``fps`` until the user disconnects.
    """
    vertices = motion["vertices"]
    num_frames = motion["num_frames"]

    body_handle = None
    if soma_faces is not None:
        body_handle = server.scene.add_mesh_simple(
            "/soma/body",
            vertices=np.asarray(vertices[0], dtype=np.float32),
            faces=np.asarray(soma_faces, dtype=np.int32),
            color=(220, 200, 180),
            opacity=0.55,
        )

    object_handle = None
    if object_mesh is not None and object_poses.shape[0] > 0:
        object_handle = server.scene.add_mesh_simple(
            "/soma/object",
            vertices=np.asarray(object_mesh.vertices, dtype=np.float32),
            faces=np.asarray(object_mesh.faces, dtype=np.int32),
            color=(120, 200, 255),
        )

    # Wireframe-style XYZ axes at the world origin so the source frame is
    # easy to interpret (X red, Y green, Z blue per viser default).
    server.scene.add_frame("/soma/origin", axes_length=0.5, axes_radius=0.005)

    frame_slider = server.gui.add_slider(
        "frame", min=0, max=num_frames - 1, step=1, initial_value=0
    )

    def _set_frame(t: int) -> None:
        if body_handle is not None:
            body_handle.vertices = np.asarray(vertices[t], dtype=np.float32)
        if object_handle is not None and t < object_poses.shape[0]:
            pose = object_poses[t]
            xyzw = R.from_matrix(pose[:3, :3]).as_quat()
            wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
            object_handle.position = np.asarray(pose[:3, 3], dtype=np.float64)
            object_handle.wxyz = wxyz

    @frame_slider.on_update
    def _on_slider(_: viser.GuiEvent) -> None:
        _set_frame(int(frame_slider.value))

    _set_frame(0)
    print(
        f"[probe] viser visualizer running on port {server.get_port()} "
        f"({num_frames} frames, fps={fps}). Ctrl+C to stop."
    )

    period = 1.0 / max(fps, 1e-6)
    frame_idx = 0
    try:
        while True:
            frame_slider.value = frame_idx
            time.sleep(period)
            frame_idx = (frame_idx + 1) % num_frames
    except KeyboardInterrupt:
        print("[probe] viser stopped.")


def main() -> int:
    """Run the probe."""
    args = parse_args()
    folder = Path(args.data_folder).resolve()
    soma_npz = folder / "soma_params.npz"
    poses_npy = folder / "poses.npy"
    ground_json = folder / "ground_plane.json"
    glb = folder / "object_mesh" / "output_aligned.glb"
    print(f"[probe] folder           : {folder}")
    print(f"[probe] soma_params.npz  : {soma_npz} (exists={soma_npz.is_file()})")
    print(f"[probe] poses.npy        : {poses_npy} (exists={poses_npy.is_file()})")
    print(f"[probe] ground_plane.json: {ground_json} (exists={ground_json.is_file()})")
    print(f"[probe] object_mesh/output_aligned.glb: {glb} (exists={glb.is_file()})")

    raw = np.load(soma_npz, allow_pickle=True)
    print()
    print("[probe] soma_params.npz fields:")
    for key in raw.files:
        arr = raw[key]
        if arr.dtype == object:
            print(f"  {key}: dtype=object shape={getattr(arr, 'shape', None)}")
        else:
            print(f"  {key}: dtype={arr.dtype} shape={getattr(arr, 'shape', None)}")
    keep_root = bool(np.asarray(raw["keep_root"]).item())
    rotation_repr = (
        str(raw["rotation_repr"].item()) if "rotation_repr" in raw.files else "rotvec"
    )
    unit = str(raw["unit"].item())
    identity_model_type = str(raw["identity_model_type"].item())
    print()
    print(
        f"[probe] keep_root={keep_root} rotation_repr={rotation_repr} unit={unit} "
        f"identity_model_type={identity_model_type}"
    )

    soma_names = [str(n) for n in raw["joint_names"]]
    if soma_names != SOMA_JOINTS_ORDER:
        diff = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(soma_names, SOMA_JOINTS_ORDER, strict=False))
            if a != b
        ]
        print(f"[probe] WARNING: SOMA_JOINTS_ORDER mismatch at indices: {diff[:5]}")
    else:
        print(f"[probe] SOMA_JOINTS_ORDER matches export ({len(soma_names)} joints).")

    print()
    print("[probe] running SOMA.load_motion ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    soma = SOMA(
        data_root=args.soma_data_root,
        identity_model_type=args.identity_model_type,
        device=device,
    )
    motion = soma.load_motion(soma_npz, normalize=args.anchor)
    if not args.anchor:
        print("[probe] body kept in raw SOMA world frame (matches retarget IK input).")
    else:
        print("[probe] --anchor: applied first-frame Hips/orient anchoring.")
    joints = motion["joints"]
    joints_wxyz = motion["joints_wxyz"]
    vertices = motion["vertices"]
    print(
        f"[probe] joints.shape={joints.shape} joints_wxyz.shape={joints_wxyz.shape} "
        f"vertices.shape={vertices.shape} num_frames={motion['num_frames']}"
    )

    hips_idx = SOMA_JOINTS_ORDER.index("Hips")
    head_idx = SOMA_JOINTS_ORDER.index("Head")
    left_foot_idx = SOMA_JOINTS_ORDER.index("LeftFoot")
    right_foot_idx = SOMA_JOINTS_ORDER.index("RightFoot")
    left_hand_idx = SOMA_JOINTS_ORDER.index("LeftHand")
    right_hand_idx = SOMA_JOINTS_ORDER.index("RightHand")
    print()
    print("[probe] frame 0 source-frame positions (after first-frame anchoring):")
    for label, idx in (
        ("Hips", hips_idx),
        ("Head", head_idx),
        ("LeftFoot", left_foot_idx),
        ("RightFoot", right_foot_idx),
        ("LeftHand", left_hand_idx),
        ("RightHand", right_hand_idx),
    ):
        print(f"  {label:>10s}: {joints[0, idx]}")

    # Transform frame-0 positions into the robot frame (x=forward, y=left,
    # z=up) using the loaded robot config's ``r_world`` to confirm the IK
    # targets land where the G1 expects them.
    R_src_to_robot = np.asarray(
        load_robot_config(args.robot_name).r_world, dtype=np.float64
    )
    print()
    print(
        "[probe] frame 0 robot-frame targets via robot_config.r_world (X=fwd Y=left Z=up):"
    )
    for label, idx in (
        ("Hips", hips_idx),
        ("Head", head_idx),
        ("LeftFoot", left_foot_idx),
        ("RightFoot", right_foot_idx),
        ("LeftHand", left_hand_idx),
        ("RightHand", right_hand_idx),
    ):
        p_robot = joints[0, idx] @ R_src_to_robot.T
        print(f"  {label:>10s}: {p_robot}")
    body_height = float(
        joints[0, head_idx, 0]
        - min(joints[0, left_foot_idx, 0], joints[0, right_foot_idx, 0])
    )
    print(f"  estimated body height (head_X - min_foot_X): {body_height:+.4f} m")

    print()
    print("[probe] foot Z stats over sequence:")
    print(
        f"  LeftFoot Z range : [{joints[:, left_foot_idx, 2].min():+.4f}, "
        f"{joints[:, left_foot_idx, 2].max():+.4f}]"
    )
    print(
        f"  RightFoot Z range: [{joints[:, right_foot_idx, 2].min():+.4f}, "
        f"{joints[:, right_foot_idx, 2].max():+.4f}]"
    )
    print(
        f"  Hips Z range     : [{joints[:, hips_idx, 2].min():+.4f}, "
        f"{joints[:, hips_idx, 2].max():+.4f}]"
    )
    print(
        f"  Head Z range     : [{joints[:, head_idx, 2].min():+.4f}, "
        f"{joints[:, head_idx, 2].max():+.4f}]"
    )

    if ground_json.is_file():
        with ground_json.open() as f:
            gp = json.load(f)
        print()
        print(f"[probe] ground_plane.json: plane={gp.get('plane')}")
        stats = gp.get("foot_plane_dist_stats", {})
        if stats:
            print(
                f"[probe] foot_plane_dist (export-side): mean={stats.get('mean'):.4f} "
                f"std={stats.get('std'):.4f} min={stats.get('min'):.4f} "
                f"max={stats.get('max'):.4f}"
            )

    object_poses_for_viz: np.ndarray | None = None
    if poses_npy.is_file():
        op_cv = np.load(poses_npy)
        print()
        print(f"[probe] poses.npy shape={op_cv.shape} dtype={op_cv.dtype}")
        if op_cv.shape[0] > 0:
            print(f"  first translation (raw, OpenCV): {op_cv[0, :3, 3]}")
            print(f"  last translation  (raw, OpenCV): {op_cv[-1, :3, 3]}")
        # Bring object into the same SOMA world frame as the body before
        # anchoring or visualization.
        op = _convert_object_poses_cv_to_soma(op_cv)
        print(f"  first translation (SOMA frame  ): {op[0, :3, 3]}")
        print(f"  last translation  (SOMA frame  ): {op[-1, :3, 3]}")
        if args.anchor:
            object_poses_for_viz = _apply_first_frame_anchor(
                op,
                transl_first=motion["first_frame_transl"],
                R_first_inv=motion["first_frame_R_inv"],
            )
        else:
            object_poses_for_viz = op

    if args.visualize:
        soma_faces = soma.faces  # (F, 3) ints, exposed by SOMA wrapper
        object_mesh = _load_object_mesh(folder)
        if object_poses_for_viz is None and object_mesh is not None:
            print("[probe] No poses.npy; object mesh will be drawn at origin only.")
            object_poses_for_viz = np.tile(np.eye(4), (motion["num_frames"], 1, 1))
        elif object_poses_for_viz is None:
            object_poses_for_viz = np.zeros((motion["num_frames"], 4, 4))

        motion_for_viz = motion
        if args.robot_frame:
            print(
                "[probe] --robot-frame: rotating body + object into robot frame "
                "(X=forward, Y=left, Z=up) via R_SOMA_TO_ROBOT for visualization."
            )
            motion_for_viz = dict(motion)
            motion_for_viz["vertices"] = _rotate_points_to_robot(
                motion["vertices"], R_src_to_robot
            )
            object_poses_for_viz = _rotate_object_poses_to_robot(
                object_poses_for_viz, R_src_to_robot
            )

        server = viser.ViserServer(host="0.0.0.0", port=args.port)
        _visualize_raw_soma(
            server=server,
            motion=motion_for_viz,
            soma_faces=soma_faces,
            object_mesh=object_mesh,
            object_poses=object_poses_for_viz,
            fps=args.fps,
        )

    print()
    print("[probe] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
