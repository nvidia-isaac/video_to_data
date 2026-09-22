# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retarget SOMA-X body and object trajectories to G1 with Pink IK."""

from __future__ import annotations

import argparse
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser
from robotic_grounding.motion_schema import MotionData, save_motion_parquet
from robotic_grounding.retarget import G1_URDF_DIR, HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.ground_alignment import (
    ReferencePlane,
    compute_object_ground_lift,
    compute_plane_leveling_transform,
    load_ground_plane_robot_frame,
)
from robotic_grounding.retarget.params import SOMA_JOINTS_ORDER
from robotic_grounding.retarget.read_soma import SOMA
from robotic_grounding.retarget.robot_config import load_robot_config
from robotic_grounding.retarget.viser_playback import LiveFrameState, ViserPlayback
from robotic_grounding.retarget.whole_body_kinematics import (
    WholeBodyKinematics,
)
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

G1_URDF = G1_URDF_DIR / "main_with_hand.urdf"
PACKAGE_DIRS = [str(G1_URDF_DIR)]
REPO_ROOT = Path(__file__).resolve().parents[2]
OBJECT_GROUND_PENETRATION_TOLERANCE_M = 5e-4
OBJECT_GROUND_CLEARANCE_M = 5e-4
OBJECT_GROUND_MAX_LIFT_M = 1e-2


def _usd_safe(name: str) -> str:
    """Make a name safe for USD prim paths (no leading digits, no @ etc.)."""
    safe = name.replace("@", "_")
    if safe and (safe[0].isdigit() or not (safe[0].isalpha() or safe[0] == "_")):
        return f"obj_{safe}"
    return safe


def _convert_glb_to_obj(glb_path: Path, dst_dir: Path) -> Path:
    """Return a cached flat OBJ, creating it from the source GLB if needed."""
    obj_path = dst_dir / "textured_mesh.obj"
    if obj_path.is_file():
        return obj_path

    dst_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(glb_path, force="scene")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"Could not convert {glb_path} to a single trimesh.Trimesh "
            f"(got {type(mesh).__name__})."
        )
    mesh.export(obj_path)
    return obj_path


def _build_object_urdf(mesh_path: str, urdf_path: Path) -> str:
    """Write a simple rigid-object URDF that references ``mesh_path``."""
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_text = f"""<?xml version="1.0"?>
<robot name="retarget_object">
  <link name="object">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    urdf_path.write_text(urdf_text, encoding="utf-8")
    return str(urdf_path.resolve())


def _compute_mesh_radius(mesh_path: str) -> float:
    """Compute max radius from mesh centroid."""
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return 0.0
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(centered, axis=1).max())


def _get_robot_joint_position_names(
    kin: WholeBodyKinematics, base_q_size: int
) -> list[str]:
    """Return names aligned to ``q[base_q_size:]`` ordering."""
    indexed_names: list[tuple[int, str]] = []
    for joint_idx in range(1, kin.robot.model.njoints):
        joint_name = str(kin.robot.model.names[joint_idx])
        q_start = int(kin.robot.model.idx_qs[joint_idx])
        q_size = int(kin.robot.model.nqs[joint_idx])
        for local_idx in range(q_size):
            q_idx = q_start + local_idx
            if q_idx < base_q_size:
                continue
            label = joint_name if q_size == 1 else f"{joint_name}[{local_idx}]"
            indexed_names.append((q_idx, label))
    indexed_names.sort(key=lambda x: x[0])
    return [name for _, name in indexed_names]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Retarget SOMA motion to G1 whole body"
    )
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Path to folder containing soma_params.npz, poses.npy, and "
            "reconstructed_mesh/output_aligned.glb"
        ),
    )
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--save", action="store_true", help="Save retargeted data")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor from SOMA to robot"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Source and output motion frame rate.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.02,
        help=(
            "Distance (m) from palm link to nearest object mesh vertex below "
            "which the side is flagged as in contact. Only used with --save."
        ),
    )
    parser.add_argument(
        "--identity-model-type",
        type=str,
        default="mhr",
        help=("SOMA identity model. Must match what was used at export time."),
    )
    parser.add_argument(
        "--soma-data-root",
        type=str,
        default=None,
        help=(
            "Override path to SOMA-X assets. Defaults to "
            "<repo>/source/robotic_grounding/robotic_grounding/assets/body_models/soma."
        ),
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="g1",
        help=(
            "Robot config folder under "
            "`source/robotic_grounding/robotic_grounding/retarget/configs/`. "
            "Defaults to `g1`. The IK end-effector targets, per-bone "
            "rotation offsets, URDF path, and ground anchoring parameters "
            "all come from `<robot>/{frame_alignment,retargeter}.json` instead of "
            "the legacy constants in `params.py`."
        ),
    )
    parser.add_argument(
        "--motion-root",
        type=Path,
        default=HUMAN_MOTION_DATA_DIR,
        help=(
            "Root directory for saved motion data. The parquet is written "
            "under <motion-root>/whole_body/<soma-subdir>/sequence_id=.../"
            "robot_name=... Defaults to the in-repo human_motion_data asset root."
        ),
    )
    parser.add_argument(
        "--soma-subdir",
        type=str,
        default="soma",
        help=(
            "Dataset subfolder under <motion-root>/whole_body for saved SOMA "
            "motion_v1 parquet. Defaults to `soma`."
        ),
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help=(
            "First source frame to retarget (0-indexed, inclusive). "
            "Use with --end-frame to focus on a specific segment. The "
            "first-frame anchoring (body normalization + object trajectory "
            "transform) always runs against the ORIGINAL sequence's frame "
            "0, so saved positions stay comparable across different "
            "[start, end) windows."
        ),
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help=(
            "One-past-last source frame to retarget (Python-slice "
            "semantics). Defaults to the full sequence length. Must be "
            "strictly greater than --start-frame."
        ),
    )
    parser.add_argument(
        "--diagnose-ik",
        action="store_true",
        help=(
            "Print per-frame IK diagnostics (which task has the largest "
            "residual, total iterations, max_iter saturation, and joint "
            "position-limit saturation) for frames whose total residual or "
            "iteration count exceeds the thresholds set by "
            "--diagnose-ik-error-threshold / --diagnose-ik-iter-fraction. "
            "Also prints a top-K worst-frame summary at the end. Use to "
            "isolate which IK targets are dominating the QP at frames "
            "where the robot looks weird."
        ),
    )
    parser.add_argument(
        "--diagnose-ik-error-threshold",
        type=float,
        default=0.05,
        help=(
            "Per-frame total task residual (sum of task position errors, "
            "in meters) above which --diagnose-ik prints a one-line "
            "report. Default 0.05 m catches obvious failures while "
            "keeping clean frames silent. Ignored without --diagnose-ik."
        ),
    )
    parser.add_argument(
        "--diagnose-ik-iter-fraction",
        type=float,
        default=0.9,
        help=(
            "Fraction of `max_iter` above which --diagnose-ik flags a "
            "frame as 'iter-saturated' and prints a one-line report. "
            "Default 0.9 (e.g. 180/200) catches solves that almost ran "
            "out of budget. Ignored without --diagnose-ik."
        ),
    )
    return parser.parse_args()


def load_data(folder_path: str) -> tuple[str, str, np.ndarray]:
    """Load SOMA paths and object poses converted from OpenCV to SOMA axes."""
    folder = Path(folder_path).resolve()
    soma_params_path = folder / "soma_params.npz"
    poses_path = folder / "poses.npy"
    object_glb_path = folder / "reconstructed_mesh" / "output_aligned.glb"

    required = {
        "soma_params.npz": soma_params_path,
        "poses.npy": poses_path,
        "reconstructed_mesh/output_aligned.glb": object_glb_path,
    }
    missing = [name for name, p in required.items() if not p.is_file()]
    if missing:
        msg = (
            f"Data folder is missing required files: {missing}\n"
            f"  Resolved folder: {folder}\n"
            f"Expected layout:\n"
            f"  {folder}/soma_params.npz\n"
            f"  {folder}/poses.npy\n"
            f"  {folder}/reconstructed_mesh/output_aligned.glb\n"
            "If you run inside Docker, host paths like /home/... are not visible "
            "unless bind-mounted. Mount your data and pass the in-container path."
        )
        raise FileNotFoundError(msg)

    obj_dst_dir = folder / "object"
    mesh_path = _convert_glb_to_obj(object_glb_path, obj_dst_dir)

    object_poses_world = np.load(poses_path)
    object_poses_world = _convert_object_poses_cv_to_soma(object_poses_world)
    return str(soma_params_path), str(mesh_path), object_poses_world


# OpenCV world (X right, Y down, Z forward) to SOMA world (X right, Y up).
_R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0, 1.0])


def _convert_object_poses_cv_to_soma(object_poses_cv: np.ndarray) -> np.ndarray:
    """Convert object poses from OpenCV world axes to SOMA world axes."""
    return np.einsum("ij,tjk->tik", _R_CV_TO_SOMA, object_poses_cv)


def _left_multiply_world_wxyz(
    quaternions_wxyz: np.ndarray,
    world_rotation: np.ndarray,
) -> np.ndarray:
    """Apply one world-frame rotation to an arbitrary batch of quaternions."""
    quaternions = np.asarray(quaternions_wxyz)
    original_shape = quaternions.shape
    if not original_shape or original_shape[-1] != 4:
        raise ValueError(
            "quaternions_wxyz must have trailing dimension 4; " f"got {original_shape}"
        )
    matrices = R.from_quat(quaternions.reshape(-1, 4), scalar_first=True).as_matrix()
    rotated = np.einsum("ij,njk->nik", world_rotation, matrices)
    return R.from_matrix(rotated).as_quat(scalar_first=True).reshape(original_shape)


def main() -> None:
    """Main function."""
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError(f"--fps must be positive, got {args.fps}.")
    save_dir = args.motion_root.expanduser().resolve() / "whole_body" / args.soma_subdir

    data_folder = Path(args.data_folder)
    soma_params_path, object_mesh_path, object_poses_world = load_data(args.data_folder)
    print(f"Loaded data from {data_folder}")
    print(f"  SOMA params: {soma_params_path}")
    print(f"  Object mesh: {object_mesh_path}")
    print(f"  Object poses: {len(object_poses_world)} frames")

    config = load_robot_config(args.robot_name)
    kin = WholeBodyKinematics(config=config)
    foot_frame_names = list(config.foot_frames)
    ankle_roll_offset = float(config.ankle_roll_offset)
    base_q_size = 7
    robot_joint_position_names = _get_robot_joint_position_names(kin, base_q_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    soma = SOMA(
        data_root=args.soma_data_root,
        identity_model_type=args.identity_model_type,
        device=device,
    )
    # Anchor body and object to the source body's first-frame root pose.
    motion = soma.load_motion(params_path=soma_params_path, normalize=True)

    transl_first = motion["first_frame_transl"]
    R_first_inv = motion["first_frame_R_inv"]
    norm_transform = np.eye(4)
    norm_transform[:3, :3] = R_first_inv
    norm_transform[:3, 3] = -R_first_inv @ transl_first
    object_poses = np.einsum("ij,tjk->tik", norm_transform, object_poses_world)

    # Transform the fitted plane through the same frame chain as the motion.
    ground_plane_path = Path(args.data_folder) / "ground_plane.json"
    reconstructed_plane = load_ground_plane_robot_frame(
        ground_plane_path,
        cv_to_source=_R_CV_TO_SOMA,
        first_frame_anchor=norm_transform,
        source_to_robot=np.asarray(config.r_world, dtype=np.float64),
    )
    joint_pos = motion["joints"]
    joint_rot_wxyz = motion["joints_wxyz"]
    vertices = motion["vertices"]

    source_to_robot = np.asarray(config.r_world, dtype=np.float64)
    if reconstructed_plane is not None:
        alignment_plane = reconstructed_plane
        ground_plane_source = "reconstructed"
        print(
            "[soma_to_g1] grounding plane: reconstructed "
            f"(normal={alignment_plane.normal}, "
            f"offset={alignment_plane.offset:+.4f})"
        )
    else:
        # Fall back to a horizontal plane at the frame-0 body-mesh minimum.
        frame0_vertices_robot = kin.transform_source_position(vertices[0])
        fallback_z = float(frame0_vertices_robot[:, 2].min())
        alignment_plane = ReferencePlane.horizontal(z=fallback_z)
        ground_plane_source = "frame0_body_mesh_fallback"
        print(
            "[soma_to_g1] grounding plane: frame-0 body-mesh fallback "
            f"(z={fallback_z:+.4f}); no ground_plane.json found at "
            f"{ground_plane_path}"
        )

    # Level body, object, and visualization geometry together before IK.
    (
        ground_level_rotation_robot,
        ground_level_translation_robot,
    ) = compute_plane_leveling_transform(alignment_plane)
    ground_level_rotation_source = (
        source_to_robot.T @ ground_level_rotation_robot @ source_to_robot
    )
    ground_level_translation_source = source_to_robot.T @ ground_level_translation_robot

    joint_pos = (
        np.einsum("ij,tkj->tki", ground_level_rotation_source, joint_pos)
        + ground_level_translation_source
    )
    joint_rot_wxyz = _left_multiply_world_wxyz(
        joint_rot_wxyz,
        ground_level_rotation_source,
    )
    if args.visualize:
        vertices = (
            np.einsum(
                "ij,tvj->tvi",
                ground_level_rotation_source,
                vertices,
            )
            + ground_level_translation_source
        )

    ground_level_transform_source = np.eye(4, dtype=np.float64)
    ground_level_transform_source[:3, :3] = ground_level_rotation_source
    ground_level_transform_source[:3, 3] = ground_level_translation_source
    object_poses = np.einsum(
        "ij,tjk->tik",
        ground_level_transform_source,
        object_poses,
    )

    _obj_verts = []
    with open(object_mesh_path) as _f:
        for _line in _f:
            if _line.startswith("v "):
                _parts = _line.split()
                _obj_verts.append(
                    [float(_parts[1]), float(_parts[2]), float(_parts[3])]
                )
    object_mesh_vertices = np.array(_obj_verts, dtype=np.float64)
    object_mesh_vertices_f32 = object_mesh_vertices.astype(np.float32)

    frame0_object_position_w = kin.transform_source_position(object_poses[0, :3, 3])
    frame0_object_rotation_w = kin.transform_world_rotation(object_poses[0, :3, :3])
    frame0_object_vertices_w = (
        object_mesh_vertices @ frame0_object_rotation_w.T + frame0_object_position_w
    )
    object_ground_lift = compute_object_ground_lift(
        frame0_object_vertices_w,
        ReferencePlane.horizontal(),
        penetration_tolerance=OBJECT_GROUND_PENETRATION_TOLERANCE_M,
        clearance=OBJECT_GROUND_CLEARANCE_M,
        max_lift=OBJECT_GROUND_MAX_LIFT_M,
    )
    lift_robot = np.array([0.0, 0.0, object_ground_lift.applied_lift], dtype=np.float64)
    object_poses[:, :3, 3] += source_to_robot.T @ lift_robot
    print(
        "[soma_to_g1] object frame-0 ground correction: "
        f"minimum={object_ground_lift.minimum_signed_distance:+.4f} m, "
        f"requested={object_ground_lift.requested_lift:+.4f} m, "
        f"applied={object_ground_lift.applied_lift:+.4f} m, "
        f"cap={OBJECT_GROUND_MAX_LIFT_M:.4f} m"
        f"{' (capped)' if object_ground_lift.capped else ''}"
    )

    tilt_degrees = float(
        np.degrees(np.arccos(np.clip(float(alignment_plane.normal[2]), -1.0, 1.0)))
    )
    print(
        "[soma_to_g1] single-pass plane leveling: "
        f"tilt={tilt_degrees:.3f} deg -> +Z, "
        f"translation_z={ground_level_translation_robot[2]:+.4f} m, "
        "final_plane=z=0"
    )

    num_frames = motion["num_frames"]
    soma_joint_names = motion["joint_names"]
    if soma_joint_names != SOMA_JOINTS_ORDER:
        raise ValueError(
            "SOMA joint_names do not match SOMA_JOINTS_ORDER. Update params.py "
            f"to track the SOMA-X release in use. Got first 5: {soma_joint_names[:5]}"
        )
    if len(object_poses) != num_frames:
        raise ValueError(
            f"poses.npy length ({len(object_poses)}) != motion frames ({num_frames})"
        )

    # Keep partial runs in the coordinate frame anchored to original frame 0.
    start_frame = int(args.start_frame)
    end_frame = int(num_frames) if args.end_frame is None else int(args.end_frame)
    if start_frame < 0 or start_frame >= num_frames:
        raise ValueError(
            f"--start-frame={start_frame} is out of range for a sequence "
            f"with {num_frames} frames (valid range: [0, {num_frames - 1}])."
        )
    if end_frame <= start_frame or end_frame > num_frames:
        raise ValueError(
            f"--end-frame={end_frame} must satisfy "
            f"start_frame ({start_frame}) < end_frame <= num_frames "
            f"({num_frames})."
        )
    if (start_frame, end_frame) != (0, num_frames):
        print(
            f"[INFO] Frame range: iterating IK over [{start_frame}, "
            f"{end_frame}) of {num_frames} source frames."
        )
        joint_pos = joint_pos[start_frame:end_frame]
        joint_rot_wxyz = joint_rot_wxyz[start_frame:end_frame]
        vertices = vertices[start_frame:end_frame]
        object_poses = object_poses[start_frame:end_frame]
    n_iter_frames = int(end_frame - start_frame)

    # Avoid duplicating the Hive partition prefix in the output path.
    sequence_id = data_folder.name.removeprefix("sequence_id=")

    head_idx = SOMA_JOINTS_ORDER.index("Head")
    root_idx = SOMA_JOINTS_ORDER.index("Hips")

    object_name = f"{sequence_id}_object"

    contact_threshold = float(args.contact_threshold)

    builder: dict[str, list] | None = None
    object_body_names: list[str] = ["object"]
    safe_object_body_names: list[str] = [_usd_safe(n) for n in object_body_names]
    object_mesh_radius: float = 0.0
    soma_identity_coeffs: list[float] = []
    soma_scale_params: list[float] = []
    frame_names_list: list[str] = list(kin.robot_frame_names.values())
    frame_name_to_idx = {name: i for i, name in enumerate(frame_names_list)}
    ee_link_name_candidates: list[str] = [
        "left_hand_palm_link",
        "right_hand_palm_link",
    ]
    ee_frame_indices: list[int] = [
        frame_names_list.index(n)
        for n in ee_link_name_candidates
        if n in frame_names_list
    ]
    ee_link_names: list[str] = [frame_names_list[i] for i in ee_frame_indices]

    hand_sides: list[str] = [
        name.split("_")[0] for name in ee_link_names if "_hand_palm_link" in name
    ]

    # SOMA fingertip joint names match the canonical MHR rig 1:1.
    hand_side_to_fingertip_source_joints: dict[str, list[int]] = {}
    for side, prefix in (("left", "Left"), ("right", "Right")):
        candidates = [
            f"{prefix}HandThumbEnd",
            f"{prefix}HandIndexEnd",
            f"{prefix}HandMiddleEnd",
            f"{prefix}HandRingEnd",
            f"{prefix}HandPinkyEnd",
        ]
        hand_side_to_fingertip_source_joints[side] = [
            SOMA_JOINTS_ORDER.index(n) for n in candidates if n in SOMA_JOINTS_ORDER
        ]

    if args.save:
        params = np.load(soma_params_path, allow_pickle=True)
        # Identity and scale are constant over the sequence.
        soma_identity_coeffs = params["identity_coeffs"][0].astype(np.float32).tolist()
        soma_scale_params = params["scale_params"][0].astype(np.float32).tolist()

        object_mesh_path = str(Path(object_mesh_path).resolve())
        object_mesh_radius = _compute_mesh_radius(object_mesh_path)

        # Keep assets outside the robot leaf replaced by the parquet writer.
        mesh_dst_dir = save_dir / f"sequence_id={sequence_id}" / "object"
        mesh_dst_dir.mkdir(parents=True, exist_ok=True)
        for src in Path(object_mesh_path).parent.iterdir():
            if not src.is_file():
                continue
            dst = mesh_dst_dir / src.name
            # Input and output partitions may share the same object directory.
            if src.resolve() == dst.resolve():
                continue
            shutil.copy2(src, dst)
        urdf_dst = mesh_dst_dir / "textured_mesh.urdf"
        _build_object_urdf(mesh_path="textured_mesh.obj", urdf_path=urdf_dst)

        copied_mesh_abs = (mesh_dst_dir / "textured_mesh.obj").resolve()
        try:
            stored_mesh_path = str(copied_mesh_abs.relative_to(REPO_ROOT))
            stored_urdf_path = str(urdf_dst.resolve().relative_to(REPO_ROOT))
        except ValueError:
            stored_mesh_path = str(copied_mesh_abs)
            stored_urdf_path = str(urdf_dst.resolve())

        expected_joint_dim = kin.robot.model.nq - base_q_size
        if len(robot_joint_position_names) != expected_joint_dim:
            raise ValueError(
                "robot_joint_names must align with robot_joint_positions. "
                f"Expected {expected_joint_dim}, got {len(robot_joint_position_names)}."
            )

        builder = {
            "robot_root_position": [],
            "robot_root_wxyz": [],
            "robot_joint_positions": [],
            "ee_pose_w": [],
            "object_articulation": [],
            "object_root_axis_angle": [],
            "object_root_position": [],
            "object_body_position": [],
            "object_body_wxyz": [],
            "hand_contact_active_per_frame": [],
            "ik_error_per_frame": [],
            "ik_num_iterations": [],
            "frame_task_errors": [],
            # Plane-leveled source values retained as opaque provenance.
            "soma_joints": [],
            "soma_joints_wxyz": [],
            "source_head_translation": [],
            "source_head_wxyz": [],
            "source_root_translation": [],
            "source_root_wxyz": [],
        }

    playback: ViserPlayback | None = None
    if args.visualize:
        server = viser.ViserServer(host="0.0.0.0", port=8080)
        playback = ViserPlayback.for_live_retarget(
            server=server,
            pin_model=kin.robot.model,
            pin_visual_model=kin.robot.visual_model,
            pin_collision_model=kin.robot.collision_model,
            object_mesh_path=object_mesh_path,
            hand_sides=tuple(hand_sides) or ("left", "right"),
        )

    q = kin.robot.q0.copy()

    foot_frame_idxs = [frame_name_to_idx[frame_name] for frame_name in foot_frame_names]

    saved_sole_z_per_frame: list[list[float]] = []
    foot_target_grounding_offsets: list[float] = []
    robot_penetration_lifts: list[float] = []

    # Diagnostic state is populated only with --diagnose-ik.
    diagnose_ik = bool(args.diagnose_ik)
    ik_task_names: list[str] = list(kin.frame_tasks.keys())
    q_lower = np.asarray(kin.robot.model.lowerPositionLimit, dtype=np.float64)
    q_upper = np.asarray(kin.robot.model.upperPositionLimit, dtype=np.float64)
    # Ignore unbounded free-flyer coordinates in saturation checks.
    finite_limit_mask = np.isfinite(q_lower) & np.isfinite(q_upper)
    # Expand joint names to q-coordinate indices once.
    q_idx_to_joint_name: dict[int, str] = {}
    for joint_idx in range(1, kin.robot.model.njoints):
        name = str(kin.robot.model.names[joint_idx])
        q_start = int(kin.robot.model.idx_qs[joint_idx])
        q_size = int(kin.robot.model.nqs[joint_idx])
        for k in range(q_size):
            label = name if q_size == 1 else f"{name}[{k}]"
            q_idx_to_joint_name[q_start + k] = label
    diag_iter_threshold = int(kin.max_iter * float(args.diagnose_ik_iter_fraction))
    diag_error_threshold = float(args.diagnose_ik_error_threshold)
    # (frame, total error, dominant task/error, iterations, saturation count/sample)
    diag_offenders: list[tuple[int, float, str, float, int, int, str]] = []

    # frame_idx is local to the selected window.
    for frame_idx in tqdm(range(n_iter_frames), desc="Retargeting"):
        positions = joint_pos[frame_idx]
        rotations = joint_rot_wxyz[frame_idx]

        result = kin.compute(
            source_joints=positions,
            source_joints_wxyz=rotations,
            source_to_robot_scale=args.scale,
            qpos=q,
            foot_target_ground_z=0.0,
        )
        q_ik = result["q"].copy()

        # Remove residual IK penetration by lifting only the robot free-flyer.
        frame_pose = np.asarray(result["frame_pose"], dtype=np.float64).copy()
        solved_sole_z = np.asarray(
            [frame_pose[i, 2] - ankle_roll_offset for i in foot_frame_idxs],
            dtype=np.float64,
        )
        robot_penetration_lift = max(0.0, -float(solved_sole_z.min()))
        if robot_penetration_lift > 0.0:
            q_ik[2] += robot_penetration_lift
            frame_pose[:, 2] += robot_penetration_lift
        q = q_ik.copy()

        saved_sole_z = np.asarray(
            [frame_pose[i, 2] - ankle_roll_offset for i in foot_frame_idxs],
            dtype=np.float64,
        )
        saved_sole_z_per_frame.append(saved_sole_z.tolist())
        foot_target_grounding_offsets.append(
            float(result["foot_target_grounding_offset_z"])
        )
        robot_penetration_lifts.append(robot_penetration_lift)

        # Record task errors after the root projection applied to saved q.
        frame_task_errors = []
        for frame_name, task in kin.frame_tasks.items():
            task_frame_idx = frame_name_to_idx[frame_name]
            target_position = task.transform_target_to_world.translation
            frame_task_errors.append(
                float(np.linalg.norm(frame_pose[task_frame_idx, :3] - target_position))
            )
        result["frame_task_errors"] = frame_task_errors

        if diagnose_ik:
            # Diagnose the saved IK result and joints within 1e-3 of a limit.
            task_errs = np.asarray(result["frame_task_errors"], dtype=np.float64)
            total_err = float(task_errs.sum())
            dom_idx = int(np.argmax(task_errs)) if task_errs.size > 0 else -1
            dom_task = ik_task_names[dom_idx] if dom_idx >= 0 else "?"
            dom_err = float(task_errs[dom_idx]) if dom_idx >= 0 else 0.0
            n_iter = int(result["num_optimization_iterations"])
            sat_tol = 1e-3
            sat_lower = (q_ik <= q_lower + sat_tol) & finite_limit_mask
            sat_upper = (q_ik >= q_upper - sat_tol) & finite_limit_mask
            sat_mask = sat_lower | sat_upper
            n_sat = int(sat_mask.sum())
            sample_sat = ""
            if n_sat > 0:
                first_sat_idx = int(np.argmax(sat_mask))
                side = "lo" if sat_lower[first_sat_idx] else "hi"
                sample_sat = (
                    f"{q_idx_to_joint_name.get(first_sat_idx, f'q[{first_sat_idx}]')}"
                    f"({side})"
                )
            iter_saturated = n_iter >= diag_iter_threshold
            # Use the absolute source index in viewer-facing diagnostics.
            abs_idx = start_frame + frame_idx
            if total_err >= diag_error_threshold or iter_saturated or n_sat > 0:
                tqdm.write(
                    f"[ik-diag] frame={abs_idx:5d} (loop_idx={frame_idx:5d}) "
                    f"total_err={total_err:.4f} "
                    f"dom={dom_task}={dom_err:.4f} "
                    f"iters={n_iter}/{kin.max_iter}"
                    f"{' SAT_ITER' if iter_saturated else ''} "
                    f"sat_joints={n_sat}"
                    f"{f' (e.g. {sample_sat})' if sample_sat else ''}"
                )
                diag_offenders.append(
                    (
                        abs_idx,
                        total_err,
                        dom_task,
                        dom_err,
                        n_iter,
                        n_sat,
                        sample_sat,
                    )
                )

        obj_pose = object_poses[frame_idx]
        obj_position = kin.transform_source_position(obj_pose[:3, 3])
        obj_rotation_mat = kin.transform_world_rotation(obj_pose[:3, :3])
        obj_rotation = R.from_matrix(obj_rotation_mat)

        head_position = kin.transform_source_position(positions[head_idx])
        head_rotation_mat = R.from_quat(
            rotations[head_idx], scalar_first=True
        ).as_matrix()
        head_rotation_mat = kin.transform_source_rotation(head_rotation_mat)
        head_rotation_wxyz = R.from_matrix(head_rotation_mat).as_quat(scalar_first=True)

        root_position = kin.transform_source_position(positions[root_idx])
        root_rotation_mat = R.from_quat(
            rotations[root_idx], scalar_first=True
        ).as_matrix()
        root_rotation_mat = kin.transform_source_rotation(root_rotation_mat)
        root_rotation_wxyz = R.from_matrix(root_rotation_mat).as_quat(scalar_first=True)

        ee_pose_t: list[list[float]] = []
        for i in ee_frame_indices:
            ee_pose_t.append(frame_pose[i].tolist())

        object_translation_w_np = obj_position.astype(np.float32)
        object_rotation_w_np = obj_rotation_mat.astype(np.float32)
        verts_w = (
            object_mesh_vertices_f32 @ object_rotation_w_np.T + object_translation_w_np
        )
        threshold_sq = contact_threshold * contact_threshold
        per_side_active: list[float] = []
        for side_idx, side in enumerate(("left", "right")[: len(ee_pose_t)]):
            fingertip_joint_ids = hand_side_to_fingertip_source_joints.get(side, [])
            points = np.empty((1 + len(fingertip_joint_ids), 3), dtype=np.float32)
            points[0] = ee_pose_t[side_idx][:3]
            for k, j in enumerate(fingertip_joint_ids, start=1):
                p = kin.transform_source_position(positions[j])
                points[k] = p
            diff = verts_w[:, None, :] - points[None, :, :]
            sq_dists = np.einsum("vki,vki->vk", diff, diff)
            min_sq = float(sq_dists.min())
            per_side_active.append(1.0 if min_sq < threshold_sq else 0.0)

        if builder is not None:
            root_pos_robot = q_ik[:3].tolist()
            root_quat_xyzw = q_ik[3:7]
            root_wxyz = [
                float(root_quat_xyzw[3]),
                float(root_quat_xyzw[0]),
                float(root_quat_xyzw[1]),
                float(root_quat_xyzw[2]),
            ]
            joint_positions = q_ik[base_q_size:].tolist()

            obj_wxyz = obj_rotation.as_quat(scalar_first=True).tolist()
            obj_body_pos = [obj_position.tolist()]
            obj_body_wxyz = [obj_wxyz]

            builder["robot_root_position"].append(root_pos_robot)
            builder["robot_root_wxyz"].append(root_wxyz)
            builder["robot_joint_positions"].append(joint_positions)
            builder["ee_pose_w"].append(ee_pose_t)
            builder["object_articulation"].append(0.0)
            object_translation_w = object_translation_w_np.tolist()
            builder["object_root_position"].append(object_translation_w)
            builder["object_root_axis_angle"].append(obj_rotation.as_rotvec().tolist())
            builder["object_body_position"].append(obj_body_pos)
            builder["object_body_wxyz"].append(obj_body_wxyz)
            builder["hand_contact_active_per_frame"].append(per_side_active)
            builder["ik_error_per_frame"].append(
                float(np.sum(result["frame_task_errors"]))
            )
            builder["ik_num_iterations"].append(
                int(result["num_optimization_iterations"])
            )
            builder["frame_task_errors"].append(list(result["frame_task_errors"]))
            builder["soma_joints"].append(positions.tolist())
            builder["soma_joints_wxyz"].append(rotations.tolist())
            builder["source_head_translation"].append(head_position.tolist())
            builder["source_head_wxyz"].append(head_rotation_wxyz.tolist())
            builder["source_root_translation"].append(root_position.tolist())
            builder["source_root_wxyz"].append(root_rotation_wxyz.tolist())

        if playback is not None:
            vertices_vis = kin.transform_source_position(vertices[frame_idx]).copy()
            q_vis = q_ik.copy()

            ik_target_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for frame_name, task in kin.frame_tasks.items():
                target_pos_vis = task.transform_target_to_world.translation.copy()
                target_wxyz = R.from_matrix(
                    task.transform_target_to_world.rotation
                ).as_quat(scalar_first=True)
                ik_target_poses[frame_name] = (target_pos_vis, target_wxyz)

            contact_wrists = [np.asarray(pose[:3]) for pose in ee_pose_t]

            playback.display(
                LiveFrameState(
                    q=q_vis,
                    object_pos=obj_position,
                    object_wxyz=obj_rotation.as_quat(scalar_first=True),
                    head_pos=head_position,
                    head_wxyz=head_rotation_wxyz,
                    root_pos=root_position,
                    root_wxyz=root_rotation_wxyz,
                    contact_wrists=contact_wrists,
                    contact_active=list(per_side_active),
                    body_vertices=vertices_vis,
                    ik_target_poses=ik_target_poses,
                ),
                body_model=soma,
            )

        if frame_idx == 0 and args.visualize:
            time.sleep(5)

    if builder is not None:
        saved_sole_z = np.asarray(saved_sole_z_per_frame, dtype=np.float64)
        target_offsets = np.asarray(foot_target_grounding_offsets, dtype=np.float64)
        penetration_lifts = np.asarray(robot_penetration_lifts, dtype=np.float64)
        print(
            "[INFO] Single-pass foot-target Z offset range: "
            f"[{float(target_offsets.min()):+.4f}, "
            f"{float(target_offsets.max()):+.4f}] m"
        )
        print(
            "[INFO] Same-frame robot-only anti-penetration lift: "
            f"[{float(penetration_lifts.min()):+.4f}, "
            f"{float(penetration_lifts.max()):+.4f}] m; "
            f"{int(np.count_nonzero(penetration_lifts > 0.0))}/"
            f"{len(penetration_lifts)} frames"
        )
        print(
            "[INFO] Saved robot sole Z range: "
            f"[{float(saved_sole_z.min()):+.4f}, "
            f"{float(saved_sole_z.max()):+.4f}] m"
        )
        print(
            "[INFO] Object trajectory: constant ground lift applied to "
            "plane-leveled raw poses; no contact-mask rewrite"
        )

        source_payload = pickle.dumps(
            {
                "soma_identity_coeffs": soma_identity_coeffs,
                "soma_scale_params": soma_scale_params,
                "source_fps": float(args.fps),
                "soma_joints": builder.pop("soma_joints"),
                "soma_joints_wxyz": builder.pop("soma_joints_wxyz"),
                "source_head_translation": builder.pop("source_head_translation"),
                "source_head_wxyz": builder.pop("source_head_wxyz"),
                "source_root_translation": builder.pop("source_root_translation"),
                "source_root_wxyz": builder.pop("source_root_wxyz"),
                "ground_alignment": {
                    "method": "single_pass_fitted_plane",
                    "ground_plane_source": ground_plane_source,
                    "ground_plane_normal": list(alignment_plane.normal),
                    "ground_plane_offset": float(alignment_plane.offset),
                    "ground_level_rotation_robot": (
                        ground_level_rotation_robot.tolist()
                    ),
                    "ground_level_translation_robot": (
                        ground_level_translation_robot.tolist()
                    ),
                    "final_ground_plane_normal": [0.0, 0.0, 1.0],
                    "final_ground_plane_offset": 0.0,
                    "ground_plane_json_path": str(ground_plane_path),
                    "foot_target_ground_z": 0.0,
                    "object_trajectory_rewritten": (
                        object_ground_lift.applied_lift > 0.0
                    ),
                    "object_trajectory_rewrite_kind": (
                        "constant_ground_lift"
                        if object_ground_lift.applied_lift > 0.0
                        else "none"
                    ),
                    "object_ground_correction": {
                        "method": "frame0_mesh_penetration",
                        "minimum_signed_distance": (
                            object_ground_lift.minimum_signed_distance
                        ),
                        "penetration_depth": object_ground_lift.penetration_depth,
                        "penetration_tolerance": (
                            OBJECT_GROUND_PENETRATION_TOLERANCE_M
                        ),
                        "clearance": OBJECT_GROUND_CLEARANCE_M,
                        "requested_lift": object_ground_lift.requested_lift,
                        "applied_lift": object_ground_lift.applied_lift,
                        "max_lift": OBJECT_GROUND_MAX_LIFT_M,
                        "capped": object_ground_lift.capped,
                    },
                },
            }
        )

        hand_contact_active: list[list[float]] = []
        if hand_sides:
            per_frame = builder.pop("hand_contact_active_per_frame")
            hand_contact_active = [
                [float(per_frame[t][s]) for t in range(len(per_frame))]
                for s in range(len(hand_sides))
            ]
            for side, series in zip(hand_sides, hand_contact_active, strict=True):
                n_active = int(sum(series))
                print(
                    f"[INFO] {side}_hand_contact_active: {n_active}/{len(series)} "
                    f"frames (threshold={contact_threshold} m)"
                )
        else:
            builder.pop("hand_contact_active_per_frame", None)

        md = MotionData(
            sequence_id=sequence_id,
            robot_name=config.robot_name,
            motion_kind="single_robot",
            source_dataset="soma",
            raw_motion_file=soma_params_path,
            fps=float(args.fps),
            coord_frame="robot_base_z_up",
            robot_joint_names=robot_joint_position_names,
            robot_root_position=builder["robot_root_position"],
            robot_root_wxyz=builder["robot_root_wxyz"],
            robot_joint_positions=builder["robot_joint_positions"],
            ee_link_names=ee_link_names,
            ee_pose_w=builder["ee_pose_w"],
            hand_sides=hand_sides,
            hand_contact_active=hand_contact_active,
            object_name=object_name,
            safe_object_name=_usd_safe(object_name),
            object_body_names=object_body_names,
            safe_object_body_names=safe_object_body_names,
            object_mesh_paths=[stored_mesh_path],
            object_urdf_paths=[stored_urdf_path],
            object_mesh_radius=[object_mesh_radius],
            object_articulation=builder["object_articulation"],
            object_root_axis_angle=builder["object_root_axis_angle"],
            object_root_position=builder["object_root_position"],
            object_body_position=builder["object_body_position"],
            object_body_wxyz=builder["object_body_wxyz"],
            ik_error_per_frame=builder["ik_error_per_frame"],
            ik_num_iterations=builder["ik_num_iterations"],
            frame_task_errors=builder["frame_task_errors"],
            source_kind="soma",
            source_payload=source_payload,
        )
        save_motion_parquet(md, root_path=str(save_dir), file_name="data.parquet")
        print(f"Saved to {save_dir}")

    if diagnose_ik:
        if not diag_offenders:
            print(
                f"[ik-diag] No frames flagged "
                f"(total_err < {diag_error_threshold} m, iters < "
                f"{diag_iter_threshold}/{kin.max_iter}, no saturated joints)."
            )
        else:
            # Rank by total task residual, then iteration count.
            top_k = min(10, len(diag_offenders))
            worst = sorted(
                diag_offenders,
                key=lambda r: (-r[1], -r[4]),
            )[:top_k]
            print(
                f"[ik-diag] {len(diag_offenders)}/{n_iter_frames} frames "
                f"flagged. Worst {top_k} by total task error:"
            )
            for (
                abs_idx,
                total_err,
                dom_task,
                dom_err,
                n_iter,
                n_sat,
                sample_sat,
            ) in worst:
                sat_str = (
                    f" sat={n_sat} (e.g. {sample_sat})"
                    if sample_sat
                    else f" sat={n_sat}"
                )
                print(
                    f"  frame={abs_idx:5d} total_err={total_err:.4f} "
                    f"dom={dom_task}={dom_err:.4f} "
                    f"iters={n_iter}/{kin.max_iter}{sat_str}"
                )
            # Summarize the dominant task across flagged frames.
            per_task_sum: dict[str, float] = dict.fromkeys(ik_task_names, 0.0)
            per_task_count: dict[str, int] = dict.fromkeys(ik_task_names, 0)
            for (
                _abs_idx,
                _total_err,
                dom_task,
                dom_err,
                _n_iter,
                _n_sat,
                _,
            ) in diag_offenders:
                per_task_sum[dom_task] += dom_err
                per_task_count[dom_task] += 1
            print("[ik-diag] dominant-task tally across flagged frames:")
            for name in ik_task_names:
                if per_task_count[name] == 0:
                    continue
                avg = per_task_sum[name] / per_task_count[name]
                print(
                    f"  {name:30s} dominant in {per_task_count[name]:4d} frames "
                    f"(avg dominant err = {avg:.4f})"
                )

    if (start_frame, end_frame) != (0, num_frames):
        print(
            f"Retargeting complete. Processed {n_iter_frames} frames "
            f"(window [{start_frame}, {end_frame}) of {num_frames})."
        )
    else:
        print(f"Retargeting complete. Processed {num_frames} frames.")
    if args.visualize:
        print("Visualization server running. Press Ctrl+C to exit.")
        # Exit cleanly so the wrapper can advance to its next stage.
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nVisualization stopped; continuing.")


if __name__ == "__main__":
    main()
