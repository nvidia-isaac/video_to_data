# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Script to retarget SOMA-X (NVlabs SOMA) motion to G1 whole body using Pink IK.

Mirrors ``scripts/retarget/nvhuman_to_g1.py`` but reads the SOMA exporter
schema (``soma_params.npz`` + object pose file + object glTF/OBJ + optional
``ground_plane.json``) instead of NVHuman's ``nova_params_opt.pt``.

The output parquet shares the same ``motion_v1`` schema used by training and
replay; only the dataset folder differs:

    HUMAN_MOTION_DATA_DIR / "whole_body" / "soma" /
        sequence_id=<folder_name>/robot_name=g1/data.parquet

Usage:
    python scripts/retarget/soma_to_g1.py <data_folder> --save
    python scripts/retarget/soma_to_g1.py <data_folder> --visualize

Where ``data_folder`` contains:
    - soma_params.npz (SOMA-X exported pose + identity parameters)
    - poses.npy (object trajectory as 4x4 transforms)
    - object_mesh/output_aligned.glb (object mesh; converted to .obj for sim)
    - ground_plane.json (optional reconstruction-side ground plane fit)
"""

from __future__ import annotations

import argparse
import os
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
    FirstPassResult,
    InteractionMaskConfig,
    ObjectCorrectionConfig,
    PlaneAlignmentConfig,
    ReferencePlane,
    compute_interaction_mask,
    compute_plane_alignment_offsets,
    correct_object_trajectory,
    load_ground_plane_robot_frame,
)
from robotic_grounding.retarget.params import SOMA_JOINTS_ORDER
from robotic_grounding.retarget.read_soma import SOMA
from robotic_grounding.retarget.robot_config import load_robot_config
from robotic_grounding.retarget.viser_playback import LiveFrameState, ViserPlayback
from robotic_grounding.retarget.whole_body_kinematics import (
    ConfigDrivenWholeBodyKinematics,
)
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

G1_URDF = G1_URDF_DIR / "main_with_hand.urdf"
PACKAGE_DIRS = [str(G1_URDF_DIR)]
# First-pass IK snapshot consumed by ``scripts/retarget/rerun_post_process.py``.
# Kept on a SOMA-specific cache path so SOMA and NVHuman runs of the same
# sequence id do not stomp on each other.
FIRST_PASS_CACHE_DIR = Path(
    os.environ.get("SOMA_FIRST_PASS_CACHE_DIR", "/tmp/soma_g1_processed_cache")
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _usd_safe(name: str) -> str:
    """Make a name safe for USD prim paths (no leading digits, no @ etc.)."""
    safe = name.replace("@", "_")
    if safe and (safe[0].isdigit() or not (safe[0].isalpha() or safe[0] == "_")):
        return f"obj_{safe}"
    return safe


def _convert_glb_to_obj(glb_path: Path, dst_dir: Path) -> Path:
    """Convert a SOMA object ``.glb`` into a flat ``textured_mesh.obj`` next to it.

    The robotic_grounding pipeline prefers ``.obj`` next to the parquet so
    Isaac Sim's URDF importer can resolve materials/textures without
    glTF-specific handling. ``trimesh`` happily concatenates a glTF scene
    into a single mesh; this is sufficient for retargeting because we use
    the geometry only for contact distance and visualization.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(glb_path, force="scene")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"Could not convert {glb_path} to a single trimesh.Trimesh "
            f"(got {type(mesh).__name__})."
        )
    obj_path = dst_dir / "textured_mesh.obj"
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
    kin: ConfigDrivenWholeBodyKinematics, base_q_size: int
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
            "object_mesh/output_aligned.glb"
        ),
    )
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--save", action="store_true", help="Save retargeted data")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor from SOMA to robot"
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
    """Locate SOMA params, raw object poses, and a usable object mesh path.

    Two transforms apply to the object trajectory between disk and the IK
    loop. This function applies the first; ``main`` applies the second:

    1. **CV -> SOMA world** (here, via ``_convert_object_poses_cv_to_soma``).
       ``poses.npy`` is in OpenCV camera convention (X=right, Y=down,
       Z=forward) while ``SOMALayer.transl`` is in SOMA's "Y up,
       Z toward camera" frame. Without this flip the body and object
       live in two different worlds.
    2. **First-frame anchoring** (deferred to ``main``). After SOMA loads
       the motion we know the frame-0 root translation/rotation; the same
       ``(p - transl_first) @ R_first_inv.T`` transform that ``SOMA.load_motion``
       applies to the body must be applied to the object trajectory or
       the object slides off in the retargeted output.

    Returns:
        Tuple of (soma_params_path, mesh_path, object_poses_world).
        ``mesh_path`` is the converted ``.obj``; ``object_poses_world`` is
        the ``poses.npy`` array of shape ``(T, 4, 4)`` after the
        CV -> SOMA flip but **before** first-frame anchoring.
    """
    folder = Path(folder_path).resolve()
    soma_params_path = folder / "soma_params.npz"
    poses_path = folder / "poses.npy"
    object_glb_path = folder / "object_mesh" / "output_aligned.glb"

    required = {
        "soma_params.npz": soma_params_path,
        "poses.npy": poses_path,
        "object_mesh/output_aligned.glb": object_glb_path,
    }
    missing = [name for name, p in required.items() if not p.is_file()]
    if missing:
        msg = (
            f"Data folder is missing required files: {missing}\n"
            f"  Resolved folder: {folder}\n"
            f"Expected layout:\n"
            f"  {folder}/soma_params.npz\n"
            f"  {folder}/poses.npy\n"
            f"  {folder}/object_mesh/output_aligned.glb\n"
            "If you run inside Docker, host paths like /home/... are not visible "
            "unless bind-mounted. Mount your data and pass the in-container path."
        )
        raise FileNotFoundError(msg)

    obj_dst_dir = folder / "object"
    mesh_path = _convert_glb_to_obj(object_glb_path, obj_dst_dir)

    object_poses_world = np.load(poses_path)
    object_poses_world = _convert_object_poses_cv_to_soma(object_poses_world)
    return str(soma_params_path), str(mesh_path), object_poses_world


# The object trajectory stored in ``poses.npy`` is in OpenCV camera
# convention (X=right, Y=down, Z=forward), while the SOMA body wrapper
# expresses ``transl`` in the body model's "Y up, Z toward camera (negative
# Z forward)" frame. Confirmed empirically on the snack_box_pick sequence:
# negating Y and Z on the object brings body-object distance from
# ~6.6 m down to ~2.0 m at sequence start (which matches the recorded
# scene where the human is ~2 m away from the box) and ~0.35 m at the
# pick-up frame.
_R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0, 1.0])


def _convert_object_poses_cv_to_soma(object_poses_cv: np.ndarray) -> np.ndarray:
    """Convert a (T, 4, 4) object pose trajectory from OpenCV to SOMA world frame.

    Applied via left-multiplication ``T_soma = R_cv2soma @ T_cv``. This
    flips the world Y and Z axes so the object lives in the same frame
    that ``SOMALayer`` uses for ``transl``. Without this, the body and
    object end up in two different worlds and the relative hand-object
    pose is meaningless.
    """
    return np.einsum("ij,tjk->tik", _R_CV_TO_SOMA, object_poses_cv)


def main() -> None:
    """Main function."""
    args = parse_args()
    save_dir = args.motion_root.expanduser().resolve() / "whole_body" / args.soma_subdir

    data_folder = Path(args.data_folder)
    soma_params_path, object_mesh_path, object_poses_world = load_data(args.data_folder)
    print(f"Loaded data from {data_folder}")
    print(f"  SOMA params: {soma_params_path}")
    print(f"  Object mesh: {object_mesh_path}")
    print(f"  Object poses: {len(object_poses_world)} frames")

    config = load_robot_config(args.robot_name)
    kin = ConfigDrivenWholeBodyKinematics(config=config)
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
    # First-frame anchoring (mirrors ``NVHuman.load_motion``): the SOMA
    # exporter writes ``transl`` in raw world coordinates, so Hips can sit
    # several meters off the origin with arbitrary heading. The in-loop
    # ground-anchoring step below assumes the body is already centered at
    # the origin with canonical heading; without this normalization the
    # frame-0 ``ground_z_offset`` is computed against an off-origin pose
    # and the retargeted robot ends up floating / sinking relative to the
    # ground plane. The same transform is applied to the object trajectory
    # below to keep relative hand-object pose intact.
    motion = soma.load_motion(params_path=soma_params_path, normalize=True)

    # Apply the SAME first-frame transform to the object trajectory so the
    # body and object stay co-located after anchoring.
    # ``object_poses_world`` is already in the SOMA world frame (the CV->SOMA
    # axis flip happened in ``load_data``), so we just left-multiply each
    # 4x4 pose by ``T_anchor = [[R_first_inv, -R_first_inv @ transl_first], [0, 1]]``.
    transl_first = motion["first_frame_transl"]
    R_first_inv = motion["first_frame_R_inv"]
    norm_transform = np.eye(4)
    norm_transform[:3, :3] = R_first_inv
    norm_transform[:3, 3] = -R_first_inv @ transl_first
    object_poses = np.einsum("ij,tjk->tik", norm_transform, object_poses_world)

    # Optional reconstruction-side ground plane. The same chain of transforms
    # that brings ``poses.npy`` into the robot frame must be applied to the
    # plane equation. Returns ``None`` when ``ground_plane.json`` is absent so
    # the post-process below falls back to the legacy horizontal plane.
    ground_plane_path = Path(args.data_folder) / "ground_plane.json"
    reconstructed_plane = load_ground_plane_robot_frame(
        ground_plane_path,
        cv_to_source=_R_CV_TO_SOMA,
        first_frame_anchor=norm_transform,
        source_to_robot=np.asarray(config.r_world, dtype=np.float64),
    )
    if reconstructed_plane is not None:
        print(
            "[soma_to_g1] grounding plane: reconstructed "
            f"(normal={reconstructed_plane.normal}, "
            f"offset={reconstructed_plane.offset:+.4f})"
        )
    else:
        print(
            "[soma_to_g1] grounding plane: fallback ReferencePlane.horizontal(z=0.0); "
            f"no ground_plane.json found at {ground_plane_path}"
        )

    joint_pos = motion["joints"]
    joint_rot_wxyz = motion["joints_wxyz"]
    vertices = motion["vertices"]
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

    # Resolve and validate --start-frame / --end-frame against the original
    # source length, then slice every per-frame array down to the requested
    # window. The first-frame anchoring above already ran against the
    # ORIGINAL frame 0, so positions in the saved trajectory stay
    # comparable across different windows.
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

    sequence_id = data_folder.name

    head_idx = SOMA_JOINTS_ORDER.index("Head")
    root_idx = SOMA_JOINTS_ORDER.index("Hips")

    object_name = f"{sequence_id}_object"

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
    contact_threshold = float(args.contact_threshold)

    builder: dict[str, list] | None = None
    object_body_names: list[str] = ["object"]
    safe_object_body_names: list[str] = [_usd_safe(n) for n in object_body_names]
    object_mesh_radius: float = 0.0
    soma_identity_coeffs: list[float] = []
    soma_scale_params: list[float] = []
    frame_names_list: list[str] = list(kin.robot_frame_names.values())
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

    # SOMA fingertip joint names match NVHuman 1:1 (no twist joints involved).
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
        # Identity/scale are constant in time; pick frame 0 for source_payload.
        soma_identity_coeffs = params["identity_coeffs"][0].astype(np.float32).tolist()
        soma_scale_params = params["scale_params"][0].astype(np.float32).tolist()

        object_mesh_path = str(Path(object_mesh_path).resolve())
        object_mesh_radius = _compute_mesh_radius(object_mesh_path)

        # Mirror the NVHuman retargeter: copy mesh + materials next to the
        # parquet partition so saved asset paths are portable. Place under a
        # sibling ``object/`` folder relative to the parquet's ``robot_name=``
        # leaf so save_motion_parquet's rmtree of the leaf does not delete
        # them.
        mesh_dst_dir = save_dir / f"sequence_id={sequence_id}" / "object"
        mesh_dst_dir.mkdir(parents=True, exist_ok=True)
        for src in Path(object_mesh_path).parent.iterdir():
            if not src.is_file():
                continue
            shutil.copy2(src, mesh_dst_dir / src.name)
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
            # Source raw -- the field names are inherited from the NVHuman
            # builder so the post-process plane-alignment helper does not
            # need a SOMA-specific FirstPassResult dataclass; for SOMA,
            # `nvhuman_*` here actually carries SOMA head/root values.
            "soma_joints": [],
            "soma_joints_wxyz": [],
            "nvhuman_head_translation": [],
            "nvhuman_head_wxyz": [],
            "nvhuman_root_translation": [],
            "nvhuman_root_wxyz": [],
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
    ground_z_offset = 0.0
    object_z_lift = 0.0

    foot_frame_idxs = [
        list(kin.robot_frame_names.values()).index(fn) for fn in foot_frame_names
    ]

    ankle_xyz_per_frame: list[list[list[float]]] = []

    # IK diagnostics scaffolding -- populated only when --diagnose-ik is
    # set. Kept out-of-band so the hot loop is unchanged for normal runs.
    diagnose_ik = bool(args.diagnose_ik)
    ik_task_names: list[str] = list(kin.frame_tasks.keys())
    q_lower = np.asarray(kin.robot.model.lowerPositionLimit, dtype=np.float64)
    q_upper = np.asarray(kin.robot.model.upperPositionLimit, dtype=np.float64)
    # Free-flyer joints have +/-inf bounds, so the saturation check needs
    # to skip rows where the URDF didn't author a finite limit. Index 0..6
    # is the free-flyer position+quat anyway.
    finite_limit_mask = np.isfinite(q_lower) & np.isfinite(q_upper)
    # `q_ik`/`q_lower`/`q_upper` all have length `nq`; joint names live at
    # the joint level (one per Pinocchio joint), so the `q_idx -> joint
    # name` map is built once here.
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
    # Per-frame diagnostic records, one tuple per "interesting" frame.
    # Format: (frame_idx, total_pos_error, dominant_task, dominant_err,
    # n_iter, n_saturated, sample_saturated_joint_name).
    diag_offenders: list[tuple[int, float, str, float, int, int, str]] = []

    # ``frame_idx`` here is local to the iterated window (0..n_iter_frames),
    # NOT the absolute index into the original SOMA sequence. The
    # ground-anchor initializer below fires on the FIRST iterated frame
    # because that is the only time the loop has IK-solved foot placements
    # to compute the offset from; if we used an absolute "== 0" check we
    # would skip initialization entirely whenever ``--start-frame > 0``.
    for frame_idx in tqdm(range(n_iter_frames), desc="Retargeting"):
        positions = joint_pos[frame_idx]
        rotations = joint_rot_wxyz[frame_idx]

        result = kin.compute(
            source_joints=positions,
            source_joints_wxyz=rotations,
            source_to_robot_scale=args.scale,
            qpos=q,
        )
        q_ik = result["q"].copy()
        q = q_ik.copy()

        if diagnose_ik:
            # Identify the dominant frame-task residual and any joints
            # whose IK solution sits within 1 mrad / 1e-3 of a position
            # limit (counting as "saturated"). The saturation check uses
            # `q_ik` -- the pre-clamp solution -- so we see what the
            # solver actually wanted, not the post-clamp value used as
            # the next warm-start.
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
            # ``frame_idx`` is local to the iterated window; show the
            # absolute source index too so it lines up with viser/replay.
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

        lowest_sole_z = (
            min(result["frame_pose"][i, 2] for i in foot_frame_idxs) - ankle_roll_offset
        )

        if frame_idx == 0:
            ground_z_offset = -lowest_sole_z
        else:
            adjusted_lowest = lowest_sole_z + ground_z_offset
            if adjusted_lowest < 0.0:
                q[2] -= adjusted_lowest

        ankle_xyz_per_frame.append(
            [
                [
                    float(result["frame_pose"][foot_frame_idxs[0], 0]),
                    float(result["frame_pose"][foot_frame_idxs[0], 1]),
                    float(result["frame_pose"][foot_frame_idxs[0], 2])
                    + ground_z_offset,
                ],
                [
                    float(result["frame_pose"][foot_frame_idxs[1], 0]),
                    float(result["frame_pose"][foot_frame_idxs[1], 1]),
                    float(result["frame_pose"][foot_frame_idxs[1], 2])
                    + ground_z_offset,
                ],
            ]
        )

        obj_pose = object_poses[frame_idx]
        obj_position = kin.transform_source_position(obj_pose[:3, 3])
        obj_rotation_mat = kin.transform_world_rotation(obj_pose[:3, :3])
        obj_rotation = R.from_matrix(obj_rotation_mat)

        obj_position[2] += ground_z_offset
        object_z_lift = 0.0

        head_position = kin.transform_source_position(positions[head_idx])
        head_position[2] += ground_z_offset
        head_rotation_mat = R.from_quat(
            rotations[head_idx], scalar_first=True
        ).as_matrix()
        head_rotation_mat = kin.transform_source_rotation(head_rotation_mat)
        head_rotation_wxyz = R.from_matrix(head_rotation_mat).as_quat(scalar_first=True)

        root_position = kin.transform_source_position(positions[root_idx])
        root_position[2] += ground_z_offset
        root_rotation_mat = R.from_quat(
            rotations[root_idx], scalar_first=True
        ).as_matrix()
        root_rotation_mat = kin.transform_source_rotation(root_rotation_mat)
        root_rotation_wxyz = R.from_matrix(root_rotation_mat).as_quat(scalar_first=True)

        frame_pose = result["frame_pose"]
        ee_pose_t: list[list[float]] = []
        for i in ee_frame_indices:
            pose = list(frame_pose[i])
            pose[2] = float(pose[2]) + ground_z_offset
            ee_pose_t.append(pose)

        object_translation_w_np = (obj_position + [0, 0, object_z_lift]).astype(
            np.float32
        )
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
                points[k, 0] = p[0]
                points[k, 1] = p[1]
                points[k, 2] = p[2] + ground_z_offset
            diff = verts_w[:, None, :] - points[None, :, :]
            sq_dists = np.einsum("vki,vki->vk", diff, diff)
            min_sq = float(sq_dists.min())
            per_side_active.append(1.0 if min_sq < threshold_sq else 0.0)

        if builder is not None:
            root_pos_robot = q_ik[:3].copy()
            root_pos_robot[2] += ground_z_offset
            root_pos_robot = root_pos_robot.tolist()
            root_quat_xyzw = q_ik[3:7]
            root_wxyz = [
                float(root_quat_xyzw[3]),
                float(root_quat_xyzw[0]),
                float(root_quat_xyzw[1]),
                float(root_quat_xyzw[2]),
            ]
            joint_positions = q_ik[base_q_size:].tolist()

            obj_wxyz = obj_rotation.as_quat(scalar_first=True).tolist()
            obj_body_pos = [(obj_position + [0, 0, object_z_lift]).tolist()]
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
            builder["nvhuman_head_translation"].append(head_position.tolist())
            builder["nvhuman_head_wxyz"].append(head_rotation_wxyz.tolist())
            builder["nvhuman_root_translation"].append(root_position.tolist())
            builder["nvhuman_root_wxyz"].append(root_rotation_wxyz.tolist())

        if playback is not None:
            vertices_vis = kin.transform_source_position(vertices[frame_idx]).copy()
            vertices_vis[:, 2] += ground_z_offset
            q_vis = q_ik.copy()
            q_vis[2] += ground_z_offset

            ik_target_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for frame_name, task in kin.frame_tasks.items():
                target_pos_vis = task.transform_target_to_world.translation.copy()
                target_pos_vis[2] += ground_z_offset
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
                # The viser playback path expects a body model with the same
                # interface NVHuman exposes for visualization. SOMA wraps
                # SOMALayer; pass it through so the helper can call
                # ``model.visualize(...)`` if implemented. The path is
                # tolerant to body wrappers that do not implement
                # ``visualize`` (it falls back to vertex-only rendering).
                nvhuman=soma,
            )

        if frame_idx == 0 and args.visualize:
            time.sleep(5)

    if builder is not None:
        first_pass = FirstPassResult(
            fps=float(kin.frequency),
            robot_root_position=np.asarray(
                builder["robot_root_position"], dtype=np.float64
            ),
            robot_root_wxyz=np.asarray(builder["robot_root_wxyz"], dtype=np.float64),
            robot_joint_positions=np.asarray(
                builder["robot_joint_positions"], dtype=np.float64
            ),
            ee_pose_w=np.asarray(builder["ee_pose_w"], dtype=np.float64),
            object_root_position=np.asarray(
                builder["object_root_position"], dtype=np.float64
            ),
            object_root_axis_angle=np.asarray(
                builder["object_root_axis_angle"], dtype=np.float64
            ),
            object_body_position=np.asarray(
                builder["object_body_position"], dtype=np.float64
            ),
            object_body_wxyz=np.asarray(builder["object_body_wxyz"], dtype=np.float64),
            hand_contact_active_per_frame=np.asarray(
                builder["hand_contact_active_per_frame"], dtype=np.float64
            ),
            nvhuman_head_translation=np.asarray(
                builder["nvhuman_head_translation"], dtype=np.float64
            ),
            nvhuman_root_translation=np.asarray(
                builder["nvhuman_root_translation"], dtype=np.float64
            ),
            ankle_frame_xyz=np.asarray(ankle_xyz_per_frame, dtype=np.float64),
        )

        first_pass_cache_extras = {
            "object_articulation": list(builder["object_articulation"]),
            "soma_joints": [list(f) for f in builder["soma_joints"]],
            "soma_joints_wxyz": [list(f) for f in builder["soma_joints_wxyz"]],
            "nvhuman_head_wxyz": [list(f) for f in builder["nvhuman_head_wxyz"]],
            "nvhuman_root_wxyz": [list(f) for f in builder["nvhuman_root_wxyz"]],
            "ik_error_per_frame": list(builder["ik_error_per_frame"]),
            "ik_num_iterations": list(builder["ik_num_iterations"]),
            "frame_task_errors": [list(f) for f in builder["frame_task_errors"]],
        }

        plane = (
            reconstructed_plane
            if reconstructed_plane is not None
            else ReferencePlane.horizontal(z=0.0)
        )
        sole_xyz = first_pass.ankle_frame_xyz.copy()
        sole_xyz[..., 2] -= ankle_roll_offset
        # Per-frame ground anchoring during the IK loop already drove the
        # body's lowest sole to z = 0 of the robot world (see
        # ``ground_z_offset`` initialization on frame 0). The reconstructed
        # ``ground_plane.json`` carries the absolute scene height instead, so
        # using its raw offset would drag the body and the carried object
        # down to that absolute level even though the body is already on the
        # floor. We therefore preserve the reconstructed plane's normal (it
        # captures any scene tilt) but shift the offset so the plane sits
        # exactly under the frame-0 lowest sole. Without this, the post-
        # process injects the reconstruction's ground-Z bias (e.g.
        # ``-1.04 m`` for the trash-can sequence) into ``robot_delta_z`` and
        # the saved object ends up below the simulator's ground plane.
        if reconstructed_plane is not None:
            n = np.asarray(plane.normal, dtype=np.float64)
            frame0_anchor = sole_xyz[0].min(axis=0)
            adjusted_offset = -float(np.dot(n, frame0_anchor))
            plane = ReferencePlane(
                normal=tuple(plane.normal),
                offset=adjusted_offset,
            )
        robot_delta_z = compute_plane_alignment_offsets(
            sole_xyz, plane, PlaneAlignmentConfig()
        )
        interaction_mask = compute_interaction_mask(first_pass, InteractionMaskConfig())
        corr_obj_root_pos, corr_obj_root_aa, corr_obj_body_pos, corr_obj_body_wxyz = (
            correct_object_trajectory(
                first_pass,
                interaction_mask,
                robot_delta_z,
                ObjectCorrectionConfig(),
            )
        )

        corr_root_pos = first_pass.robot_root_position.copy()
        corr_root_pos[:, 2] += robot_delta_z
        builder["robot_root_position"] = corr_root_pos.tolist()

        corr_ee_pose = first_pass.ee_pose_w.copy()
        corr_ee_pose[..., 2] += robot_delta_z[:, None]
        builder["ee_pose_w"] = corr_ee_pose.tolist()

        corr_nv_head = first_pass.nvhuman_head_translation.copy()
        corr_nv_head[:, 2] += robot_delta_z
        builder["nvhuman_head_translation"] = corr_nv_head.tolist()

        corr_nv_root = first_pass.nvhuman_root_translation.copy()
        corr_nv_root[:, 2] += robot_delta_z
        builder["nvhuman_root_translation"] = corr_nv_root.tolist()

        builder["object_root_position"] = corr_obj_root_pos.tolist()
        builder["object_root_axis_angle"] = corr_obj_root_aa.tolist()
        builder["object_body_position"] = corr_obj_body_pos.tolist()
        builder["object_body_wxyz"] = corr_obj_body_wxyz.tolist()

        n_interact = int(np.count_nonzero(interaction_mask))
        pre_ankle_z = first_pass.ankle_frame_xyz[:, :, 2]
        post_ankle_z = pre_ankle_z + robot_delta_z[:, None]
        pre_sole_lowest = float(pre_ankle_z.min() - ankle_roll_offset)
        pre_sole_highest = float(pre_ankle_z.max() - ankle_roll_offset)
        post_sole_lowest = float(post_ankle_z.min() - ankle_roll_offset)
        post_sole_highest = float(post_ankle_z.max() - ankle_roll_offset)
        print(
            "[INFO] Plane alignment: robot_delta_z range "
            f"[{float(robot_delta_z.min()):+.4f}, "
            f"{float(robot_delta_z.max()):+.4f}] m; "
            f"{n_interact}/{len(interaction_mask)} interaction frames."
        )
        print(
            f"[INFO] Robot sole Z pre-offset : "
            f"[{pre_sole_lowest:+.4f}, {pre_sole_highest:+.4f}] m"
        )
        print(
            f"[INFO] Robot sole Z post-offset: "
            f"[{post_sole_lowest:+.4f}, {post_sole_highest:+.4f}] m "
            f"(should be near 0)"
        )

        source_payload = pickle.dumps(
            {
                "soma_identity_coeffs": soma_identity_coeffs,
                "soma_scale_params": soma_scale_params,
                "soma_joints": builder.pop("soma_joints"),
                "soma_joints_wxyz": builder.pop("soma_joints_wxyz"),
                "nvhuman_head_translation": builder.pop("nvhuman_head_translation"),
                "nvhuman_head_wxyz": builder.pop("nvhuman_head_wxyz"),
                "nvhuman_root_translation": builder.pop("nvhuman_root_translation"),
                "nvhuman_root_wxyz": builder.pop("nvhuman_root_wxyz"),
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
            fps=float(kin.frequency),
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

        first_pass_cache_path = (
            FIRST_PASS_CACHE_DIR
            / f"sequence_id={sequence_id}"
            / f"robot_name={config.robot_name}"
            / "first_pass_cache.pkl"
        )
        first_pass_cache_path.parent.mkdir(parents=True, exist_ok=True)
        first_pass_cache = {
            "first_pass": first_pass,
            "builder_extras": first_pass_cache_extras,
            "metadata": {
                "sequence_id": sequence_id,
                "motion_params_path": soma_params_path,
                "fps": float(kin.frequency),
                "robot_joint_names": list(robot_joint_position_names),
                "ee_link_names": list(ee_link_names),
                "object_name": object_name,
                "safe_object_name": _usd_safe(object_name),
                "object_body_names": list(object_body_names),
                "safe_object_body_names": list(safe_object_body_names),
                "object_mesh_paths": [stored_mesh_path],
                "object_urdf_paths": [stored_urdf_path],
                "object_mesh_radius": [object_mesh_radius],
                "soma_identity_coeffs": list(soma_identity_coeffs),
                "soma_scale_params": list(soma_scale_params),
                "contact_threshold": float(contact_threshold),
                "g1_ankle_roll_offset": float(ankle_roll_offset),
                "ground_plane_source": (
                    "reconstructed" if reconstructed_plane is not None else "fallback"
                ),
                "ground_plane_normal": (
                    list(reconstructed_plane.normal)
                    if reconstructed_plane is not None
                    else [0.0, 0.0, 1.0]
                ),
                "ground_plane_offset": (
                    float(reconstructed_plane.offset)
                    if reconstructed_plane is not None
                    else 0.0
                ),
                "ground_plane_json_path": str(ground_plane_path),
            },
        }
        with open(first_pass_cache_path, "wb") as _f:
            pickle.dump(first_pass_cache, _f)
        print(f"[INFO] Cached first-pass snapshot -> {first_pass_cache_path}")

    if diagnose_ik:
        if not diag_offenders:
            print(
                f"[ik-diag] No frames flagged "
                f"(total_err < {diag_error_threshold} m, iters < "
                f"{diag_iter_threshold}/{kin.max_iter}, no saturated joints)."
            )
        else:
            # Top-K worst by total task residual, then by iteration count
            # as a tie-breaker. K=10 is enough to spot a band of bad
            # frames around contact while staying readable.
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
            # Per-task error contribution averaged across flagged frames
            # -- useful to decide which weight to bump.
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
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
