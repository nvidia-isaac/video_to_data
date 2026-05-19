# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Script to retarget NVHuman motion to G1 whole body using Pink IK solver.

Writes Parquet under ``HUMAN_MOTION_DATA_DIR/whole_body/nvhuman`` with
``partition_cols=(sequence_id, robot_name)``, e.g.
``.../sequence_id=<folder_name>/robot_name=g1/``. That layout is what
``SceneConfig`` / Isaac Lab use (``robot_name`` must match the registry key
``g1``).

The object mesh (and any accompanying materials) is copied to
``.../sequence_id=<folder_name>/object/`` so the parquet can reference it
with a repo-relative path — this keeps training portable across hosts and
bind mounts.

Usage:
    python scripts/retarget/nvhuman_to_g1.py <data_folder> --save
    python scripts/retarget/nvhuman_to_g1.py <data_folder> --visualize

Where data_folder contains:
    - nova_params_opt.pt (motion parameters)
    - poses.npy (object trajectory as 4x4 transforms)
    - object/textured_mesh.obj (object mesh; also used to build a URDF for sim)
"""

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
)
from robotic_grounding.retarget.params import (
    G1_ANKLE_ROLL_OFFSET,
    G1_FOOT_FRAME_NAMES,
    NVHUMAN_JOINTS_ORDER,
)
from robotic_grounding.retarget.read_nvhuman import NVHuman
from robotic_grounding.retarget.viser_playback import LiveFrameState, ViserPlayback
from robotic_grounding.retarget.whole_body_kinematics import G1WholeBodyKinematics
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

G1_URDF = G1_URDF_DIR / "main_with_hand.urdf"
PACKAGE_DIRS = [str(G1_URDF_DIR)]
SAVE_DIR = HUMAN_MOTION_DATA_DIR / "whole_body" / "nvhuman"
# First-pass IK snapshot for offline post-process reruns. Deliberately
# OUTSIDE the repo (and decoupled from ``HUMAN_MOTION_DATA_DIR``) so local dev
# runs don't litter the repo tree with large pickles; OSMO workflow runs land
# here too (pod-local ``/tmp`` is ephemeral, which is fine because pass 2 is
# a local debug loop). Override via ``FIRST_PASS_CACHE_DIR`` env var if you
# need the cache to persist across reboots or live elsewhere.
FIRST_PASS_CACHE_DIR = Path(
    os.environ.get("FIRST_PASS_CACHE_DIR", "/tmp/nvhuman_g1_processed_cache")
)
# Repo root (the `robotic_grounding/` subdir). Used to store repo-relative
# asset paths in the parquet so training works from any checkout location.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _usd_safe(name: str) -> str:
    """Make a name safe for USD prim paths (no leading digits, no @ etc.)."""
    safe = name.replace("@", "_")
    if safe and (safe[0].isdigit() or not (safe[0].isalpha() or safe[0] == "_")):
        return f"obj_{safe}"
    return safe


def _copy_mesh_assets(src_dir: Path, dst_dir: Path) -> None:
    """Copy the object mesh and any adjacent materials into ``dst_dir``.

    ``textured_mesh.obj`` is required; ``material*.mtl`` and ``material*.png``
    files are copied when present (OBJ textures referenced via the .mtl).
    Existing files at the destination are overwritten.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    obj_src = src_dir / "textured_mesh.obj"
    if not obj_src.is_file():
        raise FileNotFoundError(f"Expected mesh at {obj_src}")
    shutil.copy2(obj_src, dst_dir / "textured_mesh.obj")

    for pattern in ("material*.mtl", "material*.png"):
        for extra in src_dir.glob(pattern):
            shutil.copy2(extra, dst_dir / extra.name)


def _build_object_urdf(mesh_path: str, urdf_path: Path) -> str:
    """Write a simple rigid-object URDF that references ``mesh_path``.

    ``mesh_path`` is used verbatim inside ``<mesh filename="...">``. Pass a
    bare filename (e.g. ``"textured_mesh.obj"``) when the mesh sits next to
    the URDF so the pair remains relocatable; URDF parsers (including Isaac
    Sim's) resolve relative mesh filenames against the URDF's directory.
    """
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
    kin: G1WholeBodyKinematics, base_q_size: int
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
        description="Retarget NVHuman motion to G1 whole body"
    )
    parser.add_argument(
        "data_folder",
        type=str,
        help="Path to folder containing nova_params_opt.pt, object mesh, and poses.npy",
    )
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--save", action="store_true", help="Save retargeted data")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor from NVHuman to robot"
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
    return parser.parse_args()


def load_data(folder_path: str) -> tuple[str, str, np.ndarray]:
    """Load motion params and object data from folder.

    Args:
        folder_path: Path to folder containing nova_params_opt.pt, object mesh, poses.npy

    Returns:
        Tuple of (motion_params_path, mesh_path, poses)
        poses are normalized to match NVHuman's origin normalization
    """
    folder = Path(folder_path).resolve()
    motion_params_path = str(folder / "nova_params_opt.pt")
    mesh_path = str(folder / "object" / "textured_mesh.obj")
    poses_path = folder / "poses.npy"

    required = {
        "nova_params_opt.pt": folder / "nova_params_opt.pt",
        "poses.npy": poses_path,
        "object/textured_mesh.obj": folder / "object" / "textured_mesh.obj",
    }
    missing = [name for name, p in required.items() if not p.is_file()]
    if missing:
        msg = (
            f"Data folder is missing required files: {missing}\n"
            f"  Resolved folder: {folder}\n"
            f"Expected layout:\n"
            f"  {folder}/nova_params_opt.pt\n"
            f"  {folder}/poses.npy\n"
            f"  {folder}/object/textured_mesh.obj\n"
            "If you run inside Docker, host paths like /home/... are not visible "
            "unless bind-mounted. Mount your data (e.g. -v /host/data:/data/reconstructed_data) "
            "and pass the in-container path to this script."
        )
        raise FileNotFoundError(msg)

    object_poses = np.load(poses_path)

    params = torch.load(motion_params_path)
    global_orient_first = params["global_orient"][0].cpu().numpy()
    transl_first = params["transl"][0].cpu().numpy()

    R_first = R.from_rotvec(global_orient_first).as_matrix()
    R_first_inv = R_first.T

    norm_transform = np.eye(4)
    norm_transform[:3, :3] = R_first_inv
    norm_transform[:3, 3] = -R_first_inv @ transl_first

    object_poses_normalized = np.array([norm_transform @ pose for pose in object_poses])

    return motion_params_path, mesh_path, object_poses_normalized


def main() -> None:
    """Main function."""
    args = parse_args()

    # Load all data from folder
    data_folder = Path(args.data_folder)
    motion_params_path, object_mesh_path, object_poses = load_data(args.data_folder)
    print(f"Loaded data from {data_folder}")
    print(f"  Motion params: {motion_params_path}")
    print(f"  Object mesh: {object_mesh_path}")
    print(f"  Object poses: {len(object_poses)} frames")

    # Initialize whole body kinematics
    kin = G1WholeBodyKinematics(
        robot_asset_path=str(G1_URDF),
        package_dirs=PACKAGE_DIRS,
        source_model="nvhuman",
    )
    base_q_size = 7
    robot_joint_position_names = _get_robot_joint_position_names(kin, base_q_size)

    # Load motion data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nvhuman = NVHuman(device=device)
    motion = nvhuman.load_motion(params_path=motion_params_path)
    joint_pos = motion["joints"]
    joint_rot_wxyz = motion["joints_wxyz"]
    vertices = motion["vertices"]
    num_frames = motion["num_frames"]
    if len(object_poses) != num_frames:
        raise ValueError(
            f"poses.npy length ({len(object_poses)}) != motion frames ({num_frames})"
        )

    # Get sequence ID from folder name
    sequence_id = data_folder.name

    # Get joint indices for head/root tracking
    head_idx = NVHUMAN_JOINTS_ORDER.index("Head")
    root_idx = NVHUMAN_JOINTS_ORDER.index("Hips")

    object_name = f"{sequence_id}_object"

    # Pre-load object mesh vertices for ground-plane computation.
    _obj_verts = []
    with open(object_mesh_path) as _f:
        for _line in _f:
            if _line.startswith("v "):
                _parts = _line.split()
                _obj_verts.append(
                    [float(_parts[1]), float(_parts[2]), float(_parts[3])]
                )
    object_mesh_vertices = np.array(_obj_verts, dtype=np.float64)
    # Float32 copy used by the in-loop contact-mask computation: transform
    # verts once per frame into world, then take min distance to each side's
    # palm + fingertip stack. Thousand-fold faster than list-of-lists math.
    object_mesh_vertices_f32 = object_mesh_vertices.astype(np.float32)
    contact_threshold = float(args.contact_threshold)

    # Collected time-series for the motion_v1 parquet. These lists grow one
    # entry per frame and are materialized into a `MotionData` at save time.
    builder: dict[str, list] | None = None
    object_body_names: list[str] = ["object"]
    safe_object_body_names: list[str] = [_usd_safe(n) for n in object_body_names]
    object_mesh_radius: float = 0.0
    betas: list[float] = []
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

    # Hand sides are derived from the resolved EE links so that
    # visualization (contact markers, etc.) and the save pipeline agree
    # on the per-side ordering.
    hand_sides: list[str] = [
        name.split("_")[0] for name in ee_link_names if "_hand_palm_link" in name
    ]

    # Per-side fingertip indices in the NVHuman source-joint list. Contact
    # labeling uses palm + five fingertip joints because for large objects
    # (chair rails, backrests) the palm link can sit several cm off the
    # outer surface while fingers wrap the geometry.
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
            NVHUMAN_JOINTS_ORDER.index(n)
            for n in candidates
            if n in NVHUMAN_JOINTS_ORDER
        ]

    if args.save:
        params = torch.load(motion_params_path)
        betas_tensor = params["betas"].cpu().numpy()
        if betas_tensor.ndim > 1:
            betas = betas_tensor[0].flatten().tolist()
        else:
            betas = betas_tensor.flatten().tolist()

        object_mesh_path = str(Path(object_mesh_path).resolve())
        object_mesh_radius = _compute_mesh_radius(object_mesh_path)

        # Copy mesh + materials next to the parquet partition so the saved
        # asset paths are portable. Destination sits as a sibling of the
        # ``robot_name=g1`` partition dir, NOT inside it, because
        # ``save_motion_parquet`` rmtrees the leaf partition on each run.
        mesh_dst_dir = SAVE_DIR / f"sequence_id={sequence_id}" / "object"
        _copy_mesh_assets(src_dir=Path(object_mesh_path).parent, dst_dir=mesh_dst_dir)
        # URDF is generated at the copied-mesh location with a bare-filename
        # mesh reference so the URDF resolves the .obj via its own directory
        # regardless of cwd (matches Isaac Sim's URDF import behavior).
        urdf_dst = mesh_dst_dir / "textured_mesh.urdf"
        _build_object_urdf(mesh_path="textured_mesh.obj", urdf_path=urdf_dst)

        # Record the stored-in-parquet paths as repo-relative when the
        # copied assets land inside the repo (the common local case with
        # SAVE_DIR under ASSETS_DIR). Fall back to absolute paths when they
        # don't (e.g. OSMO workflow runs that point HUMAN_MOTION_DATA_DIR
        # at ``/tmp/human_motion_data``); those runs snapshot the whole
        # output tree, so absolute paths inside the pod are correct.
        # ``object_mesh_path`` itself is left as an absolute path so later
        # in-process reads (e.g. trimesh.load in the visualize branch) work
        # regardless of cwd.
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
            # Robot state
            "robot_root_position": [],
            "robot_root_wxyz": [],
            "robot_joint_positions": [],
            # EE pose (stacked per-frame [pos(3), wxyz(4)] for each EE link)
            "ee_pose_w": [],
            # Object
            "object_articulation": [],
            "object_root_axis_angle": [],
            "object_root_position": [],
            "object_body_position": [],
            "object_body_wxyz": [],
            # Per-side binary contact labels computed in-loop from palm +
            # fingertip proximity to the object mesh surface. Appended one
            # 2-element list per frame in order `[left, right]`.
            "hand_contact_active_per_frame": [],
            # Diagnostics
            "ik_error_per_frame": [],
            "ik_num_iterations": [],
            "frame_task_errors": [],
            # Source raw
            "nvhuman_joints": [],
            "nvhuman_joints_wxyz": [],
            "nvhuman_head_translation": [],
            "nvhuman_head_wxyz": [],
            "nvhuman_root_translation": [],
            "nvhuman_root_wxyz": [],
        }

    # Viser scene — delegated to `ViserPlayback.for_live_retarget` so the
    # scene graph matches the parquet replay tool (`scripts/replay_viser.py`)
    # exactly: `/object`, `/head`, `/root`, `/contacts/{left,right}` handles
    # plus IK-target frames lazily created under `/targets/<name>`.
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

    # Initialize qpos
    q = kin.robot.q0.copy()

    # Ground-plane Z offset: the NVHuman normalization centers the pelvis at
    # the origin, so the IK-solved robot sits around Z=0 with its feet
    # somewhere below. On frame 0 we shift the robot so its lowest IK-solved
    # foot sole sits at Z=0, then apply the same `ground_z_offset` to every
    # saved world-frame position (root, head, ees, fingertips, object) so
    # robot and scene share one floor-aligned frame. Set on frame 0 below.
    ground_z_offset = 0.0
    object_z_lift = 0.0

    # Robot ankle-roll link indices in `frame_pose`; converted to sole Z
    # per-frame by subtracting `G1_ANKLE_ROLL_OFFSET` (URDF-measured
    # distance from ankle_roll_link origin down to the foot sole).
    foot_frame_idxs = [
        list(kin.robot_frame_names.values()).index(fn) for fn in G1_FOOT_FRAME_NAMES
    ]

    # Cached per-frame ankle XYZ (post pass-1 ground offset, pre-clamp) for
    # downstream drift correction. Populated inside the loop; converted to a
    # (T, 2, 3) array and packaged into `FirstPassResult` before save.
    ankle_xyz_per_frame: list[list[list[float]]] = []

    # Processing loop
    for frame_idx in tqdm(range(num_frames), desc="Retargeting"):
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

        # Lowest IK-solved foot sole Z in the raw IK frame. ankle_roll_link
        # is the ankle joint origin; subtracting the URDF-measured
        # ankle-to-sole distance gives the contact-surface Z.
        lowest_sole_z = (
            min(result["frame_pose"][i, 2] for i in foot_frame_idxs)
            - G1_ANKLE_ROLL_OFFSET
        )

        if frame_idx == 0:
            # Anchor the whole sequence: choose an offset that lifts the
            # lowest frame-0 sole to Z=0. Do NOT mutate `q[2]` here — every
            # downstream write-site already adds `ground_z_offset` (root,
            # head, ees, fingertips, object, saved root_pos_robot), so a
            # `q[2] += ground_z_offset` would double-lift the saved root.
            ground_z_offset = -lowest_sole_z
        else:
            # Downward-drift clamp: if the lowest sole dips below Z=0 in the
            # ground-aligned frame (reconstruction noise), lift ``q[2]`` so
            # the NEXT frame's IK warm-start is floor-compliant. The clamp
            # is used *only* as IK warm-start seed, NOT for any saved data:
            # downstream writers use ``q_ik`` (pre-clamp) together with
            # ``result["frame_pose"]`` so that ``robot_root_position``,
            # ``ee_pose_w``, and ``ankle_frame_xyz`` all live in one
            # consistent reference frame. Mixing clamp-into-root with
            # pre-clamp-frame_pose-into-ankles (the earlier behavior)
            # produced physically impossible "leg extensions" up to ~1.05 m
            # in the cache, which in turn caused the post-process delta_z
            # to double-correct the robot Z.
            adjusted_lowest = lowest_sole_z + ground_z_offset
            if adjusted_lowest < 0.0:
                q[2] -= adjusted_lowest

        # Cache post-offset ankle XYZ per side for the post-process drift
        # correction module. Uses pre-clamp `frame_pose` to stay consistent
        # with the Z convention used when building `ee_pose_w` below.
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

        # Compute object pose in robot coordinate system. Object rotation is
        # a world-frame transform (not a body-local joint rotation), so we use
        # transform_world_rotation (left-multiply by R_src_to_robot) rather
        # than transform_source_rotation (similarity transform R @ M @ R^T
        # used for body-local joint rotations).
        obj_pose = object_poses[frame_idx]
        obj_position = kin.transform_source_position(obj_pose[:3, 3])
        obj_rotation_mat = kin.transform_world_rotation(obj_pose[:3, :3])
        obj_rotation = R.from_matrix(obj_rotation_mat)

        obj_position[2] += ground_z_offset

        # Object Z-offset: the object trajectory gets the SAME Z offsets as
        # the robot (ground_z_offset here in pass-1, plus the post-process
        # delta_z later). No object-only "lift to Z=0" is applied: that
        # mesh-canonicalization lift would move the chair up by +1.27 m
        # (for the skinny_wood_chair) while leaving the hand at its
        # IK-solved Z, which BREAKS the raw NVHuman hand-object grip
        # relative pose. Keep ``object_z_lift = 0`` so that
        # ``hand_saved - obj_saved == hand_raw - obj_raw`` across the whole
        # sequence. If the raw reconstruction has the object below the
        # ground (a reconstruction-calibration artifact), that will remain
        # visible in replay — that's a reconstruction issue, not something
        # to patch over here at the cost of losing the grip relationship.
        object_z_lift = 0.0

        # Extract head trajectory in robot coordinate system
        head_position = kin.transform_source_position(positions[head_idx])
        head_position[2] += ground_z_offset
        head_rotation_mat = R.from_quat(
            rotations[head_idx], scalar_first=True
        ).as_matrix()
        head_rotation_mat = kin.transform_source_rotation(head_rotation_mat)
        head_rotation_wxyz = R.from_matrix(head_rotation_mat).as_quat(scalar_first=True)

        # Extract root trajectory in robot coordinate system
        root_position = kin.transform_source_position(positions[root_idx])
        root_position[2] += ground_z_offset
        root_rotation_mat = R.from_quat(
            rotations[root_idx], scalar_first=True
        ).as_matrix()
        root_rotation_mat = kin.transform_source_rotation(root_rotation_mat)
        root_rotation_wxyz = R.from_matrix(root_rotation_mat).as_quat(scalar_first=True)

        # Ground-shifted palm positions from IK, used by both the saved
        # trajectory and the per-frame contact mask (and hence the
        # visualization marker below).
        frame_pose = result["frame_pose"]  # (K, 7), wxyz order
        ee_pose_t: list[list[float]] = []
        for i in ee_frame_indices:
            pose = list(frame_pose[i])
            pose[2] = float(pose[2]) + ground_z_offset
            ee_pose_t.append(pose)

        # Per-frame contact mask: for each side stack palm + 5 fingertip
        # positions (all in the ground-shifted world frame) and take the min
        # distance to the object mesh. Fires if any point is within
        # `contact_threshold`. Saved as `hand_contact_active` and drives the
        # on-screen contact markers.
        #
        # Implemented as one vectorized `(K, V)` sq-distance op per side to
        # keep per-frame numpy allocations minimal; repeated per-point calls
        # were surfacing spurious attribute errors from numpy's internals
        # when long sequences interact with pink/pinocchio inside a single
        # python process.
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
            # Stack palm + fingertips into (K, 3) in one go.
            fingertip_joint_ids = hand_side_to_fingertip_source_joints.get(side, [])
            points = np.empty((1 + len(fingertip_joint_ids), 3), dtype=np.float32)
            points[0] = ee_pose_t[side_idx][:3]
            for k, j in enumerate(fingertip_joint_ids, start=1):
                p = kin.transform_source_position(positions[j])
                points[k, 0] = p[0]
                points[k, 1] = p[1]
                points[k, 2] = p[2] + ground_z_offset
            # Squared distance (V, K) — compare to squared threshold; avoids
            # a per-frame sqrt and keeps the operation in one numpy call.
            diff = verts_w[:, None, :] - points[None, :, :]  # (V, K, 3)
            sq_dists = np.einsum("vki,vki->vk", diff, diff)  # (V, K)
            min_sq = float(sq_dists.min())
            per_side_active.append(1.0 if min_sq < threshold_sq else 0.0)

        # Log timestep data
        if builder is not None:
            # Pinocchio free-flyer: q[:3] = position, q[3:7] = quaternion (xyzw).
            # Use ``q_ik`` (pre-clamp) so the saved root lives in the same
            # reference frame as ``ankle_frame_xyz`` / ``ee_pose_w``, which
            # come from ``result["frame_pose"]`` (also pre-clamp). The
            # clamp mutates only the next-frame IK warm-start ``q``; it must
            # not leak into the saved root, otherwise the post-process
            # delta_z (derived from pre-clamp ankle Z) would double-correct.
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
            # Source raw motion — stashed in source_payload at save time.
            builder["nvhuman_joints"].append(positions.tolist())
            builder["nvhuman_joints_wxyz"].append(rotations.tolist())
            builder["nvhuman_head_translation"].append(head_position.tolist())
            builder["nvhuman_head_wxyz"].append(head_rotation_wxyz.tolist())
            builder["nvhuman_root_translation"].append(root_position.tolist())
            builder["nvhuman_root_wxyz"].append(root_rotation_wxyz.tolist())

        # Update visualization — one call into the shared scene graph.
        if playback is not None:
            # Ground-aligned body mesh (pre-clamp IK frame).
            vertices_vis = kin.transform_source_position(vertices[frame_idx]).copy()
            vertices_vis[:, 2] += ground_z_offset
            # Pre-clamp IK `q` lifted into the ground-aligned frame so the
            # robot stays aligned with the mesh overlay.
            q_vis = q_ik.copy()
            q_vis[2] += ground_z_offset

            # IK-target poses, lifted into the ground-aligned frame.
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
                nvhuman=nvhuman,
            )

        if frame_idx == 0 and args.visualize:
            time.sleep(5)

    # Save data
    if builder is not None:
        # ------------------------------------------------------------------
        # Post-process plane alignment + object correction
        # ------------------------------------------------------------------
        # Pack pass-1 outputs into FirstPassResult, compute a per-frame
        # robot Z-offset that drags the lowest foot sole onto the ground
        # plane (see `compute_plane_alignment_offsets`), derive an
        # interaction mask, and correct the object trajectory per the two
        # rules (interaction -> preserve hand-to-object relative pose; no
        # interaction -> anchored to adjacent contact pose). See
        # `robotic_grounding.retarget.ground_alignment` for details. Pass 1
        # itself is not modified; this step only mutates the builder arrays
        # before they are pickled / saved.
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

        # Snapshot the builder fields needed to rebuild MotionData on a
        # post-process-only rerun. Done here (pre-correction) because the
        # post-process below mutates / pops from ``builder``. The cache
        # file itself is written AFTER ``save_motion_parquet`` because
        # that call ``shutil.rmtree``s the partition directory.
        first_pass_cache_extras = {
            "object_articulation": list(builder["object_articulation"]),
            "nvhuman_joints": [list(f) for f in builder["nvhuman_joints"]],
            "nvhuman_joints_wxyz": [list(f) for f in builder["nvhuman_joints_wxyz"]],
            "nvhuman_head_wxyz": [list(f) for f in builder["nvhuman_head_wxyz"]],
            "nvhuman_root_wxyz": [list(f) for f in builder["nvhuman_root_wxyz"]],
            "ik_error_per_frame": list(builder["ik_error_per_frame"]),
            "ik_num_iterations": list(builder["ik_num_iterations"]),
            "frame_task_errors": [list(f) for f in builder["frame_task_errors"]],
        }

        # Build per-side sole XYZ from the cached ankle XYZ (ankle origin
        # sits ``G1_ANKLE_ROLL_OFFSET`` above the foot sole in the URDF).
        # The plane module is intentionally agnostic to ankle/sole semantics;
        # it just wants world-frame contact points.
        plane = ReferencePlane.horizontal(z=0.0)
        sole_xyz = first_pass.ankle_frame_xyz.copy()
        sole_xyz[..., 2] -= G1_ANKLE_ROLL_OFFSET
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

        # Apply the residual robot delta to every saved robot-side Z.
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

        # Replace object arrays with the corrected trajectory.
        builder["object_root_position"] = corr_obj_root_pos.tolist()
        builder["object_root_axis_angle"] = corr_obj_root_aa.tolist()
        builder["object_body_position"] = corr_obj_body_pos.tolist()
        builder["object_body_wxyz"] = corr_obj_body_wxyz.tolist()

        # Diagnostic report: confirms the plane-alignment post-process
        # actually ran and shows whether the offsets had a meaningful
        # effect. Pre/post sole stats come from the cached pass-1 ankle
        # XYZ (pre-offset) and the same values with the offset added
        # (post-offset), so any mismatch at runtime is obvious. The
        # post-offset lowest-sole Z should be near 0 because the module
        # drags the lowest foot per frame to the plane.
        n_interact = int(np.count_nonzero(interaction_mask))
        pre_ankle_z = first_pass.ankle_frame_xyz[:, :, 2]
        post_ankle_z = pre_ankle_z + robot_delta_z[:, None]
        pre_sole_lowest = float(pre_ankle_z.min() - G1_ANKLE_ROLL_OFFSET)
        pre_sole_highest = float(pre_ankle_z.max() - G1_ANKLE_ROLL_OFFSET)
        post_sole_lowest = float(post_ankle_z.min() - G1_ANKLE_ROLL_OFFSET)
        post_sole_highest = float(post_ankle_z.max() - G1_ANKLE_ROLL_OFFSET)
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
                "nvhuman_betas": betas,
                "nvhuman_joints": builder.pop("nvhuman_joints"),
                "nvhuman_joints_wxyz": builder.pop("nvhuman_joints_wxyz"),
                "nvhuman_head_translation": builder.pop("nvhuman_head_translation"),
                "nvhuman_head_wxyz": builder.pop("nvhuman_head_wxyz"),
                "nvhuman_root_translation": builder.pop("nvhuman_root_translation"),
                "nvhuman_root_wxyz": builder.pop("nvhuman_root_wxyz"),
            }
        )

        # Contact masks were accumulated in-loop (palm + fingertip proximity
        # to the object mesh surface). Reshape to (S, T) in hand_sides order.
        hand_contact_active: list[list[float]] = []
        if hand_sides:
            per_frame = builder.pop("hand_contact_active_per_frame")
            # per_frame[t][s] is 0.0/1.0; transpose to side-major (S, T).
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
            robot_name="g1",
            motion_kind="single_robot",
            source_dataset="nvhuman",
            raw_motion_file=motion_params_path,
            fps=float(kin.frequency),
            coord_frame="robot_base_z_up",
            # Robot state
            robot_joint_names=robot_joint_position_names,
            robot_root_position=builder["robot_root_position"],
            robot_root_wxyz=builder["robot_root_wxyz"],
            robot_joint_positions=builder["robot_joint_positions"],
            # EE
            ee_link_names=ee_link_names,
            ee_pose_w=builder["ee_pose_w"],
            # Hands (contact labels only; per-side finger joints / frames are
            # not produced by this retargeter)
            hand_sides=hand_sides,
            hand_contact_active=hand_contact_active,
            # Object
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
            # Diagnostics
            ik_error_per_frame=builder["ik_error_per_frame"],
            ik_num_iterations=builder["ik_num_iterations"],
            frame_task_errors=builder["frame_task_errors"],
            # Source raw
            source_kind="nvhuman",
            source_payload=source_payload,
        )
        save_motion_parquet(md, root_path=str(SAVE_DIR), file_name="data.parquet")
        print(f"Saved to {SAVE_DIR}")

        # Write the first-pass cache pickle to a SIBLING cache directory
        # (not inside the parquet partition) so downstream consumers that
        # scan the parquet dataset directory — e.g.
        # ``SceneConfig.from_motion_file`` in ``replay_motion.py`` uses
        # ``pq.read_table(partition_dir)`` which walks every file in the
        # partition — don't mis-detect the pickle as a parquet file.
        # ``first_pass`` is immutable through post-processing (the
        # correction copies arrays before mutating), and we snapshotted
        # the other builder fields earlier before they were popped.
        first_pass_cache_path = (
            FIRST_PASS_CACHE_DIR
            / f"sequence_id={sequence_id}"
            / "robot_name=g1"
            / "first_pass_cache.pkl"
        )
        first_pass_cache_path.parent.mkdir(parents=True, exist_ok=True)
        first_pass_cache = {
            "first_pass": first_pass,
            "builder_extras": first_pass_cache_extras,
            "metadata": {
                "sequence_id": sequence_id,
                "motion_params_path": motion_params_path,
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
                "betas": list(betas),
                "contact_threshold": float(contact_threshold),
                "g1_ankle_roll_offset": float(G1_ANKLE_ROLL_OFFSET),
            },
        }
        with open(first_pass_cache_path, "wb") as _f:
            pickle.dump(first_pass_cache, _f)
        print(f"[INFO] Cached first-pass snapshot -> {first_pass_cache_path}")

    print(f"Retargeting complete. Processed {num_frames} frames.")
    if args.visualize:
        print("Visualization server running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
