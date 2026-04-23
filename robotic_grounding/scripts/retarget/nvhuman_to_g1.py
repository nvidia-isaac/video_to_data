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
from robotic_grounding.retarget.params import (
    G1_FOOT_FRAME_NAMES,
    NVHUMAN_FOOT_JOINT_NAMES,
    NVHUMAN_JOINTS_ORDER,
)
from robotic_grounding.retarget.read_nvhuman import NVHuman
from robotic_grounding.retarget.whole_body_kinematics import G1WholeBodyKinematics
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

G1_URDF = G1_URDF_DIR / "main_with_hand.urdf"
PACKAGE_DIRS = [str(G1_URDF_DIR)]
SAVE_DIR = HUMAN_MOTION_DATA_DIR / "whole_body" / "nvhuman"
# Repo root (the `robotic_grounding/` subdir). Used to store repo-relative
# asset paths in the parquet so training works from any checkout location.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _usd_safe(name: str) -> str:
    """Make a name safe for USD prim paths (no leading digits, no @ etc.)."""
    safe = name.replace("@", "_")
    if safe and (safe[0].isdigit() or not (safe[0].isalpha() or safe[0] == "_")):
        return f"obj_{safe}"
    return safe


def _relative_to_repo(path: Path) -> str:
    """Return ``path`` as a string relative to the repo root if possible.

    Falls back to the absolute path when ``path`` lives outside ``REPO_ROOT``
    (e.g. when ``HUMAN_MOTION_DATA_DIR`` was overridden to an external
    location via the environment).
    """
    path = path.resolve()
    if path.is_relative_to(REPO_ROOT):
        return str(path.relative_to(REPO_ROOT))
    return str(path)


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

    # Collected time-series for the motion_v1 parquet. These lists grow one
    # entry per frame and are materialized into a `MotionData` at save time.
    builder: dict[str, list] | None = None
    object_body_names: list[str] = ["object"]
    safe_object_body_names: list[str] = [_usd_safe(n) for n in object_body_names]
    object_mesh_radius: float = 0.0
    object_urdf_path: str = ""
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

    if args.save:
        params = torch.load(motion_params_path)
        betas_tensor = params["betas"].cpu().numpy()
        if betas_tensor.ndim > 1:
            betas = betas_tensor[0].flatten().tolist()
        else:
            betas = betas_tensor.flatten().tolist()

        dst_object_dir = SAVE_DIR / f"sequence_id={sequence_id}" / "object"
        _copy_mesh_assets(data_folder / "object", dst_object_dir)

        dst_mesh = dst_object_dir / "textured_mesh.obj"
        dst_urdf = dst_object_dir / "textured_mesh.urdf"
        _build_object_urdf(mesh_path="textured_mesh.obj", urdf_path=dst_urdf)

        object_mesh_path = _relative_to_repo(dst_mesh)
        object_urdf_path = _relative_to_repo(dst_urdf)
        object_mesh_radius = _compute_mesh_radius(str(dst_mesh))

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

    # Visualization frames
    object_frame = None
    head_frame = None
    root_frame = None

    # Setup visualization
    server = None
    if args.visualize:
        server = viser.ViserServer(host="0.0.0.0", port=8080)

        mesh = trimesh.load(object_mesh_path)
        object_frame = server.scene.add_frame(
            "/object",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
            axes_length=0.018,
            axes_radius=0.0008,
        )
        server.scene.add_mesh_trimesh(
            "/object/mesh", mesh, position=(0, 0, 0), wxyz=(1, 0, 0, 0)
        )

        head_frame = server.scene.add_frame(
            "/head",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
            axes_length=0.1,
            axes_radius=0.002,
        )

        root_frame = server.scene.add_frame(
            "/root",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
            axes_length=0.1,
            axes_radius=0.002,
        )

    # Initialize qpos
    q = kin.robot.q0.copy()

    # Target visualization handles (created on first frame)
    target_handles: dict[str, viser.FrameHandle] = {}

    # Ground-plane Z offset: the NVHuman normalization centers the pelvis at
    # the origin, so foot-level (ground) is at a negative Z in robot coords.
    # We compute it from source foot joints on frame 0 and shift all
    # positions so ground = Z=0.
    ground_z_offset = 0.0
    object_z_lift = 0.0

    # Source foot joints used for robust ground estimation. Using only feet
    # avoids noisy non-foot joints pulling the ground estimate down.
    source_foot_joint_idxs = [
        NVHUMAN_JOINTS_ORDER.index(name)
        for name in NVHUMAN_FOOT_JOINT_NAMES
        if name in NVHUMAN_JOINTS_ORDER
    ]

    # Frame-0 foot Z reference (in IK frame) used to detect and correct
    # downward drift from noisy source reconstruction.
    foot_frame_idxs = [
        list(kin.robot_frame_names.values()).index(fn) for fn in G1_FOOT_FRAME_NAMES
    ]
    foot_z_ref: float | None = None

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

        # Foot-Z ground clamping: if the lowest ankle frame drifts below
        # its frame-0 level (reconstruction noise), lift the free-flyer
        # root so the feet stay on the ground.
        lowest_foot_z = min(result["frame_pose"][i, 2] for i in foot_frame_idxs)
        if foot_z_ref is None:
            foot_z_ref = lowest_foot_z
        else:
            drift = lowest_foot_z - foot_z_ref
            if drift < 0:
                q[2] -= drift

        # Compute object pose in robot coordinate system. Object rotation is
        # a world-frame transform (not a body-local joint rotation), so we use
        # transform_world_rotation (left-multiply by R_src_to_robot) rather
        # than transform_source_rotation (similarity transform R @ M @ R^T
        # used for body-local joint rotations).
        obj_pose = object_poses[frame_idx]
        obj_position = kin.transform_source_position(obj_pose[:3, 3])
        obj_rotation_mat = kin.transform_world_rotation(obj_pose[:3, :3])
        obj_rotation = R.from_matrix(obj_rotation_mat)

        # On frame 0, estimate ground from source feet only and shift into
        # a ground-relative frame.
        if frame_idx == 0:
            foot_joint_z = np.array(
                [
                    kin.transform_source_position(positions[i])[2]
                    for i in source_foot_joint_idxs
                ]
            )
            ground_z_offset = -float(np.min(foot_joint_z))

        obj_position[2] += ground_z_offset

        # Object-specific lift: if the mesh still starts below ground after
        # foot-based alignment (e.g. mesh origin/orientation mismatch), lift
        # object trajectories only so the lowest mesh point is at Z=0.
        # Applied only to saved data, not visualization (where it would float
        # the object above the aligned skeleton/robot).
        if frame_idx == 0:
            obj_pos_for_mesh_check = obj_position.copy()
            obj_verts_world = (
                obj_rotation_mat @ object_mesh_vertices.T
            ).T + obj_pos_for_mesh_check
            object_z_lift = max(0.0, -float(np.min(obj_verts_world[:, 2])))

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

        # Log timestep data
        if builder is not None:
            # Pinocchio free-flyer: q[:3] = position, q[3:7] = quaternion (xyzw)
            root_pos_robot = q[:3].copy()
            root_pos_robot[2] += ground_z_offset
            root_pos_robot = root_pos_robot.tolist()
            root_quat_xyzw = q[3:7]
            root_wxyz = [
                float(root_quat_xyzw[3]),
                float(root_quat_xyzw[0]),
                float(root_quat_xyzw[1]),
                float(root_quat_xyzw[2]),
            ]
            joint_positions = q[base_q_size:].tolist()

            obj_wxyz = obj_rotation.as_quat(scalar_first=True).tolist()
            obj_body_pos = [(obj_position + [0, 0, object_z_lift]).tolist()]
            obj_body_wxyz = [obj_wxyz]

            # Build per-frame EE pose from the selected frame indices.
            frame_pose = result["frame_pose"]  # (K, 7), wxyz order
            ee_pose_t = [list(frame_pose[i]) for i in ee_frame_indices]

            builder["robot_root_position"].append(root_pos_robot)
            builder["robot_root_wxyz"].append(root_wxyz)
            builder["robot_joint_positions"].append(joint_positions)
            builder["ee_pose_w"].append(ee_pose_t)
            builder["object_articulation"].append(0.0)
            builder["object_root_position"].append(
                (obj_position + [0, 0, object_z_lift]).tolist()
            )
            builder["object_root_axis_angle"].append(obj_rotation.as_rotvec().tolist())
            builder["object_body_position"].append(obj_body_pos)
            builder["object_body_wxyz"].append(obj_body_wxyz)
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

        # Update visualization
        if server is not None:
            vertices_vis = kin.transform_source_position(vertices[frame_idx]).copy()
            vertices_vis[:, 2] += ground_z_offset
            nvhuman.visualize(server, vertices=vertices_vis)
            # Visualize pre-clamp IK pose so source skeleton overlays remain
            # aligned with robot in debug view.
            q_vis = q_ik.copy()
            q_vis[2] += ground_z_offset
            kin.visualize(server, q_vis)

            # Visualize IK targets
            for frame_name, task in kin.frame_tasks.items():
                target_pos = task.transform_target_to_world.translation
                target_pos_vis = target_pos.copy()
                target_pos_vis[2] += ground_z_offset
                target_rot = task.transform_target_to_world.rotation
                target_wxyz = R.from_matrix(target_rot).as_quat(scalar_first=True)
                if frame_name not in target_handles:
                    target_handles[frame_name] = server.scene.add_frame(
                        f"/targets/{frame_name}",
                        position=target_pos_vis,
                        wxyz=target_wxyz,
                        axes_length=0.05,
                        axes_radius=0.003,
                    )
                else:
                    target_handles[frame_name].position = target_pos_vis
                    target_handles[frame_name].wxyz = target_wxyz

            if object_frame is not None:
                object_frame.position = obj_position
                object_frame.wxyz = obj_rotation.as_quat(scalar_first=True)

            if head_frame is not None:
                head_frame.position = head_position
                head_frame.wxyz = head_rotation_wxyz

            if root_frame is not None:
                root_frame.position = root_position
                root_frame.wxyz = root_rotation_wxyz

        if frame_idx == 0 and args.visualize:
            time.sleep(5)

    # Save data
    if builder is not None:
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
        md = MotionData(
            sequence_id=sequence_id,
            robot_name="g1",
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
            # Object
            object_name=object_name,
            safe_object_name=_usd_safe(object_name),
            object_body_names=object_body_names,
            safe_object_body_names=safe_object_body_names,
            object_mesh_paths=[object_mesh_path],
            object_urdf_paths=[object_urdf_path],
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
        save_motion_parquet(md, root_path=str(SAVE_DIR))
        print(f"Saved to {SAVE_DIR}")

    print(f"Retargeting complete. Processed {num_frames} frames.")
    if args.visualize:
        print("Visualization server running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
