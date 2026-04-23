# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Script to retarget NVHuman motion to Dex3 hands using Pink IK solver.

Usage:
    python scripts/retarget/nvhuman_to_dex3.py <data_folder> --visualize

Where data_folder contains:
    - nova_params_opt.pt (motion parameters)
    - poses.npy (object trajectory as 4x4 transforms)
    - object/textured_mesh.obj (object mesh)
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser
from robotic_grounding.motion_schema import MotionData, save_motion_parquet
from robotic_grounding.retarget import G1_URDF_DIR, HUMAN_MOTION_DATA_DIR, MESHES_DIR
from robotic_grounding.retarget.hand_kinematics import Dex3HandKinematics
from robotic_grounding.retarget.params import (
    NVHUMAN_JOINTS_ORDER,
    R_PALM_CORRECTION_LEFT,
    R_PALM_CORRECTION_RIGHT,
)
from robotic_grounding.retarget.read_nvhuman import NVHuman
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

LEFT_URDF = G1_URDF_DIR / "dex3_left.urdf"
RIGHT_URDF = G1_URDF_DIR / "dex3_right.urdf"
PACKAGE_DIRS = [str(MESHES_DIR)]
SAVE_DIR = HUMAN_MOTION_DATA_DIR / "nvhuman_dex3_processed"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Retarget NVHuman motion to Dex3 hands"
    )
    parser.add_argument(
        "data_folder",
        type=str,
        help="Path to folder containing nova_params_opt.pt, object.urdf, and poses.npy",
    )
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--save", action="store_true", help="Save retargeted data")
    return parser.parse_args()


def load_data(folder_path: str) -> tuple[str, str, np.ndarray]:
    """Load motion params and object data from folder.

    Args:
        folder_path: Path to folder containing nova_params_opt.pt, object mesh, poses.npy

    Returns:
        Tuple of (motion_params_path, mesh_path, poses)
        poses are normalized to match NVHuman's origin normalization
    """
    folder = Path(folder_path)
    motion_params_path = str(folder / "nova_params_opt.pt")
    mesh_path = str(folder / "object" / "textured_mesh.obj")
    object_poses = np.load(folder / "poses.npy")

    # Apply same normalization as NVHuman (normalize to origin based on first frame)
    params = torch.load(motion_params_path)
    global_orient_first = params["global_orient"][0].cpu().numpy()
    transl_first = params["transl"][0].cpu().numpy()

    # Compute inverse rotation
    R_first = R.from_rotvec(global_orient_first).as_matrix()
    R_first_inv = R_first.T

    # Build normalization transform to match NVHuman's origin normalization
    norm_transform = np.eye(4)
    norm_transform[:3, :3] = R_first_inv
    norm_transform[:3, 3] = -R_first_inv @ transl_first

    # Apply normalization to all object poses
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

    # Initialize hand kinematics
    left_kin = Dex3HandKinematics(
        side="left",
        robot_asset_path=str(LEFT_URDF),
        package_dirs=PACKAGE_DIRS,
        source_model="nvskel",
    )
    right_kin = Dex3HandKinematics(
        side="right",
        robot_asset_path=str(RIGHT_URDF),
        package_dirs=PACKAGE_DIRS,
        source_model="nvskel",
    )

    # Load motion data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nvhuman = NVHuman(device=device)
    motion = nvhuman.load_motion(params_path=motion_params_path)
    joint_pos = motion["joints"]
    joint_rot_wxyz = motion["joints_wxyz"]
    vertices = motion["vertices"]
    num_frames = motion["num_frames"]

    # Get sequence ID from folder name
    sequence_id = data_folder.name

    # Visualization frames (set up later if visualizing)
    object_frame = None
    head_frame = None
    root_frame = None

    # Setup visualization
    server = None
    if args.visualize:
        server = viser.ViserServer(host="0.0.0.0", port=8080)

        # Setup object visualization
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

        # Setup head visualization
        head_frame = server.scene.add_frame(
            "/head",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
            axes_length=0.1,
            axes_radius=0.002,
        )

        # Setup root visualization
        root_frame = server.scene.add_frame(
            "/root",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
            axes_length=0.1,
            axes_radius=0.002,
        )

    # Setup data logger (optional)
    builder: dict[str, list] | None = None
    betas: list[float] = []
    right_finger_joint_names: list[str] = []
    left_finger_joint_names: list[str] = []
    right_frame_names: list[str] = []
    left_frame_names: list[str] = []
    if args.save:
        # Load betas from motion params (use first frame if per-frame, or just the betas)
        params = torch.load(motion_params_path)
        betas_tensor = params["betas"].cpu().numpy()
        if betas_tensor.ndim > 1:
            betas = betas_tensor[0].flatten().tolist()  # Take first frame
        else:
            betas = betas_tensor.flatten().tolist()

        right_finger_joint_names = [
            str(right_kin.robot.model.names[i])
            for i in range(1, right_kin.robot.model.njoints)
        ]
        left_finger_joint_names = [
            str(left_kin.robot.model.names[i])
            for i in range(1, left_kin.robot.model.njoints)
        ]
        right_frame_names = [
            str(right_kin.robot.model.frames[i].name)
            for i in range(len(right_kin.robot.model.frames))
        ]
        left_frame_names = [
            str(left_kin.robot.model.frames[i].name)
            for i in range(len(left_kin.robot.model.frames))
        ]
        builder = {
            # EE poses: [left, right] per frame, each [x, y, z, qw, qx, qy, qz].
            "ee_pose_w": [],
            # Per-side hand series.
            "left_frames": [],
            "right_frames": [],
            "left_finger_joints": [],
            "right_finger_joints": [],
            # Object
            "object_articulation": [],
            "object_root_axis_angle": [],
            "object_root_position": [],
            "object_body_position": [],
            "object_body_wxyz": [],
            # Diagnostics (combined both-hand IK error)
            "frame_task_errors": [],
            # Source raw
            "nvhuman_joints": [],
            "nvhuman_joints_wxyz": [],
            "nvhuman_head_translation": [],
            "nvhuman_head_wxyz": [],
            "nvhuman_root_translation": [],
            "nvhuman_root_wxyz": [],
        }

    # Get joint indices
    left_hand_idx = NVHUMAN_JOINTS_ORDER.index("LeftHand")
    right_hand_idx = NVHUMAN_JOINTS_ORDER.index("RightHand")
    head_idx = NVHUMAN_JOINTS_ORDER.index("Head")
    root_idx = NVHUMAN_JOINTS_ORDER.index("Hips")

    # Initialize qpos with first frame hand positions
    left_palm_correction = np.array(R_PALM_CORRECTION_LEFT, dtype=np.float64)
    right_palm_correction = np.array(R_PALM_CORRECTION_RIGHT, dtype=np.float64)

    left_q = left_kin.robot.q0.copy()
    left_q[:3] = left_kin.transform_source_position(joint_pos[0][left_hand_idx])
    left_rot = R.from_quat(
        joint_rot_wxyz[0][left_hand_idx], scalar_first=True
    ).as_matrix()
    left_rot_corrected = (
        left_kin.transform_source_rotation(left_rot) @ left_palm_correction
    )
    left_q[3:6] = R.from_matrix(left_rot_corrected).as_euler("XYZ")

    right_q = right_kin.robot.q0.copy()
    right_q[:3] = right_kin.transform_source_position(joint_pos[0][right_hand_idx])
    right_rot = R.from_quat(
        joint_rot_wxyz[0][right_hand_idx], scalar_first=True
    ).as_matrix()
    right_rot_corrected = (
        right_kin.transform_source_rotation(right_rot) @ right_palm_correction
    )
    right_q[3:6] = R.from_matrix(right_rot_corrected).as_euler("XYZ")

    # Processing loop
    for frame_idx in tqdm(range(num_frames), desc="Retargeting"):
        # Get current frame data
        positions = joint_pos[frame_idx]
        rotations = joint_rot_wxyz[frame_idx]

        # Solve IK
        left_result = left_kin.compute(
            source_joints=positions,
            source_joints_wxyz=rotations,
            qpos=left_q,
        )
        left_q = left_result["q"]

        right_result = right_kin.compute(
            source_joints=positions,
            source_joints_wxyz=rotations,
            qpos=right_q,
        )
        right_q = right_result["q"]

        # Compute object pose in robot coordinate system
        obj_pose = object_poses[frame_idx]
        obj_position = left_kin.transform_source_position(obj_pose[:3, 3])
        R_coord = left_kin._R_nvhuman_to_robot
        obj_rotation_mat = R_coord @ obj_pose[:3, :3]
        obj_rotation = R.from_matrix(obj_rotation_mat)

        # Extract head trajectory in robot coordinate system
        head_position = left_kin.transform_source_position(positions[head_idx])
        head_rotation_mat = R.from_quat(
            rotations[head_idx], scalar_first=True
        ).as_matrix()
        head_rotation_mat = left_kin.transform_source_rotation(head_rotation_mat)
        head_rotation_wxyz = R.from_matrix(head_rotation_mat).as_quat(scalar_first=True)

        # Extract root trajectory in robot coordinate system
        root_position = left_kin.transform_source_position(positions[root_idx])
        root_rotation_mat = R.from_quat(
            rotations[root_idx], scalar_first=True
        ).as_matrix()
        root_rotation_mat = left_kin.transform_source_rotation(root_rotation_mat)
        root_rotation = R.from_matrix(root_rotation_mat)
        root_rotation_wxyz = root_rotation.as_quat(scalar_first=True)

        # Log timestep data
        if builder is not None:
            # Convert XYZ Euler wrist rotation to wxyz for the unified layout.
            left_wxyz = (
                R.from_euler("XYZ", left_q[3:6]).as_quat(scalar_first=True).tolist()
            )
            right_wxyz = (
                R.from_euler("XYZ", right_q[3:6]).as_quat(scalar_first=True).tolist()
            )
            left_ee = list(left_q[:3]) + list(left_wxyz)
            right_ee = list(right_q[:3]) + list(right_wxyz)
            obj_wxyz = obj_rotation.as_quat(scalar_first=True).tolist()

            builder["ee_pose_w"].append([left_ee, right_ee])
            builder["left_frames"].append(left_result["frame_pose"].tolist())
            builder["right_frames"].append(right_result["frame_pose"].tolist())
            builder["left_finger_joints"].append(left_q[6:].tolist())
            builder["right_finger_joints"].append(right_q[6:].tolist())
            builder["object_articulation"].append(0.0)
            builder["object_root_position"].append(obj_position.tolist())
            builder["object_root_axis_angle"].append(obj_rotation.as_rotvec().tolist())
            builder["object_body_position"].append([obj_position.tolist()])
            builder["object_body_wxyz"].append([obj_wxyz])
            builder["frame_task_errors"].append(
                [
                    float(np.sum(left_result["frame_task_errors"])),
                    float(np.sum(right_result["frame_task_errors"])),
                ]
            )
            builder["nvhuman_joints"].append(positions.tolist())
            builder["nvhuman_joints_wxyz"].append(rotations.tolist())
            builder["nvhuman_head_translation"].append(head_position.tolist())
            builder["nvhuman_head_wxyz"].append(head_rotation_wxyz.tolist())
            builder["nvhuman_root_translation"].append(root_position.tolist())
            builder["nvhuman_root_wxyz"].append(root_rotation_wxyz.tolist())

        # Update visualization
        if server is not None:
            vertices_vis = left_kin.transform_source_position(vertices[frame_idx])
            nvhuman.visualize(server, vertices=vertices_vis)
            left_kin.visualize(server, left_q)
            right_kin.visualize(server, right_q)

            # Update object pose
            if object_frame is not None:
                object_frame.position = obj_position
                object_frame.wxyz = obj_rotation.as_quat(scalar_first=True)

            # Update head pose
            if head_frame is not None:
                head_frame.position = head_position
                head_frame.wxyz = head_rotation_wxyz

            # Update root pose
            if root_frame is not None:
                root_frame.position = root_position
                root_frame.wxyz = root_rotation.as_quat(scalar_first=True)

        if frame_idx == 0 and args.visualize:
            time.sleep(5)

    # Save data
    if builder is not None:
        T = len(builder["ee_pose_w"])
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
        # Dex3 is a pair of floating hands; whole-body joint state is empty.
        md = MotionData(
            sequence_id=sequence_id,
            robot_name="dex3",
            source_dataset="nvhuman",
            raw_motion_file=motion_params_path,
            fps=float(left_kin.frequency),
            coord_frame="robot_base_z_up",
            robot_joint_names=[],
            robot_root_position=[[0.0, 0.0, 0.0] for _ in range(T)],
            robot_root_wxyz=[[1.0, 0.0, 0.0, 0.0] for _ in range(T)],
            robot_joint_positions=[[] for _ in range(T)],
            ee_link_names=["left_wrist_link", "right_wrist_link"],
            ee_pose_w=builder["ee_pose_w"],
            object_name=f"{sequence_id}_object",
            safe_object_name=f"{sequence_id}_object",
            object_body_names=["object"],
            safe_object_body_names=["object"],
            object_mesh_paths=[],
            object_urdf_paths=[],
            object_articulation=builder["object_articulation"],
            object_root_axis_angle=builder["object_root_axis_angle"],
            object_root_position=builder["object_root_position"],
            object_body_position=builder["object_body_position"],
            object_body_wxyz=builder["object_body_wxyz"],
            hand_sides=["left", "right"],
            hand_frame_names=[left_frame_names, right_frame_names],
            hand_frames_w=[builder["left_frames"], builder["right_frames"]],
            hand_finger_joint_names=[left_finger_joint_names, right_finger_joint_names],
            hand_finger_joints=[
                builder["left_finger_joints"],
                builder["right_finger_joints"],
            ],
            frame_task_errors=builder["frame_task_errors"],
            source_kind="nvhuman",
            source_payload=source_payload,
        )
        save_motion_parquet(md, root_path=str(SAVE_DIR))
        print(f"Saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
