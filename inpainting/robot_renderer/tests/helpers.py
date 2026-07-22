from __future__ import annotations

from pathlib import Path

import numpy as np


LEFT_JOINTS = (
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
)
RIGHT_JOINTS = tuple(name.replace("left_", "right_", 1) for name in LEFT_JOINTS)
ARM_JOINTS = tuple(
    [*(f"L_arm_j{index}" for index in range(1, 8)), *(f"R_arm_j{index}" for index in range(1, 8))]
)


def trajectory_arrays(
    frame_count: int = 3,
    *,
    coordinate_frame: str = "world",
    z: float = 2.0,
) -> dict[str, np.ndarray]:
    identity_quaternion = np.tile(
        np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32), (frame_count, 1)
    )
    left_position = np.tile(np.array((-0.1, 0.0, z), dtype=np.float32), (frame_count, 1))
    right_position = np.tile(np.array((0.1, 0.0, z), dtype=np.float32), (frame_count, 1))
    return {
        "schema_version": np.asarray("v2d.inpainting.robot-trajectory/v1"),
        "coordinate_frame": np.asarray(coordinate_frame),
        "robot": np.asarray("dexmate_vega"),
        "gripper": np.asarray("sharpa_wave"),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
        "left_valid": np.ones(frame_count, dtype=bool),
        "right_valid": np.ones(frame_count, dtype=bool),
        "left_wrist_position": left_position,
        "right_wrist_position": right_position,
        "left_wrist_wxyz": identity_quaternion.copy(),
        "right_wrist_wxyz": identity_quaternion.copy(),
        "left_finger_joints": np.zeros((frame_count, len(LEFT_JOINTS)), dtype=np.float32),
        "right_finger_joints": np.zeros((frame_count, len(RIGHT_JOINTS)), dtype=np.float32),
        "left_finger_joint_names": np.asarray(LEFT_JOINTS),
        "right_finger_joint_names": np.asarray(RIGHT_JOINTS),
    }


def save_inputs(
    root: Path,
    *,
    frame_count: int = 3,
    coordinate_frame: str = "world",
    z: float = 2.0,
) -> tuple[Path, Path, Path]:
    trajectory = root / "robot_trajectory.npz"
    intrinsic = root / "intrinsic.txt"
    world_to_camera = root / "world_to_camera.npy"
    np.savez(trajectory, **trajectory_arrays(frame_count, coordinate_frame=coordinate_frame, z=z))
    np.savetxt(intrinsic, np.array(((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0, 0, 1))))
    np.save(world_to_camera, np.tile(np.eye(4), (frame_count, 1, 1)))
    return trajectory, intrinsic, world_to_camera


def _urdf(name: str, links: tuple[str, ...], joints: tuple[str, ...], mesh_ref: str) -> str:
    link_xml = []
    for index, link in enumerate(links):
        visual = (
            f'<visual><geometry><mesh filename="{mesh_ref}"/></geometry></visual>'
            if index == 0
            else ""
        )
        link_xml.append(f'<link name="{link}">{visual}</link>')
    joint_xml = []
    for index, joint in enumerate(joints):
        parent = links[min(index, len(links) - 1)]
        child = links[min(index + 1, len(links) - 1)]
        joint_xml.append(
            f'<joint name="{joint}" type="revolute"><parent link="{parent}"/>'
            f'<child link="{child}"/><axis xyz="0 0 1"/>'
            '<limit lower="-1" upper="2" effort="1" velocity="1"/></joint>'
        )
    return f'<robot name="{name}">{"".join(link_xml)}{"".join(joint_xml)}</robot>'


def create_synthetic_asset_tree(root: Path) -> Path:
    arms_dir = root / "urdfs" / "vega_sharpa"
    hands_dir = root / "urdfs" / "sharpawave"
    mesh_dir = root / "meshes"
    arms_dir.mkdir(parents=True)
    hands_dir.mkdir(parents=True)
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "dummy.stl").write_text("solid dummy\nendsolid dummy\n")
    arms_links = tuple(["arm_center", "L_arm_l8", "R_arm_l8", *(f"arm_extra_{i}" for i in range(14))])
    hand_links = tuple(["C_MC", *(f"finger_link_{i}" for i in range(22))])
    (arms_dir / "vega_arms_only.urdf").write_text(
        _urdf("arms", arms_links, ARM_JOINTS, "../../meshes/dummy.stl")
    )
    (hands_dir / "left_sharpa_wave.urdf").write_text(
        _urdf("left", hand_links, LEFT_JOINTS, "../../meshes/dummy.stl")
    )
    (hands_dir / "right_sharpa_wave.urdf").write_text(
        _urdf("right", hand_links, RIGHT_JOINTS, "../../meshes/dummy.stl")
    )
    return root


def create_fake_scene_utils(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "arm_ik.py").write_text("class ArmIK:\n    pass\n")
    (root / "arm_mount_opt.py").write_text(
        "from .arm_ik import ArmIK\n"
        "from .arm_replay import build_hand_mount_inverses\n"
        "def place_hub_from_wrists(*args, **kwargs):\n"
        "    return ((0,0,0), (1,0,0,0), (0,0,0))\n"
    )
    return root
