from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inpainting.parallel_jaw_renderer.bundle import BUNDLE_SCHEMA
from inpainting.parallel_jaw_renderer.inputs import TARGET_SCHEMA


def target_arrays(frame_count: int = 3) -> dict[str, np.ndarray]:
    left_position = np.tile(((-0.1, 0.0, 2.0),), (frame_count, 1)).astype(np.float32)
    right_position = np.tile(((0.1, 0.0, 2.0),), (frame_count, 1)).astype(np.float32)
    quaternion = np.tile(((1.0, 0.0, 0.0, 0.0),), (frame_count, 1)).astype(np.float32)
    return {
        "schema_version": np.asarray(TARGET_SCHEMA),
        "tracker": np.asarray("synthetic"),
        "coordinate_frame": np.asarray("world"),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
        "left_valid": np.ones(frame_count, dtype=np.bool_),
        "right_valid": np.ones(frame_count, dtype=np.bool_),
        "left_position": left_position,
        "right_position": right_position,
        "left_wxyz": quaternion.copy(),
        "right_wxyz": quaternion.copy(),
        "left_aperture_m": np.linspace(0.0, 0.08, frame_count).astype(np.float32),
        "right_aperture_m": np.linspace(0.08, 0.0, frame_count).astype(np.float32),
    }


def save_inputs(root: Path, frame_count: int = 3) -> tuple[Path, Path, Path]:
    target = root / "target.npz"
    intrinsic = root / "intrinsic.txt"
    world_to_camera = root / "world_to_camera.npy"
    np.savez(target, **target_arrays(frame_count))
    np.savetxt(
        intrinsic,
        np.asarray(((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))),
    )
    np.save(world_to_camera, np.tile(np.eye(4), (frame_count, 1, 1)))
    return target, intrinsic, world_to_camera


def _joint(
    name: str,
    parent: str,
    child: str,
    *,
    joint_type: str = "revolute",
    mimic: str | None = None,
    lower: float = -2.0,
    upper: float = 2.0,
) -> str:
    mimic_xml = f'<mimic joint="{mimic}" multiplier="1"/>' if mimic else ""
    return (
        f'<joint name="{name}" type="{joint_type}">'
        f'<parent link="{parent}"/><child link="{child}"/>'
        '<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>'
        f'<limit lower="{lower}" upper="{upper}" effort="1" velocity="1"/>'
        f"{mimic_xml}</joint>"
    )


def make_bundle(
    root: Path,
    *,
    mapping_kind: str = "galbot_four_bar",
) -> Path:
    render_urdf = root / "render.urdf"
    ik_urdf = root / "ik.urdf"
    render_links = (
        "root",
        "body",
        "left_tcp",
        "right_tcp",
        "left_finger",
        "left_mimic_link",
        "right_finger",
        "right_mimic_link",
    )
    render_joints = (
        _joint("root_posture", "root", "body"),
        _joint("left_arm", "body", "left_tcp"),
        _joint("right_arm", "body", "right_tcp"),
        _joint(
            "left_gripper",
            "left_tcp",
            "left_finger",
            lower=0.0 if mapping_kind == "galbot_four_bar" else -0.1,
            upper=2.0 if mapping_kind == "galbot_four_bar" else 0.1,
            joint_type=(
                "revolute" if mapping_kind == "galbot_four_bar" else "prismatic"
            ),
        ),
        _joint(
            "left_mimic",
            "left_tcp",
            "left_mimic_link",
            mimic="left_gripper",
        ),
        _joint(
            "right_gripper",
            "right_tcp",
            "right_finger",
            lower=0.0 if mapping_kind == "galbot_four_bar" else -0.1,
            upper=2.0 if mapping_kind == "galbot_four_bar" else 0.1,
            joint_type=(
                "revolute" if mapping_kind == "galbot_four_bar" else "prismatic"
            ),
        ),
        _joint(
            "right_mimic",
            "right_tcp",
            "right_mimic_link",
            mimic="right_gripper",
        ),
    )
    render_urdf.write_text(
        '<robot name="render">'
        + "".join(f'<link name="{name}"/>' for name in render_links)
        + "".join(render_joints)
        + "</robot>"
    )
    ik_urdf.write_text(
        '<robot name="ik">'
        '<link name="root"/><link name="left_tcp"/><link name="right_tcp"/>'
        + _joint("left_arm", "root", "left_tcp")
        + _joint("right_arm", "root", "right_tcp")
        + "</robot>"
    )
    if mapping_kind == "galbot_four_bar":
        mapping = {
            "kind": mapping_kind,
            "joint_names": {
                "left": ["left_gripper"],
                "right": ["right_gripper"],
            },
            "params": {
                "inner_pivot_half_gap_m": 0.026,
                "pad_inset_m": 0.0062,
                "finger_link_length_m": 0.045,
                "knuckle_angle_rad": 1.2465,
                "joint_lower_rad": 0.0,
                "joint_upper_rad": 1.703,
            },
        }
    else:
        # Use the mimic-independent synthetic URDF only for bundle schema tests;
        # mapping-specific two-independent-joint tests build a spec directly.
        mapping = {
            "kind": mapping_kind,
            "joint_names": {
                "left": ["left_gripper"],
                "right": ["right_gripper"],
            },
            "params": {},
        }
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "robot_id": "synthetic",
        "render_urdf": render_urdf.name,
        "ik_urdf": ik_urdf.name,
        "tcp_frames": {"left": "left_tcp", "right": "right_tcp"},
        "T_robot_root_hub": [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "semantic_target_to_tcp_rotation": np.eye(3).tolist(),
        "arm_joint_names": ["left_arm", "right_arm"],
        "gripper_mapping": mapping,
        "fixed_root_posture": {
            "joint_values": {"root_posture": 0.25},
            "provenance": {"source": "synthetic test"},
        },
        "asset_provenance": {"source": "synthetic test"},
    }
    path = root / "bundle.json"
    path.write_text(json.dumps(payload))
    return path
