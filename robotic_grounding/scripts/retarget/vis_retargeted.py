# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import time
from typing import Any

import numpy as np
import trimesh
import viser
from arctic_to_sharpa import setup_sharpa_kinematics
from robotic_grounding.retarget import ASSETS_DIR, HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import ManoSharpaData
from robotic_grounding.retarget.distance_utils import MANO_FINGERTIP_INDICES
from robotic_grounding.retarget.hand_kinematics import HandKinematics

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def distance_to_color(d: float) -> tuple[int, int, int]:
    """Map distance to a green-to-red color gradient.

    Green (0, 255, 0) at d <= 0.01m (contact), red (255, 0, 0) at d >= 0.05m (far).
    """
    t = np.clip((d - 0.01) / (0.05 - 0.01), 0.0, 1.0)
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    return (r, g, 0)


ARCTIC_MOTION_DIR = HUMAN_MOTION_DATA_DIR / "arctic"
ARCTIC_URDF_DIR = ASSETS_DIR / "urdfs" / "arctic"
ARCTIC_MESH_DIR = ASSETS_DIR / "meshes" / "arctic"
SAVE_DIR = HUMAN_MOTION_DATA_DIR / "arctic_processed"


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-tid", "--trajectory-id", type=int, default=0)
    return parser.parse_args()


def visualize_one_trajectory(
    viser_server: viser.ViserServer,
    right_sharpa_kinematics: HandKinematics,
    left_sharpa_kinematics: HandKinematics,
    viser_object_handles: dict[str, Any],
    trajectory_id: int,
) -> dict[str, Any]:
    """Visualize one trajectory."""
    # Clear object handles in viser server
    for _, handle in viser_object_handles.items():
        handle.remove()
    viser_object_handles.clear()

    # Load logger data
    logger_data = ManoSharpaData.from_parquet(
        root_path=str(SAVE_DIR),
        filters=[
            ("robot_name", "=", "sharpa_wave"),
            ("sequence_id", "contains", "box_grab"),
        ],
        trajectory_id=trajectory_id,
    )
    H = len(logger_data.robot_right_wrist_position)

    # Setup viser object handles for visualization
    viser_object_handles["frame"] = viser_server.scene.add_frame(
        name="/object/frame",
        position=np.array([0, 0, 0]),
        wxyz=np.array([1, 0, 0, 0]),
        axes_length=0.2,
        axes_radius=0.007,
    )
    for part in logger_data.object_body_names:
        mesh = trimesh.load(
            str(
                ARCTIC_MESH_DIR
                / logger_data.object_name
                / f"{part}_watertight_tiny.obj"
            )
        )
        viser_object_handles[part] = viser_server.scene.add_mesh_trimesh(
            name=f"/object/{part}",
            mesh=mesh,
            position=np.array([0, 0, 0]),
            wxyz=np.array([1, 0, 0, 0]),
        )

    for frame_id in range(H):
        # Visualize right hand
        right_qpos = right_sharpa_kinematics.robot.q0.copy()
        right_qpos[:3] = np.array(logger_data.robot_right_wrist_position[frame_id])
        right_qpos[3:7] = np.array(logger_data.robot_right_wrist_wxyz[frame_id])[
            [1, 2, 3, 0]
        ]
        right_qpos[7:] = np.array(logger_data.robot_right_finger_joints[frame_id])
        right_sharpa_kinematics.visualize(viser_server, right_qpos)
        # Visualize left hand
        left_qpos = left_sharpa_kinematics.robot.q0.copy()
        left_qpos[:3] = np.array(logger_data.robot_left_wrist_position[frame_id])
        left_qpos[3:7] = np.array(logger_data.robot_left_wrist_wxyz[frame_id])[
            [1, 2, 3, 0]
        ]
        left_qpos[7:] = np.array(logger_data.robot_left_finger_joints[frame_id])
        left_sharpa_kinematics.visualize(viser_server, left_qpos)

        for object_body_idx, object_body_name in enumerate(
            logger_data.object_body_names
        ):
            viser_object_handles[object_body_name].position = np.asarray(
                logger_data.object_body_position[frame_id][object_body_idx]
            )
            viser_object_handles[object_body_name].wxyz = np.asarray(
                logger_data.object_body_wxyz[frame_id][object_body_idx]
            )

        # Visualize fingertip distance spheres (if distance data is available)
        for side, joints_data, dist_data in [
            (
                "right",
                logger_data.mano_right_joints,
                logger_data.mano_right_tips_distance,
            ),
            ("left", logger_data.mano_left_joints, logger_data.mano_left_tips_distance),
        ]:
            if not dist_data:
                continue
            fingertip_positions = np.array(joints_data[frame_id])[
                MANO_FINGERTIP_INDICES
            ]
            distances = dist_data[frame_id]
            for i, finger_name in enumerate(FINGER_NAMES):
                viser_server.scene.add_icosphere(
                    name=f"/tips/{side}_{finger_name}",
                    radius=0.005,
                    color=distance_to_color(distances[i]),
                    position=fingertip_positions[i],
                )

        time.sleep(1.0 / logger_data.fps)

    return viser_object_handles


def main(args: argparse.Namespace) -> None:
    """Main function."""
    # Setup viser server
    viser_server = viser.ViserServer()
    viser_object_handles: dict[str, Any] = {}

    # Setup IK solvers for visualization
    right_sharpa_kinematics = setup_sharpa_kinematics(
        side="right", frame_tasks_converged_threshold=1e-6
    )
    left_sharpa_kinematics = setup_sharpa_kinematics(
        side="left", frame_tasks_converged_threshold=1e-6
    )

    viser_object_handles = visualize_one_trajectory(
        viser_server,
        right_sharpa_kinematics,
        left_sharpa_kinematics,
        viser_object_handles,
        args.trajectory_id,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
