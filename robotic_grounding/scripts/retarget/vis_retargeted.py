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

import mujoco
import numpy as np
import trimesh
import viser
from arctic_to_sharpa import setup_sharpa_kinematics
from robotic_grounding.retarget import ASSETS_DIR, HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import ManoSharpaData
from robotic_grounding.retarget.hand_kinematics import HandKinematics
from scipy.spatial.transform import Rotation as R

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
        ],
        trajectory_id=trajectory_id,
    )
    H = len(logger_data.robot_right_qpos)

    # Setup mujoco object model and data for visualization
    object_urdf_path = ARCTIC_URDF_DIR / f"{logger_data.object_name}.urdf"
    mujoco_object_model = mujoco.MjModel.from_xml_path(str(object_urdf_path))
    mujoco_object_data = mujoco.MjData(mujoco_object_model)
    top_body_id = mujoco_object_model.body("top").id
    rotation_joint_id = mujoco_object_model.joint("rotation").id

    # Setup viser object handles for visualization
    for part in ["top", "bottom"]:
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
        right_sharpa_kinematics.visualize(
            viser_server,
            np.array(logger_data.robot_right_qpos[frame_id]),
            visualize_sites=True,
        )
        # Visualize left hand
        left_sharpa_kinematics.visualize(
            viser_server,
            np.array(logger_data.robot_left_qpos[frame_id]),
            visualize_sites=True,
        )

        # Compute top and bottom object poses and visualize
        world_p_bottom = np.array(logger_data.object_translation[frame_id])
        world_q_bottom = R.from_rotvec(logger_data.object_axis_angle[frame_id]).as_quat(
            scalar_first=True
        )
        world_t_bottom = np.eye(4)
        world_t_bottom[:3, :3] = R.from_quat(
            world_q_bottom, scalar_first=True
        ).as_matrix()
        world_t_bottom[:3, 3] = world_p_bottom
        viser_object_handles["bottom"].position = world_p_bottom
        viser_object_handles["bottom"].wxyz = world_q_bottom

        mujoco_object_data.qpos[rotation_joint_id] = np.array(
            logger_data.object_articulation[frame_id]
        )
        mujoco.mj_forward(mujoco_object_model, mujoco_object_data)
        bottom_t_top = np.eye(4)
        bottom_t_top[:3, :3] = np.array(mujoco_object_data.xmat[top_body_id]).reshape(
            3, 3
        )
        bottom_t_top[:3, 3] = mujoco_object_data.xpos[top_body_id]
        world_t_top = world_t_bottom @ bottom_t_top
        viser_object_handles["top"].position = world_t_top[:3, 3]
        viser_object_handles["top"].wxyz = R.from_matrix(world_t_top[:3, :3]).as_quat(
            scalar_first=True
        )

        time.sleep(1.0 / 30)

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
