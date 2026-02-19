# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import logging
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import torch
import trimesh
import viser
from robotic_grounding.retarget import (
    ASSETS_DIR,
    HUMAN_MOTION_DATA_DIR,
    SHARPA_WAVE_XMLS_DIR,
)
from robotic_grounding.retarget.data_logger import ManoSharpaData
from robotic_grounding.retarget.distance_utils import compute_tips_distance
from robotic_grounding.retarget.hand_kinematics import HandKinematics
from robotic_grounding.retarget.read_mano import MANO
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Suppress warnings about joint limits being slightly out of bounds
logging.getLogger().setLevel(logging.ERROR)

ARCTIC_MOTION_DIR = HUMAN_MOTION_DATA_DIR / "arctic"
ARCTIC_URDF_DIR = ASSETS_DIR / "urdfs" / "arctic"
ARCTIC_MESH_DIR = ASSETS_DIR / "meshes" / "arctic"
SAVE_DIR = HUMAN_MOTION_DATA_DIR / "arctic_processed"


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--mano_to_robot_scale", type=float, default=1.2)
    return parser.parse_args()


def setup_sharpa_kinematics(
    side: Literal["right", "left"],
    frequency: float = 200.0,
    frame_tasks_converged_threshold: float = 1e-6,
) -> HandKinematics:
    """Setup the Sharpa hand kinematics."""
    robot_xml = SHARPA_WAVE_XMLS_DIR / f"{side}_sharpawave.xml"
    return HandKinematics(
        side=side,
        robot_xml=str(robot_xml),
        frequency=frequency,
        frame_tasks_converged_threshold=frame_tasks_converged_threshold,
    )


def main(args: argparse.Namespace) -> None:
    """Main function."""
    device = torch.device(args.device)

    if args.visualize:
        viser_server = viser.ViserServer()

    if args.save:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    #############################################################
    # IK solver
    #############################################################

    right_sharpa_kinematics = setup_sharpa_kinematics(
        side="right", frame_tasks_converged_threshold=1e-6
    )
    left_sharpa_kinematics = setup_sharpa_kinematics(
        side="left", frame_tasks_converged_threshold=1e-6
    )

    #############################################################
    # Motion data and MANO
    #############################################################

    # Exclude scissor as no URDF is available
    arctic_mano_files = sorted(
        [f for f in ARCTIC_MOTION_DIR.glob("*/*.mano.npy") if "scissor" not in f.name]
    )

    object_body_names = ["bottom", "top"]

    print(f"Found {len(arctic_mano_files)} arctic trajectories")

    mano = MANO(
        gender="neutral",
        device=device,
        flat_hand_mean=False,  # Arctic requires flat hand mean to be False
    )

    #############################################################
    # Optimization loop
    #############################################################

    for mano_data_file in tqdm(arctic_mano_files):

        # 0. Read MANO data
        mano_data = np.load(mano_data_file, allow_pickle=True).item()

        right_global_orient = torch.from_numpy(mano_data["right"]["rot"])  # (H, 3)
        right_finger_pose = torch.from_numpy(mano_data["right"]["pose"])  # (H, 45)
        right_trans = torch.from_numpy(mano_data["right"]["trans"])  # (H, 3)
        right_betas = torch.from_numpy(mano_data["right"]["shape"])  # (10,)
        right_fitting_err = torch.tensor(mano_data["right"]["fitting_err"])  # (H,)

        left_global_orient = torch.from_numpy(mano_data["left"]["rot"])  # (H, 3)
        left_finger_pose = torch.from_numpy(mano_data["left"]["pose"])  # (H, 45)
        left_trans = torch.from_numpy(mano_data["left"]["trans"])  # (H, 3)
        left_betas = torch.from_numpy(mano_data["left"]["shape"])  # (10,)
        left_fitting_err = torch.tensor(mano_data["left"]["fitting_err"])  # (H,)

        H = len(right_global_orient)

        # 1. Prepare logging
        object_name = mano_data_file.name.split("_")[0]
        raw_motion_file = str(Path(*mano_data_file.parts[-3:]))[:-9]

        if args.save:
            logger_data = ManoSharpaData(
                sequence_id=raw_motion_file.replace("/", "_"),
                raw_motion_file=raw_motion_file,
                object_name=object_name,
                robot_name="sharpa_wave",
                fps=30.0,
                mano_to_robot_scale=args.mano_to_robot_scale,
                mano_right_betas=right_betas.tolist(),
                mano_left_betas=left_betas.tolist(),
                right_robot_finger_joint_names=list(
                    right_sharpa_kinematics.robot_finger_joint_names.values()
                ),
                right_robot_frame_names=list(
                    right_sharpa_kinematics.robot_frame_names.values()
                ),
                right_robot_frame_task_names=list(
                    right_sharpa_kinematics.frame_tasks.keys()
                ),
                left_robot_finger_joint_names=list(
                    left_sharpa_kinematics.robot_finger_joint_names.values()
                ),
                left_robot_frame_names=list(
                    left_sharpa_kinematics.robot_frame_names.values()
                ),
                left_robot_frame_task_names=list(
                    left_sharpa_kinematics.frame_tasks.keys()
                ),
                object_body_names=object_body_names,
            )

        # 2. Forward pass of the MANO model
        right_mano_results = mano.forward(
            side="right",
            global_orient=right_global_orient,
            finger_pose=right_finger_pose,
            transl=right_trans,
            betas=right_betas,
        )
        left_mano_results = mano.forward(
            side="left",
            global_orient=left_global_orient,
            finger_pose=left_finger_pose,
            transl=left_trans,
            betas=left_betas,
        )

        # 3. Read object data for visualization and logging
        object_data_path = str(mano_data_file).replace(".mano.", ".object.")
        object_data = np.load(object_data_path, allow_pickle=True)  # (H, 7)
        object_articulation = object_data[:, 0]  # (H,)
        object_axis_angle = object_data[:, 1:4]  # (H, 3)
        object_translation = object_data[:, 4:] / 1000.0  # (H, 3)

        # FIXME: We should load the mesh given the URDF rather than manually combining the meshes.
        # Load object meshes for tips_distance computation (ManipTrans approach)
        # Keep top and bottom meshes separate so we can apply articulation transforms
        object_mesh_meshes: dict[str, Any] = {}
        object_mesh_verts: dict[str, torch.Tensor] = {}
        object_mesh_faces: dict[str, torch.Tensor] = {}
        compute_tips_dist = False
        for part in object_body_names:
            mesh_path = ARCTIC_MESH_DIR / object_name / f"{part}_watertight_tiny.obj"
            if mesh_path.exists():
                mesh = trimesh.load(str(mesh_path))
                object_mesh_meshes[part] = mesh
                object_mesh_verts[part] = (
                    torch.from_numpy(mesh.vertices).float().to(device)
                )
                object_mesh_faces[part] = torch.from_numpy(mesh.faces).long().to(device)
                compute_tips_dist = True
            else:
                print(
                    f"Warning: Watertight mesh not found for {object_name}, skipping {part}"
                )

        # Set up MuJoCo object model for articulation transforms
        # (needed for both distance computation and visualization)
        object_urdf_path = ARCTIC_URDF_DIR / f"{object_name}.urdf"
        mujoco_object_model = mujoco.MjModel.from_xml_path(str(object_urdf_path))
        mujoco_object_data = mujoco.MjData(mujoco_object_model)
        top_body_id = mujoco_object_model.body("top").id
        rotation_joint_id = mujoco_object_model.joint("rotation").id

        if args.visualize:
            # Set viser object handles for visualization if enabled
            viser_object_handles = {}
            for part in object_body_names:
                viser_object_handles[part] = viser_server.scene.add_mesh_trimesh(
                    name=f"/object/{part}",
                    mesh=object_mesh_meshes[part],
                    position=np.array([0, 0, 0]),
                    wxyz=np.array([1, 0, 0, 0]),
                )

        # 4. Solve IK for each frame with wrist initialization
        right_qpos = None
        left_qpos = None

        for frame_id in range(40, H - 50):

            # Initialize wrist pose to the first frame of the MANO data
            if right_qpos is None:
                right_qpos = right_sharpa_kinematics.robot.q0.copy()
                right_qpos[:3] = right_mano_results["joints"][frame_id][0].cpu().numpy()
                right_qpos[3:7] = (
                    R.from_quat(
                        right_mano_results["joints_wxyz"][frame_id][0].cpu().numpy(),
                        scalar_first=True,
                    )
                    * R.from_quat(
                        [0.5, -0.5, 0.5, 0.5], scalar_first=True
                    ).inv()  # Rotation offset from link frame to site
                ).as_quat(scalar_first=False)

            if left_qpos is None:
                left_qpos = left_sharpa_kinematics.robot.q0.copy()
                left_qpos[:3] = left_mano_results["joints"][frame_id][0].cpu().numpy()
                left_qpos[3:7] = (
                    R.from_quat(
                        left_mano_results["joints_wxyz"][frame_id][0].cpu().numpy(),
                        scalar_first=True,
                    )
                    * R.from_quat(
                        [0.5, -0.5, 0.5, 0.5], scalar_first=True
                    ).inv()  # Rotation offset from link frame to site
                ).as_quat(scalar_first=False)

            # Right hand
            right_kinematics_results = right_sharpa_kinematics.compute(
                right_mano_results["joints"][frame_id],
                right_mano_results["joints_wxyz"][frame_id],
                mano_to_robot_scale=args.mano_to_robot_scale,
                qpos=right_qpos,
            )
            right_qpos = right_kinematics_results["q"]

            # Left hand
            left_kinematics_results = left_sharpa_kinematics.compute(
                left_mano_results["joints"][frame_id],
                left_mano_results["joints_wxyz"][frame_id],
                mano_to_robot_scale=args.mano_to_robot_scale,
                qpos=left_qpos,
            )
            left_qpos = left_kinematics_results["q"]

            # Compute object world transforms (shared by visualization and distance)
            world_p_bottom = object_translation[frame_id]
            world_q_bottom = R.from_rotvec(object_axis_angle[frame_id]).as_quat(
                scalar_first=True
            )
            world_t_bottom = np.eye(4)
            world_t_bottom[:3, :3] = R.from_quat(
                world_q_bottom, scalar_first=True
            ).as_matrix()
            world_t_bottom[:3, 3] = world_p_bottom

            # Top body: apply articulation via MuJoCo forward kinematics
            mujoco_object_data.qpos[rotation_joint_id] = object_articulation[frame_id]
            mujoco.mj_forward(mujoco_object_model, mujoco_object_data)
            bottom_t_top = np.eye(4)
            bottom_t_top[:3, :3] = np.array(
                mujoco_object_data.xmat[top_body_id]
            ).reshape(3, 3)
            bottom_t_top[:3, 3] = mujoco_object_data.xpos[top_body_id]
            world_t_top = world_t_bottom @ bottom_t_top
            world_p_top = world_t_top[:3, 3]
            world_q_top = R.from_matrix(world_t_top[:3, :3]).as_quat(scalar_first=True)

            world_p_objects = np.vstack([world_p_bottom, world_p_top])
            world_q_objects = np.vstack([world_q_bottom, world_q_top])

            # Visualize results if enabled
            if args.visualize:
                mano.visualize(
                    viser_server,
                    "right",
                    right_mano_results["vertices"][frame_id],
                    right_mano_results["faces"],
                    right_mano_results["joints"][frame_id],
                    right_mano_results["joints_wxyz"][frame_id],
                )
                right_sharpa_kinematics.visualize(
                    viser_server,
                    right_qpos,
                )
                mano.visualize(
                    viser_server,
                    "left",
                    left_mano_results["vertices"][frame_id],
                    left_mano_results["faces"],
                    left_mano_results["joints"][frame_id],
                    left_mano_results["joints_wxyz"][frame_id],
                )
                left_sharpa_kinematics.visualize(
                    viser_server,
                    left_qpos,
                )

                # Visualize object parts
                viser_object_handles["bottom"].position = world_p_bottom
                viser_object_handles["bottom"].wxyz = world_q_bottom
                viser_object_handles["top"].position = world_p_top
                viser_object_handles["top"].wxyz = world_q_top

            # Compute tips_distance (ManipTrans approach)
            # Distance from MANO fingertips to object surface, accounting for articulation
            right_tips_dist = None
            left_tips_dist = None
            if compute_tips_dist:
                # Transform each part's vertices to world frame separately
                obj_R_bottom = (
                    torch.from_numpy(world_t_bottom[:3, :3]).float().to(device)
                )
                obj_t_bottom = (
                    torch.from_numpy(world_t_bottom[:3, 3]).float().to(device)
                )
                obj_R_top = torch.from_numpy(world_t_top[:3, :3]).float().to(device)
                obj_t_top = torch.from_numpy(world_t_top[:3, 3]).float().to(device)

                bottom_world = (
                    obj_R_bottom @ object_mesh_verts["bottom"].T
                ).T + obj_t_bottom
                top_world = (obj_R_top @ object_mesh_verts["top"].T).T + obj_t_top

                # Combine in world frame
                combined_world_verts = torch.cat([bottom_world, top_world], dim=0)
                n_bottom_verts = object_mesh_verts["bottom"].shape[0]
                combined_faces = torch.cat(
                    [
                        object_mesh_faces["bottom"],
                        object_mesh_faces["top"] + n_bottom_verts,
                    ],
                    dim=0,
                )

                # Pass identity transform since vertices are already in world frame
                identity_R = torch.eye(3, device=device)
                zero_t = torch.zeros(3, device=device)

                right_tips_dist = (
                    compute_tips_distance(
                        right_mano_results["joints"][frame_id],
                        combined_world_verts,
                        combined_faces,
                        identity_R,
                        zero_t,
                    )
                    .cpu()
                    .tolist()
                )

                left_tips_dist = (
                    compute_tips_distance(
                        left_mano_results["joints"][frame_id],
                        combined_world_verts,
                        combined_faces,
                        identity_R,
                        zero_t,
                    )
                    .cpu()
                    .tolist()
                )

            # Log data
            if args.save:
                logger_data.log_timestep(
                    # MANO right hand
                    mano_right_trans=right_trans[frame_id].cpu().tolist(),
                    mano_right_global_orient=right_global_orient[frame_id]
                    .cpu()
                    .tolist(),
                    mano_right_finger_pose=right_finger_pose[frame_id].cpu().tolist(),
                    mano_right_joints=right_mano_results["joints"][frame_id]
                    .cpu()
                    .tolist(),
                    mano_right_joints_wxyz=right_mano_results["joints_wxyz"][frame_id]
                    .cpu()
                    .tolist(),
                    mano_right_fitting_err=right_fitting_err[frame_id].cpu().item(),
                    mano_right_tips_distance=right_tips_dist,
                    # MANO left hand
                    mano_left_trans=left_trans[frame_id].cpu().tolist(),
                    mano_left_global_orient=left_global_orient[frame_id].cpu().tolist(),
                    mano_left_finger_pose=left_finger_pose[frame_id].cpu().tolist(),
                    mano_left_joints=left_mano_results["joints"][frame_id]
                    .cpu()
                    .tolist(),
                    mano_left_joints_wxyz=left_mano_results["joints_wxyz"][frame_id]
                    .cpu()
                    .tolist(),
                    mano_left_fitting_err=left_fitting_err[frame_id].cpu().item(),
                    mano_left_tips_distance=left_tips_dist,
                    # Object
                    object_articulation=object_articulation[frame_id].tolist(),
                    object_root_axis_angle=object_axis_angle[frame_id].tolist(),
                    object_root_position=object_translation[frame_id].tolist(),
                    object_body_position=world_p_objects.tolist(),
                    object_body_wxyz=world_q_objects.tolist(),
                    # Robot
                    robot_right_wrist_position=right_kinematics_results["q"][
                        :3
                    ].tolist(),
                    robot_right_wrist_wxyz=right_kinematics_results["q"][3:7][
                        [3, 0, 1, 2]
                    ].tolist(),
                    robot_right_finger_joints=right_kinematics_results["q"][
                        7:
                    ].tolist(),
                    robot_right_frames=right_kinematics_results["frame_pose"].tolist(),
                    robot_right_frame_task_errors=right_kinematics_results[
                        "frame_task_errors"
                    ],
                    robot_right_num_optimization_iterations=right_kinematics_results[
                        "num_optimization_iterations"
                    ],
                    robot_left_wrist_position=left_kinematics_results["q"][:3].tolist(),
                    robot_left_wrist_wxyz=left_kinematics_results["q"][3:7][
                        [3, 0, 1, 2]
                    ].tolist(),
                    robot_left_finger_joints=left_kinematics_results["q"][7:].tolist(),
                    robot_left_frames=left_kinematics_results["frame_pose"].tolist(),
                    robot_left_frame_task_errors=left_kinematics_results[
                        "frame_task_errors"
                    ],
                    robot_left_num_optimization_iterations=left_kinematics_results[
                        "num_optimization_iterations"
                    ],
                )

        # Save logger data to Parquet file
        if args.save:
            logger_data.save_to_parquet(
                root_path=str(SAVE_DIR),
                partition_cols=[
                    "sequence_id",
                    "robot_name",
                    "mano_to_robot_scale",
                ],
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
