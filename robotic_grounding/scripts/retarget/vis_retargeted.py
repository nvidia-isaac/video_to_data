# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Visualize retargeted hand-object data from any dataset (ARCTIC, TACO, etc.).

Loads ManoSharpaData Parquet from a given directory; object meshes are loaded from
the object_mesh_paths field (one path per object body). Plays back robot hands + object poses in viser.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import viser
from pxr import Usd, UsdGeom
from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import ManoSharpaData, list_sequence_ids
from robotic_grounding.retarget.hand_kinematics import HandKinematics
from robotic_grounding.retarget.params import MANO_FINGERTIP_INDICES, MANO_HAND_LINKS
from robotic_grounding.retarget.read_mano import MANO
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def distance_to_color(d: float) -> tuple[int, int, int]:
    """Map distance to a green-to-red color gradient.

    Green (0, 255, 0) at d <= 0.01m (contact), red (255, 0, 0) at d >= 0.05m (far).
    """
    t = np.clip((d - 0.01) / (0.05 - 0.01), 0.0, 1.0)
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    return (r, g, 0)


def load_object_meshes_from_paths(
    viser_server: viser.ViserServer,
    object_mesh_paths: list[str],
    object_body_names: list[str],
) -> dict[str, Any]:
    """Load object meshes from schema paths (one per body) and add them to the viser scene.

    Paths ending with _cm.obj are scaled by 0.01 (cm -> m). Returns dict mapping body name to handle.
    """
    handles: dict[str, Any] = {}
    for part, path in zip(object_body_names, object_mesh_paths, strict=True):
        if not path or not Path(path).exists():
            continue
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if path.endswith("_cm.obj"):
            mesh.vertices *= 0.01
        handles[part] = viser_server.scene.add_mesh_trimesh(
            name=f"/object/{part}",
            mesh=mesh,
            position=np.array([0.0, 0.0, 0.0]),
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
    return handles


def load_support_surfaces_from_usd(
    viser_server: viser.ViserServer,
    usd_path: Path,
) -> dict[str, Any]:
    """Load support-surface disks from a .usda file and add them to the viser scene.

    Each ``UsdGeom.Cylinder`` prim is converted to a trimesh cylinder and rendered
    with the ``displayColor`` stored in the USD (falls back to light-blue if absent).

    Args:
        viser_server: The running viser server.
        usd_path: Path to the ``.usda`` file produced by ``reconstruct_support_surfaces.py``.

    Returns:
        Dict mapping prim path to the viser mesh handle.
    """
    stage = Usd.Stage.Open(str(usd_path))
    handles: dict[str, Any] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Cylinder):
            continue
        cyl = UsdGeom.Cylinder(prim)
        radius = cyl.GetRadiusAttr().Get()
        height = cyl.GetHeightAttr().Get()
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        translate = ops[0].Get() if ops else (0.0, 0.0, 0.0)

        display_color = cyl.GetDisplayColorAttr().Get()
        if display_color and len(display_color) > 0:
            c = display_color[0]
            rgba = [int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 180]
        else:
            rgba = [150, 200, 255, 160]

        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
        mesh.apply_translation(translate)
        mesh.visual.face_colors = rgba

        prim_path = str(prim.GetPath())
        handles[prim_path] = viser_server.scene.add_mesh_trimesh(
            name=prim_path,
            mesh=mesh,
        )
    return handles


DATASET_DIRS: dict[str, str] = {
    "arctic": "arctic_processed",
    "taco": "taco_processed",
    "oakink2": "oakink2_processed",
}


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize retargeted Parquet data (hands + optional object meshes)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_DIRS),
        default=None,
        help=(
            "Dataset shorthand; sets --input_dir to the corresponding processed directory "
            f"under HUMAN_MOTION_DATA_DIR. Choices: {list(DATASET_DIRS)}. "
            "Ignored when --input_dir is set explicitly."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help=(
            "Root directory of retargeted Parquet (e.g. .../arctic_processed). "
            "Defaults to arctic_processed when neither --input_dir nor --dataset is given."
        ),
    )
    parser.add_argument(
        "--sequence_id",
        type=str,
        default=None,
        help="Sequence to visualize. If not set, iterate through all sequences in input_dir.",
    )
    parser.add_argument(
        "-tid",
        "--trajectory_id",
        type=int,
        default=0,
        help="Row index when multiple rows match filters (default 0).",
    )
    parser.add_argument(
        "--show_mano",
        action="store_true",
        default=False,
        help="Show MANO hand meshes and joint frames alongside robot hands.",
    )
    parser.add_argument(
        "--visualize_contacts",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--visualize_fingertip_distances",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--support_usd",
        type=Path,
        default=None,
        help="Explicit path to a support-surface .usda file (overrides auto-discovery).",
    )
    return parser.parse_args()


def mano_kwargs_from_data(logger_data: Any) -> dict[str, Any]:
    """Read MANO model kwargs stored in the Parquet data."""
    kwargs: dict[str, Any] = {}
    if hasattr(logger_data, "mano_flat_hand_mean"):
        kwargs["flat_hand_mean"] = logger_data.mano_flat_hand_mean
    if (
        hasattr(logger_data, "mano_center_idx")
        and logger_data.mano_center_idx is not None
    ):
        kwargs["center_idx"] = logger_data.mano_center_idx
    return kwargs


def run_mano_forward_from_data(
    mano: MANO,
    logger_data: Any,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run MANO forward pass for both hands using stored Parquet parameters.

    Returns dict keyed by "right"/"left", each containing MANO forward outputs.
    """
    results: dict[str, dict[str, torch.Tensor]] = {}
    for side in ("right", "left"):
        trans = torch.tensor(
            getattr(logger_data, f"mano_{side}_trans"),
            dtype=torch.float32,
            device=device,
        )
        global_orient = torch.tensor(
            getattr(logger_data, f"mano_{side}_global_orient"),
            dtype=torch.float32,
            device=device,
        )
        finger_pose = torch.tensor(
            getattr(logger_data, f"mano_{side}_finger_pose"),
            dtype=torch.float32,
            device=device,
        )
        betas = torch.tensor(
            getattr(logger_data, f"mano_{side}_betas"),
            dtype=torch.float32,
            device=device,
        )
        results[side] = mano.forward(
            side=side,
            global_orient=global_orient,
            finger_pose=finger_pose,
            transl=trans,
            betas=betas,
        )
    return results


def find_support_usd(input_dir: Path, sequence_id: str) -> Path | None:
    """Try to find a support-surface .usda file for the given sequence.

    Looks in ``<input_dir_parent>/reconstructed_stage/<sequence_id>_support.usda``.
    """
    candidate = input_dir.parent / "reconstructed_stage" / f"{sequence_id}_support.usda"
    if candidate.exists():
        return candidate
    return None


def visualize_one_trajectory(
    viser_server: viser.ViserServer,
    right_sharpa_kinematics: HandKinematics,
    left_sharpa_kinematics: HandKinematics,
    viser_object_handles: dict[str, Any],
    input_dir: Path,
    sequence_id: str,
    trajectory_id: int,
    show_mano: bool = False,
    visualize_contacts: bool = False,
    visualize_fingertip_distances: bool = False,
    support_usd: Path | None = None,
) -> dict[str, Any]:
    """Load one sequence and visualize playback (hands + objects from object_mesh_paths)."""
    for _, handle in viser_object_handles.items():
        handle.remove()
    viser_object_handles.clear()

    # Auto-discover or use explicit support-surface USD
    usd_path = support_usd or find_support_usd(input_dir, sequence_id)
    if usd_path is not None:
        print(f"  Loading support surfaces from {usd_path}")
        load_support_surfaces_from_usd(viser_server, usd_path)

    contact_points_handles: list[Any] = []

    logger_data = ManoSharpaData.from_parquet(
        root_path=str(input_dir),
        filters=[("sequence_id", "=", sequence_id)],
        trajectory_id=trajectory_id,
    )
    H = len(logger_data.robot_right_wrist_position)

    mano: MANO | None = None
    mano_results: dict[str, dict[str, torch.Tensor]] | None = None
    if show_mano:
        mano_kwargs = mano_kwargs_from_data(logger_data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mano = MANO(gender="neutral", device=device, **mano_kwargs)
        print(f"MANO model initialized ({mano_kwargs}) on {device}")
        mano_results = run_mano_forward_from_data(mano, logger_data, device)

    # Optional: world frame
    viser_object_handles["frame"] = viser_server.scene.add_frame(
        name="/object/frame",
        position=np.array([0, 0, 0]),
        wxyz=np.array([1, 0, 0, 0]),
        axes_length=0.2,
        axes_radius=0.007,
    )

    # Object meshes from schema paths (one per body); placeholders for any missing
    object_mesh_paths = getattr(logger_data, "object_mesh_paths", None) or []
    if object_mesh_paths and len(object_mesh_paths) == len(
        logger_data.object_body_names
    ):
        handles = load_object_meshes_from_paths(
            viser_server,
            object_mesh_paths,
            logger_data.object_body_names,
        )
        viser_object_handles.update(handles)
    for part in logger_data.object_body_names:
        if part not in viser_object_handles:
            viser_object_handles[part] = viser_server.scene.add_icosphere(
                name=f"/object/{part}",
                radius=0.02,
                color=(128, 128, 128),
                position=np.array([0.0, 0.0, 0.0]),
            )

    for frame_id in range(H):
        # Right hand
        right_qpos = right_sharpa_kinematics.robot.q0.copy()
        right_qpos[:3] = np.array(logger_data.robot_right_wrist_position[frame_id])
        right_qpos[3:7] = np.array(logger_data.robot_right_wrist_wxyz[frame_id])[
            [1, 2, 3, 0]
        ]
        right_qpos[7:] = np.array(logger_data.robot_right_finger_joints[frame_id])
        right_sharpa_kinematics.visualize(viser_server, right_qpos)
        # Left hand
        left_qpos = left_sharpa_kinematics.robot.q0.copy()
        left_qpos[:3] = np.array(logger_data.robot_left_wrist_position[frame_id])
        left_qpos[3:7] = np.array(logger_data.robot_left_wrist_wxyz[frame_id])[
            [1, 2, 3, 0]
        ]
        left_qpos[7:] = np.array(logger_data.robot_left_finger_joints[frame_id])
        left_sharpa_kinematics.visualize(viser_server, left_qpos)

        # Update object poses from Parquet
        for object_body_idx, object_body_name in enumerate(
            logger_data.object_body_names
        ):
            if object_body_name not in viser_object_handles:
                continue
            handle = viser_object_handles[object_body_name]
            handle.position = np.asarray(
                logger_data.object_body_position[frame_id][object_body_idx]
            )
            if hasattr(handle, "wxyz"):
                handle.wxyz = np.asarray(
                    logger_data.object_body_wxyz[frame_id][object_body_idx]
                )

        # Fingertip distance spheres (if available)
        if visualize_fingertip_distances:
            for side, joints_data, dist_data in [
                (
                    "right",
                    logger_data.mano_right_joints,
                    logger_data.mano_right_tips_distance,
                ),
                (
                    "left",
                    logger_data.mano_left_joints,
                    logger_data.mano_left_tips_distance,
                ),
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

        # Link contact visualization
        if visualize_contacts:
            for contact_handle in contact_points_handles:
                contact_handle.remove()
            contact_points_handles.clear()
            for side, contact_positions in [
                ("right", logger_data.mano_right_object_contact_positions[frame_id]),
                ("left", logger_data.mano_left_object_contact_positions[frame_id]),
            ]:
                for contact_position, (link_name, _) in zip(
                    contact_positions, MANO_HAND_LINKS.items(), strict=False
                ):
                    if np.sum(contact_position) > 0.0:
                        contact_handle = viser_server.scene.add_icosphere(
                            name=f"/mano/{side}_contact_points/{link_name}",
                            radius=0.005,
                            color=np.array([0, 0, 255]),
                            position=np.array(contact_position[:3]),
                        )
                        contact_points_handles.append(contact_handle)

        # MANO hand mesh + joint frames
        if mano is not None and mano_results is not None:
            for side in ("right", "left"):
                mano.visualize(
                    viser_server,
                    side,
                    vertices=mano_results[side]["vertices"][frame_id],
                    faces=mano_results[side]["faces"],
                    joints=mano_results[side]["joints"][frame_id],
                    joints_wxyz=mano_results[side]["joints_wxyz"][frame_id],
                )

        time.sleep(1.0 / logger_data.fps)

    return viser_object_handles


def main(args: argparse.Namespace) -> None:
    """List or use sequence, setup kinematics, run visualization."""
    if args.input_dir is not None:
        input_dir = args.input_dir
    elif args.dataset is not None:
        input_dir = HUMAN_MOTION_DATA_DIR / DATASET_DIRS[args.dataset]
    else:
        input_dir = HUMAN_MOTION_DATA_DIR / DATASET_DIRS["arctic"]

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    available = list_sequence_ids(str(input_dir))
    if not available:
        raise ValueError(f"No sequences found in {input_dir}")

    if args.sequence_id is not None:
        sequence_ids = [args.sequence_id]
    else:
        sequence_ids = available
        print(
            f"No sequence_id specified, will iterate through all {len(sequence_ids)} sequences."
        )

    viser_server = viser.ViserServer()
    viser_object_handles: dict[str, Any] = {}

    support_usd = args.support_usd
    if support_usd is not None and not support_usd.exists():
        raise FileNotFoundError(f"Support USD not found: {support_usd}")

    right_sharpa_kinematics = setup_sharpa_kinematics(
        side="right", frame_tasks_converged_threshold=1e-6
    )
    left_sharpa_kinematics = setup_sharpa_kinematics(
        side="left", frame_tasks_converged_threshold=1e-6
    )

    for seq_idx, sequence_id in enumerate(sequence_ids):
        print(
            f"[{seq_idx + 1}/{len(sequence_ids)}] Visualizing sequence: {sequence_id}"
        )
        visualize_one_trajectory(
            viser_server,
            right_sharpa_kinematics,
            left_sharpa_kinematics,
            viser_object_handles,
            input_dir=input_dir,
            sequence_id=sequence_id,
            trajectory_id=args.trajectory_id,
            show_mano=args.show_mano,
            visualize_contacts=args.visualize_contacts,
            visualize_fingertip_distances=args.visualize_fingertip_distances,
            support_usd=support_usd,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
