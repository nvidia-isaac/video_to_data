# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Base class for loading hand-object datasets into ManoSharpaData schema.

Subclass and implement the abstract methods to add a new dataset (e.g. ARCTIC, TACO).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from robotic_grounding.retarget.contact_utils import (
    approximate_contact_with_id,
    find_link_contact_positions,
    find_object_contact_positions,
)
from robotic_grounding.retarget.data_logger import ManoSharpaData
from robotic_grounding.retarget.params import MANO_HAND_LINKS, NUM_MANO_LINKS
from robotic_grounding.retarget.read_mano import MANO
from robotic_grounding.retarget.retarget_utils import (
    DEFAULT_PARTITION_COLS,
    compute_tips_distance_for_mesh,
)


@dataclass
class SequenceInfo:
    """Metadata for one sequence to process."""

    sequence_id: str
    raw_motion_file: str
    object_name: str
    object_body_names: list[str]
    source: Any = None  # Subclass-specific (e.g. Path to .mano.npy, or (hand_dir, N))


@dataclass
class FrameObjectPoses:
    """Per-frame object poses and optional combined mesh for tips-distance."""

    object_body_position: list[list[float]]
    object_body_wxyz: list[list[float]]
    object_root_axis_angle: list[float]
    object_root_position: list[float]
    object_articulation: float
    tips_verts: torch.Tensor | None = None
    tips_faces: torch.Tensor | None = None


def poses_to_root_position_and_axis_angle(
    poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract root position and axis-angle from (N, 4, 4) world transform matrices.

    Returns:
        root_position: (N, 3) translation column.
        root_axis_angle: (N, 3) axis-angle from rotation matrices.
    """
    root_position = poses[:, :3, 3]
    root_axis_angle = np.array(
        [R.from_matrix(poses[i, :3, :3]).as_rotvec() for i in range(len(poses))]
    )
    return root_position, root_axis_angle


def load_meshes_to_device(
    mesh_paths: dict[str, str],
    device: torch.device,
    vertex_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], bool]:
    """Load trimesh objects by part name, convert verts/faces to device tensors.

    Args:
        mesh_paths: Mapping of part name to mesh file path.
        device: Target device for tensors.
        vertex_scale: Factor to scale vertex positions (e.g. 0.01 for cm->m).

    Returns:
        (meshes, verts, faces, compute_tips_dist) -- meshes is the raw trimesh
        per part; verts/faces are on device; compute_tips_dist is False if any
        part mesh was not found.
    """
    meshes: dict[str, Any] = {}
    verts: dict[str, torch.Tensor] = {}
    faces: dict[str, torch.Tensor] = {}
    compute_tips_dist = True
    for part, path_str in mesh_paths.items():
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Mesh not found at {path}, skipping {part}")
            compute_tips_dist = False
            continue
        mesh = trimesh.load(str(path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if vertex_scale != 1.0:
            mesh.vertices *= vertex_scale
        meshes[part] = mesh
        verts[part] = torch.from_numpy(mesh.vertices).float().to(device)
        faces[part] = torch.from_numpy(mesh.faces).long().to(device)
    return meshes, verts, faces, compute_tips_dist


def build_combined_world_mesh(
    object_mesh_verts: dict[str, torch.Tensor],
    object_mesh_faces: dict[str, torch.Tensor],
    body_names: list[str],
    world_transforms: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine per-body meshes transformed to world frame for tips-distance.

    Args:
        object_mesh_verts: Vertices per body name (on device).
        object_mesh_faces: Faces per body name (on device).
        body_names: Order of bodies (must match world_transforms).
        world_transforms: List of 4x4 world matrices, one per body in body_names order.
        device: Target device for output tensors.

    Returns:
        combined_verts: (V, 3) all vertices in world frame.
        combined_faces: (F, 3) face indices with offset for concatenated verts.
    """
    all_verts: list[torch.Tensor] = []
    all_faces: list[torch.Tensor] = []
    offset = 0
    for name, T in zip(body_names, world_transforms, strict=True):
        if name not in object_mesh_verts or name not in object_mesh_faces:
            continue
        R_mat = torch.from_numpy(np.asarray(T)[:3, :3]).float().to(device)
        t_vec = torch.from_numpy(np.asarray(T)[:3, 3]).float().to(device)
        verts = (R_mat @ object_mesh_verts[name].T).T + t_vec
        all_verts.append(verts)
        all_faces.append(object_mesh_faces[name] + offset)
        offset += verts.shape[0]
    if not all_verts:
        raise ValueError("No mesh parts found to combine")
    return torch.cat(all_verts, dim=0), torch.cat(all_faces, dim=0)


class DatasetLoaderBase(ABC):
    """Base class for loading a dataset into ManoSharpaData (MANO + object only).

    Subclasses implement: list_sequences, load_mano_data, load_object_data,
    load_object_meshes, get_mano_kwargs, get_fps. Optionally override
    get_frame_object_poses, get_object_mesh_paths, get_frame_range.
    """

    @abstractmethod
    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        """List sequences to process. Uses args for dataset paths/filters."""
        ...

    @abstractmethod
    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        """Load MANO parameters for the sequence.

        Returns dict with: H, right_global_orient, right_finger_pose, right_trans,
        right_betas, right_fitting_err, left_global_orient, left_finger_pose,
        left_trans, left_betas, left_fitting_err. All tensors on device.
        """
        ...

    @abstractmethod
    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        """Load object pose data (passed to get_frame_object_poses).

        Return a dictionary mapping each object/body name to a tuple
        (pose, root_position, root_axis_angle, articulation):
        - pose: (N, 4, 4) world transform per frame
        - root_position: (N, 3) object root position per frame
        - root_axis_angle: (N, 3) object root axis-angle per frame
        - articulation: (N,) array or None if no articulation
        """
        ...

    @abstractmethod
    def load_object_meshes(
        self,
        sequence_info: SequenceInfo,
        device: torch.device,
    ) -> tuple[
        dict[str, Any],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        bool,
    ]:
        """Load object part meshes.

        Returns:
            object_mesh_meshes: trimesh (or Scene) per part, for visualization.
            object_mesh_verts: vertices per part on device.
            object_mesh_faces: faces per part on device.
            compute_tips_dist: whether tips-distance can be computed.
        """
        ...

    def get_frame_object_poses(
        self,
        frame_id: int,
        object_data: dict[str, Any],
        object_mesh_verts: dict[str, torch.Tensor],
        object_mesh_faces: dict[str, torch.Tensor],
        device: torch.device,
    ) -> FrameObjectPoses:
        """Compute per-body world poses and optional combined mesh for one frame.

        Uses the (pose, root_position, root_axis_angle, articulation) tuple
        convention from load_object_data(). Override if custom logic is needed.
        """
        body_names = list(object_data.keys())
        world_transforms = [object_data[name][0][frame_id] for name in body_names]

        object_body_position = [t[:3, 3].tolist() for t in world_transforms]
        object_body_wxyz = [
            R.from_matrix(t[:3, :3]).as_quat(scalar_first=True).tolist()
            for t in world_transforms
        ]

        _pose, root_position, root_axis_angle, articulation = object_data[body_names[0]]

        tips_verts = None
        tips_faces = None
        if all(name in object_mesh_verts for name in body_names):
            tips_verts, tips_faces = build_combined_world_mesh(
                object_mesh_verts,
                object_mesh_faces,
                body_names,
                world_transforms,
                device,
            )

        return FrameObjectPoses(
            object_body_position=object_body_position,
            object_body_wxyz=object_body_wxyz,
            object_root_axis_angle=root_axis_angle[frame_id].tolist(),
            object_root_position=root_position[frame_id].tolist(),
            object_articulation=(
                float(articulation[frame_id]) if articulation is not None else 0.0
            ),
            tips_verts=tips_verts,
            tips_faces=tips_faces,
        )

    @abstractmethod
    def get_mano_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for MANO(...), e.g. flat_hand_mean, center_idx."""
        ...

    @abstractmethod
    def get_fps(self) -> float:
        """Frames per second for this dataset."""
        ...

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return paths to object mesh files (one per object body, same order as object_body_names). Override in subclass."""
        return []

    def get_frame_range(self, num_frames: int) -> tuple[int, int]:
        """Return (start, end) frame indices to process. Override to trim frames."""
        return 0, num_frames

    def run(self, args: Any) -> None:
        """Common pipeline: list sequences, load MANO/object, log timesteps, save."""
        self._args = args
        device = torch.device(args.device)

        if args.visualize:
            viser_server = viser.ViserServer()
        else:
            viser_server = None

        if args.save:
            args.output_dir.mkdir(parents=True, exist_ok=True)

        sequences = self.list_sequences(args)
        print(f"Found {len(sequences)} sequences")

        mano_kwargs = self.get_mano_kwargs()
        mano = MANO(gender="neutral", device=device, **mano_kwargs)
        viser_object_handles: dict[str, Any] = {}

        for sequence_info in tqdm(sequences):
            try:
                raw_data = self.load_mano_data(sequence_info, device)
            except (FileNotFoundError, ValueError, KeyError) as e:
                print(f"Skipping {sequence_info.sequence_id}: {e}")
                continue

            # Number of frames
            H = raw_data["H"]
            try:
                object_data = self.load_object_data(sequence_info)
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping {sequence_info.sequence_id}: {e}")
                continue

            (
                object_mesh_meshes,
                object_mesh_verts,
                object_mesh_faces,
                compute_tips_dist,
            ) = self.load_object_meshes(sequence_info, device)

            object_surface_points: dict[str, np.ndarray] = {}
            for part in sequence_info.object_body_names:
                if part in object_mesh_meshes:
                    pts, _ = trimesh.sample.sample_surface_even(
                        object_mesh_meshes[part], 2048
                    )
                    object_surface_points[part] = np.asarray(pts)

            if args.visualize and viser_server is not None:
                for handle in viser_object_handles.values():
                    handle.remove()
                viser_object_handles.clear()
                for part in sequence_info.object_body_names:
                    if part not in object_mesh_meshes:
                        continue
                    mesh = object_mesh_meshes[part]
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                    viser_object_handles[part] = viser_server.scene.add_mesh_trimesh(
                        name=f"/object/{part}",
                        mesh=mesh,
                        position=np.array([0.0, 0.0, 0.0]),
                        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                    )

            if args.save:
                logger_data = ManoSharpaData(
                    sequence_id=sequence_info.sequence_id,
                    raw_motion_file=sequence_info.raw_motion_file,
                    object_name=sequence_info.object_name,
                    robot_name="sharpa_wave",
                    fps=self.get_fps(),
                    mano_flat_hand_mean=mano_kwargs.get("flat_hand_mean", True),
                    mano_center_idx=mano_kwargs.get("center_idx", None),
                    mano_to_robot_scale=args.mano_to_robot_scale,
                    mano_right_betas=raw_data["right_betas"].tolist(),
                    mano_left_betas=raw_data["left_betas"].tolist(),
                    right_robot_finger_joint_names=[],
                    right_robot_frame_names=[],
                    right_robot_frame_task_names=[],
                    left_robot_finger_joint_names=[],
                    left_robot_frame_names=[],
                    left_robot_frame_task_names=[],
                    object_body_names=sequence_info.object_body_names,
                    object_mesh_paths=self.get_object_mesh_paths(sequence_info),
                    mano_link_names=list(MANO_HAND_LINKS.keys()),
                )

            mano_results: dict[str, Any] = {}
            for side in ("right", "left"):
                mano_results[side] = mano.forward(
                    side=side,
                    global_orient=raw_data[f"{side}_global_orient"],
                    finger_pose=raw_data[f"{side}_finger_pose"],
                    transl=raw_data[f"{side}_trans"],
                    betas=raw_data[f"{side}_betas"],
                )

            frame_start, frame_end = self.get_frame_range(H)
            for frame_id in range(frame_start, frame_end):
                joints: dict[str, torch.Tensor] = {}
                joints_wxyz: dict[str, torch.Tensor] = {}
                for side in ("right", "left"):
                    joints[side] = mano_results[side]["joints"][frame_id]
                    joints_wxyz[side] = mano_results[side]["joints_wxyz"][frame_id]

                frame_poses = self.get_frame_object_poses(
                    frame_id,
                    object_data,
                    object_mesh_verts,
                    object_mesh_faces,
                    device,
                )

                if args.visualize and viser_server is not None:
                    for side in ("right", "left"):
                        mano.visualize(
                            viser_server,
                            side,
                            vertices=mano_results[side]["vertices"][frame_id],
                            faces=mano_results[side]["faces"],
                            joints=joints[side],
                            joints_wxyz=joints_wxyz[side],
                        )
                    for idx, part in enumerate(sequence_info.object_body_names):
                        if part in viser_object_handles and idx < len(
                            frame_poses.object_body_position
                        ):
                            pos = frame_poses.object_body_position[idx]
                            wxyz = frame_poses.object_body_wxyz[idx]
                            viser_object_handles[part].position = (
                                np.asarray(pos)
                                if not isinstance(pos, np.ndarray)
                                else pos
                            )
                            viser_object_handles[part].wxyz = (
                                np.asarray(wxyz)
                                if not isinstance(wxyz, np.ndarray)
                                else wxyz
                            )

                tips_dist: dict[str, Any] = {}
                if (
                    compute_tips_dist
                    and frame_poses.tips_verts is not None
                    and frame_poses.tips_faces is not None
                ):
                    for side in ("right", "left"):
                        tips_dist[side] = compute_tips_distance_for_mesh(
                            joints[side],
                            frame_poses.tips_verts,
                            frame_poses.tips_faces,
                        )

                contact_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                body_names = list(object_data.keys())
                obj_verts_world_parts: list[np.ndarray] = []
                obj_part_ids_parts: list[np.ndarray] = []
                for body_idx, name in enumerate(body_names):
                    if name not in object_surface_points:
                        continue
                    world_t = object_data[name][0][frame_id]
                    pts_local = object_surface_points[name]
                    pts_world = (pts_local @ world_t[:3, :3].T) + world_t[:3, 3]
                    obj_verts_world_parts.append(pts_world)
                    obj_part_ids_parts.append(np.full(pts_world.shape[0], body_idx + 1))

                if obj_verts_world_parts:
                    obj_verts_world = np.vstack(obj_verts_world_parts)
                    obj_part_ids = np.concatenate(obj_part_ids_parts)
                    for side in ("right", "left"):
                        hand_verts = (
                            mano_results[side]["vertices"][frame_id].cpu().numpy()
                        )
                        joints_np = joints[side].cpu().numpy()
                        _, contacts_on_hand = approximate_contact_with_id(
                            obj_verts_world,
                            obj_part_ids,
                            hand_verts,
                            threshold=0.01,
                        )
                        link_contacts = np.zeros((NUM_MANO_LINKS, 4))
                        obj_contacts = np.zeros((NUM_MANO_LINKS, 4))
                        if contacts_on_hand.shape[0] > 0:
                            link_contacts = np.asarray(
                                find_link_contact_positions(contacts_on_hand, joints_np)
                            )
                            obj_contacts = find_object_contact_positions(
                                link_contacts, obj_verts_world, obj_part_ids
                            )
                        contact_data[side] = (link_contacts, obj_contacts)

                if args.save:
                    right_link = (
                        contact_data["right"][0].tolist()
                        if "right" in contact_data
                        else None
                    )
                    right_obj = (
                        contact_data["right"][1].tolist()
                        if "right" in contact_data
                        else None
                    )
                    left_link = (
                        contact_data["left"][0].tolist()
                        if "left" in contact_data
                        else None
                    )
                    left_obj = (
                        contact_data["left"][1].tolist()
                        if "left" in contact_data
                        else None
                    )
                    logger_data.log_timestep(
                        mano_right_trans=raw_data["right_trans"][frame_id]
                        .cpu()
                        .tolist(),
                        mano_right_global_orient=raw_data["right_global_orient"][
                            frame_id
                        ]
                        .cpu()
                        .tolist(),
                        mano_right_finger_pose=raw_data["right_finger_pose"][frame_id]
                        .cpu()
                        .tolist(),
                        mano_right_joints=joints["right"].cpu().tolist(),
                        mano_right_joints_wxyz=joints_wxyz["right"].cpu().tolist(),
                        mano_right_fitting_err=raw_data["right_fitting_err"][frame_id]
                        .cpu()
                        .item(),
                        mano_right_tips_distance=tips_dist.get("right"),
                        mano_right_link_contact_positions=right_link,
                        mano_right_object_contact_positions=right_obj,
                        mano_left_trans=raw_data["left_trans"][frame_id].cpu().tolist(),
                        mano_left_global_orient=raw_data["left_global_orient"][frame_id]
                        .cpu()
                        .tolist(),
                        mano_left_finger_pose=raw_data["left_finger_pose"][frame_id]
                        .cpu()
                        .tolist(),
                        mano_left_joints=joints["left"].cpu().tolist(),
                        mano_left_joints_wxyz=joints_wxyz["left"].cpu().tolist(),
                        mano_left_fitting_err=raw_data["left_fitting_err"][frame_id]
                        .cpu()
                        .item(),
                        mano_left_tips_distance=tips_dist.get("left"),
                        mano_left_link_contact_positions=left_link,
                        mano_left_object_contact_positions=left_obj,
                        object_articulation=frame_poses.object_articulation,
                        object_root_axis_angle=frame_poses.object_root_axis_angle,
                        object_root_position=frame_poses.object_root_position,
                        object_body_position=frame_poses.object_body_position,
                        object_body_wxyz=frame_poses.object_body_wxyz,
                    )

            if args.save:
                logger_data.save_to_parquet(
                    root_path=str(args.output_dir),
                    partition_cols=DEFAULT_PARTITION_COLS,
                )
