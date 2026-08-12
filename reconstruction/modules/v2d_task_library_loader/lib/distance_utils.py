# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for computing fingertip-to-object surface distances.

This module implements the ManipTrans approach for pre-computing reference distances
from MANO fingertips to object mesh surfaces during data processing.
"""

import logging

import torch
import trimesh

# Try to import PyTorch3D, fall back to trimesh-based sampling if unavailable
try:
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes

    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    sample_points_from_meshes = None  # type: ignore[misc, assignment]
    Meshes = None  # type: ignore[misc, assignment]

# MANO fingertip joint indices (5 per hand)
# From params.py: wrist(0), thumb1-4(1-4), index1-4(5-8), middle1-4(9-12), ring1-4(13-16), pinky1-4(17-20)
# Tips are: thumb4(4), index4(8), middle4(12), ring4(16), pinky4(20)
MANO_FINGERTIP_INDICES = [4, 8, 12, 16, 20]


def _sample_points_from_mesh_trimesh(
    vertices: torch.Tensor, faces: torch.Tensor, num_samples: int
) -> torch.Tensor:
    """Sample points from mesh surface using trimesh (CPU fallback).

    Args:
        vertices: Mesh vertices (V, 3).
        faces: Mesh faces (F, 3).
        num_samples: Number of points to sample.

    Returns:
        Sampled surface points (num_samples, 3).
    """
    # Convert to numpy for trimesh
    verts_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()

    # Create trimesh and sample
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)

    return torch.from_numpy(points).float().to(vertices.device)


def _sample_points_from_mesh_pytorch3d(
    vertices: torch.Tensor, faces: torch.Tensor, num_samples: int
) -> torch.Tensor:
    """Sample points from mesh surface using PyTorch3D (GPU when available).

    Tries the input device first (GPU if vertices are on CUDA). If PyTorch3D
    was not compiled with GPU support, falls back to CPU and moves the result
    back. For maximum speed, install PyTorch3D with CUDA (see PyTorch3D docs).

    Args:
        vertices: Mesh vertices (V, 3).
        faces: Mesh faces (F, 3).
        num_samples: Number of points to sample.

    Returns:
        Sampled surface points (num_samples, 3) on the same device as vertices.
    """
    device = vertices.device
    mesh = Meshes(verts=[vertices], faces=[faces])
    try:
        return sample_points_from_meshes(mesh, num_samples)[0]
    except RuntimeError as e:
        err_msg = str(e)
        if "Not compiled with GPU support" in err_msg or "CUDA" in err_msg:
            logging.getLogger(__name__).warning(
                "PyTorch3D GPU path failed (%s). Using CPU for mesh sampling. "
                "For faster runs, install PyTorch3D with CUDA support.",
                err_msg[:80],
            )
            mesh_cpu = Meshes(verts=[vertices.cpu()], faces=[faces.cpu()])
            out = sample_points_from_meshes(mesh_cpu, num_samples)[0]
            return out.to(device)
        raise


def compute_tips_distance(
    mano_joints: torch.Tensor,
    obj_mesh_verts: torch.Tensor,
    obj_mesh_faces: torch.Tensor,
    obj_rotation: torch.Tensor,
    obj_translation: torch.Tensor,
    num_samples: int = 1000,
) -> torch.Tensor:
    """Compute distance from MANO fingertips to object surface.

    This follows the ManipTrans approach of pre-computing chamfer distances
    from the reference (MANO) fingertips to the object mesh surface.

    Args:
        mano_joints: MANO joint positions (21, 3).
        obj_mesh_verts: Object mesh vertices in local frame (V, 3).
        obj_mesh_faces: Object mesh faces (F, 3).
        obj_rotation: Object rotation matrix (3, 3).
        obj_translation: Object translation (3,).
        num_samples: Number of points to sample from mesh surface. Default: 1000.

    Returns:
        Distances from each fingertip to nearest point on object surface (5,).
        Order: thumb, index, middle, ring, pinky.
    """
    device = mano_joints.device

    # Ensure tensors are on same device and correct dtype
    obj_mesh_verts = obj_mesh_verts.to(device=device, dtype=torch.float32)
    obj_mesh_faces = obj_mesh_faces.to(device=device, dtype=torch.int64)
    obj_rotation = obj_rotation.to(device=device, dtype=torch.float32)
    obj_translation = obj_translation.to(device=device, dtype=torch.float32)

    # Sample points on object surface
    if PYTORCH3D_AVAILABLE:
        surface_points = _sample_points_from_mesh_pytorch3d(
            obj_mesh_verts, obj_mesh_faces, num_samples
        )
    else:
        surface_points = _sample_points_from_mesh_trimesh(
            obj_mesh_verts, obj_mesh_faces, num_samples
        )

    # Transform surface points to world frame
    # world_points = R @ local_points^T + t
    surface_points_world = (
        obj_rotation @ surface_points.T
    ).T + obj_translation  # (num_samples, 3)

    # Get fingertip positions from MANO joints
    fingertips = mano_joints[MANO_FINGERTIP_INDICES]  # (5, 3)

    # Compute pairwise distances: (5, num_samples)
    dists = torch.cdist(
        fingertips.unsqueeze(0), surface_points_world.unsqueeze(0)
    ).squeeze(0)

    # Get minimum distance from each fingertip to surface
    min_dists = dists.min(dim=-1).values  # (5,)

    return min_dists


def compute_tip_to_object_surface_distance(
    mano_joints: torch.Tensor,
    object_surface_points_world: torch.Tensor,
) -> list[list[float]] | None:
    """Compute MANO fingertip-to-surface distances for one hand.

    Object surface points are assumed to be in world frame.

    Args:
        mano_joints: MANO joints (21, 3) on same device as object_surface_points_world.
        object_surface_points_world: Object surface points in world frame (V, 3).

    Returns:
        List of 5 lists (one per fingertip) of float distances, or None if
        computation is not possible.
    """
    fingertips = mano_joints[MANO_FINGERTIP_INDICES]

    dists = torch.cdist(
        fingertips.unsqueeze(0), object_surface_points_world.unsqueeze(0)
    ).squeeze(0)

    # Get minimum distance from each fingertip to surface
    min_dists = dists.amin(dim=-1)  # (5,)

    return min_dists.cpu().tolist()


def load_object_mesh(mesh_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load object mesh and return vertices and faces as tensors.

    Args:
        mesh_path: Path to the mesh file (OBJ, PLY, etc.).

    Returns:
        Tuple of (vertices, faces) tensors.
        - vertices: (V, 3) float tensor
        - faces: (F, 3) int tensor
    """
    mesh = trimesh.load(mesh_path)
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    return vertices, faces
