# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact computation utilities for MANO hand–object contact (dexmachina format)."""

import torch

from robotic_grounding.retarget.params import MANO_HAND_LINKS, NUM_MANO_LINKS


def approximate_contact_with_id(
    object_surface_points_world: torch.Tensor,
    object_surface_normals_world: torch.Tensor,
    object_surface_points_part_ids: torch.Tensor,
    hand_verts: torch.Tensor,
    hand_normals: torch.Tensor,
    threshold: float = 0.01,
    dist_min: float = 0.0,
    dist_max: float = 100.0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Find contact points between object and hand with object part IDs.

    For each object vertex, find the closest hand vertex.

    Args:
        object_surface_points_world: Object surface points in world frame (N, 3).
        object_surface_normals_world: Object surface normals in world frame (N, 3), pointing inward.
        object_surface_points_part_ids: Part ID per object surface point (N,).
        hand_verts: Hand mesh vertices in world frame (M, 3).
        hand_normals: Hand mesh normals in world frame (M, 3), pointing outward.
        threshold: Max distance for a pair to count as contact.
        dist_min: Lower clip for distances.
        dist_max: Upper clip for distances.

    Returns:
        object_contact_points_world: (num_contact, 3) — world_xyz.
        object_contact_normals_world: (num_contact, 3) — world_normal.
        object_contact_part_ids: (num_contact,) — part_id.
        hand_contact_points_world: (num_contact, 3) — world_xyz.
        hand_contact_normals_world: (num_contact, 3) — world_normal.
        contact_dists: (num_contact,) — distance from object to hand.
    """
    dists = torch.clamp(
        torch.cdist(object_surface_points_world, hand_verts), min=dist_min, max=dist_max
    )  # (N, M)

    object_to_hand_closest_dist = dists.amin(dim=-1)  # (N,)
    contact_mask = object_to_hand_closest_dist < threshold  # (N,)

    closest_hand_idx = dists.argmin(dim=-1)  # (N,)
    closet_hand_verts = hand_verts[closest_hand_idx]  # (N, 3)
    closet_hand_normals = hand_normals[closest_hand_idx]  # (N, 3)

    if contact_mask.sum() == 0:
        return (
            torch.zeros((0, 3), device=object_surface_points_world.device),
            torch.zeros((0, 3), device=object_surface_points_world.device),
            torch.zeros((0, 1), device=object_surface_points_world.device),
            torch.zeros((0, 3), device=object_surface_points_world.device),
            torch.zeros((0, 3), device=object_surface_points_world.device),
            torch.zeros((0, 1), device=object_surface_points_world.device),
        )

    object_contact_points_world = object_surface_points_world[contact_mask]
    object_contact_normals_world = object_surface_normals_world[contact_mask]
    object_contact_part_ids = object_surface_points_part_ids[contact_mask]

    hand_contact_points_world = closet_hand_verts[contact_mask]
    hand_contact_normals_world = closet_hand_normals[contact_mask]
    contact_dists = object_to_hand_closest_dist[contact_mask]

    return (
        object_contact_points_world,
        object_contact_normals_world,
        object_contact_part_ids,
        hand_contact_points_world,
        hand_contact_normals_world,
        contact_dists,
    )


def compute_hand_link_contact_positions(
    joint_points: torch.Tensor,
    object_contact_part_ids: torch.Tensor,
    hand_contact_points_world: torch.Tensor,
    hand_contact_normals_world: torch.Tensor,
    contact_dists: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign contact points to MANO links and compute one contact position and part_id per link.

    For each link, average distance from each contact to the link's joints is
    computed; each contact is assigned to the closest link. For each link with
    at least one contact, returns weighted average position (by inverse
    distance) and voted part_id. Links with no contact get zeros (4,).

    Args:
        joint_points: (21, 3) MANO joint positions in world frame.
        object_contact_part_ids: (N,) object part IDs with values 1 or 2.
        hand_contact_points_world: (N, 3) hand contact points in world frame.
        hand_contact_normals_world: (N, 3) hand contact normals in world frame.
        contact_dists: (N,) distances from object to hand.

    Returns:
        hand_link_contact_positions: (NUM_MANO_LINKS, 3) contact positions in world frame.
        hand_link_contact_normals: (NUM_MANO_LINKS, 3) contact normals in world frame.
        hand_link_contact_part_ids: (NUM_MANO_LINKS,) contact part IDs.
    """
    # Compute MANO link position by averaging the joint positions.
    hand_link_positions = torch.zeros(
        (NUM_MANO_LINKS, 3), device=joint_points.device
    )  # (NUM_MANO_LINKS, 3)
    for link_idx, joint_idxs in enumerate(MANO_HAND_LINKS.values()):
        hand_link_positions[link_idx] = joint_points[joint_idxs].mean(dim=0)  # (3,)

    # Compute distance from contact points to link
    dists = torch.cdist(
        hand_contact_points_world, hand_link_positions
    )  # (N, NUM_MANO_LINKS)
    closest_link_idx = dists.argmin(dim=-1)  # (N,)

    # Average contact positions and vote part id
    hand_link_contact_positions = torch.zeros(
        (NUM_MANO_LINKS, 3), device=joint_points.device
    )  # (NUM_MANO_LINKS, 3)
    hand_link_contact_normals = torch.zeros(
        (NUM_MANO_LINKS, 3), device=joint_points.device
    )  # (NUM_MANO_LINKS, 3)
    hand_link_contact_part_ids = torch.zeros(
        (NUM_MANO_LINKS,), device=joint_points.device
    )  # (NUM_MANO_LINKS,)

    for link_idx in range(NUM_MANO_LINKS):
        mask = closest_link_idx == link_idx
        if not torch.any(mask):
            continue
        # Average contact positions by inverse distance
        weights = torch.nn.functional.softmax(-contact_dists[mask], dim=-1)
        hand_link_contact_positions[link_idx] = (
            hand_contact_points_world[mask] * weights[:, None]
        ).sum(dim=0)
        hand_link_contact_normals[link_idx] = (
            hand_contact_normals_world[mask] * weights[:, None]
        ).sum(dim=0)
        hand_link_contact_normals[link_idx] /= hand_link_contact_normals[
            link_idx
        ].norm()

        # Vote part id
        part_ids = object_contact_part_ids[mask]
        voted_part_id = part_ids.mode().values.item()
        hand_link_contact_part_ids[link_idx] = voted_part_id

    return (
        hand_link_contact_positions,
        hand_link_contact_normals,
        hand_link_contact_part_ids,
    )


def find_object_contact_positions(
    hand_link_contact_positions: torch.Tensor,
    object_surface_points: torch.Tensor,
    object_surface_normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign contact points to object vertices and normals per contact.

    The object contact positions are the closest object vertices to the hand link contact positions that
    has surface normal pointing towards the hand link contact positions.

    Args:
        hand_link_contact_positions: (NUM_MANO_LINKS, 3) link contact positions in world frame.
        object_surface_points: (N, 3) object vertices in world frame.
        object_surface_normals: (N, 3) object normals in world frame.

    Returns:
        object_contact_positions: (NUM_MANO_LINKS, 3) object contact positions in world frame.
        object_contact_normals: (NUM_MANO_LINKS, 3) object contact normals in world frame.
    """
    dists = torch.cdist(
        hand_link_contact_positions, object_surface_points
    )  # (NUM_MANO_LINKS, N)

    closest_object_idx = dists.argmin(dim=-1)  # (NUM_MANO_LINKS,)
    object_contact_positions = object_surface_points[
        closest_object_idx
    ]  # (NUM_MANO_LINKS, 3)
    object_contact_normals = object_surface_normals[
        closest_object_idx
    ]  # (NUM_MANO_LINKS, 3)

    # Set invalid contacts to zero.
    invalid_contact_mask = (
        hand_link_contact_positions.norm(dim=-1) < 1e-3
    )  # (NUM_MANO_LINKS,)
    object_contact_positions[invalid_contact_mask] = 0.0
    object_contact_normals[invalid_contact_mask] = 0.0

    return object_contact_positions, object_contact_normals
