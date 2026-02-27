# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Contact computation utilities for MANO hand–object contact (dexmachina format)."""

import numpy as np
from sklearn.neighbors import KDTree

from robotic_grounding.retarget.params import MANO_HAND_LINKS, NUM_MANO_LINKS


def approximate_contact_with_id(
    obj_verts: np.ndarray,
    obj_part_ids: np.ndarray,
    hand_verts: np.ndarray,
    threshold: float = 0.01,
    dist_min: float = 0.0,
    dist_max: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find contact points between object and hand with object part IDs.

    For each object vertex, find the closest hand vertex. If within threshold,
    record contact on object and on hand, with part_id from the object vertex.
    Part IDs are 1 or 2 (e.g. top/bottom).

    Args:
        obj_verts: Object vertices in world frame (N, 3).
        obj_part_ids: Part ID per object vertex (N,) with values 1 or 2.
        hand_verts: Hand mesh vertices in world frame (M, 3).
        threshold: Max distance for a pair to count as contact.
        dist_min: Lower clip for distances.
        dist_max: Upper clip for distances.

    Returns:
        contact_on_obj: (num_contact, 4) — xyz + part_id.
        contact_on_hand: (num_contact, 4) — xyz + part_id (from object).
    """
    tree_hand = KDTree(hand_verts)
    dist, idx = tree_hand.query(obj_verts, k=1)
    dist = np.clip(dist, dist_min, dist_max)
    if dist.ndim == 2:
        dist = dist.squeeze(axis=1)
        idx = idx.squeeze(axis=1)
    contact_mask = dist < threshold
    if not np.any(contact_mask):
        return np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)

    contact_on_hand = hand_verts[idx[contact_mask]]
    contact_on_obj = obj_verts[contact_mask]
    part_id_on_obj = obj_part_ids[contact_mask]

    return (
        np.concatenate([contact_on_obj, part_id_on_obj[:, None]], axis=-1),
        np.concatenate([contact_on_hand, part_id_on_obj[:, None]], axis=-1),
    )


def find_link_contact_positions(
    contact_points: np.ndarray,
    joint_points: np.ndarray,
) -> list[np.ndarray]:
    """Assign contact points to MANO links and compute one (x,y,z,part_id) per link.

    For each link, average distance from each contact to the link's joints is
    computed; each contact is assigned to the closest link. For each link with
    at least one contact, returns weighted average position (by inverse
    distance) and voted part_id. Links with no contact get zeros (4,).

    Args:
        contact_points: (N, 3) or (N, 4) contact positions (last dim may be part_id).
        joint_points: (21, 3) MANO joint positions in world frame.

    Returns:
        List of length NUM_MANO_LINKS; each element is (4,) — xyz + part_id, or zeros.
    """
    if contact_points.size == 0:
        return [np.zeros(4, dtype=np.float64) for _ in range(NUM_MANO_LINKS)]

    if contact_points.ndim == 1:
        contact_points = contact_points.reshape(1, -1)
    if contact_points.shape[1] == 3:
        contact_points = np.concatenate(
            [contact_points, np.zeros((contact_points.shape[0], 1))], axis=-1
        )

    dist_links_to_contacts = []
    for _link_name, joint_idxs in MANO_HAND_LINKS.items():
        dists = []  # (num_joints, num_contacts)
        for joint_idx in joint_idxs:
            joint_point = joint_points[int(joint_idx)]
            # Compute distance from contact to one joint
            d = np.linalg.norm(contact_points[:, :3] - joint_point, axis=-1)
            dists.append(d)
        dist_link_to_contacts = np.mean(np.stack(dists, axis=0), axis=0)
        dist_links_to_contacts.append(dist_link_to_contacts)
    dist_links_to_contacts = np.stack(
        dist_links_to_contacts, axis=0
    )  # (NUM_MANO_LINKS, num_contacts)

    closest_link_idx = np.argmin(dist_links_to_contacts, axis=0)  # (num_contacts,)
    closest_link_dist = np.min(dist_links_to_contacts, axis=0)  # (num_contacts,)
    avg_contact_positions = [np.zeros(4) for _ in range(NUM_MANO_LINKS)]

    for idx, (_link_name, _joint_idxs) in enumerate(MANO_HAND_LINKS.items()):
        mask = closest_link_idx == idx
        if not np.any(mask):
            continue
        part_ids = contact_points[mask, 3]
        voted_part_id = np.argmax(np.bincount(part_ids.astype(int)))
        weights = 1.0 / (np.clip(closest_link_dist[mask], 1e-8, None))
        weighted_avg = np.average(contact_points[mask][:, :3], axis=0, weights=weights)
        avg_contact_positions[idx] = np.concatenate([weighted_avg, [voted_part_id]])

    return np.asarray(avg_contact_positions)


def find_object_contact_positions(
    link_contact_positions: np.ndarray,
    obj_verts: np.ndarray,
    obj_part_ids: np.ndarray,
) -> np.ndarray:
    """Assign contact points to object vertices and parts and compute one (x,y,z,part_id) per contact.

    All inputs and outputs are in world frame.

    Args:
        link_contact_positions: (NUM_MANO_LINKS, 4) link contact positions in world frame (xyz + part_id).
        obj_verts: (N, 3) object vertices in world frame.
        obj_part_ids: (N,) object part IDs with values 1 or 2.

    Returns:
        (NUM_MANO_LINKS, 4) object contact positions in world frame (xyz + part_id).
    """
    object_contact_positions = np.zeros_like(link_contact_positions)
    for contact_idx, link_contact_position in enumerate(link_contact_positions):
        if link_contact_position.sum() > 0.0:
            closest_vertex_idx = np.argmin(
                np.linalg.norm(obj_verts - link_contact_position[:3], axis=-1)
            )
            closest_vertex_position = obj_verts[closest_vertex_idx]
            closest_part_idx = obj_part_ids[closest_vertex_idx]
            object_contact_positions[contact_idx] = np.concatenate(
                [closest_vertex_position, [closest_part_idx]]
            )
    return object_contact_positions
