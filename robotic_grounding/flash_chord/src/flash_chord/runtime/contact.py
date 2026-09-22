# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode Newton rigid contacts to world-frame arrays.

Geometry (position/normal/bodies) is produced by the collision pipeline
(``model.collide``). ``contact_force_w`` additionally requires the solver's
``update_contacts(contacts, state)`` to have run with the ``"force"`` contact
attribute requested (``builder.request_contact_attributes("force")``); otherwise it
is zero. Host-side (numpy) readout for tests, wiring, and debug viz; the
per-(hand-link, object-body) on-device bucketing comes with the env.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout
from flash_chord.embodiments.frames import DeviceBodyFrameMap
from flash_chord.utils.quat import quat_mul_xyzw, quat_rotate_xyzw


@dataclass
class ContactReadout:
    """Active rigid contacts in world frame (length ``n``).

    ``contact_normal_w`` points shape0 -> shape1; ``contact_force_w`` /
    ``contact_torque_w`` act on shape0 (reaction on shape1 is the negative).
    """

    body0_ids: np.ndarray  # (n,) body of shape0 (-1 = world)
    body1_ids: np.ndarray  # (n,) body of shape1 (-1 = world)
    contact_pos_w: np.ndarray  # (n, 3) world contact point [m]
    contact_normal_w: np.ndarray  # (n, 3) unit normal, shape0 -> shape1
    contact_force_w: np.ndarray  # (n, 3) linear contact force on shape0 [N]
    contact_torque_w: np.ndarray  # (n, 3) contact torque on shape0 [N·m]

    def __len__(self) -> int:
        return len(self.body0_ids)


def read_contacts_w(model, state, contacts) -> ContactReadout:
    """Decode the active rigid contacts of ``contacts`` into world-frame arrays."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        empty3 = np.zeros((0, 3))
        empty_id = np.zeros((0,), dtype=int)
        return ContactReadout(empty_id, empty_id, empty3, empty3, empty3, empty3)

    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]
    point0 = contacts.rigid_contact_point0.numpy()[:n]  # body-frame point on shape0
    normal = contacts.rigid_contact_normal.numpy()[:n]
    shape_body = np.asarray(model.shape_body.numpy() if hasattr(model.shape_body, "numpy") else model.shape_body)
    body_q = state.body_q.numpy()  # (num_bodies, 7): pos(3) + quat xyzw(4)

    force_attr = getattr(contacts, "force", None)
    force6 = force_attr.numpy()[:n] if force_attr is not None else np.zeros((n, 6))

    def body_of(shape: int) -> int:
        return int(shape_body[shape]) if 0 <= shape < len(shape_body) else -1

    body0_ids = np.array([body_of(int(s)) for s in shape0], dtype=int)
    body1_ids = np.array([body_of(int(s)) for s in shape1], dtype=int)

    pos_w = np.zeros((n, 3))
    for i in range(n):
        b0 = int(body0_ids[i])
        if b0 < 0:  # world body: shape-0 point is already in world frame
            pos_w[i] = point0[i]
        else:
            xf = body_q[b0]
            pos_w[i] = xf[:3] + quat_rotate_xyzw(np.asarray(xf[3:7]), np.asarray(point0[i]))

    return ContactReadout(
        body0_ids=body0_ids,
        body1_ids=body1_ids,
        contact_pos_w=pos_w,
        contact_normal_w=np.asarray(normal, dtype=float),
        contact_force_w=np.asarray(force6[:, :3], dtype=float),
        contact_torque_w=np.asarray(force6[:, 3:6], dtype=float),
    )


def contact_gaps_w(model, state, contacts) -> np.ndarray:
    """Signed separation per active contact along the normal (negative = penetration depth) [m].

    Gap = ``(point1_w - point0_w) · normal``; the contact points are the closest points on each shape,
    so a negative value is how deep the shapes interpenetrate. Empty if there are no contacts.
    """
    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        return np.zeros((0,))
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]
    point0 = contacts.rigid_contact_point0.numpy()[:n]  # shape0 body-frame
    point1 = contacts.rigid_contact_point1.numpy()[:n]  # shape1 body-frame
    normal = contacts.rigid_contact_normal.numpy()[:n]
    shape_body = np.asarray(model.shape_body.numpy() if hasattr(model.shape_body, "numpy") else model.shape_body)
    body_q = state.body_q.numpy()

    def to_world(pt, shape):
        b = int(shape_body[shape]) if 0 <= shape < len(shape_body) else -1
        if b < 0:
            return np.asarray(pt, dtype=float)
        xf = body_q[b]
        return xf[:3] + quat_rotate_xyzw(np.asarray(xf[3:7]), np.asarray(pt))

    return np.array(
        [float(np.dot(normal[i], to_world(point1[i], shape1[i]) - to_world(point0[i], shape0[i]))) for i in range(n)]
    )


@dataclass(frozen=True)
class ContactTrackerConfig:
    """Live contact aggregation parameters.

    ``force_threshold`` is accepted for resolved-config compatibility but does not gate
    contact presence; IsaacLab semantics use the number of raw simulator contacts.
    """

    force_threshold: float = 0.0
    sides: tuple[str, ...] = ("right", "left")

    def __post_init__(self) -> None:
        object.__setattr__(self, "sides", tuple(self.sides))
        if not math.isfinite(self.force_threshold) or self.force_threshold < 0.0:
            raise ValueError(f"force_threshold must be finite and non-negative, got {self.force_threshold}")
        if not self.sides or len(set(self.sides)) != len(self.sides):
            raise ValueError(f"contact sides must be unique and nonempty, got {self.sides}")


@dataclass(frozen=True)
class ContactLayout:
    """Per-world hand-link/object-body contact slot mapping."""

    sides: tuple[str, ...]
    num_hands: int
    num_objects: int
    slots_per_world: int
    hand_slot_starts: tuple[int, ...]
    link_counts: tuple[int, ...]
    wrist_frames: tuple[BodyFrame, ...]
    object_body_ids: tuple[int, ...]
    shapes_per_world: int
    shape_hand_ids: tuple[int, ...]
    shape_link_ids: tuple[int, ...]
    body_object_ids: tuple[int, ...]
    slot_hand_ids: tuple[int, ...]
    slot_object_ids: tuple[int, ...]

    @classmethod
    def build(
        cls,
        embodiment: EmbodimentLayout,
        object_body_ids: tuple[int, ...],
        bodies_per_world: int,
        shapes_per_world: int,
        sides: tuple[str, ...] = ("right", "left"),
    ) -> "ContactLayout":
        """Derive contact roles and flattened slot offsets from scene layouts."""
        if set(sides) != set(embodiment.sides) or len(sides) != len(embodiment.sides):
            raise ValueError(f"contact sides {sides} must match embodiment sides {embodiment.sides}")
        if bodies_per_world <= 0 or shapes_per_world <= 0:
            raise ValueError(
                f"contact layout requires positive bodies/shapes per world, got {bodies_per_world}/{shapes_per_world}"
            )

        shape_hand_ids = [-1] * shapes_per_world
        shape_link_ids = [-1] * shapes_per_world
        hand_slot_starts: list[int] = []
        link_counts: list[int] = []
        wrist_frames: list[BodyFrame] = []
        slot_hand_ids: list[int] = []
        slot_object_ids: list[int] = []
        slots_per_world = 0
        for hand_id, side in enumerate(sides):
            hand = embodiment.hand(side)
            hand_slot_starts.append(slots_per_world)
            link_counts.append(hand.contact_link_count)
            if hand.palm_frame.body_id >= bodies_per_world:
                raise ValueError(f"{side} palm body {hand.palm_frame.body_id} is outside the per-world body range")
            wrist_frames.append(hand.palm_frame)
            for link_id, link in enumerate(hand.contact_links):
                for shape_id in link.shape_ids:
                    if shape_id >= shapes_per_world:
                        raise ValueError(f"{side} contact shape {shape_id} is outside the per-world shape range")
                    if shape_hand_ids[shape_id] >= 0:
                        raise ValueError(f"shape {shape_id} is mapped to more than one hand contact link")
                    shape_hand_ids[shape_id] = hand_id
                    shape_link_ids[shape_id] = link_id
            for object_id in range(len(object_body_ids)):
                slot_hand_ids.extend([hand_id] * hand.contact_link_count)
                slot_object_ids.extend([object_id] * hand.contact_link_count)
            slots_per_world += len(object_body_ids) * hand.contact_link_count

        body_object_ids = [-1] * bodies_per_world
        for object_id, body_id in enumerate(object_body_ids):
            if body_id < 0 or body_id >= bodies_per_world:
                raise ValueError(f"object body {body_id} is outside the per-world body range")
            if body_object_ids[body_id] >= 0:
                raise ValueError(f"body {body_id} is mapped to more than one object")
            body_object_ids[body_id] = object_id

        return cls(
            sides=sides,
            num_hands=len(sides),
            num_objects=len(object_body_ids),
            slots_per_world=slots_per_world,
            hand_slot_starts=tuple(hand_slot_starts),
            link_counts=tuple(link_counts),
            wrist_frames=tuple(wrist_frames),
            object_body_ids=object_body_ids,
            shapes_per_world=shapes_per_world,
            shape_hand_ids=tuple(shape_hand_ids),
            shape_link_ids=tuple(shape_link_ids),
            body_object_ids=tuple(body_object_ids),
            slot_hand_ids=tuple(slot_hand_ids),
            slot_object_ids=tuple(slot_object_ids),
        )


@wp.kernel
def clear_contact_slots(
    position_sum_w: wp.array(dtype=wp.vec3),
    force_sum_w: wp.array(dtype=wp.vec3),
    raw_contact_count: wp.array(dtype=wp.int32),
) -> None:
    """Clear one aggregation slot."""
    slot = wp.tid()
    position_sum_w[slot] = wp.vec3(0.0)
    force_sum_w[slot] = wp.vec3(0.0)
    raw_contact_count[slot] = 0


@wp.kernel
def aggregate_hand_object_contacts(
    contact_count: wp.array(dtype=wp.int32),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_force: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=wp.int32),
    replicated_shape_count: int,
    shapes_per_world: int,
    body_q: wp.array(dtype=wp.transform),
    bodies_per_world: int,
    slots_per_world: int,
    shape_hand_ids: wp.array(dtype=wp.int32),
    shape_link_ids: wp.array(dtype=wp.int32),
    body_object_ids: wp.array(dtype=wp.int32),
    hand_slot_starts: wp.array(dtype=wp.int32),
    link_counts: wp.array(dtype=wp.int32),
    position_sum_w: wp.array(dtype=wp.vec3),
    force_sum_w: wp.array(dtype=wp.vec3),
    raw_contact_count: wp.array(dtype=wp.int32),
) -> None:
    """Accumulate raw simulator contacts into IsaacLab-compatible fixed slots."""
    contact_id = wp.tid()
    if contact_id >= contact_count[0]:
        return

    shape0 = contact_shape0[contact_id]
    shape1 = contact_shape1[contact_id]
    if shape0 < 0 or shape1 < 0:
        return
    if shape0 < 0 or shape0 >= replicated_shape_count or shape1 < 0 or shape1 >= replicated_shape_count:
        return
    shape_world0 = shape0 // shapes_per_world
    shape_world1 = shape1 // shapes_per_world
    if shape_world0 != shape_world1:
        return
    local_shape0 = shape0 - shape_world0 * shapes_per_world
    local_shape1 = shape1 - shape_world1 * shapes_per_world
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    if body0 < 0 or body1 < 0:
        return
    world0 = body0 // bodies_per_world
    world1 = body1 // bodies_per_world
    if world0 != world1 or world0 != shape_world0:
        return

    local0 = body0 - world0 * bodies_per_world
    local1 = body1 - world1 * bodies_per_world
    hand = int(-1)
    link = int(-1)
    object_id = int(-1)
    object_body = int(-1)
    point_object = wp.vec3(0.0)
    force_object_w = wp.vec3(0.0)
    force0_w = wp.spatial_top(contact_force[contact_id])
    if shape_hand_ids[local_shape0] >= 0 and body_object_ids[local1] >= 0:
        hand = shape_hand_ids[local_shape0]
        link = shape_link_ids[local_shape0]
        object_id = body_object_ids[local1]
        object_body = body1
        point_object = contact_point1[contact_id]
        force_object_w = -force0_w
    elif body_object_ids[local0] >= 0 and shape_hand_ids[local_shape1] >= 0:
        hand = shape_hand_ids[local_shape1]
        link = shape_link_ids[local_shape1]
        object_id = body_object_ids[local0]
        object_body = body0
        point_object = contact_point0[contact_id]
        force_object_w = force0_w
    else:
        return

    point_w = wp.transform_point(body_q[object_body], point_object)
    slot = world0 * slots_per_world + hand_slot_starts[hand] + object_id * link_counts[hand] + link
    wp.atomic_add(position_sum_w, slot, point_w)
    wp.atomic_add(force_sum_w, slot, force_object_w)
    wp.atomic_add(raw_contact_count, slot, 1)


@wp.kernel
def finalize_contact_slots(
    body_q: wp.array(dtype=wp.transform),
    bodies_per_world: int,
    slots_per_world: int,
    slot_hand_ids: wp.array(dtype=wp.int32),
    slot_object_ids: wp.array(dtype=wp.int32),
    wrist_frame_body_ids: wp.array(dtype=wp.int32),
    wrist_body_to_frame_pos: wp.array(dtype=wp.vec3),
    wrist_body_to_frame_quat: wp.array(dtype=wp.quat),
    object_body_ids: wp.array(dtype=wp.int32),
    position_sum_w: wp.array(dtype=wp.vec3),
    force_sum_w: wp.array(dtype=wp.vec3),
    raw_contact_count: wp.array(dtype=wp.int32),
    contact_pos_w: wp.array(dtype=wp.vec3),
    contact_force_w: wp.array(dtype=wp.vec3),
    contact_pos_b: wp.array(dtype=wp.vec3),
    contact_force_direction_b: wp.array(dtype=wp.vec3),
    contact_pos_o: wp.array(dtype=wp.vec3),
    contact_force_direction_o: wp.array(dtype=wp.vec3),
    contact_active: wp.array(dtype=wp.int32),
) -> None:
    """Normalize aggregate slots and transform them into wrist/object frames."""
    slot = wp.tid()
    world = slot // slots_per_world
    local_slot = slot - world * slots_per_world
    hand = slot_hand_ids[local_slot]
    object_id = slot_object_ids[local_slot]
    force_w = force_sum_w[slot]
    active = raw_contact_count[slot] > 0

    contact_pos_w[slot] = wp.vec3(0.0)
    contact_force_w[slot] = wp.vec3(0.0)
    contact_pos_b[slot] = wp.vec3(0.0)
    contact_force_direction_b[slot] = wp.vec3(0.0)
    contact_pos_o[slot] = wp.vec3(0.0)
    contact_force_direction_o[slot] = wp.vec3(0.0)
    contact_active[slot] = 0
    if not active:
        return

    pos_w = position_sum_w[slot] / float(raw_contact_count[slot])
    direction_w = wp.vec3(0.0)
    if wp.length(force_w) > 1.0e-8:
        direction_w = wp.normalize(force_w)
    wrist_body_xf = body_q[world * bodies_per_world + wrist_frame_body_ids[hand]]
    wrist_xf = wp.transform(
        wp.transform_point(wrist_body_xf, wrist_body_to_frame_pos[hand]),
        wp.normalize(
            quat_mul_xyzw(
                wp.transform_get_rotation(wrist_body_xf),
                wrist_body_to_frame_quat[hand],
            )
        ),
    )
    object_xf = body_q[world * bodies_per_world + object_body_ids[object_id]]
    wrist_inv = wp.transform_inverse(wrist_xf)
    object_inv = wp.transform_inverse(object_xf)
    contact_pos_w[slot] = pos_w
    contact_force_w[slot] = force_w
    contact_pos_b[slot] = wp.transform_point(wrist_inv, pos_w)
    contact_force_direction_b[slot] = wp.transform_vector(wrist_inv, direction_w)
    contact_pos_o[slot] = wp.transform_point(object_inv, pos_w)
    contact_force_direction_o[slot] = wp.transform_vector(object_inv, direction_w)
    contact_active[slot] = 1


@dataclass
class ContactTracker:
    """Fixed-slot live hand/object contacts and frame-transformed aggregates."""

    layout: ContactLayout
    config: ContactTrackerConfig
    world_count: int
    bodies_per_world: int
    shapes_per_world: int
    hand_slot_starts: wp.array
    link_counts: wp.array
    wrist_frames: DeviceBodyFrameMap
    object_body_ids: wp.array
    shape_hand_ids: wp.array
    shape_link_ids: wp.array
    body_object_ids: wp.array
    slot_hand_ids: wp.array
    slot_object_ids: wp.array
    position_sum_w: wp.array
    force_sum_w: wp.array
    raw_contact_count: wp.array
    contact_pos_w: wp.array
    contact_force_w: wp.array
    contact_pos_b: wp.array
    contact_force_direction_b: wp.array
    contact_pos_o: wp.array
    contact_force_direction_o: wp.array
    contact_active: wp.array

    @classmethod
    def build(
        cls,
        embodiment: EmbodimentLayout,
        object_body_ids: tuple[int, ...],
        world_count: int,
        bodies_per_world: int,
        shapes_per_world: int,
        config: ContactTrackerConfig | None = None,
        sides: tuple[str, ...] | None = None,
        device=None,
    ) -> "ContactTracker":
        """Allocate contact mappings and fixed output slots."""
        config = config or ContactTrackerConfig()
        sides = config.sides if sides is None else sides
        layout = ContactLayout.build(
            embodiment,
            object_body_ids,
            bodies_per_world,
            shapes_per_world,
            sides=sides,
        )
        i32 = lambda values: wp.array(values, dtype=wp.int32, device=device)  # noqa: E731
        vec3 = lambda: wp.zeros(world_count * layout.slots_per_world, dtype=wp.vec3, device=device)  # noqa: E731
        i32_slots = lambda: wp.zeros(  # noqa: E731
            world_count * layout.slots_per_world,
            dtype=wp.int32,
            device=device,
        )
        return cls(
            layout=layout,
            config=config,
            world_count=world_count,
            bodies_per_world=bodies_per_world,
            shapes_per_world=shapes_per_world,
            hand_slot_starts=i32(layout.hand_slot_starts),
            link_counts=i32(layout.link_counts),
            wrist_frames=DeviceBodyFrameMap.build(layout.wrist_frames, device=device),
            object_body_ids=i32(layout.object_body_ids),
            shape_hand_ids=i32(layout.shape_hand_ids),
            shape_link_ids=i32(layout.shape_link_ids),
            body_object_ids=i32(layout.body_object_ids),
            slot_hand_ids=i32(layout.slot_hand_ids),
            slot_object_ids=i32(layout.slot_object_ids),
            position_sum_w=vec3(),
            force_sum_w=vec3(),
            raw_contact_count=i32_slots(),
            contact_pos_w=vec3(),
            contact_force_w=vec3(),
            contact_pos_b=vec3(),
            contact_force_direction_b=vec3(),
            contact_pos_o=vec3(),
            contact_force_direction_o=vec3(),
            contact_active=wp.zeros(
                world_count * layout.slots_per_world,
                dtype=wp.int32,
                device=device,
            ),
        )

    def update(self, model, state, contacts) -> None:
        """Aggregate current MuJoCo contacts into persistent fixed slots."""
        num_slots = self.world_count * self.layout.slots_per_world
        if num_slots == 0:
            return
        wp.launch(
            clear_contact_slots,
            dim=num_slots,
            outputs=[self.position_sum_w, self.force_sum_w, self.raw_contact_count],
        )
        wp.launch(
            aggregate_hand_object_contacts,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.force,
                model.shape_body,
                self.world_count * self.shapes_per_world,
                self.shapes_per_world,
                state.body_q,
                self.bodies_per_world,
                self.layout.slots_per_world,
                self.shape_hand_ids,
                self.shape_link_ids,
                self.body_object_ids,
                self.hand_slot_starts,
                self.link_counts,
            ],
            outputs=[
                self.position_sum_w,
                self.force_sum_w,
                self.raw_contact_count,
            ],
        )
        wp.launch(
            finalize_contact_slots,
            dim=num_slots,
            inputs=[
                state.body_q,
                self.bodies_per_world,
                self.layout.slots_per_world,
                self.slot_hand_ids,
                self.slot_object_ids,
                self.wrist_frames.body_ids,
                self.wrist_frames.body_to_frame_pos,
                self.wrist_frames.body_to_frame_quat,
                self.object_body_ids,
                self.position_sum_w,
                self.force_sum_w,
                self.raw_contact_count,
            ],
            outputs=[
                self.contact_pos_w,
                self.contact_force_w,
                self.contact_pos_b,
                self.contact_force_direction_b,
                self.contact_pos_o,
                self.contact_force_direction_o,
                self.contact_active,
            ],
        )
