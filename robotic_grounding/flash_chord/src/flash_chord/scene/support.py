# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add a sequence's support surfaces (kinematic collision shapes) from its support USDA.

The support USDA holds inline ``Cylinder`` / ``Cube`` prims (a translate + dimensions). They
are added as static (kinematic) collision shapes the objects rest on. Newton cylinders extend
along Z, matching the USDA's ``axis = "Z"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import newton

from flash_chord.scene.collision import ShapeSpan


@dataclass(frozen=True)
class SupportBinding:
    """Resolved body, free-root coordinates, and shapes owned by one support component."""

    body_id: int
    free_q_ids: tuple[int, ...]
    free_dof_ids: tuple[int, ...]
    shapes: ShapeSpan

    def __post_init__(self) -> None:
        object.__setattr__(self, "free_q_ids", tuple(self.free_q_ids))
        object.__setattr__(self, "free_dof_ids", tuple(self.free_dof_ids))
        if self.body_id < 0:
            raise ValueError("support body ID must be nonnegative")
        if (
            len(self.free_q_ids) != 7
            or self.free_q_ids != tuple(range(self.free_q_ids[0], self.free_q_ids[0] + 7))
            or self.free_q_ids[0] < 0
        ):
            raise ValueError("support must bind exactly seven contiguous nonnegative free-joint q IDs")
        if (
            len(self.free_dof_ids) != 6
            or self.free_dof_ids != tuple(range(self.free_dof_ids[0], self.free_dof_ids[0] + 6))
            or self.free_dof_ids[0] < 0
        ):
            raise ValueError("support must bind exactly six contiguous nonnegative free-joint DOF IDs")
        if not len(self.shapes):
            raise ValueError("support must own at least one shape")


def add_support_surfaces(builder: newton.ModelBuilder, support_usda_path) -> SupportBinding:
    """Add the support USDA's prims as kinematic collision shapes on one static body.

    Returns the exact component topology. An explicitly requested missing/invalid file fails.
    """
    path = Path(support_usda_path)
    if not path.is_file():
        raise FileNotFoundError(f"support surface does not exist: {path}")

    import warp as wp
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"failed to open support surface: {path}")
    shapes = []
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name not in ("Cylinder", "Cube"):
            continue
        translate = prim.GetAttribute("xformOp:translate").Get()
        position = (
            (0.0, 0.0, 0.0)
            if translate is None
            else (float(translate[0]), float(translate[1]), float(translate[2]))
        )
        if not all(math.isfinite(value) for value in position):
            raise ValueError(f"support shape {prim.GetPath()} has a non-finite translation")
        if type_name == "Cylinder":
            dimensions = (
                float(prim.GetAttribute("radius").Get()),
                float(prim.GetAttribute("height").Get()),
            )
        else:
            size = prim.GetAttribute("size").Get()
            dimensions = (float(size if size is not None else 2.0),)
        if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
            raise ValueError(f"support shape {prim.GetPath()} must have positive finite dimensions")
        shapes.append((type_name, prim.GetName(), position, dimensions))
    if not shapes:
        raise ValueError(f"support surface contains no Cylinder or Cube collision shapes: {path}")

    body_start = builder.body_count
    joint_start = builder.joint_count
    q_start = builder.joint_coord_count
    dof_start = builder.joint_dof_count
    shape_start = builder.shape_count
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        is_kinematic=True,
        label="support",
    )
    for type_name, name, position, dimensions in shapes:
        xform = wp.transform(wp.vec3(*position), wp.quat_identity())
        if type_name == "Cylinder":
            radius, height = dimensions
            builder.add_shape_cylinder(
                body,
                xform=xform,
                radius=radius,
                half_height=0.5 * height,
                label=name,
            )
        else:  # Cube: USD `size` is the full edge length
            half = 0.5 * dimensions[0]
            builder.add_shape_box(body, xform=xform, hx=half, hy=half, hz=half, label=name)
    if (
        body != body_start
        or builder.body_count != body_start + 1
        or builder.joint_count != joint_start + 1
        or builder.joint_coord_count != q_start + 7
        or builder.joint_dof_count != dof_start + 6
    ):
        raise RuntimeError("support body creation did not append one Newton free-root component")
    joint = joint_start
    if (
        builder.joint_type[joint] != newton.JointType.FREE
        or builder.joint_parent[joint] != -1
        or builder.joint_child[joint] != body
        or builder.joint_q_start[joint] != q_start
        or builder.joint_qd_start[joint] != dof_start
    ):
        raise RuntimeError("support body does not expose the expected world-parented Newton free joint")
    return SupportBinding(
        body_id=body,
        free_q_ids=tuple(range(q_start, q_start + 7)),
        free_dof_ids=tuple(range(dof_start, dof_start + 6)),
        shapes=ShapeSpan(shape_start, builder.shape_count),
    )
