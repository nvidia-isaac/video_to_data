# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from robotic_grounding.assets import ASSET_DIR


@dataclass
class ObjectSpec:
    """Specification for a loadable object."""

    usd_path: str | None = None
    urdf_path: str | None = None
    rigid_urdf_path: str | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    articulated: bool = False


_ARCTIC_URDF_DIR = f"{ASSET_DIR}/urdfs/arctic"

OBJECT_REGISTRY: dict[str, ObjectSpec] = {
    # Arctic objects (articulated URDFs, base joints removed)
    "box": ObjectSpec(
        urdf_path=f"{_ARCTIC_URDF_DIR}/box_art.urdf",
        rigid_urdf_path=f"{_ARCTIC_URDF_DIR}/box_rigid.urdf",
        articulated=True,
    ),
    "capsulemachine": ObjectSpec(
        urdf_path=f"{_ARCTIC_URDF_DIR}/capsulemachine_art.urdf",
        rigid_urdf_path=f"{_ARCTIC_URDF_DIR}/capsulemachine_rigid.urdf",
        articulated=True,
    ),
    "espressomachine": ObjectSpec(
        urdf_path=f"{_ARCTIC_URDF_DIR}/espressomachine_art.urdf",
        rigid_urdf_path=f"{_ARCTIC_URDF_DIR}/espressomachine_rigid.urdf",
        articulated=True,
    ),
    "ketchup": ObjectSpec(
        urdf_path=f"{_ARCTIC_URDF_DIR}/ketchup_art.urdf",
        rigid_urdf_path=f"{_ARCTIC_URDF_DIR}/ketchup_rigid.urdf",
        articulated=True,
    ),
    "laptop": ObjectSpec(
        urdf_path=f"{_ARCTIC_URDF_DIR}/laptop_art.urdf",
        rigid_urdf_path=f"{_ARCTIC_URDF_DIR}/laptop_rigid.urdf",
        articulated=True,
    ),
    "microwave": ObjectSpec(
        urdf_path=f"{_ARCTIC_URDF_DIR}/microwave_art.urdf",
        rigid_urdf_path=f"{_ARCTIC_URDF_DIR}/microwave_rigid.urdf",
        articulated=True,
    ),
    "mixer": ObjectSpec(
        urdf_path=f"{_ARCTIC_URDF_DIR}/mixer_art.urdf",
        rigid_urdf_path=f"{_ARCTIC_URDF_DIR}/mixer_rigid.urdf",
        articulated=True,
    ),
    "notebook": ObjectSpec(
        urdf_path=f"{_ARCTIC_URDF_DIR}/notebook_art.urdf",
        rigid_urdf_path=f"{_ARCTIC_URDF_DIR}/notebook_rigid.urdf",
        articulated=True,
    ),
    "phone": ObjectSpec(
        urdf_path=f"{_ARCTIC_URDF_DIR}/phone_art.urdf",
        rigid_urdf_path=f"{_ARCTIC_URDF_DIR}/phone_rigid.urdf",
        articulated=True,
    ),
    "waffleiron": ObjectSpec(
        urdf_path=f"{_ARCTIC_URDF_DIR}/waffleiron_art.urdf",
        rigid_urdf_path=f"{_ARCTIC_URDF_DIR}/waffleiron_rigid.urdf",
        articulated=True,
    ),
}


def get_object_spec(name: str) -> ObjectSpec | None:
    """Return the object spec for the given name, or None if not found."""
    return OBJECT_REGISTRY.get(name)
