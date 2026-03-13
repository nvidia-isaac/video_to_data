"""Joint ordering registry for different robots."""

from __future__ import annotations

from robotic_grounding.assets.g1 import (
    DEX3_PARQUET_JOINT_ORDER as G1_DEX3_PARQUET_JOINT_ORDER,
)
from robotic_grounding.assets.g1 import MUJOCO_JOINT_ORDER as G1_MUJOCO_JOINT_ORDER

JOINT_ORDER_REGISTRY: dict[str, dict[str, list[str]]] = {
    "g1": {
        "mujoco": G1_MUJOCO_JOINT_ORDER,
    },
    "g1_dex3": {
        "parquet": G1_DEX3_PARQUET_JOINT_ORDER,
    },
    "dex3": {
        "parquet": G1_DEX3_PARQUET_JOINT_ORDER,
    },
}


def get_joint_order(robot_type: str, ordering: str) -> list[str] | None:
    """Get joint ordering for a robot type.

    Args:
        robot_type: Robot type (e.g., "g1", "h1").
        ordering: Ordering name (e.g., "mujoco", "isaaclab").

    Returns:
        List of joint names in the specified order, or None if not found.
    """
    robot_orders = JOINT_ORDER_REGISTRY.get(robot_type)
    if robot_orders is None:
        return None
    return robot_orders.get(ordering)
