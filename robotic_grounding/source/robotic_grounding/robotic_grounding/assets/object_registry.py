from __future__ import annotations

from dataclasses import dataclass

from robotic_grounding.assets import OBJECTS_ASSET_DIR


@dataclass
class ObjectSpec:
    """Specification for a loadable object (USD path and scale)."""

    usd_path: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


OBJECT_REGISTRY: dict[str, ObjectSpec] = {
    "apple": ObjectSpec(
        usd_path=f"{OBJECTS_ASSET_DIR}/apple/apple_simple.usda",
        scale=(0.8, 0.8, 0.8),
    ),
    "object": ObjectSpec(
        usd_path=f"{OBJECTS_ASSET_DIR}/apple/apple_simple.usda",
        scale=(0.8, 0.8, 0.8),
    ),
    "apple_green": ObjectSpec(
        usd_path=f"{OBJECTS_ASSET_DIR}/apple/apple_simple_green.usda",
        scale=(0.8, 0.8, 0.8),
    ),
    "table": ObjectSpec(
        usd_path=f"{OBJECTS_ASSET_DIR}/ikea_4x2_shelf/ikea_4x2_shelf.usd",
    ),
}


def get_object_spec(name: str) -> ObjectSpec | None:
    """Return the object spec for the given name, or None if not found."""
    return OBJECT_REGISTRY.get(name)
