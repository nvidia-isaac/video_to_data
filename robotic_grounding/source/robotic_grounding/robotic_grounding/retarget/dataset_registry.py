# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Central dataset registry — single source of truth for dataset properties.

Every script that needs to know about datasets (loaders, CSS tools, URDF
generation, training validation) should import from here instead of
maintaining its own hardcoded constants.

Usage::

    from robotic_grounding.retarget.dataset_registry import (
        get_dataset_config,
        get_all_dataset_names,
    )

    config = get_dataset_config("taco")
    print(config.fps)          # 30.0
    print(config.mesh_format)  # "obj"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BASE_CSS_PREFIX = "v2d/human_motion_data"


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable configuration for a single dataset.

    Attributes:
        name: Short identifier (e.g. "taco", "arctic").
        fps: Frames per second of the source motion data.
        mano_kwargs: Keyword arguments for ManoLayer forward kinematics
            (flat_hand_mean, center_idx).
        mesh_vertex_scale: Scale factor to convert mesh vertices to meters.
            0.01 for TACO (centimeters), 1.0 for datasets already in meters.
        mesh_format: Source mesh file format ("obj" or "glb").
        has_articulated_objects: Whether objects have articulated parts
            (e.g. Arctic drawers/scissors). Affects URDF generation.
        has_contact_data: Whether processed parquets include per-link
            contact positions/normals.
        has_support_surfaces: Whether support surface reconstruction is
            expected to produce output.
        link_to_site_quat_wxyz: Quaternion (w, x, y, z) for the MANO
            link-to-site transform used during IK retargeting. None if
            no transform is needed.
        loaded_suffix: Suffix for the loaded data directory name
            (e.g. "_loaded" -> "{name}_loaded").
        processed_suffix: Suffix for the processed data directory name
            (e.g. "_processed" -> "{name}_processed").
        css_raw_prefix: Subdirectory under the dataset's CSS path for raw
            data. Empty string means "dataset/" (the default).
            TACO overrides this to "dataset/Hand_Poses/".
        loader_script: Path to the loader script relative to repo root.
        retarget_script: Path to the retarget script relative to repo root.
    """

    # Identity
    name: str

    # Loader metadata
    fps: float
    mano_kwargs: dict[str, Any] = field(default_factory=dict)
    mesh_vertex_scale: float = 1.0
    mesh_format: str = "obj"

    # Capabilities
    has_articulated_objects: bool = False
    has_contact_data: bool = True
    has_support_surfaces: bool = True

    # IK retargeting
    link_to_site_quat_wxyz: tuple[float, ...] | None = None

    # Storage path conventions (relative to HUMAN_MOTION_DATA_DIR/{name}/)
    loaded_suffix: str = "_loaded"
    processed_suffix: str = "_processed"

    # CSS storage
    css_raw_prefix: str = ""

    # Script dispatch (relative to repo root)
    loader_script: str = ""
    retarget_script: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "taco": DatasetConfig(
        name="taco",
        fps=30.0,
        mano_kwargs={"flat_hand_mean": True, "center_idx": 0},
        mesh_vertex_scale=0.01,
        mesh_format="obj",
        has_articulated_objects=False,
        has_contact_data=True,
        css_raw_prefix="dataset/Hand_Poses/",
        loader_script="scripts/retarget/taco_loader.py",
        retarget_script="scripts/retarget/taco_to_sharpa.py",
    ),
    "arctic": DatasetConfig(
        name="arctic",
        fps=30.0,
        mano_kwargs={"flat_hand_mean": False, "center_idx": None},
        mesh_vertex_scale=1.0,
        mesh_format="obj",
        has_articulated_objects=True,
        has_contact_data=True,
        link_to_site_quat_wxyz=(0.5, -0.5, 0.5, 0.5),
        loader_script="scripts/retarget/arctic_loader.py",
        retarget_script="scripts/retarget/arctic_to_sharpa.py",
    ),
    "oakink2": DatasetConfig(
        name="oakink2",
        fps=120.0,
        mano_kwargs={"flat_hand_mean": True, "center_idx": 0},
        mesh_vertex_scale=1.0,
        mesh_format="obj",
        has_articulated_objects=False,
        has_contact_data=True,
        loader_script="scripts/retarget/oakink2_loader.py",
        retarget_script="scripts/retarget/oakink2_to_sharpa.py",
    ),
    "hot3d": DatasetConfig(
        name="hot3d",
        fps=30.0,
        mano_kwargs={"flat_hand_mean": False, "center_idx": None},
        mesh_vertex_scale=1.0,
        mesh_format="glb",
        has_articulated_objects=False,
        has_contact_data=True,
        loader_script="scripts/retarget/hot3d_loader.py",
        retarget_script="scripts/retarget/hot3d_to_sharpa.py",
    ),
    "h2o": DatasetConfig(
        name="h2o",
        fps=30.0,
        mano_kwargs={"flat_hand_mean": False, "center_idx": None},
        mesh_vertex_scale=1.0,
        mesh_format="obj",
        has_articulated_objects=False,
        has_contact_data=True,
        loader_script="scripts/retarget/h2o_loader.py",
        retarget_script="scripts/retarget/h2o_to_sharpa.py",
    ),
    "grab": DatasetConfig(
        name="grab",
        fps=120.0,
        mano_kwargs={"flat_hand_mean": False, "center_idx": None},
        mesh_vertex_scale=1.0,
        mesh_format="obj",
        has_articulated_objects=False,
        has_contact_data=True,
        loader_script="scripts/retarget/grab_loader.py",
        retarget_script="scripts/retarget/grab_to_sharpa.py",
    ),
    "dexycb": DatasetConfig(
        name="dexycb",
        fps=30.0,
        mano_kwargs={"flat_hand_mean": False, "center_idx": None},
        mesh_vertex_scale=1.0,
        mesh_format="obj",
        has_articulated_objects=False,
        has_contact_data=True,
        loader_script="scripts/retarget/dexycb_loader.py",
        retarget_script="scripts/retarget/dexycb_to_sharpa.py",
    ),
    "v2d_reconstruction": DatasetConfig(
        name="v2d_reconstruction",
        fps=25.0,
        mano_kwargs={"flat_hand_mean": False, "center_idx": None},
        mesh_vertex_scale=1.0,
        mesh_format="obj",
        has_articulated_objects=False,
        has_contact_data=False,
        has_support_surfaces=False,
        loader_script="scripts/retarget/v2d_reconstruction_loader.py",
        retarget_script="scripts/retarget/v2d_reconstruction_to_sharpa.py",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_dataset_config(name: str) -> DatasetConfig:
    """Return the configuration for a dataset.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in DATASET_CONFIGS:
        available = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return DATASET_CONFIGS[name]


def get_all_dataset_names() -> tuple[str, ...]:
    """Return all registered dataset names (insertion order)."""
    return tuple(DATASET_CONFIGS)


def get_css_stage_prefixes(name: str) -> dict[str, str]:
    """Derive the CSS S3 stage prefixes for a dataset.

    Returns a dict with keys "raw", "loaded", "processed" mapping to
    S3 prefix strings under the ``BASE_CSS_PREFIX``.
    """
    config = get_dataset_config(name)
    base = f"{BASE_CSS_PREFIX}/{name}"
    raw_prefix = config.css_raw_prefix or "dataset/"
    return {
        "raw": f"{base}/{raw_prefix}",
        "loaded": f"{base}/{name}{config.loaded_suffix}/",
        "processed": f"{base}/{name}{config.processed_suffix}/",
    }
