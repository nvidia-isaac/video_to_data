# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time composition of a task reference and its simulation scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flash_chord.assets.registry import support_usda_for_reference
from flash_chord.data import frame_window, load_reference
from flash_chord.data.reference import Reference
from flash_chord.embodiments.base import Embodiment
from flash_chord.scene.builder import Scene, build_scene
from flash_chord.scene.collision import CollisionPolicy


@dataclass(frozen=True)
class SceneSetup:
    """Resolved source, reference, support, and built scene for one task."""

    parquet_path: Path
    reference: Reference
    support_usda: Path | None
    scene: Scene


def setup_scene(
    *,
    parquet: str | Path,
    control_fps: float | None,
    motion_speed: float,
    embodiment: Embodiment,
    collision: CollisionPolicy,
    world_count: int,
    include_support: bool,
    decompose_objects: bool,
    object_scale_min: float = 1.0,
    object_scale_max: float = 1.0,
    object_scale_seed: int = 0,
    contact_friction: float = 1.0,
    object_free_joint_damping: float = 0.0,
    source_frame_playback: bool = False,
    motion_start_frame: int = 0,
    motion_end_frame: int = -1,
) -> SceneSetup:
    """Load one reference, resolve its optional support, and build its scene once.

    This function owns only task-scene composition. Device scopes, Hydra config,
    environments, algorithms, replay, and visualization remain caller concerns.
    """
    reference = load_reference(
        parquet,
        control_fps=control_fps,
        motion_speed=motion_speed,
        source_frame_playback=source_frame_playback,
    )
    reference = frame_window(reference, motion_start_frame, motion_end_frame)
    parquet_path = Path(reference.metadata.source_path)
    support_usda = support_usda_for_reference(reference) if include_support else None
    if include_support and support_usda is None:
        raise FileNotFoundError(
            "scene support is enabled, but no support asset could be resolved from the reference metadata: "
            f"{parquet_path}"
        )

    scene_kwargs = {
        "world_count": world_count,
        "support_usda": support_usda,
        "collision": collision,
        "decompose_objects": decompose_objects,
    }
    if object_scale_min != 1.0 or object_scale_max != 1.0:
        scene_kwargs.update(
            object_scale_min=object_scale_min,
            object_scale_max=object_scale_max,
            object_scale_seed=object_scale_seed,
        )
    if contact_friction != 1.0 or object_free_joint_damping != 0.0:
        scene_kwargs.update(
            contact_friction=contact_friction,
            object_free_joint_damping=object_free_joint_damping,
        )
    scene = build_scene(embodiment, reference, **scene_kwargs)
    return SceneSetup(
        parquet_path=parquet_path,
        reference=reference,
        support_usda=support_usda,
        scene=scene,
    )
