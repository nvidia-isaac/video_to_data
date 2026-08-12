# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene loading utilities and config (motion_file → SceneConfig → apply to env_cfg)."""

import os

from .apply_scene_config import (
    apply_scene_commands,
    apply_scene_config,
    apply_scene_contact_sensors,
    apply_scene_objects,
    apply_scene_robot,
)
from .scene_config import (
    ArticulatedObjectConfig,
    ObjectConfig,
    SceneConfig,
)

SCENE_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "config"))

__all__ = [
    "ArticulatedObjectConfig",
    "ObjectConfig",
    "SceneConfig",
    "apply_scene_commands",
    "apply_scene_config",
    "apply_scene_contact_sensors",
    "apply_scene_objects",
    "apply_scene_robot",
    "SCENE_CONFIG_DIR",
]
