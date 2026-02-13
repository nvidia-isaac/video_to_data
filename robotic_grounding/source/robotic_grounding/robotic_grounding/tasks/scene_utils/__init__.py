"""Scene loading utilities and config (YAML → SceneConfig, apply to env_cfg)."""

import os

from .apply_scene_config import apply_scene_config
from .scene_config import ObjectConfig, SceneConfig

# Directory containing scene YAML configs (e.g. apple_pick.yaml)
SCENE_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "config"))

__all__ = [
    "ObjectConfig",
    "SceneConfig",
    "apply_scene_config",
    "SCENE_CONFIG_DIR",
]
