# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable photorealistic rendering infrastructure for robotic-grounding tasks.

The package root deliberately exports only host-safe configuration and lifecycle APIs.
Isaac Sim integrations live in submodules and import ``carb``, ``pxr``, ``omni``, and
Isaac Lab lazily wherever possible.
"""

from .assets import RenderAssetRegistry
from .config import HeroLightSettings, RendererSettings, RenderProfile
from .lifecycle import prepare_mdl_environment, register_mdl_search_paths

__all__ = [
    "HeroLightSettings",
    "RenderAssetRegistry",
    "RenderProfile",
    "RendererSettings",
    "prepare_mdl_environment",
    "register_mdl_search_paths",
]
