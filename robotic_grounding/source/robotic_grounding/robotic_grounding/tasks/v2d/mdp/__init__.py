# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contains the functions that are specific to the locomotion environments."""

from isaaclab.envs.mdp import *  # noqa: F403

# Visual-DR event functions live in the shared rendering package. Re-exported here so
# existing `mdp.randomize_*` references in event configs keep resolving.
# `randomize_visual_texture_material` is IsaacLab's built-in and arrives via the star
# import above.
from robotic_grounding.rendering.dr.isaaclab_events import (  # noqa: F401
    prewarm_textures,
    randomize_distant_light,
    randomize_dome_light,
    randomize_visual_texture_on_prims,
)

from .actions import *  # noqa: F403
from .commands.commands_cfg import *  # noqa: F403
from .curriculum import *  # noqa: F403
from .events import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
