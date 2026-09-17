# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-embodiment closed-loop profiles.

Importing this package registers all built-in profiles (via ``register_profile``), so
``embodiment.get_profile(name)`` can resolve them. Add a new robot by creating a module
here that builds an ``EmbodimentProfile`` and calls ``register_profile``, then importing
it below.
"""

from . import (
    sharpa_dual_hand_three_camera,  # noqa: F401
    vega_sharpa_joint,  # noqa: F401  (vega whole-body 58-joint contract)
)
