# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
def __getattr__(name):
    if name == "Renderer":
        from .renderer import Renderer
        return Renderer
    if name == "Wis3DScene":
        from .wis3d_helper import Wis3DScene
        return Wis3DScene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Renderer", "Wis3DScene"]
