# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration shared by rendering consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RenderMode = Literal["realtime", "path_tracing"]


@dataclass(frozen=True)
class RendererSettings:
    """RTX renderer settings that can be applied after the simulation app starts."""

    mode: RenderMode = "realtime"
    samples_per_pixel: int = 48
    max_bounces: int = 4
    max_specular_bounces: int = 6
    denoiser: bool = True
    clamp_spp: float | None = None
    exposure: float | None = None

    def __post_init__(self) -> None:
        """Validate renderer values before an Isaac runtime is launched."""
        if self.mode not in ("realtime", "path_tracing"):
            raise ValueError(f"Unsupported render mode: {self.mode!r}")
        if self.samples_per_pixel < 1:
            raise ValueError("samples_per_pixel must be positive")
        if self.max_bounces < 0 or self.max_specular_bounces < 0:
            raise ValueError("bounce counts cannot be negative")
        if self.clamp_spp is not None and self.clamp_spp <= 0:
            raise ValueError("clamp_spp must be positive when supplied")


@dataclass(frozen=True)
class HeroLightSettings:
    """Optional point-key and dome-fill settings for deterministic captures."""

    point_intensity: float | None = None
    point_height: float = 2.5
    point_radius: float = 0.25
    point_xy: tuple[float, float] = (0.1, 0.05)
    dome_intensity: float | None = None
    disable_distant_lights: bool = True

    def __post_init__(self) -> None:
        """Validate deterministic-light settings."""
        if self.point_radius < 0:
            raise ValueError("point_radius cannot be negative")


@dataclass(frozen=True)
class RenderProfile:
    """Composable rendering profile used by recording or task-specific adapters."""

    renderer: RendererSettings = field(default_factory=RendererSettings)
    lighting: HeroLightSettings = field(default_factory=HeroLightSettings)
    use_vmaterials: bool = False
    use_visual_domain_randomization: bool = False
