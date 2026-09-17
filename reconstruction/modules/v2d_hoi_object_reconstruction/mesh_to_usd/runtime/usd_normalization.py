# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize a referenced USD asset into the v2d meter/Z-up asset frame."""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdShade


GLTF_MDL_SOURCE = "gltf/pbr.mdl"


def force_opaque_gltf_materials(visual_asset: str | Path) -> dict:
    stage = Usd.Stage.Open(str(visual_asset))
    if stage is None:
        raise RuntimeError(f"Could not inspect visual USD: {visual_asset}")

    modified_shaders = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        source_asset = shader.GetSourceAsset("mdl")
        source_path = getattr(source_asset, "path", "")
        source_identifier = shader.GetSourceAssetSubIdentifier("mdl")
        if source_path != GLTF_MDL_SOURCE or source_identifier != "gltf_material":
            continue

        transmission_input = shader.GetInput("transmission_factor")
        if not transmission_input:
            continue
        transmission_factor = transmission_input.Get()
        if transmission_factor is None or float(transmission_factor) <= 0.0:
            continue

        transmission_input.Set(0.0)
        modified_shaders.append(
            {
                "shader_path": str(prim.GetPath()),
                "original_transmission_factor": float(transmission_factor),
            }
        )

    if modified_shaders:
        stage.GetRootLayer().Save()

    return {
        "policy": "force_opaque_gltf_transmission",
        "modified_shader_count": len(modified_shaders),
        "modified_shaders": modified_shaders,
    }


def normalize_visual_reference(
    visual_xform: UsdGeom.Xform,
    visual_asset: str | Path,
) -> dict:
    source_stage = Usd.Stage.Open(str(visual_asset))
    if source_stage is None:
        raise RuntimeError(f"Could not inspect visual USD: {visual_asset}")
    source_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(source_stage))
    if not math.isfinite(source_meters_per_unit) or source_meters_per_unit <= 0:
        raise ValueError("Source USD metersPerUnit must be positive and finite")
    source_up_axis = str(UsdGeom.GetStageUpAxis(source_stage))

    if not math.isclose(source_meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1e-12):
        visual_xform.AddScaleOp().Set(
            Gf.Vec3f(
                source_meters_per_unit,
                source_meters_per_unit,
                source_meters_per_unit,
            )
        )
    if source_up_axis == str(UsdGeom.Tokens.y):
        visual_xform.AddRotateXOp().Set(90.0)
    elif source_up_axis != str(UsdGeom.Tokens.z):
        raise ValueError(f"Unsupported source USD up axis: {source_up_axis}")

    return {
        "source_meters_per_unit": source_meters_per_unit,
        "source_up_axis": source_up_axis,
        "target_meters_per_unit": 1.0,
        "target_up_axis": "Z",
    }
