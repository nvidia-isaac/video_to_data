#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a rigid USD with explicit colliders and mass properties."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from explicit_colliders import (
    ColliderBuildConfig,
    build_explicit_colliders,
    calculate_mass_properties,
    collider_set_to_report,
)
from geometry import triangulate_faces
from physics import physics_contact_config
from recorded_support import file_sha256


GENERATOR_VERSION = "0.6.0"
SUPPORTED_INPUTS = {".usd", ".usda", ".usdc", ".obj", ".fbx", ".gltf", ".glb", ".stl"}
USD_INPUTS = {".usd", ".usda", ".usdc"}


@dataclass(frozen=True)
class GeometryMetrics:
    extents_m: tuple[float, float, float]
    vertex_count: int
    mesh_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mass-kg", type=float, required=True)
    parser.add_argument(
        "--mass-source",
        choices=("command_line", "assumed_grounding_default"),
        required=True,
    )
    parser.add_argument("--static-friction", type=float, default=0.5)
    parser.add_argument("--dynamic-friction", type=float, default=0.4)
    parser.add_argument("--restitution", type=float, default=0.1)
    parser.add_argument("--max-convex-hulls", type=int, default=16)
    parser.add_argument("--hull-vertex-limit", type=int, default=64)
    parser.add_argument("--coacd-threshold", type=float, default=0.05)
    parser.add_argument("--coacd-resolution", type=int, default=2000)
    parser.add_argument("--max-decomposition-source-faces", type=int, default=20_000)
    parser.add_argument(
        "--simplify-decomposition-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify dense source triangles before CoACD (default: enabled)",
    )
    parser.add_argument("--coacd-seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
INPUT_ASSET = Path(ARGS.asset).expanduser().resolve()
OUTPUT_DIR = Path(ARGS.output_dir).expanduser().resolve()
if not INPUT_ASSET.is_file() or INPUT_ASSET.suffix.lower() not in SUPPORTED_INPUTS:
    raise SystemExit(f"error: unsupported or missing asset: {INPUT_ASSET}")
if ARGS.mass_kg <= 0:
    raise SystemExit("error: mass-kg must be positive")
if not 0 <= ARGS.dynamic_friction <= ARGS.static_friction:
    raise SystemExit("error: expected 0 <= dynamic-friction <= static-friction")
if not 0 <= ARGS.restitution <= 1:
    raise SystemExit("error: restitution must be in [0, 1]")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BUILD_CONFIG = ColliderBuildConfig(
    coacd_threshold=ARGS.coacd_threshold,
    max_convex_hulls=ARGS.max_convex_hulls,
    coacd_resolution=ARGS.coacd_resolution,
    max_decomposition_source_faces=ARGS.max_decomposition_source_faces,
    simplify_decomposition_source=ARGS.simplify_decomposition_source,
    max_hull_vertices=ARGS.hull_vertex_limit,
    coacd_seed=ARGS.coacd_seed,
)


from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless})

import numpy as np
from isaacsim.core.experimental.utils.app import enable_extension
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from usd_normalization import force_opaque_gltf_materials, normalize_visual_reference


async def convert_to_usd(input_path: Path, output_path: Path) -> None:
    import omni.kit.asset_converter

    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.use_meter_as_world_unit = True
    context.create_world_as_default_root_prim = True
    context.convert_stage_up_z = True
    context.disabling_instancing = True
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(input_path), str(output_path), lambda _progress, _total: None, context
    )
    if not await task.wait_until_finished():
        raise RuntimeError(f"Asset conversion failed: {input_path}")


def prepare_visual_asset() -> Path:
    visual_asset = OUTPUT_DIR / "visual_asset.usd"
    if INPUT_ASSET.suffix.lower() in USD_INPUTS:
        source_stage = Usd.Stage.Open(str(INPUT_ASSET))
        if source_stage is None or not source_stage.Flatten().Export(str(visual_asset)):
            raise RuntimeError(f"Could not flatten USD asset: {INPUT_ASSET}")
        return visual_asset

    enable_extension("omni.kit.asset_converter")
    simulation_app.update()
    from omni.kit.async_engine import run_coroutine

    conversion = run_coroutine(convert_to_usd(INPUT_ASSET, visual_asset))
    while not conversion.done():
        simulation_app.update()
    conversion.result()
    if not visual_asset.is_file():
        raise RuntimeError(f"Converter did not create {visual_asset}")
    return visual_asset


def gather_visual_geometry(
    root_prim: Usd.Prim,
) -> tuple[GeometryMetrics, list[Usd.Prim], np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    meshes = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.IsInstanceProxy():
            raise RuntimeError(f"Instance proxy cannot be made reproducible: {prim.GetPath()}")
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        if not points:
            continue
        transform = cache.GetLocalToWorldTransform(prim)
        transformed = [transform.Transform(point) for point in points]
        offset = len(vertices)
        vertices.extend(tuple(float(value) for value in point) for point in transformed)
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        faces.extend(triangulate_faces(counts, indices, offset))
        meshes.append(prim)
    if not meshes or len(vertices) < 4 or len(faces) < 4:
        raise RuntimeError("The composed asset has no usable triangle mesh")

    vertices_array = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    minimum = vertices_array.min(axis=0)
    maximum = vertices_array.max(axis=0)
    extents = maximum - minimum
    if np.any(extents <= 1e-9):
        raise RuntimeError(f"Degenerate asset extents: {extents.tolist()}")
    metrics = GeometryMetrics(
        extents_m=tuple(float(value) for value in extents),
        vertex_count=len(vertices_array),
        mesh_count=len(meshes),
    )
    return metrics, meshes, vertices_array, faces_array


def strip_visual_physics(root_prim: Usd.Prim) -> None:
    """Remove imported physics so only the generated rigid root owns physics."""
    schemas = (
        UsdPhysics.CollisionAPI,
        UsdPhysics.MeshCollisionAPI,
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.MassAPI,
        UsdPhysics.ArticulationRootAPI,
    )
    for prim in Usd.PrimRange(root_prim):
        for api in schemas:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)


def author_explicit_colliders(
    stage: Usd.Stage,
    colliders,
    contact_config: dict,
) -> list[Usd.Prim]:
    root = UsdGeom.Xform.Define(stage, "/Asset/Colliders")
    root.GetPrim().CreateAttribute(
        "v2d:colliderType", Sdf.ValueTypeNames.String, custom=True
    ).Set(colliders.collider_type)
    collider_prims = []
    for part in colliders.parts:
        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/Colliders/{part.name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in part.vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(part.faces))
        mesh.CreateFaceVertexIndicesAttr(part.faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
            UsdPhysics.Tokens.convexHull
        )
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_collision.CreateRestOffsetAttr().Set(contact_config["rest_offset_m"])
        physx_collision.CreateContactOffsetAttr().Set(contact_config["contact_offset_m"])
        prim.CreateAttribute("v2d:geometrySha256", Sdf.ValueTypeNames.String, custom=True).Set(
            part.sha256
        )
        collider_prims.append(prim)
    return collider_prims


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def author_physics_material(stage: Usd.Stage, colliders: list[Usd.Prim]) -> dict:
    material = UsdShade.Material.Define(stage, "/Asset/PhysicsMaterials/Default")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    static = clamp(ARGS.static_friction, 0.0, 10.0)
    dynamic = clamp(ARGS.dynamic_friction, 0.0, static)
    restitution = clamp(ARGS.restitution, 0.0, 1.0)
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(restitution)
    for prim in colliders:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
    return {
        "path": str(material.GetPath()),
        "static_friction": static,
        "dynamic_friction": dynamic,
        "restitution": restitution,
    }


def author_mass_properties(root: Usd.Prim, properties) -> None:
    api = UsdPhysics.MassAPI.Apply(root)
    api.CreateMassAttr(properties.mass_kg)
    api.CreateCenterOfMassAttr(Gf.Vec3f(*properties.center_of_mass_m))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*properties.diagonal_inertia_kg_m2))
    w, x, y, z = properties.principal_axes_wxyz
    api.CreatePrincipalAxesAttr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def author_provenance(
    root: Usd.Prim,
    colliders,
    source_asset_file_sha256: str,
) -> None:
    values = {
        "v2d:generator": "v2d_hoi_mesh_to_usd",
        "v2d:generatorVersion": GENERATOR_VERSION,
        "v2d:sourceAssetFileSha256": source_asset_file_sha256,
        "v2d:sourceMeshSha256": colliders.source_mesh_sha256,
        "v2d:colliderGenerator": colliders.generator,
        "v2d:colliderGeneratorVersion": colliders.generator_version,
        "v2d:colliderConfig": json.dumps(asdict(colliders.config), sort_keys=True),
        "v2d:massSource": ARGS.mass_source,
    }
    for name, value in values.items():
        root.CreateAttribute(name, Sdf.ValueTypeNames.String, custom=True).Set(value)


def validate_authored_asset(
    root: Usd.Prim,
    visual_meshes: list[Usd.Prim],
    collider_prims: list[Usd.Prim],
) -> list[str]:
    errors = []
    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        errors.append("/Asset is missing RigidBodyAPI")
    if not root.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        errors.append("/Asset is missing PhysxRigidBodyAPI")
    mass = UsdPhysics.MassAPI(root)
    required = (
        ("mass", mass.GetMassAttr()),
        ("centerOfMass", mass.GetCenterOfMassAttr()),
        ("diagonalInertia", mass.GetDiagonalInertiaAttr()),
        ("principalAxes", mass.GetPrincipalAxesAttr()),
    )
    errors.extend(
        f"MassAPI {name} is not authored"
        for name, attr in required
        if not attr.HasAuthoredValue()
    )
    if not collider_prims:
        errors.append("No explicit collider prims")
    for prim in collider_prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            errors.append(f"Collider lacks CollisionAPI: {prim.GetPath()}")
        physx_collision = PhysxSchema.PhysxCollisionAPI(prim)
        if not physx_collision.GetRestOffsetAttr().HasAuthoredValue():
            errors.append(f"Collider lacks authored restOffset: {prim.GetPath()}")
        if not physx_collision.GetContactOffsetAttr().HasAuthoredValue():
            errors.append(f"Collider lacks authored contactOffset: {prim.GetPath()}")
    for prim in visual_meshes:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            errors.append(f"Visual mesh still has CollisionAPI: {prim.GetPath()}")
    return errors


def run() -> dict:
    source_asset_file_sha256 = file_sha256(INPUT_ASSET)
    visual_asset = prepare_visual_asset()
    visual_material_normalization = force_opaque_gltf_materials(visual_asset)
    output_usd = OUTPUT_DIR / "rigid_object.usd"
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    visuals = UsdGeom.Xform.Define(stage, "/Asset/Visuals")
    visuals.GetPrim().GetReferences().AddReference("./visual_asset.usd")
    normalization = normalize_visual_reference(visuals, visual_asset)

    metrics, visual_meshes, vertices, faces = gather_visual_geometry(root.GetPrim())
    contact_config = physics_contact_config(max(metrics.extents_m))
    selected = {
        "collider_type": "convexDecomposition",
        "reasons": ["policy=coacd_default"],
    }
    explicit = build_explicit_colliders(
        vertices,
        faces,
        selected["collider_type"],
        BUILD_CONFIG,
    )
    mass_properties = calculate_mass_properties(explicit, ARGS.mass_kg)

    strip_visual_physics(visuals.GetPrim())
    collider_prims = author_explicit_colliders(stage, explicit, contact_config)
    rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    physx_rigid.CreateSolverPositionIterationCountAttr().Set(
        contact_config["solver_position_iterations"]
    )
    physx_rigid.CreateSolverVelocityIterationCountAttr().Set(
        contact_config["solver_velocity_iterations"]
    )
    physx_rigid.CreateEnableCCDAttr().Set(contact_config["ccd_enabled"])
    author_mass_properties(root.GetPrim(), mass_properties)
    material = author_physics_material(stage, collider_prims)
    author_provenance(root.GetPrim(), explicit, source_asset_file_sha256)
    errors = validate_authored_asset(root.GetPrim(), visual_meshes, collider_prims)
    if errors:
        raise RuntimeError("Authoring validation failed: " + "; ".join(errors))
    stage.GetRootLayer().Save()

    report = {
        "schema_version": 2,
        "status": "generated",
        "input_asset": str(INPUT_ASSET),
        "input_asset_file_sha256": source_asset_file_sha256,
        "visual_asset": str(visual_asset),
        "output_usd": str(output_usd),
        "rigid_body": True,
        "articulation": False,
        "stage_metadata": {
            "meters_per_unit": 1.0,
            "kilograms_per_unit": 1.0,
            "up_axis": "Z",
        },
        "normalization": normalization,
        "visual_material_normalization": visual_material_normalization,
        "geometry_metrics": asdict(metrics),
        "selected_collider": selected,
        "explicit_colliders": collider_set_to_report(explicit),
        "mass_properties": asdict(mass_properties),
        "mass_source": ARGS.mass_source,
        "inertia_source": "explicit_from_generated_colliders",
        "physics_material": material,
        "physics_contact_config": contact_config,
        "generator": {"name": "v2d_hoi_mesh_to_usd", "version": GENERATOR_VERSION},
        "authoring_validation": {"passed": True, "errors": []},
        "drop_test_recommended": True,
    }
    (OUTPUT_DIR / "generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


try:
    print(json.dumps(run(), indent=2))
except Exception as error:
    failure = {
        "schema_version": 2,
        "status": "error",
        "input_asset": str(INPUT_ASSET),
        "error": f"{type(error).__name__}: {error}",
    }
    (OUTPUT_DIR / "generation_report.json").write_text(
        json.dumps(failure, indent=2), encoding="utf-8"
    )
    print(json.dumps(failure, indent=2))
    raise
finally:
    simulation_app.close()
