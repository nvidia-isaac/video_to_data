#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an Isaac Sim rigid-body/collider hold-and-drop test."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from geometry import principal_axis_pose_candidates, triangulate_faces
from physics import (
    PoseSettleTracker,
    SettleTracker,
    aggregate_pose_results,
    classify_drop_test_outcome,
    classify_standing,
    ground_metrics,
    is_within_ground_tolerance,
    physics_contact_config,
)
from recorded_support import recorded_support_pose_load


SUPPORTED_INPUTS = {".usd", ".usda", ".usdc", ".obj", ".fbx", ".gltf", ".glb", ".stl"}
USD_INPUTS = {".usd", ".usda", ".usdc"}
DEFAULT_LIFT_HEIGHT_RATIO = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, help="USD or supported mesh asset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record MP4 evidence (default: enabled)",
    )
    parser.add_argument(
        "--fail-if-not-standing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the final pose to satisfy the standing diagnostic (default: enabled)",
    )

    parser.add_argument("--physics-hz", type=int, default=60)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--pre-lift-seconds", type=float, default=1.0)
    parser.add_argument("--lift-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument(
        "--lift-height",
        type=float,
        help=(
            "Lift distance in meters; defaults to lift-height-ratio times the "
            "object's largest extent"
        ),
    )
    parser.add_argument(
        "--lift-height-ratio",
        type=float,
        default=DEFAULT_LIFT_HEIGHT_RATIO,
        help=(
            "Fraction of the object's largest extent to lift "
            f"(default: {DEFAULT_LIFT_HEIGHT_RATIO})"
        ),
    )
    parser.add_argument("--max-drop-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--post-settle-seconds", type=float, default=1.0)
    parser.add_argument("--initial-clearance", type=float, default=0.005)

    parser.add_argument("--fallback-mass-kg", type=float, default=0.3)
    parser.add_argument("--asset-scale", type=float, default=1.0)
    parser.add_argument(
        "--collision-approximation",
        choices=("convexHull", "convexDecomposition"),
        default="convexHull",
    )
    parser.add_argument("--local-up", nargs=3, type=float)
    parser.add_argument(
        "--initial-pose",
        choices=(
            "as-authored",
            "principal-6",
            "principal-6-support",
            "recorded-support",
        ),
        default="principal-6",
        help="Initial orientation policy (default: principal-6)",
    )
    parser.add_argument(
        "--recorded-support",
        help="Strict recorded-support JSON; required only for recorded-support mode",
    )
    parser.add_argument("--standing-angle", type=float, default=15.0)
    parser.add_argument("--linear-speed-limit", type=float, default=0.01)
    parser.add_argument("--angular-speed-limit", type=float, default=0.05)
    parser.add_argument(
        "--pose-settle-position-ratio",
        type=float,
        default=0.01,
        help=(
            "Maximum full-pose translation as a fraction of object extent "
            "(default: 0.01; clamped to 0.001-0.01 m)"
        ),
    )
    parser.add_argument(
        "--pose-settle-angle",
        type=float,
        default=2.0,
        help="Maximum full-pose rotation in degrees (default: 2)",
    )
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--ground-tolerance", type=float, default=0.02)
    parser.add_argument("--contact-force-threshold", type=float, default=1e-4)

    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--camera-eye", nargs=3, type=float)
    parser.add_argument("--camera-target", nargs=3, type=float)
    parser.add_argument("--camera-distance-scale", type=float, default=2.2)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    asset = Path(args.asset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not asset.is_file() or asset.suffix.lower() not in SUPPORTED_INPUTS:
        raise ValueError(f"Unsupported or missing asset: {asset}")
    scalar_values = (
        args.pre_lift_seconds,
        args.lift_seconds,
        args.hold_seconds,
        args.lift_height_ratio,
        args.max_drop_seconds,
        args.settle_seconds,
        args.post_settle_seconds,
        args.initial_clearance,
        args.fallback_mass_kg,
        args.asset_scale,
        args.standing_angle,
        args.linear_speed_limit,
        args.angular_speed_limit,
        args.pose_settle_position_ratio,
        args.pose_settle_angle,
        args.ground_z,
        args.ground_tolerance,
        args.contact_force_threshold,
        args.camera_distance_scale,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("drop-test numeric options must be finite")
    if args.physics_hz <= 0 or args.fallback_mass_kg <= 0 or args.asset_scale <= 0:
        raise ValueError(
            "physics-hz, fallback-mass-kg, and asset-scale must be positive"
        )
    if args.video_fps <= 0 or args.video_fps > args.physics_hz:
        raise ValueError("video-fps must be positive and no greater than physics-hz")
    if args.lift_height is not None and (
        not math.isfinite(args.lift_height) or args.lift_height <= 0
    ):
        raise ValueError("lift-height must be positive and finite")
    if args.lift_height_ratio <= 0 or args.camera_distance_scale <= 0:
        raise ValueError("lift-height-ratio and camera-distance-scale must be positive")
    if (
        args.pose_settle_position_ratio <= 0
        or not 0 < args.pose_settle_angle <= 180
    ):
        raise ValueError(
            "pose settle position ratio must be positive and angle must be in (0, 180]"
        )
    durations = (
        args.pre_lift_seconds,
        args.lift_seconds,
        args.hold_seconds,
        args.max_drop_seconds,
        args.settle_seconds,
        args.post_settle_seconds,
    )
    if any(duration < 0 for duration in durations):
        raise ValueError("simulation durations cannot be negative")
    if args.initial_clearance < 0:
        raise ValueError("initial-clearance cannot be negative")
    if not 0 <= args.standing_angle <= 180:
        raise ValueError("standing-angle must be in [0, 180]")
    if min(
        args.linear_speed_limit,
        args.angular_speed_limit,
        args.ground_tolerance,
        args.contact_force_threshold,
    ) < 0:
        raise ValueError("diagnostic thresholds cannot be negative")
    if args.local_up is not None:
        if not all(math.isfinite(value) for value in args.local_up):
            raise ValueError("local-up must be finite")
        if math.sqrt(sum(value * value for value in args.local_up)) == 0:
            raise ValueError("local-up cannot be the zero vector")
    if args.initial_pose.startswith("principal-6") and args.local_up is not None:
        raise ValueError(
            "local-up cannot be combined with a principal-6 initial pose"
        )
    if args.initial_pose == "recorded-support":
        if args.recorded_support is None:
            raise ValueError(
                "recorded-support initial pose requires --recorded-support"
            )
        if args.local_up is not None:
            raise ValueError(
                "local-up is read from recorded-support JSON and cannot be overridden"
            )
        support_path = Path(args.recorded_support).expanduser().resolve()
        if not support_path.is_file():
            raise ValueError(f"missing recorded-support file: {support_path}")
    elif args.recorded_support is not None:
        raise ValueError(
            "--recorded-support can only be used with recorded-support initial pose"
        )
    if any(value <= 0 for value in args.resolution):
        raise ValueError("resolution values must be positive")
    for name, vector in (
        ("camera-eye", args.camera_eye),
        ("camera-target", args.camera_target),
    ):
        if vector is not None and not all(math.isfinite(value) for value in vector):
            raise ValueError(f"{name} must be finite")
    output_dir.mkdir(parents=True, exist_ok=True)
    return asset, output_dir


ARGS = parse_args()
try:
    INPUT_ASSET, OUTPUT_DIR = validate_args(ARGS)
except ValueError as error:
    raise SystemExit(f"error: {error}") from error


# Isaac/Omniverse imports must happen after SimulationApp construction.
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": ARGS.resolution[0],
        "height": ARGS.resolution[1],
    }
)

import carb
import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.experimental.utils.app import enable_extension
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics


OBJECT_PATH = "/World/Object"
GROUND_PATH = "/World/Ground"


@dataclass(frozen=True)
class PoseCandidate:
    pose_id: str
    local_up: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float] | None
    metadata: dict


@dataclass(frozen=True)
class SceneState:
    object_prim: Usd.Prim
    object_view: RigidPrim
    body_api: UsdPhysics.RigidBodyAPI
    translate_op: UsdGeom.XformOp
    mesh_count: int
    start_root_z: float
    lift_height: float
    object_height: float
    characteristic_extent: float
    contact_config: dict
    preserved_authored_physics: bool
    effective_mass: float
    used_authored_mass: bool
    local_up: tuple[float, float, float]
    pose_selection: dict


@dataclass
class CaptureState:
    viewport: object
    frames_dir: Path
    frame_stem: str
    frame_index: int = 0
    physics_step_index: int = 0


def get_or_add_xform_op(xformable, op_type):
    for operation in xformable.GetOrderedXformOps():
        if operation.GetOpType() == op_type:
            return operation
    if op_type == UsdGeom.XformOp.TypeTranslate:
        return xformable.AddTranslateOp()
    if op_type == UsdGeom.XformOp.TypeScale:
        return xformable.AddScaleOp()
    if op_type == UsdGeom.XformOp.TypeOrient:
        return xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    raise ValueError(f"Unsupported transform operation type: {op_type}")


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

    converter = omni.kit.asset_converter.get_instance()
    task = converter.create_converter_task(
        str(input_path),
        str(output_path),
        lambda _progress, _total: None,
        context,
    )
    if not await task.wait_until_finished():
        raise RuntimeError(f"Asset conversion failed: {input_path}")


def prepare_asset() -> Path:
    if INPUT_ASSET.suffix.lower() in USD_INPUTS:
        return INPUT_ASSET
    enable_extension("omni.kit.asset_converter")
    simulation_app.update()
    converted = OUTPUT_DIR / "converted_asset.usd"
    from omni.kit.async_engine import run_coroutine

    conversion = run_coroutine(convert_to_usd(INPUT_ASSET, converted))
    while not conversion.done():
        simulation_app.update()
    conversion.result()
    if not converted.is_file():
        raise RuntimeError(f"Converter did not create {converted}")
    return converted


def pose_geometry_root(object_prim: Usd.Prim) -> Usd.Prim:
    visuals = object_prim.GetChild("Visuals")
    return visuals if visuals and visuals.IsValid() else object_prim


def collect_pose_geometry(root_prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    return collect_mesh_geometry(root_prim, visible_only=True)


def collect_collision_geometry(root_prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    return collect_mesh_geometry(root_prim, collision_only=True)


def collect_mesh_geometry(
    root_prim: Usd.Prim,
    *,
    visible_only: bool = False,
    collision_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    triangles = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if collision_only and not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if visible_only:
            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        if not points:
            continue
        transform = cache.GetLocalToWorldTransform(prim)
        offset = len(vertices)
        vertices.extend(
            tuple(float(value) for value in transform.Transform(point))
            for point in points
        )
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if counts or indices:
            triangles.extend(triangulate_faces(counts, indices, offset=offset))
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
    )


def set_object_orientation(
    object_xform: UsdGeom.Xform,
    quaternion_wxyz: tuple[float, float, float, float],
) -> None:
    operation = get_or_add_xform_op(object_xform, UsdGeom.XformOp.TypeOrient)
    w, x, y, z = quaternion_wxyz
    if operation.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        operation.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        operation.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def build_pose_candidates(asset_path: Path) -> tuple[PoseCandidate, ...]:
    if ARGS.initial_pose == "recorded-support":
        recorded = recorded_support_pose_load(ARGS.recorded_support)
        stage = Usd.Stage.CreateInMemory()
        object_prim = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
        object_prim.GetReferences().AddReference(str(asset_path))
        source_hash = object_prim.GetAttribute(
            "v2d:sourceAssetFileSha256"
        ).Get()
        if not source_hash:
            raise RuntimeError(
                "recorded-support requires a USD generated with source-file hash provenance"
            )
        if source_hash != recorded.mesh_file_sha256:
            raise RuntimeError(
                "recorded-support mesh hash does not match the generated USD source"
            )
        return (
            PoseCandidate(
                pose_id="recorded_support",
                local_up=recorded.local_up,
                rotation_wxyz=recorded.initial_rotation_wxyz,
                metadata={
                    "mode": "recorded-support",
                    "pose_source": "recorded",
                    "fallback_used": False,
                    "mesh_hash_validated": True,
                    "recorded_support": recorded.to_dict(),
                },
            ),
        )

    if ARGS.initial_pose == "as-authored":
        local_up = tuple(ARGS.local_up or (0.0, 0.0, 1.0))
        return (
            PoseCandidate(
                pose_id="as_authored",
                local_up=local_up,
                rotation_wxyz=None,
                metadata={
                    "mode": "as-authored",
                    "local_up": list(local_up),
                },
            ),
        )

    stage = Usd.Stage.CreateInMemory()
    object_xform = UsdGeom.Xform.Define(stage, "/Object")
    object_prim = object_xform.GetPrim()
    object_prim.GetReferences().AddReference(str(asset_path))
    get_or_add_xform_op(object_xform, UsdGeom.XformOp.TypeScale).Set(
        Gf.Vec3f(ARGS.asset_scale, ARGS.asset_scale, ARGS.asset_scale)
    )
    points, faces = collect_pose_geometry(pose_geometry_root(object_prim))
    if len(points) < 3:
        raise RuntimeError("Principal-axis pose generation requires visual geometry")
    refine_support_plane = ARGS.initial_pose == "principal-6-support"
    support_points = points
    support_faces = faces
    support_geometry_source = "visual_geometry"
    if refine_support_plane:
        collision_points, collision_faces = collect_collision_geometry(object_prim)
        if len(collision_points) >= 3 and len(collision_faces):
            support_points = collision_points
            support_faces = collision_faces
            support_geometry_source = "authored_collision_geometry"
    principal_candidates = principal_axis_pose_candidates(
        points,
        faces,
        refine_support_plane=refine_support_plane,
        support_points=support_points,
        support_faces=support_faces,
    )

    def support_metadata(candidate) -> dict:
        refinement = candidate.support_plane
        if refinement is None:
            return {}
        return {
            "support_plane": {
                "applied": refinement.applied,
                "reason": refinement.reason,
                "correction_degrees": refinement.correction_degrees,
                "face_count": refinement.face_count,
                "surface_area_m2": refinement.surface_area,
                "projected_area_m2": refinement.projected_area,
                "normal_before": (
                    None
                    if refinement.normal_before is None
                    else list(refinement.normal_before)
                ),
                "plane_rms_error_m": refinement.plane_rms_error,
                "geometry_source": support_geometry_source,
                "geometry_vertex_count": len(support_points),
                "geometry_triangle_count": len(support_faces),
            },
            "principal_rotation_wxyz": list(candidate.principal_rotation_wxyz),
        }

    return tuple(
        PoseCandidate(
            pose_id=candidate.pose_id,
            local_up=candidate.local_up,
            rotation_wxyz=candidate.rotation_wxyz,
            metadata={
                "mode": ARGS.initial_pose,
                "axis_source": "visual_geometry_pca",
                "axis_index": candidate.axis_index,
                "axis_sign": candidate.axis_sign,
                "axis_extent_m": candidate.axis_extent,
                "singular_value": candidate.singular_value,
                "rotation_wxyz": list(candidate.rotation_wxyz),
                "local_up": list(candidate.local_up),
                **support_metadata(candidate),
            },
        )
        for candidate in principal_candidates
    )


def create_scene(asset_path: Path, pose_candidate: PoseCandidate) -> SceneState:
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetFramesPerSecond(ARGS.physics_hz)
    stage.SetTimeCodesPerSecond(ARGS.physics_hz)

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    ground = UsdGeom.Cube.Define(stage, GROUND_PATH)
    ground.CreateSizeAttr(1.0)
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.35)])
    ground_xform = UsdGeom.Xformable(ground.GetPrim())
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ARGS.ground_z - 0.05))
    ground_xform.AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1000.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

    object_xform = UsdGeom.Xform.Define(stage, OBJECT_PATH)
    object_prim = object_xform.GetPrim()
    object_prim.GetReferences().AddReference(str(asset_path))
    translate_op = get_or_add_xform_op(
        object_xform, UsdGeom.XformOp.TypeTranslate
    )
    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    get_or_add_xform_op(object_xform, UsdGeom.XformOp.TypeScale).Set(
        Gf.Vec3f(ARGS.asset_scale, ARGS.asset_scale, ARGS.asset_scale)
    )
    if pose_candidate.rotation_wxyz is not None:
        set_object_orientation(object_xform, pose_candidate.rotation_wxyz)

    preserve_authored_physics = object_prim.HasAPI(UsdPhysics.RigidBodyAPI) and any(
        prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(object_prim)
    )
    authored_mass = None
    if preserve_authored_physics:
        value = UsdPhysics.MassAPI(object_prim).GetMassAttr().Get()
        if value is not None and float(value) > 0:
            authored_mass = float(value)
    effective_mass = authored_mass or ARGS.fallback_mass_kg
    mesh_count = 0
    for prim in Usd.PrimRange(object_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.IsInstanceProxy():
            raise RuntimeError(
                f"Cannot author collision on instance proxy {prim.GetPath()}; "
                "convert with instancing disabled"
            )
        if not preserve_authored_physics:
            UsdPhysics.CollisionAPI.Apply(prim)
            collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            collision.CreateApproximationAttr().Set(ARGS.collision_approximation)
        mesh_count += 1
    if mesh_count == 0:
        raise RuntimeError(
            f"No UsdGeom.Mesh prims were composed below {OBJECT_PATH}. "
            "The USD asset must have a default prim."
        )

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox_cache.Clear()
    initial_bounds = bbox_cache.ComputeWorldBound(object_prim).ComputeAlignedRange()
    initial_bottom = float(initial_bounds.GetMin()[2])
    initial_extent = initial_bounds.GetMax() - initial_bounds.GetMin()
    object_height = float(initial_extent[2])
    characteristic_extent = max(float(value) for value in initial_extent)
    contact_config = physics_contact_config(characteristic_extent)
    for prim in Usd.PrimRange(object_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_collision.CreateRestOffsetAttr().Set(contact_config["rest_offset_m"])
        physx_collision.CreateContactOffsetAttr().Set(contact_config["contact_offset_m"])
    ground_collision = PhysxSchema.PhysxCollisionAPI.Apply(ground.GetPrim())
    ground_collision.CreateRestOffsetAttr().Set(contact_config["rest_offset_m"])
    ground_collision.CreateContactOffsetAttr().Set(contact_config["contact_offset_m"])
    physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(object_prim)
    physx_rigid.CreateSolverPositionIterationCountAttr().Set(
        contact_config["solver_position_iterations"]
    )
    physx_rigid.CreateSolverVelocityIterationCountAttr().Set(
        contact_config["solver_velocity_iterations"]
    )
    physx_rigid.CreateEnableCCDAttr().Set(False)
    lift_height = (
        ARGS.lift_height
        if ARGS.lift_height is not None
        else ARGS.lift_height_ratio * characteristic_extent
    )
    start_root_z = ARGS.ground_z + ARGS.initial_clearance - initial_bottom
    translate_op.Set(Gf.Vec3d(0.0, 0.0, start_root_z))

    object_view = RigidPrim(
        OBJECT_PATH,
        masses=None if authored_mass is not None else [effective_mass],
        contact_filter_paths=GROUND_PATH,
        max_contact_count=32,
    )
    object_view.set_enabled_contact_tracking([True], threshold=0.0)
    body_api = UsdPhysics.RigidBodyAPI(object_prim)
    body_api.CreateRigidBodyEnabledAttr(True)
    body_api.CreateKinematicEnabledAttr(True)
    return SceneState(
        object_prim=object_prim,
        object_view=object_view,
        body_api=body_api,
        translate_op=translate_op,
        mesh_count=mesh_count,
        start_root_z=start_root_z,
        lift_height=lift_height,
        object_height=object_height,
        characteristic_extent=characteristic_extent,
        contact_config=contact_config,
        preserved_authored_physics=preserve_authored_physics,
        effective_mass=effective_mass,
        used_authored_mass=authored_mass is not None,
        local_up=pose_candidate.local_up,
        pose_selection=pose_candidate.metadata,
    )


def configure_camera(
    object_prim: Usd.Prim,
    lift_height: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    simulation_app.update()
    camera = ViewportManager.get_camera()
    if camera is None:
        raise RuntimeError("No active viewport camera is available")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = bbox_cache.ComputeWorldBound(object_prim).ComputeAlignedRange()
    size = bounds.GetSize()
    bottom = float(bounds.GetMin()[2])
    top_at_release = float(bounds.GetMax()[2]) + lift_height
    framing_extent = max(
        top_at_release - bottom,
        float(size[0]),
        float(size[1]),
        0.01,
    )
    camera_target = tuple(
        ARGS.camera_target or (0.0, 0.0, 0.5 * (bottom + top_at_release))
    )
    distance = ARGS.camera_distance_scale * framing_extent
    camera_eye = tuple(
        ARGS.camera_eye
        or (
            0.85 * distance,
            0.85 * distance,
            camera_target[2] + 0.4 * distance,
        )
    )
    ViewportManager.set_camera_view(camera, eye=camera_eye, target=camera_target)
    ViewportManager.set_resolution(ARGS.resolution)
    simulation_app.update()
    return camera_eye, camera_target


def start_video_capture(frame_stem: str) -> CaptureState | None:
    if not ARGS.video:
        return None
    import omni.kit.viewport.utility as viewport_utils

    viewport = viewport_utils.get_active_viewport()
    if viewport is None:
        raise RuntimeError("MP4 capture requires an active viewport")
    frames_dir = OUTPUT_DIR / f"{frame_stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in frames_dir.glob(f"{frame_stem}.*.png"):
        stale_frame.unlink()
    return CaptureState(
        viewport=viewport,
        frames_dir=frames_dir,
        frame_stem=frame_stem,
    )


def capture_live_frame(capture: CaptureState | None) -> None:
    if capture is None:
        return
    import omni.kit.renderer_capture
    import omni.kit.viewport.utility as viewport_utils
    from omni.kit.async_engine import run_coroutine

    settings = carb.settings.get_settings()
    play_simulations_setting = "/app/player/playSimulations"
    was_simulating = settings.get_as_bool(play_simulations_setting)
    if was_simulating:
        settings.set_bool(play_simulations_setting, False)
    frame_path = (
        capture.frames_dir / f"{capture.frame_stem}.{capture.frame_index:04d}.png"
    )
    try:
        helper = viewport_utils.capture_viewport_to_file(
            capture.viewport, file_path=str(frame_path)
        )
        completion = run_coroutine(helper.wait_for_result(completion_frames=30))
        while not completion.done():
            simulation_app.update()
        if not completion.result():
            raise RuntimeError(f"Failed to capture live frame {frame_path}")
        omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        capture.frame_index += 1
    finally:
        if was_simulating:
            settings.set_bool(play_simulations_setting, True)


def step_simulation(capture: CaptureState | None) -> None:
    SimulationManager.step(update_fabric=SimulationManager.is_fabric_enabled())
    RenderingManager.render()
    if capture is None:
        return
    capture.physics_step_index += 1
    next_frame = capture.frame_index + 1
    if (
        capture.physics_step_index * ARGS.video_fps
        >= next_frame * ARGS.physics_hz
    ):
        capture_live_frame(capture)


def set_kinematic_position(
    object_view: RigidPrim,
    translate_op: UsdGeom.XformOp,
    position: list[float],
) -> None:
    """Move the kinematic body to an exact world-space position."""
    object_view.set_world_poses(positions=[position])
    translate_op.Set(Gf.Vec3d(*position))


def run_drop(scene: SceneState, capture: CaptureState | None) -> dict:
    pre_lift_frames = max(0, round(ARGS.pre_lift_seconds * ARGS.physics_hz))
    lift_frames = max(1, round(ARGS.lift_seconds * ARGS.physics_hz))
    hold_frames = max(0, round(ARGS.hold_seconds * ARGS.physics_hz))
    max_drop_frames = max(1, round(ARGS.max_drop_seconds * ARGS.physics_hz))
    required_settle_frames = max(1, round(ARGS.settle_seconds * ARGS.physics_hz))
    post_settle_frames = max(
        0,
        round(ARGS.post_settle_seconds * ARGS.physics_hz),
    )
    release_root_z = scene.start_root_z + scene.lift_height

    for _ in range(pre_lift_frames):
        set_kinematic_position(
            scene.object_view,
            scene.translate_op,
            [0.0, 0.0, scene.start_root_z],
        )
        step_simulation(capture)
    for frame in range(lift_frames):
        alpha = (frame + 1) / lift_frames
        z = scene.start_root_z + alpha * scene.lift_height
        set_kinematic_position(
            scene.object_view,
            scene.translate_op,
            [0.0, 0.0, z],
        )
        step_simulation(capture)
    for _ in range(hold_frames):
        set_kinematic_position(
            scene.object_view,
            scene.translate_op,
            [0.0, 0.0, release_root_z],
        )
        step_simulation(capture)

    release_positions, release_orientations = scene.object_view.get_world_poses()
    release_position = release_positions.numpy()[0].tolist()
    release_orientation = release_orientations.numpy()[0].tolist()
    release_tilt = classify_standing(
        quaternion_wxyz=release_orientation,
        settled=False,
        contact_seen=False,
        touching_ground=False,
        max_tilt_degrees=ARGS.standing_angle,
        local_up=scene.local_up,
    ).tilt_degrees
    scene.body_api.GetKinematicEnabledAttr().Set(False)
    PhysxSchema.PhysxRigidBodyAPI(scene.body_api.GetPrim()).GetEnableCCDAttr().Set(True)
    RenderingManager.render()

    first_contact_frame = None
    simulated_frames = 0
    linear_velocity = np.zeros(3)
    angular_velocity = np.zeros(3)
    contact_force = np.zeros(3)
    settle_tracker = SettleTracker(
        required_frames=required_settle_frames,
        linear_speed_limit=ARGS.linear_speed_limit,
        angular_speed_limit=ARGS.angular_speed_limit,
    )
    pose_settle_position_tolerance = max(
        0.001,
        min(
            0.01,
            ARGS.pose_settle_position_ratio * scene.characteristic_extent,
        ),
    )
    pose_settle_tracker = PoseSettleTracker(
        required_frames=required_settle_frames,
        position_tolerance=pose_settle_position_tolerance,
        angular_tolerance_degrees=ARGS.pose_settle_angle,
    )

    for frame in range(max_drop_frames):
        step_simulation(capture)
        simulated_frames = frame + 1
        positions, orientations = scene.object_view.get_world_poses()
        linear, angular = scene.object_view.get_velocities()
        linear_velocity = linear.numpy()[0]
        angular_velocity = angular.numpy()[0]
        contact_force = scene.object_view.get_net_contact_forces(
            dt=1.0 / ARGS.physics_hz
        ).numpy()[0]
        contact_active = (
            np.linalg.norm(contact_force) > ARGS.contact_force_threshold
        )
        if contact_active and not settle_tracker.contact_seen:
            first_contact_frame = frame
        velocity_settled = settle_tracker.observe(
            contact_active=contact_active,
            linear_speed=float(np.linalg.norm(linear_velocity)),
            angular_speed=float(np.linalg.norm(angular_velocity)),
        )
        pose_settle_tracker.observe(
            contact_active=contact_active,
            position=positions.numpy()[0],
            orientation_wxyz=orientations.numpy()[0],
        )
        if velocity_settled:
            break

    settled_before_post_observation = (
        settle_tracker.settled or pose_settle_tracker.settled
    )
    timed_out_before_post_observation = (
        simulated_frames >= max_drop_frames
        and not settled_before_post_observation
    )
    for post_frame in range(post_settle_frames):
        step_simulation(capture)
        positions, orientations = scene.object_view.get_world_poses()
        linear, angular = scene.object_view.get_velocities()
        linear_velocity = linear.numpy()[0]
        angular_velocity = angular.numpy()[0]
        contact_force = scene.object_view.get_net_contact_forces(
            dt=1.0 / ARGS.physics_hz
        ).numpy()[0]
        contact_active = (
            np.linalg.norm(contact_force) > ARGS.contact_force_threshold
        )
        if contact_active and not settle_tracker.contact_seen:
            first_contact_frame = simulated_frames + post_frame
        settle_tracker.observe(
            contact_active=contact_active,
            linear_speed=float(np.linalg.norm(linear_velocity)),
            angular_speed=float(np.linalg.norm(angular_velocity)),
        )
        pose_settle_tracker.observe(
            contact_active=contact_active,
            position=positions.numpy()[0],
            orientation_wxyz=orientations.numpy()[0],
        )

    velocity_settled = settle_tracker.settled
    pose_settled = pose_settle_tracker.settled
    settled = velocity_settled or pose_settled
    timed_out = simulated_frames >= max_drop_frames and not settled
    positions, orientations = scene.object_view.get_world_poses()
    return {
        "contact_seen": settle_tracker.contact_seen,
        "first_contact_seconds_after_release": (
            None if first_contact_frame is None else first_contact_frame / ARGS.physics_hz
        ),
        "settled_before_post_observation": settled_before_post_observation,
        "settled": settled,
        "settled_by_velocity": velocity_settled,
        "settled_by_pose_window": pose_settled,
        "pose_settle_metric": "full-translation-and-orientation",
        "post_settle_observation_frames": post_settle_frames,
        "final_consecutive_settled_frames": settle_tracker.consecutive_frames,
        "pose_settle_window_frames": len(pose_settle_tracker.positions),
        "pose_settle_position_tolerance_m": pose_settle_position_tolerance,
        "pose_settle_position_span_m": pose_settle_tracker.position_span,
        "pose_settle_angle_tolerance_degrees": ARGS.pose_settle_angle,
        "pose_settle_angular_span_degrees": (
            pose_settle_tracker.angular_span_degrees
        ),
        "timed_out_before_post_observation": timed_out_before_post_observation,
        "timed_out": timed_out,
        "drop_simulation_seconds": simulated_frames / ARGS.physics_hz,
        "position": positions.numpy()[0].tolist(),
        "orientation_wxyz": orientations.numpy()[0].tolist(),
        "linear_speed": float(np.linalg.norm(linear_velocity)),
        "angular_speed": float(np.linalg.norm(angular_velocity)),
        "last_sampled_contact_force": contact_force.tolist(),
        "release_root_z": release_root_z,
        "release_position": release_position,
        "release_orientation_wxyz": release_orientation,
        "release_tilt_degrees": release_tilt,
    }


def transformed_mesh_bounds(
    root: Usd.Prim,
    *,
    include_invisible: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return exact world-space bounds from transformed mesh vertices below root."""

    if not root or not root.IsValid():
        raise RuntimeError("Missing geometry root for ground-clearance measurement")
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    vertex_count = 0
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if (
            not include_invisible
            and imageable
            and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
        ):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get(Usd.TimeCode.Default())
        if not points:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        for point in points:
            world = local_to_world.Transform(Gf.Vec3d(point))
            minimum = np.minimum(minimum, (world[0], world[1], world[2]))
            maximum = np.maximum(maximum, (world[0], world[1], world[2]))
        vertex_count += len(points)
    if vertex_count == 0:
        raise RuntimeError(f"No mesh vertices found below {root.GetPath()}")
    return minimum, maximum, vertex_count


def classify_final_state(
    result: dict,
    object_prim: Usd.Prim,
    local_up: tuple[float, float, float],
) -> None:
    visuals_prim = object_prim.GetChild("Visuals")
    visual_root = visuals_prim if visuals_prim and visuals_prim.IsValid() else object_prim
    visual_minimum, visual_maximum, visual_vertex_count = transformed_mesh_bounds(
        visual_root
    )
    visual = ground_metrics(visual_minimum, visual_maximum, ARGS.ground_z)
    collider_fields = {}
    support = visual
    colliders_prim = object_prim.GetChild("Colliders")
    if colliders_prim and colliders_prim.IsValid():
        collider_minimum, collider_maximum, collider_vertex_count = transformed_mesh_bounds(
            colliders_prim, include_invisible=True
        )
        collider = ground_metrics(collider_minimum, collider_maximum, ARGS.ground_z)
        support = collider
        collider_fields = {
            "collider_min_z": collider["minimum_z"],
            "collider_ground_clearance": collider["clearance"],
            "collider_ground_gap": collider["gap"],
            "collider_ground_penetration": collider["penetration"],
            "collider_vertex_count": collider_vertex_count,
            "visual_collider_min_z_offset": (
                visual["minimum_z"] - collider["minimum_z"]
            ),
        }
    classification = classify_standing(
        quaternion_wxyz=result["orientation_wxyz"],
        settled=result["settled"],
        contact_seen=result["contact_seen"],
        touching_ground=is_within_ground_tolerance(
            support["clearance"], ARGS.ground_tolerance
        ),
        max_tilt_degrees=ARGS.standing_angle,
        local_up=local_up,
    )
    outcome = classify_drop_test_outcome(
        contact_seen=result["contact_seen"],
        settled=result["settled"],
        timed_out=result["timed_out"],
        standing=classification.standing,
        standing_required=ARGS.fail_if_not_standing,
    )
    result.update(
        {
            "object_min_z": visual["minimum_z"],
            "visual_ground_clearance": visual["clearance"],
            "visual_ground_distance": visual["distance"],
            "visual_ground_gap": visual["gap"],
            "visual_ground_penetration": visual["penetration"],
            "visual_ground_gap_ratio": visual["gap_ratio"],
            "visual_ground_penetration_ratio": visual["penetration_ratio"],
            "visual_characteristic_extent_m": visual["characteristic_extent"],
            "visual_vertex_count": visual_vertex_count,
            **collider_fields,
            "touching_ground": classification.touching_ground,
            "tilt_degrees": classification.tilt_degrees,
            "standing": classification.standing,
            "standing_required": ARGS.fail_if_not_standing,
            "standing_requirement_passed": outcome.standing_requirement_passed,
            "rigid_body_test_passed": outcome.rigid_body_test_passed,
            "passed": outcome.passed,
            "failure_reasons": list(outcome.failure_reasons),
            "status": "passed" if outcome.passed else "failed",
        }
    )


def finish_capture(capture: CaptureState | None) -> list[str]:
    if capture is None:
        return []
    frame_count = capture.frame_index
    if frame_count == 0:
        raise RuntimeError("No live simulation frames were captured")
    video_path = OUTPUT_DIR / f"{capture.frame_stem}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(ARGS.video_fps),
            "-i",
            str(capture.frames_dir / f"{capture.frame_stem}.%04d.png"),
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )
    if not video_path.is_file():
        raise RuntimeError(f"FFmpeg did not create {video_path}")
    for frame_path in capture.frames_dir.glob(f"{capture.frame_stem}.*.png"):
        frame_path.unlink()
    capture.frames_dir.rmdir()
    return [str(video_path)]


def refresh_final_motion(result: dict, object_view: RigidPrim) -> None:
    positions, orientations = object_view.get_world_poses()
    linear, angular = object_view.get_velocities()
    result.update(
        {
            "position": positions.numpy()[0].tolist(),
            "orientation_wxyz": orientations.numpy()[0].tolist(),
            "linear_speed": float(np.linalg.norm(linear.numpy()[0])),
            "angular_speed": float(np.linalg.norm(angular.numpy()[0])),
        }
    )


def run_pose_candidate(
    prepared_asset: Path,
    pose_candidate: PoseCandidate,
    pose_index: int,
    timeline,
    capture: CaptureState | None,
) -> tuple[dict, CaptureState | None]:
    scene = create_scene(prepared_asset, pose_candidate)
    SimulationManager.set_physics_dt(1.0 / ARGS.physics_hz)
    timeline.set_time_codes_per_second(ARGS.physics_hz)
    camera_eye, camera_target = configure_camera(
        scene.object_prim,
        scene.lift_height,
    )

    if capture is None:
        capture = start_video_capture("drop_test")
    segment_start_frame = capture.frame_index if capture is not None else None
    timeline.play()
    simulation_app.update()
    try:
        result = run_drop(scene, capture)
        refresh_final_motion(result, scene.object_view)
        classify_final_state(result, scene.object_prim, scene.local_up)
        candidate_details = {
            "pose_index": pose_index,
            "pose_id": pose_candidate.pose_id,
            "input_asset": str(INPUT_ASSET),
            "prepared_asset": str(prepared_asset),
            "mesh_count": scene.mesh_count,
            "preserved_authored_physics": scene.preserved_authored_physics,
            "mass_kg": scene.effective_mass,
            "mass_source": (
                "authored_asset" if scene.used_authored_mass else "fallback_argument"
            ),
            "physics_contact_config": scene.contact_config,
            "asset_scale": ARGS.asset_scale,
            "object_height_m": scene.object_height,
            "characteristic_extent_m": scene.characteristic_extent,
            "lift_height_m": scene.lift_height,
            "lift_height_ratio": ARGS.lift_height_ratio,
            "pre_lift_seconds": ARGS.pre_lift_seconds,
            "post_settle_seconds": ARGS.post_settle_seconds,
            "local_up": list(scene.local_up),
            "initial_pose": ARGS.initial_pose,
            "pose_selection": scene.pose_selection,
            "camera_eye": list(camera_eye),
            "camera_target": list(camera_target),
            "camera_distance_scale": ARGS.camera_distance_scale,
            "ground_tolerance_m": ARGS.ground_tolerance,
            "standing_angle_limit_degrees": ARGS.standing_angle,
            "video_files": [],
        }
        if capture is not None and segment_start_frame is not None:
            candidate_details["video_segment"] = {
                "start_frame": segment_start_frame,
                "end_frame": capture.frame_index,
                "start_seconds": segment_start_frame / ARGS.video_fps,
                "end_seconds": capture.frame_index / ARGS.video_fps,
            }
        result.update(candidate_details)
        return result, capture
    finally:
        timeline.stop()
        simulation_app.update()


def run_workflow() -> int:
    prepared_asset = prepare_asset()
    pose_candidates = build_pose_candidates(prepared_asset)
    timeline = omni.timeline.get_timeline_interface()
    pose_results = []
    capture = None
    for pose_index, pose_candidate in enumerate(pose_candidates):
        print(
            f"Running pose {pose_index + 1}/{len(pose_candidates)}: "
            f"{pose_candidate.pose_id}"
        )
        pose_result, capture = run_pose_candidate(
            prepared_asset,
            pose_candidate,
            pose_index,
            timeline,
            capture,
        )
        pose_results.append(pose_result)

    video_files = finish_capture(capture)
    for pose_result in pose_results:
        pose_result["video_files"] = video_files

    result = aggregate_pose_results(
        pose_results,
        standing_required=ARGS.fail_if_not_standing,
    )
    first_pose = pose_results[0]
    result.update(
        {
            "input_asset": str(INPUT_ASSET),
            "prepared_asset": str(prepared_asset),
            "mesh_count": first_pose["mesh_count"],
            "preserved_authored_physics": first_pose["preserved_authored_physics"],
            "mass_kg": first_pose["mass_kg"],
            "mass_source": first_pose["mass_source"],
            "physics_contact_config": first_pose["physics_contact_config"],
            "asset_scale": ARGS.asset_scale,
            "lift_height_m": first_pose["lift_height_m"],
            "lift_height_ratio": ARGS.lift_height_ratio,
            "initial_pose": ARGS.initial_pose,
            "pose_selection": {
                "mode": ARGS.initial_pose,
                "candidate_count": len(pose_results),
                "selection_policy": (
                    "single_recorded_support_pose"
                    if ARGS.initial_pose == "recorded-support"
                    else "pass_if_any_candidate_meets_configured_checks"
                ),
                "representative_candidate_index": result[
                    "representative_candidate_index"
                ],
                "representative_pose_id": result["representative_pose_id"],
            },
            "ground_tolerance_m": ARGS.ground_tolerance,
            "standing_angle_limit_degrees": ARGS.standing_angle,
            "video_fps": ARGS.video_fps,
            "pose_results": pose_results,
            "video_files": video_files,
        }
    )
    result_path = OUTPUT_DIR / "drop_test_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "rigid_body_pass_count": result["rigid_body_pass_count"],
                "standing_pass_count": result["standing_pass_count"],
                "representative_pose_id": result["representative_pose_id"],
                "video_files": result["video_files"],
            },
            indent=2,
        )
    )
    print(f"Result written to {result_path}")
    return 0 if result["passed"] else 2


try:
    EXIT_CODE = run_workflow()
except Exception as error:
    failure = {
        "status": "error",
        "input_asset": str(INPUT_ASSET),
        "error": f"{type(error).__name__}: {error}",
    }
    (OUTPUT_DIR / "drop_test_result.json").write_text(
        json.dumps(failure, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(failure, indent=2), file=sys.stderr)
    EXIT_CODE = 1
finally:
    simulation_app.close()

raise SystemExit(EXIT_CODE)
