"""Convert the pinned YAMLab single-arm USD into a bimanual URDF bundle.

The converter is intentionally separate from the renderer.  It runs offline in
the existing ``robotic-grounding:photo-render-v6`` container, where ``pxr``,
``trimesh``, ``yourdfpy``, ``pinocchio``, and ``pyrender`` are already pinned.
USD- and mesh-specific packages are imported only while conversion or runtime
validation is executing, so importing this module on the host remains cheap.

Visual meshes are embedded in the source USD.  Each authored visual primitive
is transformed into its owning rigid-body frame and exported as a binary STL;
the bound USD material's diffuse color is represented by an inline URDF color.
The two arms share these immutable meshes but have independently prefixed links
and joints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


BUNDLE_SCHEMA = "v2d.inpainting.parallel-jaw-robot-bundle/v1"
PINNED_YAMLAB_COMMIT = "ec0455d2b4ce35f21fc126418ea5e74ac567133d"
PINNED_YAM_USD_SHA256 = (
    "8b997d9c864a53a00abba54d2381d3aebf71853f7918450305afc11e73aeb499"
)
PINNED_YAM_CONFIG_SHA256 = (
    "edb89ddf2b418a48a0b46632aafea6a082a306048df4f7c2f2a2301cced17dd1"
)
PINNED_REPOSITORY_URL = "https://github.com/ARISE-Initiative/yamlab.git"

SOURCE_USD_RELATIVE = Path("yamlab/robot/yam/arm/yam.usd")
SOURCE_CONFIG_RELATIVE = Path("yamlab/configs/robot/yam.yaml")
SOURCE_LICENSE_RELATIVE = Path("LICENSE")

RENDER_URDF_NAME = "yam_bimanual_render.urdf"
IK_URDF_NAME = "yam_bimanual_ik.urdf"
MANIFEST_NAME = "bundle_manifest.json"
ROBOT_ROOT_LINK = "robot_root"
ARM_ROOT_BODY = "arm"
SIDE_OFFSETS_Y = {"left": 0.305, "right": -0.305}
TCP_OFFSET_LINK_6 = (0.0, 0.0, 0.14256)
MATERIAL_FALLBACK_RGBA = (0.18, 0.18, 0.20, 1.0)

# Fixed embodiment calibration for the common semantic parallel-jaw target.
#
# ``T_HUB_ROBOT_ROOT`` is composed on the right of each clip's shared
# ``T_world_hub``.  It moves the wider, shorter YAM pair 15 cm forward and
# rolls it -10 degrees in the hub frame.  The bundle contract stores the
# inverse, ``T_robot_root_hub``.
#
# A two-finger parallel jaw is contact-equivalent after a 180-degree roll
# around its approach axis.  Selecting that equivalent roll for the left YAM
# wrist keeps the finite joint-6 range away from a branch cut; the right side
# already uses the reachable representative.  This is one embodiment mapping
# shared by clips and trackers, not a per-condition fit.
YAM_MOUNT_FORWARD_X_M = 0.15
YAM_MOUNT_ROLL_DEG = -10.0
_YAM_MOUNT_COS = math.cos(math.radians(YAM_MOUNT_ROLL_DEG))
_YAM_MOUNT_SIN = math.sin(math.radians(YAM_MOUNT_ROLL_DEG))
YAM_T_HUB_ROBOT_ROOT = (
    (1.0, 0.0, 0.0, YAM_MOUNT_FORWARD_X_M),
    (0.0, _YAM_MOUNT_COS, -_YAM_MOUNT_SIN, 0.0),
    (0.0, _YAM_MOUNT_SIN, _YAM_MOUNT_COS, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
YAM_T_ROBOT_ROOT_HUB = (
    (1.0, 0.0, 0.0, -YAM_MOUNT_FORWARD_X_M),
    (0.0, _YAM_MOUNT_COS, _YAM_MOUNT_SIN, 0.0),
    (0.0, -_YAM_MOUNT_SIN, _YAM_MOUNT_COS, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
YAM_SEMANTIC_TARGET_TO_TCP_ROTATION = {
    "left": (
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "right": (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
}

EXPECTED_BODY_NAMES = (
    "arm",
    "link_1",
    "link_2",
    "link_3",
    "link_4",
    "link_5",
    "link_6",
    "left_finger",
    "right_finger",
    "camera_mount",
    "camera_d405",
    "camera_frame",
)
EXPECTED_JOINT_TYPES = {
    "rootJoint_arm": "fixed",
    "joint1": "revolute",
    "joint2": "revolute",
    "joint3": "revolute",
    "joint4": "revolute",
    "joint5": "revolute",
    "joint6": "revolute",
    "left_finger": "prismatic",
    "right_finger": "prismatic",
    "camera_mount": "fixed",
    "camera_d405": "fixed",
    "camera_frame": "fixed",
}
EMITTED_JOINT_ORDER = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "left_finger",
    "right_finger",
    "camera_mount",
    "camera_d405",
    "camera_frame",
)
FINGER_JOINTS = frozenset({"left_finger", "right_finger"})
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class YamConversionError(ValueError):
    """Raised when the pinned input or generated bundle violates its contract."""


@dataclass(frozen=True)
class MaterialSpec:
    rgba: tuple[float, float, float, float]
    provenance: str


@dataclass(frozen=True)
class VisualSpec:
    body: str
    source_prim: str
    mesh_path: str
    material: MaterialSpec
    body_from_mesh: tuple[tuple[float, ...], ...]
    vertex_count: int
    triangle_count: int


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    joint_type: str
    parent_from_child_zero: tuple[tuple[float, ...], ...]
    axis_child: tuple[float, float, float] | None
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None
    source_axis: str | None
    source_lower: float | None
    source_upper: float | None
    authored_zero_pose_max_abs_residual: float

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.parent_from_child_zero, dtype=np.float64)


@dataclass(frozen=True)
class ArmModel:
    bodies: tuple[str, ...]
    visuals: tuple[VisualSpec, ...]
    joints: tuple[JointSpec, ...]
    meters_per_unit: float
    finger_open_position: float
    finger_closed_position: float
    closed_aperture_m: float
    open_aperture_m: float
    fingertip_keypoints: Mapping[str, tuple[float, float, float]]

    def joint(self, name: str) -> JointSpec:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(name)


@dataclass(frozen=True)
class ConversionResult:
    bundle_dir: Path
    render_urdf: Path
    ik_urdf: Path
    manifest: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "bundle_dir": str(self.bundle_dir),
            "render_urdf": str(self.render_urdf),
            "ik_urdf": str(self.ik_urdf),
            "manifest": str(self.manifest),
        }


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash one regular file without loading it into memory at once."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, recorded_path: str) -> dict[str, Any]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": recorded_path,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _git_command(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "-C",
            str(repository),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise YamConversionError(
            f"git {' '.join(arguments)} failed in {repository}: {detail}"
        )
    return completed


def _repository_record(
    repository: Path,
    source_paths: Sequence[Path],
    *,
    expected_commit: str,
) -> dict[str, Any]:
    root = repository.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise YamConversionError("expected repository commit must be 40 lowercase hex digits")
    commit = _git_command(root, "rev-parse", "HEAD").stdout.strip()
    if commit != expected_commit:
        raise YamConversionError(
            f"YAMLab commit {commit!r} does not match pinned {expected_commit!r}"
        )
    tree = _git_command(root, "rev-parse", "HEAD^{tree}").stdout.strip()
    if COMMIT_RE.fullmatch(tree) is None:
        raise YamConversionError(f"invalid Git tree ID {tree!r}")

    source_records: dict[str, Any] = {}
    for source in source_paths:
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise YamConversionError(
                f"input {resolved} is outside repository {root}"
            ) from exc
        relative_text = relative.as_posix()
        _git_command(root, "ls-files", "--error-unmatch", "--", relative_text)
        dirty = _git_command(
            root, "diff", "--quiet", "HEAD", "--", relative_text, check=False
        )
        if dirty.returncode != 0:
            raise YamConversionError(
                f"tracked input {relative_text} differs from pinned commit {commit}"
            )
        blob = _git_command(root, "rev-parse", f"HEAD:{relative_text}").stdout.strip()
        if COMMIT_RE.fullmatch(blob) is None:
            raise YamConversionError(f"invalid Git blob ID for {relative_text}: {blob!r}")
        source_records[relative_text] = {"git_blob_sha1": blob}

    remote = _git_command(root, "remote", "get-url", "origin").stdout.strip()
    normalized_remote = remote.removesuffix(".git")
    normalized_expected = PINNED_REPOSITORY_URL.removesuffix(".git")
    if normalized_remote != normalized_expected:
        raise YamConversionError(
            f"YAMLab origin {remote!r} does not match {PINNED_REPOSITORY_URL!r}"
        )
    return {
        "url": PINNED_REPOSITORY_URL,
        "commit_sha1": commit,
        "tree_sha1": tree,
        "tracked_inputs": source_records,
    }


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise YamConversionError("transform must be a finite 4x4 matrix")
    return tuple(tuple(float(value) for value in row) for row in array)


def _format_float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise YamConversionError(f"cannot serialize non-finite value {number}")
    if abs(number) < 5e-16:
        number = 0.0
    return f"{number:.17g}"


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(_format_float(value) for value in values)


def _rotation_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return URDF fixed-axis roll, pitch, yaw from a proper rotation matrix."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise YamConversionError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-6):
        raise YamConversionError("joint origin rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=2e-6):
        raise YamConversionError("joint origin rotation is not proper")

    horizontal = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    pitch = math.atan2(-float(matrix[2, 0]), horizontal)
    if horizontal > 1e-10:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = math.atan2(-float(matrix[1, 2]), float(matrix[1, 1]))
        yaw = 0.0
    return roll, pitch, yaw


def _origin_from_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise YamConversionError("origin must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise YamConversionError("origin has an invalid homogeneous bottom row")
    xyz = tuple(float(value) for value in transform[:3, 3])
    return xyz, _rotation_to_rpy(transform[:3, :3])


def _joint_pose(joint: JointSpec, position: float) -> np.ndarray:
    if joint.axis_child is None:
        if position != 0.0:
            raise YamConversionError(f"fixed joint {joint.name} cannot be displaced")
        return joint.matrix
    motion = np.eye(4, dtype=np.float64)
    axis = np.asarray(joint.axis_child, dtype=np.float64)
    if joint.joint_type == "prismatic":
        motion[:3, 3] = axis * float(position)
    elif joint.joint_type == "revolute":
        x, y, z = axis
        angle = float(position)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        motion[:3, :3] = (
            cosine * np.eye(3)
            + (1.0 - cosine) * np.outer(axis, axis)
            + sine * cross
        )
    else:
        raise YamConversionError(f"unsupported moving joint type {joint.joint_type!r}")
    return joint.matrix @ motion


def _triangulate(
    face_counts: Sequence[int],
    face_indices: Sequence[int],
    *,
    hole_indices: frozenset[int],
    reverse_winding: bool,
) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for face_index, raw_count in enumerate(face_counts):
        count = int(raw_count)
        if count < 3:
            raise YamConversionError(f"USD face {face_index} has only {count} vertices")
        polygon = [int(value) for value in face_indices[cursor : cursor + count]]
        if len(polygon) != count:
            raise YamConversionError("USD faceVertexCounts exceeds faceVertexIndices")
        cursor += count
        if face_index in hole_indices:
            continue
        for offset in range(1, count - 1):
            triangle = (polygon[0], polygon[offset], polygon[offset + 1])
            if reverse_winding:
                triangle = (triangle[0], triangle[2], triangle[1])
            triangles.append(triangle)
    if cursor != len(face_indices):
        raise YamConversionError("USD faceVertexIndices contains unused values")
    if not triangles:
        raise YamConversionError("USD visual mesh contains no rendered triangles")
    return np.asarray(triangles, dtype=np.int64)


def _material_from_usd(mesh_prim: Any, UsdGeom: Any, UsdShade: Any) -> MaterialSpec:
    mesh = UsdGeom.Mesh(mesh_prim)
    material, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
    color: tuple[float, float, float] | None = None
    opacity: float | None = None
    provenance = ""

    if material:
        descendants: list[Any] = []
        pending = list(material.GetPrim().GetChildren())
        while pending:
            descendant = pending.pop(0)
            descendants.append(descendant)
            pending.extend(descendant.GetChildren())
        for input_name in ("diffuse_color_constant", "diffuseColor", "base_color"):
            for prim in descendants:
                value = prim.GetAttribute(f"inputs:{input_name}").Get()
                if value is not None and len(value) >= 3:
                    color = tuple(float(value[index]) for index in range(3))
                    provenance = f"{material.GetPath()}/inputs:{input_name}"
                    break
            if color is not None:
                break
        for input_name in ("opacity_constant", "opacity"):
            for prim in descendants:
                value = prim.GetAttribute(f"inputs:{input_name}").Get()
                if value is not None:
                    opacity = float(value)
                    break
            if opacity is not None:
                break

    if color is None:
        display = mesh.GetDisplayColorAttr().Get()
        if display:
            color = tuple(float(display[0][index]) for index in range(3))
            provenance = f"{mesh_prim.GetPath()}/primvars:displayColor"
    if opacity is None:
        display_opacity = mesh.GetDisplayOpacityAttr().Get()
        if display_opacity:
            opacity = float(display_opacity[0])
    if color is None:
        color = MATERIAL_FALLBACK_RGBA[:3]
        provenance = "converter_fallback"
    if opacity is None:
        opacity = MATERIAL_FALLBACK_RGBA[3]

    rgba = tuple(float(np.clip(value, 0.0, 1.0)) for value in (*color, opacity))
    if not np.isfinite(rgba).all():
        raise YamConversionError(f"material for {mesh_prim.GetPath()} is not finite")
    return MaterialSpec(rgba=rgba, provenance=provenance)


def _gf_local_matrix(Gf: Any, position: Any, quaternion: Any) -> Any:
    imaginary = quaternion.GetImaginary()
    normalized = Gf.Quatd(
        float(quaternion.GetReal()),
        Gf.Vec3d(*(float(value) for value in imaginary)),
    ).GetNormalized()
    return Gf.Matrix4d(
        Gf.Rotation(normalized),
        Gf.Vec3d(*(float(value) for value in position)),
    )


def _load_robot_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # lazy: only conversion needs the source YAML
        raise YamConversionError("PyYAML is required in the conversion container") from exc
    try:
        payload = yaml.safe_load(config_path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise YamConversionError(f"cannot parse YAM robot config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise YamConversionError("YAM robot config root must be an object")
    try:
        gripper = payload["gripper"]
        fingers = payload["fingers"]
        open_values = (
            float(gripper["left_finger_open_pos"]),
            float(gripper["right_finger_open_pos"]),
        )
        closed_values = (
            float(gripper["left_finger_closed_pos"]),
            float(gripper["right_finger_closed_pos"]),
        )
        tips = {
            "left_finger": tuple(float(value) for value in fingers["lf"]["keypoints"][0]),
            "right_finger": tuple(float(value) for value in fingers["rf"]["keypoints"][0]),
        }
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise YamConversionError("YAM config lacks valid gripper positions/keypoints") from exc
    if not np.isfinite((*open_values, *closed_values, *tips["left_finger"], *tips["right_finger"])).all():
        raise YamConversionError("YAM gripper config contains non-finite values")
    if not math.isclose(open_values[0], open_values[1], abs_tol=1e-12):
        raise YamConversionError("left/right YAM open positions differ")
    if not math.isclose(closed_values[0], closed_values[1], abs_tol=1e-12):
        raise YamConversionError("left/right YAM closed positions differ")
    return {
        "open_position": open_values[0],
        "closed_position": closed_values[0],
        "tips": tips,
    }


def _measure_fingertip_aperture(
    joints: Mapping[str, JointSpec],
    tips: Mapping[str, tuple[float, float, float]],
    position: float,
) -> float:
    points: list[np.ndarray] = []
    for name in ("left_finger", "right_finger"):
        point = np.ones(4, dtype=np.float64)
        point[:3] = tips[name]
        transformed = _joint_pose(joints[name], position) @ point
        points.append(transformed[:3])
    return float(np.linalg.norm(points[0] - points[1]))


def _extract_arm_model(
    source_usd: Path,
    config: Mapping[str, Any],
    output_root: Path,
) -> ArmModel:
    try:
        import trimesh
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # lazy: host-side module import must not require these
        raise YamConversionError(
            "pxr and trimesh are required; run inside the pinned photo-render container"
        ) from exc

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise YamConversionError(f"could not open USD stage {source_usd}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise YamConversionError("pinned YAM USD must use a Z-up stage")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise YamConversionError(f"invalid USD metersPerUnit {meters_per_unit}")
    external_layers = [
        layer.identifier
        for layer in stage.GetUsedLayers()
        if not layer.anonymous and Path(layer.realPath).resolve() != source_usd.resolve()
    ]
    if external_layers:
        raise YamConversionError(
            f"YAM USD is not self-contained; external layers: {external_layers}"
        )

    joint_prims: dict[str, Any] = {}
    type_by_usd = {
        "PhysicsRevoluteJoint": "revolute",
        "PhysicsPrismaticJoint": "prismatic",
        "PhysicsFixedJoint": "fixed",
    }
    for prim in stage.Traverse():
        joint_type = type_by_usd.get(prim.GetTypeName())
        if joint_type is not None:
            name = prim.GetName()
            if name in joint_prims:
                raise YamConversionError(f"duplicate USD joint name {name!r}")
            joint_prims[name] = prim
    if set(joint_prims) != set(EXPECTED_JOINT_TYPES):
        raise YamConversionError(
            "pinned USD joint set changed: "
            f"missing={sorted(set(EXPECTED_JOINT_TYPES) - set(joint_prims))}, "
            f"extra={sorted(set(joint_prims) - set(EXPECTED_JOINT_TYPES))}"
        )
    for name, expected_type in EXPECTED_JOINT_TYPES.items():
        actual_type = type_by_usd[joint_prims[name].GetTypeName()]
        if actual_type != expected_type:
            raise YamConversionError(
                f"joint {name!r} type {actual_type!r} != {expected_type!r}"
            )

    root_joint = joint_prims["rootJoint_arm"]
    root_body0 = root_joint.GetRelationship("physics:body0").GetTargets()
    root_body1 = root_joint.GetRelationship("physics:body1").GetTargets()
    if root_body0 or len(root_body1) != 1 or root_body1[0].name != ARM_ROOT_BODY:
        raise YamConversionError("rootJoint_arm no longer anchors the expected arm body")
    body_container = stage.GetPrimAtPath(root_body1[0]).GetParent()
    if not body_container:
        raise YamConversionError("could not resolve YAM rigid-body container")

    body_prims: dict[str, Any] = {}
    for child in body_container.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            body_prims[child.GetName()] = child
    if set(body_prims) != set(EXPECTED_BODY_NAMES):
        raise YamConversionError(
            "pinned USD rigid-body set changed: "
            f"missing={sorted(set(EXPECTED_BODY_NAMES) - set(body_prims))}, "
            f"extra={sorted(set(body_prims) - set(EXPECTED_BODY_NAMES))}"
        )

    cache = UsdGeom.XformCache()
    joints: list[JointSpec] = []
    for name in EMITTED_JOINT_ORDER:
        prim = joint_prims[name]
        targets0 = prim.GetRelationship("physics:body0").GetTargets()
        targets1 = prim.GetRelationship("physics:body1").GetTargets()
        if len(targets0) != 1 or len(targets1) != 1:
            raise YamConversionError(f"joint {name!r} must connect exactly two bodies")
        parent = targets0[0].name
        child = targets1[0].name
        if parent not in body_prims or child not in body_prims:
            raise YamConversionError(f"joint {name!r} targets an unknown rigid body")

        position0 = prim.GetAttribute("physics:localPos0").Get()
        position1 = prim.GetAttribute("physics:localPos1").Get()
        rotation0 = prim.GetAttribute("physics:localRot0").Get()
        rotation1 = prim.GetAttribute("physics:localRot1").Get()
        if None in (position0, position1, rotation0, rotation1):
            raise YamConversionError(f"joint {name!r} lacks authored local frames")
        local0_row = _gf_local_matrix(Gf, position0, rotation0)
        local1_row = _gf_local_matrix(Gf, position1, rotation1)
        # Gf uses row-vector transforms.  This is the column-vector equivalent
        # of T_parent_joint0 @ inverse(T_child_joint1).
        origin = np.array(local1_row.GetInverse() * local0_row, dtype=np.float64).T
        origin[:3, 3] *= meters_per_unit

        parent_world = cache.GetLocalToWorldTransform(body_prims[parent])
        child_world = cache.GetLocalToWorldTransform(body_prims[child])
        authored = np.array(
            child_world * parent_world.GetInverse(), dtype=np.float64
        ).T
        authored[:3, 3] *= meters_per_unit
        authored_residual = float(np.max(np.abs(origin - authored)))
        # The physics joint frames are authoritative.  The six arm joints and
        # fingers agree with the display-stage body transforms to sub-micron
        # precision.  At the pinned revision, camera_d405 differs by 20 um and
        # the visual-less camera_frame body differs by 7.3 mm; refusing those
        # fixed auxiliary bodies would replace exact authored joint frames with
        # stale display transforms.
        if name not in {"camera_d405", "camera_frame"} and authored_residual > 2e-6:
            raise YamConversionError(
                f"joint-frame origin for {name!r} does not reproduce authored zero pose"
            )
        if authored_residual > 0.01:
            raise YamConversionError(
                f"joint {name!r} authored zero-pose residual {authored_residual} is too large"
            )

        joint_type = EXPECTED_JOINT_TYPES[name]
        axis_child: tuple[float, float, float] | None = None
        source_axis: str | None = None
        lower: float | None = None
        upper: float | None = None
        effort: float | None = None
        velocity: float | None = None
        source_lower: float | None = None
        source_upper: float | None = None
        if joint_type != "fixed":
            source_axis = str(prim.GetAttribute("physics:axis").Get())
            basis_by_axis = {
                "X": np.array((1.0, 0.0, 0.0)),
                "Y": np.array((0.0, 1.0, 0.0)),
                "Z": np.array((0.0, 0.0, 1.0)),
            }
            if source_axis not in basis_by_axis:
                raise YamConversionError(
                    f"joint {name!r} has unsupported USD axis {source_axis!r}"
                )
            local1 = np.array(local1_row, dtype=np.float64).T
            axis = local1[:3, :3] @ basis_by_axis[source_axis]
            norm = float(np.linalg.norm(axis))
            if not math.isfinite(norm) or norm < 1e-10:
                raise YamConversionError(f"joint {name!r} has a degenerate axis")
            axis_child = tuple(float(value) for value in axis / norm)
            source_lower = float(prim.GetAttribute("physics:lowerLimit").Get())
            source_upper = float(prim.GetAttribute("physics:upperLimit").Get())
            if not math.isfinite(source_lower) or not math.isfinite(source_upper):
                raise YamConversionError(f"joint {name!r} has non-finite limits")
            if joint_type == "revolute":
                lower = math.radians(source_lower)
                upper = math.radians(source_upper)
                drive = "drive:angular:physics:maxForce"
            else:
                lower = source_lower * meters_per_unit
                upper = source_upper * meters_per_unit
                drive = "drive:linear:physics:maxForce"
            if lower > upper:
                raise YamConversionError(f"joint {name!r} has reversed limits")
            raw_effort = prim.GetAttribute(drive).Get()
            effort = float(raw_effort) if raw_effort is not None else 0.0
            if effort <= 0.0:
                effort = 100.0  # YAMLab actuator config for both fingers
            velocity = 100.0  # YAMLab actuator config at the pinned revision

        joints.append(
            JointSpec(
                name=name,
                parent=parent,
                child=child,
                joint_type=joint_type,
                parent_from_child_zero=_matrix_tuple(origin),
                axis_child=axis_child,
                lower=lower,
                upper=upper,
                effort=effort,
                velocity=velocity,
                source_axis=source_axis,
                source_lower=source_lower,
                source_upper=source_upper,
                authored_zero_pose_max_abs_residual=authored_residual,
            )
        )

    joint_map = {joint.name: joint for joint in joints}
    open_position = float(config["open_position"]) * meters_per_unit
    closed_position = float(config["closed_position"]) * meters_per_unit
    tips = {
        name: tuple(float(value) * meters_per_unit for value in point)
        for name, point in config["tips"].items()
    }
    for name in FINGER_JOINTS:
        joint = joint_map[name]
        assert joint.lower is not None and joint.upper is not None
        if not math.isclose(open_position, joint.lower, abs_tol=2e-8):
            raise YamConversionError(
                f"config open position {open_position} != USD lower limit {joint.lower}"
            )
        if not math.isclose(closed_position, joint.upper, abs_tol=2e-8):
            raise YamConversionError(
                f"config closed position {closed_position} != USD upper limit {joint.upper}"
            )
    closed_aperture = _measure_fingertip_aperture(
        joint_map, tips, closed_position
    )
    open_aperture = _measure_fingertip_aperture(joint_map, tips, open_position)
    if not (0.0 < closed_aperture < open_aperture):
        raise YamConversionError(
            f"invalid measured apertures closed={closed_aperture}, open={open_aperture}"
        )

    mesh_dir = output_root / "meshes"
    mesh_dir.mkdir()
    visuals: list[VisualSpec] = []
    for body in EXPECTED_BODY_NAMES:
        body_prim = body_prims[body]
        visuals_prim = body_prim.GetChild("visuals")
        mesh_prims = (
            sorted(
                (
                    prim
                    for prim in Usd.PrimRange(visuals_prim)
                    if prim.GetTypeName() == "Mesh"
                ),
                key=lambda prim: str(prim.GetPath()),
            )
            if visuals_prim
            else []
        )
        for ordinal, mesh_prim in enumerate(mesh_prims):
            usd_mesh = UsdGeom.Mesh(mesh_prim)
            if usd_mesh.GetSubdivisionSchemeAttr().Get() != UsdGeom.Tokens.none:
                raise YamConversionError(
                    f"visual {mesh_prim.GetPath()} uses unsupported subdivision"
                )
            points = usd_mesh.GetPointsAttr().Get()
            counts = usd_mesh.GetFaceVertexCountsAttr().Get()
            indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not counts or not indices:
                raise YamConversionError(f"visual {mesh_prim.GetPath()} is empty")

            mesh_world = cache.GetLocalToWorldTransform(mesh_prim)
            body_world = cache.GetLocalToWorldTransform(body_prim)
            mesh_to_body = mesh_world * body_world.GetInverse()
            vertices = np.asarray(
                [
                    tuple(mesh_to_body.Transform(Gf.Vec3d(*(float(v) for v in point))))
                    for point in points
                ],
                dtype=np.float64,
            )
            vertices *= meters_per_unit
            if not np.isfinite(vertices).all():
                raise YamConversionError(f"visual {mesh_prim.GetPath()} has non-finite points")
            holes = usd_mesh.GetHoleIndicesAttr().Get() or ()
            triangles = _triangulate(
                counts,
                indices,
                hole_indices=frozenset(int(value) for value in holes),
                reverse_winding=(
                    usd_mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
                ),
            )
            if triangles.min() < 0 or triangles.max() >= len(vertices):
                raise YamConversionError(
                    f"visual {mesh_prim.GetPath()} has out-of-range face indices"
                )

            safe_prim_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", mesh_prim.GetName())
            relative_path = Path("meshes") / f"{body}_{ordinal:02d}_{safe_prim_name}.stl"
            export_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=triangles,
                process=False,
            )
            encoded = export_mesh.export(file_type="stl")
            if isinstance(encoded, str):
                encoded = encoded.encode("utf-8")
            if not isinstance(encoded, bytes) or not encoded:
                raise YamConversionError(
                    f"trimesh returned no STL bytes for {mesh_prim.GetPath()}"
                )
            (output_root / relative_path).write_bytes(encoded)

            body_from_mesh = np.array(mesh_to_body, dtype=np.float64).T
            body_from_mesh[:3, 3] *= meters_per_unit
            visuals.append(
                VisualSpec(
                    body=body,
                    source_prim=str(mesh_prim.GetPath()),
                    mesh_path=relative_path.as_posix(),
                    material=_material_from_usd(mesh_prim, UsdGeom, UsdShade),
                    body_from_mesh=_matrix_tuple(body_from_mesh),
                    vertex_count=len(vertices),
                    triangle_count=len(triangles),
                )
            )
    bodies_with_visuals = {visual.body for visual in visuals}
    if set(EXPECTED_BODY_NAMES) - bodies_with_visuals != {"camera_frame"}:
        raise YamConversionError(
            "every pinned rigid body except camera_frame must contain a visual mesh"
        )
    return ArmModel(
        bodies=EXPECTED_BODY_NAMES,
        visuals=tuple(visuals),
        joints=tuple(joints),
        meters_per_unit=meters_per_unit,
        finger_open_position=open_position,
        finger_closed_position=closed_position,
        closed_aperture_m=closed_aperture,
        open_aperture_m=open_aperture,
        fingertip_keypoints=tips,
    )


def _prefixed(side: str, name: str) -> str:
    if side not in SIDE_OFFSETS_Y:
        raise YamConversionError(f"unknown side {side!r}")
    return f"{side}_{name}"


def _add_origin(joint_element: ET.Element, matrix: np.ndarray) -> None:
    xyz, rpy = _origin_from_matrix(matrix)
    ET.SubElement(
        joint_element,
        "origin",
        {"xyz": _format_vector(xyz), "rpy": _format_vector(rpy)},
    )


def _add_joint_element(
    root: ET.Element,
    *,
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    origin: np.ndarray,
    axis: Sequence[float] | None = None,
    lower: float | None = None,
    upper: float | None = None,
    effort: float | None = None,
    velocity: float | None = None,
) -> None:
    element = ET.SubElement(root, "joint", {"name": name, "type": joint_type})
    ET.SubElement(element, "parent", {"link": parent})
    ET.SubElement(element, "child", {"link": child})
    _add_origin(element, origin)
    if joint_type == "fixed":
        return
    if axis is None or None in (lower, upper, effort, velocity):
        raise YamConversionError(f"moving joint {name!r} lacks axis/limits")
    ET.SubElement(element, "axis", {"xyz": _format_vector(axis)})
    ET.SubElement(
        element,
        "limit",
        {
            "lower": _format_float(float(lower)),
            "upper": _format_float(float(upper)),
            "effort": _format_float(float(effort)),
            "velocity": _format_float(float(velocity)),
        },
    )


def _build_urdf(model: ArmModel, *, fixed_open_fingers: bool) -> bytes:
    variant = "ik" if fixed_open_fingers else "render"
    root = ET.Element("robot", {"name": f"yam_bimanual_{variant}"})
    root.append(
        ET.Comment(
            " Generated from pinned YAMLab USD; meshes are baked into rigid-link frames. "
        )
    )
    ET.SubElement(root, "link", {"name": ROBOT_ROOT_LINK})
    visuals_by_body: dict[str, list[VisualSpec]] = {
        body: [] for body in model.bodies
    }
    for visual in model.visuals:
        visuals_by_body[visual.body].append(visual)

    for side in ("left", "right"):
        for body in model.bodies:
            link = ET.SubElement(root, "link", {"name": _prefixed(side, body)})
            for ordinal, visual in enumerate(visuals_by_body[body]):
                visual_element = ET.SubElement(
                    link,
                    "visual",
                    {"name": f"{side}_{body}_visual_{ordinal:02d}"},
                )
                geometry = ET.SubElement(visual_element, "geometry")
                ET.SubElement(
                    geometry,
                    "mesh",
                    {"filename": visual.mesh_path, "scale": "1 1 1"},
                )
                material = ET.SubElement(
                    visual_element,
                    "material",
                    {"name": f"{side}_{body}_material_{ordinal:02d}"},
                )
                ET.SubElement(
                    material,
                    "color",
                    {"rgba": _format_vector(visual.material.rgba)},
                )
        ET.SubElement(root, "link", {"name": f"{side}_tcp"})

    identity = np.eye(4, dtype=np.float64)
    for side in ("left", "right"):
        mount = identity.copy()
        mount[1, 3] = SIDE_OFFSETS_Y[side]
        _add_joint_element(
            root,
            name=f"{side}_mount",
            joint_type="fixed",
            parent=ROBOT_ROOT_LINK,
            child=_prefixed(side, ARM_ROOT_BODY),
            origin=mount,
        )
        for joint in model.joints:
            joint_type = joint.joint_type
            origin = joint.matrix
            axis = joint.axis_child
            lower = joint.lower
            upper = joint.upper
            effort = joint.effort
            velocity = joint.velocity
            if fixed_open_fingers and joint.name in FINGER_JOINTS:
                joint_type = "fixed"
                origin = _joint_pose(joint, model.finger_open_position)
                axis = None
                lower = upper = effort = velocity = None
            _add_joint_element(
                root,
                name=_prefixed(side, joint.name),
                joint_type=joint_type,
                parent=_prefixed(side, joint.parent),
                child=_prefixed(side, joint.child),
                origin=origin,
                axis=axis,
                lower=lower,
                upper=upper,
                effort=effort,
                velocity=velocity,
            )
        tcp = identity.copy()
        tcp[:3, 3] = TCP_OFFSET_LINK_6
        _add_joint_element(
            root,
            name=f"{side}_tcp_joint",
            joint_type="fixed",
            parent=_prefixed(side, "link_6"),
            child=f"{side}_tcp",
            origin=tcp,
        )

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _runtime_validate_bundle(
    bundle_root: Path,
    *,
    render_joint_names: Sequence[str],
    ik_joint_names: Sequence[str],
) -> dict[str, Any]:
    """Load both URDFs through all three consumer stacks without rendering a frame."""

    try:
        import pinocchio as pin
        import pyrender
        import trimesh
        import yourdfpy
    except ImportError as exc:
        raise YamConversionError(
            "runtime validation requires yourdfpy, pinocchio, pyrender, and trimesh"
        ) from exc

    results: dict[str, Any] = {}
    expected_by_variant = {
        "render": tuple(render_joint_names),
        "ik": tuple(ik_joint_names),
    }
    for variant, filename in (("render", RENDER_URDF_NAME), ("ik", IK_URDF_NAME)):
        urdf_path = bundle_root / filename
        urdf = yourdfpy.URDF.load(
            str(urdf_path),
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_meshes=True,
            load_collision_meshes=False,
            force_mesh=False,
        )
        actual_names = tuple(str(name) for name in urdf.actuated_joint_names)
        if actual_names != expected_by_variant[variant]:
            raise YamConversionError(
                f"yourdfpy {variant} joints {actual_names} != "
                f"{expected_by_variant[variant]}"
            )
        visual_count = 0
        for geometry in urdf.scene.geometry.values():
            if isinstance(geometry, trimesh.Scene):
                meshes = tuple(geometry.geometry.values())
            else:
                meshes = (geometry,)
            for mesh in meshes:
                pyrender.Mesh.from_trimesh(mesh, smooth=False)
                visual_count += 1
        if visual_count == 0:
            raise YamConversionError(f"yourdfpy loaded no {variant} visual geometry")

        pin_model = pin.buildModelFromUrdf(str(urdf_path))
        if pin_model.nq != len(expected_by_variant[variant]):
            raise YamConversionError(
                f"pinocchio {variant} nq {pin_model.nq} != "
                f"{len(expected_by_variant[variant])}"
            )
        pin_geometry = pin.buildGeomFromUrdf(
            pin_model,
            str(urdf_path),
            pin.GeometryType.VISUAL,
            None,
            [str(bundle_root)],
        )
        if len(pin_geometry.geometryObjects) == 0:
            raise YamConversionError(f"pinocchio loaded no {variant} visual geometry")
        results[variant] = {
            "independent_joint_names": list(actual_names),
            "yourdfpy_visual_geometry_count": visual_count,
            "pinocchio_nq": int(pin_model.nq),
            "pinocchio_visual_geometry_count": len(pin_geometry.geometryObjects),
        }
    results["versions"] = {
        "pinocchio": str(getattr(pin, "__version__", "unknown")),
        "pyrender": str(getattr(pyrender, "__version__", "unknown")),
        "trimesh": str(getattr(trimesh, "__version__", "unknown")),
        "yourdfpy": str(getattr(yourdfpy, "__version__", "unknown")),
    }
    return results


def _joint_provenance(joint: JointSpec) -> dict[str, Any]:
    return {
        "source_name": joint.name,
        "source_parent_body": joint.parent,
        "source_child_body": joint.child,
        "type": joint.joint_type,
        "parent_from_child_zero": [list(row) for row in joint.parent_from_child_zero],
        "axis_child": list(joint.axis_child) if joint.axis_child is not None else None,
        "source_axis": joint.source_axis,
        "source_lower": joint.source_lower,
        "source_upper": joint.source_upper,
        "urdf_lower": joint.lower,
        "urdf_upper": joint.upper,
        "authored_zero_pose_max_abs_residual": (
            joint.authored_zero_pose_max_abs_residual
        ),
        "revolute_limit_conversion": (
            "degrees_to_radians" if joint.joint_type == "revolute" else None
        ),
    }


def _build_manifest(
    staging: Path,
    *,
    source_usd: Path,
    config_path: Path,
    repository_record: Mapping[str, Any],
    model: ArmModel,
    runtime_validation: Mapping[str, Any],
    converter_source: Path,
) -> dict[str, Any]:
    arm_joint_names = [
        _prefixed(side, f"joint{index}")
        for side in ("left", "right")
        for index in range(1, 7)
    ]
    gripper_names = {
        side: [
            _prefixed(side, "left_finger"),
            _prefixed(side, "right_finger"),
        ]
        for side in ("left", "right")
    }
    output_records = [
        _file_record(path, recorded_path=path.relative_to(staging).as_posix())
        for path in sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(staging).as_posix(),
        )
    ]
    visuals = [
        {
            "body": visual.body,
            "source_prim": visual.source_prim,
            "output_mesh": visual.mesh_path,
            "body_from_mesh": [list(row) for row in visual.body_from_mesh],
            "vertex_count": visual.vertex_count,
            "triangle_count": visual.triangle_count,
            "material_rgba": list(visual.material.rgba),
            "material_provenance": visual.material.provenance,
        }
        for visual in model.visuals
    ]
    converter_record_path = "inpainting/robot_assets/yam_usd_to_urdf.py"
    return {
        "schema_version": BUNDLE_SCHEMA,
        "robot_id": "yam_bimanual",
        "render_urdf": RENDER_URDF_NAME,
        "ik_urdf": IK_URDF_NAME,
        "tcp_frames": {"left": "left_tcp", "right": "right_tcp"},
        "T_robot_root_hub": [list(row) for row in YAM_T_ROBOT_ROOT_HUB],
        "semantic_target_to_tcp_rotation": {
            side: [list(row) for row in rotation]
            for side, rotation in YAM_SEMANTIC_TARGET_TO_TCP_ROTATION.items()
        },
        "arm_joint_names": arm_joint_names,
        "arm_seed": {
            "left": [0.0] * 6,
            "right": [0.0] * 6,
        },
        "gripper_mapping": {
            "kind": "mirrored_prismatic",
            "joint_names": gripper_names,
            "params": {
                "closed_aperture_m": model.closed_aperture_m,
                "open_aperture_m": model.open_aperture_m,
                "closed_joint_position_m": [model.finger_closed_position] * 2,
                "open_joint_position_m": [model.finger_open_position] * 2,
            },
        },
        "fixed_root_posture": {
            "joint_values": {},
            "provenance": {
                "method": "baked_fixed_structure",
                "robot_root": ROBOT_ROOT_LINK,
                "hub_definition": "midpoint of the two arm roots",
                "left_arm_root_xyz_m": [0.0, SIDE_OFFSETS_Y["left"], 0.0],
                "right_arm_root_xyz_m": [0.0, SIDE_OFFSETS_Y["right"], 0.0],
                "source_fixed_joints": [
                    joint.name for joint in model.joints if joint.joint_type == "fixed"
                ],
            },
        },
        "asset_provenance": {
            "hash_algorithm": "sha256",
            "source_repository": dict(repository_record),
            "source_files": {
                "usd": _file_record(
                    source_usd, recorded_path=SOURCE_USD_RELATIVE.as_posix()
                ),
                "robot_config": _file_record(
                    config_path, recorded_path=SOURCE_CONFIG_RELATIVE.as_posix()
                ),
            },
            "converter_source": _file_record(
                converter_source, recorded_path=converter_record_path
            ),
            "conversion": {
                "stage_up_axis": "Z",
                "meters_per_unit": model.meters_per_unit,
                "visual_transform_policy": (
                    "USD authored local-to-world transformed into owning rigid-body "
                    "frame and baked into binary STL vertices"
                ),
                "material_policy": (
                    "bound USD diffuse input, then displayColor, then converter fallback"
                ),
                "arm_root_offsets_y_m": dict(SIDE_OFFSETS_Y),
                "tcp_offset_from_link_6_m": list(TCP_OFFSET_LINK_6),
                "ik_finger_policy": "fixed at authored fully-open joint position",
                "fingertip_keypoints_by_finger_link_m": {
                    key: list(value) for key, value in model.fingertip_keypoints.items()
                },
                "measured_fingertip_aperture_m": {
                    "closed": model.closed_aperture_m,
                    "open": model.open_aperture_m,
                },
                "parallel_jaw_retarget_calibration": {
                    "policy": (
                        "one fixed YAM embodiment mapping shared by all clips "
                        "and tracker conditions"
                    ),
                    "evaluation_context": {
                        "clips": [
                            "taco_cut__knife__plate_20231013_105",
                            "taco_dust__brush__cup_20231005_253",
                        ],
                        "tracker_conditions": [
                            "ground_truth",
                            "v2d",
                            "phantom",
                        ],
                        "strict_gates": {
                            "max_position_residual_m": 0.01,
                            "max_orientation_residual_deg": 20.0,
                            "max_joint_step_rad": 0.4,
                        },
                        "solver_policy": (
                            "parallel_jaw_renderer Pink trajectory solve with "
                            "dt=0.5, lm_damping=1e-4, orientation_cost=0.010, "
                            "and no Vega elbow bias"
                        ),
                    },
                    "mount": {
                        "description": (
                            "robot root is 0.15 m forward and rolled -10 degrees "
                            "relative to the shared clip hub"
                        ),
                        "translation_hub_m": [
                            YAM_MOUNT_FORWARD_X_M,
                            0.0,
                            0.0,
                        ],
                        "roll_hub_deg": YAM_MOUNT_ROLL_DEG,
                        "T_hub_robot_root": [
                            list(row) for row in YAM_T_HUB_ROBOT_ROOT
                        ],
                        "stored_bundle_field": (
                            "T_robot_root_hub is the exact inverse of "
                            "T_hub_robot_root"
                        ),
                    },
                    "semantic_target_to_tcp_rotation": {
                        side: [list(row) for row in rotation]
                        for side, rotation in (
                            YAM_SEMANTIC_TARGET_TO_TCP_ROTATION.items()
                        )
                    },
                    "left_jaw_symmetry": {
                        "rotation": "Rz(pi)",
                        "physical_basis": (
                            "the two opposing jaw contact surfaces are unchanged "
                            "when jaw-axis signs are reversed while approach is fixed"
                        ),
                        "selection_reason": (
                            "choose the contact-equivalent wrist representative "
                            "that avoids the finite joint-6 branch cut"
                        ),
                    },
                },
                "joints": [_joint_provenance(joint) for joint in model.joints],
                "visuals": visuals,
            },
            "runtime_validation": dict(runtime_validation),
            "output_files": output_records,
            "manifest_self_hash": (
                "omitted: a manifest cannot contain its own stable cryptographic hash"
            ),
        },
    }


def _install_permissions(staging: Path, output_parent: Path) -> None:
    """Make a root-run container bundle readable and owned like its bind parent."""

    parent_stat = output_parent.stat()
    paths = sorted(staging.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in (*paths, staging):
        path.chmod(0o755 if path.is_dir() else 0o644)
        try:
            os.chown(path, parent_stat.st_uid, parent_stat.st_gid)
        except PermissionError as exc:
            current = path.stat()
            if (current.st_uid, current.st_gid) != (
                parent_stat.st_uid,
                parent_stat.st_gid,
            ):
                raise YamConversionError(
                    f"cannot assign output ownership of {path} to "
                    f"{parent_stat.st_uid}:{parent_stat.st_gid}"
                ) from exc


def convert_yam_usd(
    *,
    source_usd: str | Path,
    input_repository: str | Path,
    output_dir: str | Path,
    expected_commit: str = PINNED_YAMLAB_COMMIT,
    expected_usd_sha256: str = PINNED_YAM_USD_SHA256,
    expected_config_sha256: str = PINNED_YAM_CONFIG_SHA256,
    runtime_validation: bool = True,
) -> ConversionResult:
    """Create one atomic, self-contained YAM bimanual bundle.

    ``output_dir`` must not already exist.  The converter writes to a sibling
    temporary directory and installs the completed bundle with one rename.
    """

    source = Path(source_usd).resolve()
    repository = Path(input_repository).resolve()
    destination = Path(output_dir).resolve()
    converter_source = Path(__file__).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if SHA256_RE.fullmatch(expected_usd_sha256) is None:
        raise YamConversionError("expected USD SHA-256 must be 64 lowercase hex digits")
    if SHA256_RE.fullmatch(expected_config_sha256) is None:
        raise YamConversionError("expected config SHA-256 must be 64 lowercase hex digits")
    if source != (repository / SOURCE_USD_RELATIVE).resolve():
        raise YamConversionError(
            f"source USD must be the pinned repository path {SOURCE_USD_RELATIVE}"
        )
    config_path = (repository / SOURCE_CONFIG_RELATIVE).resolve()
    license_path = (repository / SOURCE_LICENSE_RELATIVE).resolve()
    for required in (config_path, license_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    actual_usd_hash = sha256_file(source)
    if actual_usd_hash != expected_usd_sha256:
        raise YamConversionError(
            f"source USD SHA-256 {actual_usd_hash} != pinned {expected_usd_sha256}"
        )
    actual_config_hash = sha256_file(config_path)
    if actual_config_hash != expected_config_sha256:
        raise YamConversionError(
            f"robot config SHA-256 {actual_config_hash} != pinned {expected_config_sha256}"
        )
    repository_record = _repository_record(
        repository,
        (source, config_path, license_path),
        expected_commit=expected_commit,
    )
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing YAM bundle directory {destination}"
        )
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {destination.parent}")
    if destination == Path(destination.anchor):
        raise YamConversionError("output directory cannot be a filesystem root")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        config = _load_robot_config(config_path)
        model = _extract_arm_model(source, config, temporary)
        (temporary / RENDER_URDF_NAME).write_bytes(
            _build_urdf(model, fixed_open_fingers=False)
        )
        (temporary / IK_URDF_NAME).write_bytes(
            _build_urdf(model, fixed_open_fingers=True)
        )
        shutil.copyfile(license_path, temporary / "LICENSE.yamlab")

        arm_joint_names = [
            _prefixed(side, f"joint{index}")
            for side in ("left", "right")
            for index in range(1, 7)
        ]
        gripper_joint_names = [
            _prefixed(side, finger)
            for side in ("left", "right")
            for finger in ("left_finger", "right_finger")
        ]
        validation = (
            _runtime_validate_bundle(
                temporary,
                render_joint_names=(*arm_joint_names[:6], *gripper_joint_names[:2],
                                    *arm_joint_names[6:], *gripper_joint_names[2:]),
                ik_joint_names=arm_joint_names,
            )
            if runtime_validation
            else {"status": "skipped_by_caller"}
        )
        manifest = _build_manifest(
            temporary,
            source_usd=source,
            config_path=config_path,
            repository_record=repository_record,
            model=model,
            runtime_validation=validation,
            converter_source=converter_source,
        )
        manifest_path = temporary / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for record in manifest["asset_provenance"]["output_files"]:
            payload = temporary / record["path"]
            if payload.stat().st_size != record["bytes"]:
                raise YamConversionError(f"output size changed before commit: {payload}")
            if sha256_file(payload) != record["sha256"]:
                raise YamConversionError(f"output hash changed before commit: {payload}")
        _install_permissions(temporary, destination.parent)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return ConversionResult(
        bundle_dir=destination,
        render_urdf=destination / RENDER_URDF_NAME,
        ik_urdf=destination / IK_URDF_NAME,
        manifest=destination / MANIFEST_NAME,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-usd", required=True, type=Path)
    parser.add_argument("--input-repository", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-commit", default=PINNED_YAMLAB_COMMIT)
    parser.add_argument("--expected-usd-sha256", default=PINNED_YAM_USD_SHA256)
    parser.add_argument("--expected-config-sha256", default=PINNED_YAM_CONFIG_SHA256)
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Skip yourdfpy/pinocchio/pyrender compatibility checks (not recommended).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = convert_yam_usd(
        source_usd=args.source_usd,
        input_repository=args.input_repository,
        output_dir=args.output_dir,
        expected_commit=args.expected_commit,
        expected_usd_sha256=args.expected_usd_sha256,
        expected_config_sha256=args.expected_config_sha256,
        runtime_validation=not args.skip_runtime_validation,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
