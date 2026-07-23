"""Explicit, embodiment-neutral robot bundle schema and URDF inspection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np

from .transforms import (
    invert_transform,
    rotation_from_json,
    validate_transform,
)


BUNDLE_SCHEMA = "v2d.inpainting.parallel-jaw-robot-bundle/v1"
GRIPPER_KINDS = frozenset({"galbot_four_bar", "mirrored_prismatic"})


class BundleError(ValueError):
    """Raised when a robot bundle is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class JointLimit:
    joint_type: str
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class UrdfInspection:
    path: Path
    links: tuple[str, ...]
    joints: tuple[str, ...]
    independent_joint_names: tuple[str, ...]
    mimic_joint_names: tuple[str, ...]
    joint_limits: Mapping[str, JointLimit]
    visual_mesh_paths: tuple[Path, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "link_count": len(self.links),
            "joint_count": len(self.joints),
            "independent_joint_names": list(self.independent_joint_names),
            "mimic_joint_names": list(self.mimic_joint_names),
            "visual_mesh_paths": [str(path) for path in self.visual_mesh_paths],
        }


@dataclass(frozen=True)
class GripperMappingSpec:
    kind: str
    joint_names: Mapping[str, tuple[str, ...]]
    params: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "joint_names": {
                side: list(self.joint_names[side]) for side in ("left", "right")
            },
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class RobotBundle:
    source_path: Path
    robot_id: str
    render_urdf: Path
    ik_urdf: Path
    tcp_frames: Mapping[str, str]
    T_robot_root_hub: np.ndarray
    semantic_target_to_tcp_rotation: Mapping[str, np.ndarray]
    arm_joint_names: tuple[str, ...]
    gripper_mapping: GripperMappingSpec
    fixed_root_joint_values: Mapping[str, float]
    fixed_root_provenance: Mapping[str, Any]
    asset_provenance: Mapping[str, Any]
    render_inspection: UrdfInspection
    ik_inspection: UrdfInspection

    @property
    def T_hub_robot_root(self) -> np.ndarray:
        return invert_transform(self.T_robot_root_hub)

    def world_robot_root(self, T_world_hub: np.ndarray) -> np.ndarray:
        """Place the robot root from the caller's explicit shared world hub."""

        return validate_transform(
            validate_transform(T_world_hub, name="T_world_hub") @ self.T_hub_robot_root,
            name="T_world_robot_root",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BUNDLE_SCHEMA,
            "source_path": str(self.source_path),
            "robot_id": self.robot_id,
            "render_urdf": str(self.render_urdf),
            "ik_urdf": str(self.ik_urdf),
            "tcp_frames": dict(self.tcp_frames),
            "T_robot_root_hub": self.T_robot_root_hub.tolist(),
            "semantic_target_to_tcp_rotation": {
                side: self.semantic_target_to_tcp_rotation[side].tolist()
                for side in ("left", "right")
            },
            "arm_joint_names": list(self.arm_joint_names),
            "gripper_mapping": self.gripper_mapping.as_dict(),
            "fixed_root_posture": {
                "joint_values": dict(self.fixed_root_joint_values),
                "provenance": dict(self.fixed_root_provenance),
            },
            "asset_provenance": dict(self.asset_provenance),
            "render_urdf_inspection": self.render_inspection.as_dict(),
            "ik_urdf_inspection": self.ik_inspection.as_dict(),
        }


def _require_object(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BundleError(f"{name} must be a JSON object")
    return value


def _require_nonempty_object(value: object, *, name: str) -> Mapping[str, Any]:
    result = _require_object(value, name=name)
    if not result:
        raise BundleError(f"{name} must not be empty")
    return result


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundleError(f"{name} must be a non-empty string")
    return value


def _resolve_bundle_path(value: object, *, name: str, base_dir: Path) -> Path:
    if isinstance(value, dict):
        raise BundleError(
            f"{name} must be a path string, not a provenance record object; "
            f"put hashes under asset_provenance"
        )
    text = _require_string(value, name=name)
    candidate = Path(text)
    result = (
        candidate.resolve()
        if candidate.is_absolute()
        else (base_dir / candidate).resolve()
    )
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _resolve_visual_mesh(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        raise BundleError(
            f"{urdf_path} uses unsupported package URI {filename!r}; rewrite it "
            "to a bundle-relative path or an absolute /robot_assets path"
        )
    parsed = urlparse(filename)
    if parsed.scheme and parsed.scheme != "file":
        raise BundleError(
            f"{urdf_path} visual mesh URI {filename!r} is not a local file"
        )
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else filename
    candidate = Path(raw_path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (urdf_path.parent / candidate).resolve()
    )


def inspect_urdf(
    path: str | Path, *, require_visual_assets: bool = True
) -> UrdfInspection:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise BundleError(f"invalid URDF XML {source}: {exc}") from exc
    if root.tag != "robot":
        raise BundleError(f"{source} root must be <robot>")

    links = tuple(element.attrib.get("name", "") for element in root.findall("link"))
    if not links or any(not name for name in links) or len(set(links)) != len(links):
        raise BundleError(f"{source} has missing or duplicate link names")

    joint_names: list[str] = []
    independent: list[str] = []
    mimic: list[str] = []
    limits: dict[str, JointLimit] = {}
    for element in root.findall("joint"):
        name = element.attrib.get("name", "")
        joint_type = element.attrib.get("type", "")
        if not name or name in joint_names:
            raise BundleError(f"{source} has a missing or duplicate joint name")
        joint_names.append(name)
        if joint_type in ("fixed", "floating"):
            continue
        mimic_element = element.find("mimic")
        if mimic_element is not None:
            parent = mimic_element.attrib.get("joint", "")
            if not parent:
                raise BundleError(f"mimic joint {name!r} has no source joint")
            mimic.append(name)
        else:
            independent.append(name)
        lower: float | None = None
        upper: float | None = None
        if joint_type != "continuous":
            limit = element.find("limit")
            if (
                limit is None
                or "lower" not in limit.attrib
                or "upper" not in limit.attrib
            ):
                raise BundleError(
                    f"joint {name!r} in {source} lacks lower/upper limits"
                )
            try:
                lower = float(limit.attrib["lower"])
                upper = float(limit.attrib["upper"])
            except ValueError as exc:
                raise BundleError(f"joint {name!r} has invalid limits") from exc
            if not np.isfinite((lower, upper)).all() or lower > upper:
                raise BundleError(
                    f"joint {name!r} has invalid limits [{lower}, {upper}]"
                )
        limits[name] = JointLimit(joint_type=joint_type, lower=lower, upper=upper)

    independent_set = set(independent)
    for element in root.findall("joint"):
        mimic_element = element.find("mimic")
        if mimic_element is not None:
            parent = mimic_element.attrib["joint"]
            if parent not in independent_set:
                raise BundleError(
                    f"mimic joint {element.attrib['name']!r} references non-independent "
                    f"joint {parent!r}"
                )

    visual_meshes: list[Path] = []
    for mesh in root.findall("./link/visual/geometry/mesh"):
        filename = mesh.attrib.get("filename", "")
        if not filename:
            raise BundleError(f"{source} has a visual mesh with no filename")
        resolved = _resolve_visual_mesh(source, filename)
        if require_visual_assets and not resolved.is_file():
            raise BundleError(
                f"visual mesh {filename!r} from {source} is missing at {resolved}; "
                "mount the asset repository at the URDF-authored /robot_assets path"
            )
        visual_meshes.append(resolved)
    return UrdfInspection(
        path=source,
        links=links,
        joints=tuple(joint_names),
        independent_joint_names=tuple(independent),
        mimic_joint_names=tuple(mimic),
        joint_limits=limits,
        visual_mesh_paths=tuple(dict.fromkeys(visual_meshes)),
    )


def _parse_side_names(value: object, *, name: str) -> Mapping[str, tuple[str, ...]]:
    mapping = _require_object(value, name=name)
    if set(mapping) != {"left", "right"}:
        raise BundleError(f"{name} must contain exactly left and right")
    result: dict[str, tuple[str, ...]] = {}
    for side in ("left", "right"):
        raw = mapping[side]
        if not isinstance(raw, list) or not raw:
            raise BundleError(f"{name}.{side} must be a non-empty string list")
        names = tuple(_require_string(item, name=f"{name}.{side}[]") for item in raw)
        if len(names) != len(set(names)):
            raise BundleError(f"{name}.{side} contains duplicates")
        result[side] = names
    if set(result["left"]) & set(result["right"]):
        raise BundleError(f"{name} left and right joint names overlap")
    return result


def _parse_semantic_rotation(value: object) -> Mapping[str, np.ndarray]:
    if isinstance(value, dict):
        if set(value) != {"left", "right"}:
            raise BundleError(
                "semantic_target_to_tcp_rotation object must contain exactly left and right"
            )
        return {
            side: rotation_from_json(
                value[side],
                name=f"semantic_target_to_tcp_rotation.{side}",
            )
            for side in ("left", "right")
        }
    shared = rotation_from_json(value, name="semantic_target_to_tcp_rotation")
    return {"left": shared.copy(), "right": shared.copy()}


def _parse_finite_float_map(value: object, *, name: str) -> Mapping[str, float]:
    mapping = _require_object(value, name=name)
    result: dict[str, float] = {}
    for key, raw in mapping.items():
        joint = _require_string(key, name=f"{name} key")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise BundleError(f"{name}.{joint} must be numeric")
        number = float(raw)
        if not np.isfinite(number):
            raise BundleError(f"{name}.{joint} must be finite")
        result[joint] = number
    return result


def _validate_joint_limit(
    value: float, joint: str, inspection: UrdfInspection, *, label: str
) -> None:
    limit = inspection.joint_limits[joint]
    tolerance = 1e-6
    if limit.lower is not None and value < limit.lower - tolerance:
        raise BundleError(
            f"{label}.{joint}={value:.8g} is below URDF lower limit {limit.lower:.8g}"
        )
    if limit.upper is not None and value > limit.upper + tolerance:
        raise BundleError(
            f"{label}.{joint}={value:.8g} is above URDF upper limit {limit.upper:.8g}"
        )


def load_robot_bundle(
    path: str | Path,
    *,
    require_visual_assets: bool = True,
) -> RobotBundle:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"could not parse robot bundle {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BundleError("robot bundle root must be an object")
    schema = payload.get("schema_version")
    if schema != BUNDLE_SCHEMA:
        raise BundleError(
            f"robot bundle schema {schema!r} != {BUNDLE_SCHEMA!r}; legacy embodiment "
            "asset manifests must be converted to the renderer bundle schema"
        )

    robot_id = _require_string(payload.get("robot_id"), name="robot_id")
    render_urdf = _resolve_bundle_path(
        payload.get("render_urdf"),
        name="render_urdf",
        base_dir=source.parent,
    )
    ik_urdf = _resolve_bundle_path(
        payload.get("ik_urdf"),
        name="ik_urdf",
        base_dir=source.parent,
    )
    render_inspection = inspect_urdf(
        render_urdf, require_visual_assets=require_visual_assets
    )
    ik_inspection = inspect_urdf(ik_urdf, require_visual_assets=False)

    tcp_raw = _require_object(payload.get("tcp_frames"), name="tcp_frames")
    if set(tcp_raw) != {"left", "right"}:
        raise BundleError("tcp_frames must contain exactly left and right")
    tcp_frames = {
        side: _require_string(tcp_raw[side], name=f"tcp_frames.{side}")
        for side in ("left", "right")
    }
    ik_frames = set(ik_inspection.links) | set(ik_inspection.joints)
    for side, frame in tcp_frames.items():
        if frame not in ik_frames:
            raise BundleError(f"tcp_frames.{side}={frame!r} is absent from ik_urdf")

    has_root_from_hub = "T_robot_root_hub" in payload
    has_hub_from_root = "T_hub_robot_root" in payload
    if has_root_from_hub == has_hub_from_root:
        raise BundleError(
            "bundle must contain exactly one of T_robot_root_hub or T_hub_robot_root"
        )
    if has_root_from_hub:
        T_robot_root_hub = validate_transform(
            np.asarray(payload["T_robot_root_hub"], dtype=np.float64),
            name="T_robot_root_hub",
        )
    else:
        T_robot_root_hub = invert_transform(
            validate_transform(
                np.asarray(payload["T_hub_robot_root"], dtype=np.float64),
                name="T_hub_robot_root",
            )
        )

    semantic_rotation = _parse_semantic_rotation(
        payload.get("semantic_target_to_tcp_rotation")
    )
    arm_raw = payload.get("arm_joint_names")
    if not isinstance(arm_raw, list) or not arm_raw:
        raise BundleError("arm_joint_names must be a non-empty string list")
    arm_joint_names = tuple(
        _require_string(item, name="arm_joint_names[]") for item in arm_raw
    )
    if len(arm_joint_names) != len(set(arm_joint_names)):
        raise BundleError("arm_joint_names contains duplicates")
    if set(ik_inspection.independent_joint_names) != set(arm_joint_names):
        raise BundleError(
            "ik_urdf must be arm-only and exactly cover arm_joint_names; "
            f"missing={sorted(set(arm_joint_names) - set(ik_inspection.independent_joint_names))}, "
            f"extra={sorted(set(ik_inspection.independent_joint_names) - set(arm_joint_names))}"
        )

    gripper_raw = _require_object(
        payload.get("gripper_mapping"), name="gripper_mapping"
    )
    kind = _require_string(gripper_raw.get("kind"), name="gripper_mapping.kind")
    if kind not in GRIPPER_KINDS:
        raise BundleError(
            f"gripper_mapping.kind {kind!r} is unsupported; expected {sorted(GRIPPER_KINDS)}"
        )
    gripper_names = _parse_side_names(
        gripper_raw.get("joint_names"), name="gripper_mapping.joint_names"
    )
    gripper_params = _require_object(
        gripper_raw.get("params"), name="gripper_mapping.params"
    )
    gripper_mapping = GripperMappingSpec(
        kind=kind,
        joint_names=gripper_names,
        params=dict(gripper_params),
    )

    fixed_raw = _require_object(
        payload.get("fixed_root_posture"), name="fixed_root_posture"
    )
    if set(fixed_raw) != {"joint_values", "provenance"}:
        raise BundleError(
            "fixed_root_posture must contain exactly joint_values and provenance"
        )
    fixed_values = _parse_finite_float_map(
        fixed_raw["joint_values"], name="fixed_root_posture.joint_values"
    )
    fixed_provenance = _require_nonempty_object(
        fixed_raw["provenance"], name="fixed_root_posture.provenance"
    )
    asset_provenance = _require_nonempty_object(
        payload.get("asset_provenance"), name="asset_provenance"
    )

    arm_set = set(arm_joint_names)
    gripper_set = set(gripper_names["left"]) | set(gripper_names["right"])
    fixed_set = set(fixed_values)
    if arm_set & gripper_set or arm_set & fixed_set or gripper_set & fixed_set:
        raise BundleError(
            "arm, gripper, and fixed-root posture joint sets must be disjoint"
        )
    render_set = set(render_inspection.independent_joint_names)
    covered = arm_set | gripper_set | fixed_set
    if render_set != covered:
        raise BundleError(
            "render_urdf independent joints are not covered exactly once; "
            f"missing={sorted(render_set - covered)}, extra={sorted(covered - render_set)}"
        )
    mimic_overlap = set(render_inspection.mimic_joint_names) & covered
    if mimic_overlap:
        raise BundleError(
            "mimic followers must not be commanded explicitly; remove "
            f"{sorted(mimic_overlap)} from bundle joint lists"
        )
    for joint, value in fixed_values.items():
        _validate_joint_limit(
            value,
            joint,
            render_inspection,
            label="fixed_root_posture.joint_values",
        )

    return RobotBundle(
        source_path=source,
        robot_id=robot_id,
        render_urdf=render_urdf,
        ik_urdf=ik_urdf,
        tcp_frames=tcp_frames,
        T_robot_root_hub=T_robot_root_hub,
        semantic_target_to_tcp_rotation=semantic_rotation,
        arm_joint_names=arm_joint_names,
        gripper_mapping=gripper_mapping,
        fixed_root_joint_values=fixed_values,
        fixed_root_provenance=fixed_provenance,
        asset_provenance=asset_provenance,
        render_inspection=render_inspection,
        ik_inspection=ik_inspection,
    )
