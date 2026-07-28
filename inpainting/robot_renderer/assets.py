"""Resolve and inspect the complete Vega/Sharpa asset tree without importing Isaac."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np

from .provenance import file_record


ARMS_URDF = Path("urdfs/vega_sharpa/vega_arms_only.urdf")
LEFT_HAND_URDF = Path("urdfs/sharpawave/left_sharpa_wave.urdf")
RIGHT_HAND_URDF = Path("urdfs/sharpawave/right_sharpa_wave.urdf")


class AssetError(ValueError):
    """Raised when a URDF tree is missing, incomplete, or semantically incompatible."""


@dataclass(frozen=True)
class JointLimit:
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class UrdfInspection:
    path: Path
    links: tuple[str, ...]
    actuated_joint_names: tuple[str, ...]
    joint_limits: Mapping[str, JointLimit]
    mesh_paths: tuple[Path, ...]
    mesh_bytes: int

    def as_dict(self, *, asset_root: Path | None = None) -> dict:
        root = asset_root.resolve() if asset_root is not None else None

        def recorded_path(path: Path) -> str:
            if root is None:
                return str(path)
            try:
                return path.relative_to(root).as_posix()
            except ValueError as exc:  # defensive: inspection already enforces this
                raise AssetError(f"asset {path} is outside provenance root {root}") from exc

        return {
            "path": str(self.path),
            "link_count": len(self.links),
            "actuated_joint_count": len(self.actuated_joint_names),
            "actuated_joint_names": list(self.actuated_joint_names),
            "mesh_count": len(self.mesh_paths),
            "mesh_bytes": self.mesh_bytes,
            "urdf_file": file_record(
                self.path, recorded_path=recorded_path(self.path)
            ),
            "referenced_asset_files": [
                file_record(path, recorded_path=recorded_path(path))
                for path in sorted(self.mesh_paths, key=recorded_path)
            ],
        }


@dataclass(frozen=True)
class RobotAssets:
    root: Path
    arms: UrdfInspection
    left_hand: UrdfInspection
    right_hand: UrdfInspection

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "arms": self.arms.as_dict(asset_root=self.root),
            "left_hand": self.left_hand.as_dict(asset_root=self.root),
            "right_hand": self.right_hand.as_dict(asset_root=self.root),
        }


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError as exc:
        raise AssetError(f"cannot read asset {path}: {exc}") from exc


def _resolve_mesh(urdf: Path, asset_root: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        raise AssetError(
            f"{urdf} uses package URI {filename!r}; this offline renderer requires "
            "self-contained relative mesh paths"
        )
    mesh = (urdf.parent / filename).resolve()
    try:
        mesh.relative_to(asset_root)
    except ValueError as exc:
        raise AssetError(f"mesh {filename!r} in {urdf} escapes asset root {asset_root}") from exc
    if not mesh.is_file():
        raise AssetError(f"mesh {filename!r} referenced by {urdf} is missing at {mesh}")
    if mesh.stat().st_size <= 0:
        raise AssetError(f"mesh {mesh} is empty")
    if _is_lfs_pointer(mesh):
        raise AssetError(f"mesh {mesh} is a Git LFS pointer, not renderable content")
    return mesh


def _resolve_dependent_resource(owner: Path, asset_root: Path, uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme:
        raise AssetError(f"mesh resource {uri!r} in {owner} is not a local relative file")
    resource = (owner.parent / unquote(parsed.path)).resolve()
    try:
        resource.relative_to(asset_root)
    except ValueError as exc:
        raise AssetError(f"mesh resource {uri!r} in {owner} escapes asset root") from exc
    if not resource.is_file():
        raise AssetError(f"mesh resource {uri!r} referenced by {owner} is missing at {resource}")
    if resource.stat().st_size <= 0:
        raise AssetError(f"mesh resource {resource} is empty")
    if _is_lfs_pointer(resource):
        raise AssetError(f"mesh resource {resource} is a Git LFS pointer")
    return resource


def _mesh_dependencies(mesh: Path, asset_root: Path) -> tuple[Path, ...]:
    """Resolve external buffers/textures referenced by glTF or COLLADA files."""

    uris: list[str] = []
    if mesh.suffix.lower() == ".gltf":
        try:
            document = json.loads(mesh.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AssetError(f"invalid glTF JSON at {mesh}: {exc}") from exc
        for entry in (*document.get("buffers", []), *document.get("images", [])):
            uri = entry.get("uri")
            if uri and not str(uri).startswith("data:"):
                uris.append(str(uri))
    elif mesh.suffix.lower() == ".dae":
        try:
            root = ET.parse(mesh).getroot()
        except ET.ParseError as exc:
            raise AssetError(f"invalid COLLADA XML at {mesh}: {exc}") from exc
        # Only library_images/init_from contains a file URI. Other COLLADA
        # init_from elements refer to in-document material/sampler identifiers.
        for library in root.findall(".//{*}library_images"):
            for image in library.findall("{*}image"):
                init_from = image.find("{*}init_from")
                if init_from is not None and init_from.text and init_from.text.strip():
                    uris.append(init_from.text.strip())
    return tuple(
        _resolve_dependent_resource(mesh, asset_root, uri)
        for uri in dict.fromkeys(uris)
    )


def inspect_urdf(path: str | Path, *, asset_root: str | Path) -> UrdfInspection:
    """Parse one URDF and verify every visual mesh resolves to real content."""

    root = Path(asset_root).resolve()
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise AssetError(f"URDF {source} is outside asset root {root}") from exc
    if _is_lfs_pointer(source):
        raise AssetError(f"URDF {source} is a Git LFS pointer, not XML content")
    try:
        xml_root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise AssetError(f"invalid URDF XML at {source}: {exc}") from exc
    if xml_root.tag != "robot":
        raise AssetError(f"{source} root element must be <robot>, got <{xml_root.tag}>")

    links = tuple(link.attrib.get("name", "") for link in xml_root.findall("link"))
    if not links or any(not name for name in links) or len(set(links)) != len(links):
        raise AssetError(f"{source} has missing or duplicate link names")

    joint_names: list[str] = []
    limits: dict[str, JointLimit] = {}
    for joint in xml_root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        if joint_type in ("fixed", "floating"):
            continue
        name = joint.attrib.get("name", "")
        if not name:
            raise AssetError(f"{source} has an actuated joint without a name")
        if name in limits:
            raise AssetError(f"{source} has duplicate actuated joint {name!r}")
        limit = joint.find("limit")
        lower: float | None = None
        upper: float | None = None
        if joint_type != "continuous":
            if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                raise AssetError(f"joint {name!r} in {source} has no finite lower/upper limit")
            try:
                lower = float(limit.attrib["lower"])
                upper = float(limit.attrib["upper"])
            except ValueError as exc:
                raise AssetError(f"joint {name!r} in {source} has invalid limits") from exc
            if not np.isfinite((lower, upper)).all() or lower > upper:
                raise AssetError(
                    f"joint {name!r} in {source} has invalid limits [{lower}, {upper}]"
                )
        joint_names.append(name)
        limits[name] = JointLimit(lower=lower, upper=upper)

    mesh_paths: list[Path] = []
    for mesh_element in xml_root.findall("./link/visual/geometry/mesh"):
        filename = mesh_element.attrib.get("filename", "")
        if not filename:
            raise AssetError(f"{source} contains a visual mesh with no filename")
        mesh_paths.append(_resolve_mesh(source, root, filename))
    if not mesh_paths:
        raise AssetError(f"{source} contains no visual meshes")
    unique_meshes = tuple(dict.fromkeys(mesh_paths))
    dependent_resources = tuple(
        dict.fromkeys(
            resource
            for mesh in unique_meshes
            for resource in _mesh_dependencies(mesh, root)
        )
    )
    all_resources = tuple(dict.fromkeys((*unique_meshes, *dependent_resources)))
    return UrdfInspection(
        path=source,
        links=links,
        actuated_joint_names=tuple(joint_names),
        joint_limits=limits,
        mesh_paths=all_resources,
        mesh_bytes=sum(mesh.stat().st_size for mesh in all_resources),
    )


def resolve_robot_assets(asset_root: str | Path) -> RobotAssets:
    """Resolve the exact arms-only Vega and two articulated Sharpa URDFs."""

    root = Path(asset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    assets = RobotAssets(
        root=root,
        arms=inspect_urdf(root / ARMS_URDF, asset_root=root),
        left_hand=inspect_urdf(root / LEFT_HAND_URDF, asset_root=root),
        right_hand=inspect_urdf(root / RIGHT_HAND_URDF, asset_root=root),
    )
    required_arm_links = {"arm_center", "L_arm_l8", "R_arm_l8"}
    missing_links = sorted(required_arm_links - set(assets.arms.links))
    if missing_links:
        raise AssetError(f"Vega arms URDF lacks required links {missing_links}")
    expected_arm_joints = {
        *(f"L_arm_j{index}" for index in range(1, 8)),
        *(f"R_arm_j{index}" for index in range(1, 8)),
    }
    actual_arm_joints = set(assets.arms.actuated_joint_names)
    if actual_arm_joints != expected_arm_joints:
        raise AssetError(
            "Vega arms actuated joints differ from the expected dual 7-DOF model: "
            f"missing={sorted(expected_arm_joints - actual_arm_joints)}, "
            f"extra={sorted(actual_arm_joints - expected_arm_joints)}"
        )
    return assets


def validate_named_joint_trajectory(
    values: np.ndarray,
    names: np.ndarray,
    inspection: UrdfInspection,
    *,
    label: str,
    tolerance: float = 1e-5,
) -> None:
    """Check exact joint coverage and enforce URDF limits without clamping."""

    data = np.asarray(values, dtype=np.float64)
    columns = tuple(str(name) for name in np.asarray(names).tolist())
    expected = set(inspection.actuated_joint_names)
    actual = set(columns)
    if len(columns) != len(actual):
        raise AssetError(f"{label} contains duplicate joint names")
    if actual != expected:
        raise AssetError(
            f"{label} does not exactly cover {inspection.path.name}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise AssetError(
            f"{label} values must have one column per name; got {data.shape} and {len(columns)}"
        )
    for column, name in enumerate(columns):
        limit = inspection.joint_limits[name]
        series = data[:, column]
        if limit.lower is not None and np.any(series < limit.lower - tolerance):
            frame = int(np.argmin(series))
            raise AssetError(
                f"{label}[{frame},{name}]={series[frame]:.8g} is below URDF lower "
                f"limit {limit.lower:.8g}"
            )
        if limit.upper is not None and np.any(series > limit.upper + tolerance):
            frame = int(np.argmax(series))
            raise AssetError(
                f"{label}[{frame},{name}]={series[frame]:.8g} is above URDF upper "
                f"limit {limit.upper:.8g}"
            )


def validate_finger_trajectories(
    assets: RobotAssets,
    trajectory: Mapping[str, np.ndarray],
) -> None:
    validate_named_joint_trajectory(
        trajectory["left_finger_joints"],
        trajectory["left_finger_joint_names"],
        assets.left_hand,
        label="left_finger_joints",
    )
    validate_named_joint_trajectory(
        trajectory["right_finger_joints"],
        trajectory["right_finger_joint_names"],
        assets.right_hand,
        label="right_finger_joints",
    )
