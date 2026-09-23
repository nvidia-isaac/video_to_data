#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the fixed-base 58-DOF Vega Sharpa v2 asset from its vendored upstream snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASSET_ROOT = _REPO_ROOT / "src" / "flash_chord" / "assets"
_VERSION_ROOT = _ASSET_ROOT / "urdfs" / "vega_sharpa" / "v2"
_SOURCE_URDF = _VERSION_ROOT / "upstream" / "vega_sharpa.urdf"
_OUTPUT_URDF = _VERSION_ROOT / "vega_sharpa_58dof.urdf"
_MESH_ROOT = _ASSET_ROOT / "meshes" / "vega_sharpa" / "v2"
_MANIFEST = _VERSION_ROOT / "SOURCE.json"
_OUTPUT_MESH_PREFIX = "../../../meshes/vega_sharpa/v2"

_UPSTREAM_REVISION = "7cd29eecbd1c41bb299100421f7a2068b9fc1bce"

_FINGER_SUFFIXES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
_ACTIVE_JOINTS = tuple(
    name
    for arm_side, hand_side in (("L", "left"), ("R", "right"))
    for name in (
        *(f"{arm_side}_arm_j{index}" for index in range(1, 8)),
        *(f"{hand_side}_{suffix}" for suffix in _FINGER_SUFFIXES),
    )
)

# Values are URDF joint coordinates. The torso axes (-Y, +Y, -Y) turn these
# positive values into the fixed RPY values (-0.78, +1.57, -0.44) in the v1 asset.
_FROZEN_JOINT_POSITIONS = {
    "B_wheel_j1": 0.0,
    "B_wheel_j2": 0.0,
    "R_wheel_j1": 0.0,
    "R_wheel_j2": 0.0,
    "L_wheel_j1": 0.0,
    "L_wheel_j2": 0.0,
    "torso_j1": 0.78,
    "torso_j2": 1.57,
    "torso_j3": 0.44,
    "head_j1": 0.0,
    "head_j2": 0.0,
    "head_j3": 0.0,
}
_ROOT_TRANSFORM = {
    "parent": "root",
    "child": "base",
    "xyz": "0 0.4 0.05393118",
    "rpy": "0 0 -1.5708",
}


def _vector(value: str | None, size: int, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    parsed = tuple(float(component) for component in value.split())
    if len(parsed) != size or not all(math.isfinite(component) for component in parsed):
        raise ValueError(f"expected {size} finite values, got {value!r}")
    return parsed


def _matmul(left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...]):
    return tuple(
        tuple(sum(left[row][inner] * right[inner][column] for inner in range(3)) for column in range(3))
        for row in range(3)
    )


def _rpy_matrix(rpy: tuple[float, float, float]):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _axis_angle_matrix(axis: tuple[float, float, float], angle: float):
    norm = math.sqrt(sum(component * component for component in axis))
    if norm <= 1.0e-12:
        raise ValueError(f"joint axis must have nonzero norm, got {axis}")
    x, y, z = (component / norm for component in axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    return (
        (cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine),
        (y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine),
        (z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement),
    )


def _matrix_rpy(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, float, float]:
    horizontal = math.hypot(matrix[0][0], matrix[1][0])
    pitch = math.atan2(-matrix[2][0], horizontal)
    if horizontal > 1.0e-10:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        yaw = 0.0
    return roll, pitch, yaw


def _format_vector(values: tuple[float, ...]) -> str:
    return " ".join("0" if abs(value) < 1.0e-15 else f"{value:.15g}" for value in values)


def _freeze_joint(joint: ET.Element, position: float) -> None:
    origin = joint.find("origin")
    if origin is None:
        origin = ET.Element("origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        joint.insert(0, origin)
    rpy = _vector(origin.get("rpy"), 3, default=(0.0, 0.0, 0.0))
    axis_element = joint.find("axis")
    axis = _vector(
        None if axis_element is None else axis_element.get("xyz"),
        3,
        default=(1.0, 0.0, 0.0),
    )
    rotation = _matmul(_rpy_matrix(rpy), _axis_angle_matrix(axis, position))
    origin.set("rpy", _format_vector(_matrix_rpy(rotation)))
    joint.set("type", "fixed")
    for tag in ("axis", "calibration", "dynamics", "limit", "mimic", "safety_controller"):
        child = joint.find(tag)
        if child is not None:
            joint.remove(child)


def _add_root_transform(robot: ET.Element) -> None:
    if robot.find("./link[@name='root']") is not None or robot.find("./joint[@name='base_offset']") is not None:
        raise ValueError("upstream asset unexpectedly defines the Flash Chord root transform")
    base = robot.find("./link[@name='base']")
    if base is None:
        raise ValueError("upstream asset is missing its base link")

    root_link = ET.Element("link", {"name": "root"})
    inertial = ET.SubElement(root_link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "1.0"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "0.1", "ixy": "0.0", "ixz": "0.0", "iyy": "0.1", "iyz": "0.0", "izz": "0.1"},
    )
    base_offset = ET.Element("joint", {"name": "base_offset", "type": "fixed"})
    ET.SubElement(base_offset, "origin", {"xyz": _ROOT_TRANSFORM["xyz"], "rpy": _ROOT_TRANSFORM["rpy"]})
    ET.SubElement(base_offset, "parent", {"link": _ROOT_TRANSFORM["parent"]})
    ET.SubElement(base_offset, "child", {"link": _ROOT_TRANSFORM["child"]})

    robot.insert(0, root_link)
    robot.insert(list(robot).index(base) + 1, base_offset)


def generate_urdf(source_path: Path = _SOURCE_URDF) -> bytes:
    root = ET.parse(source_path).getroot()
    root.set("name", "vega-sharpa-58dof-v2")
    _add_root_transform(root)
    movable = {joint.get("name") for joint in root.findall("joint") if joint.get("type") not in (None, "fixed")}
    expected = set(_ACTIVE_JOINTS) | set(_FROZEN_JOINT_POSITIONS)
    if movable != expected:
        raise ValueError(
            "upstream movable-joint topology changed: "
            f"missing={sorted(expected - movable)}, unexpected={sorted(movable - expected)}"
        )

    for joint in root.findall("joint"):
        name = joint.get("name")
        if name in _FROZEN_JOINT_POSITIONS:
            _freeze_joint(joint, _FROZEN_JOINT_POSITIONS[name])

    active = tuple(joint.get("name") for joint in root.findall("joint") if joint.get("type") not in (None, "fixed"))
    if len(active) != 58 or set(active) != set(_ACTIVE_JOINTS):
        raise ValueError(f"generated asset must expose exactly the approved 58 joints, got {active}")

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename is None or not filename.startswith("meshes/"):
            raise ValueError(f"upstream mesh path must be relative to its asset root, got {filename!r}")
        mesh.set("filename", f"{_OUTPUT_MESH_PREFIX}/{filename.removeprefix('meshes/')}")

    ET.indent(root, space="  ")
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True, short_empty_elements=True)
    return payload + b"\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_hashes(mesh_root: Path = _MESH_ROOT) -> dict[str, str]:
    files = sorted(path for path in mesh_root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"vendored mesh tree is empty: {mesh_root}")
    return {path.relative_to(mesh_root).as_posix(): _sha256(path) for path in files}


def _tree_sha256(file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_hash in file_hashes.items():
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(generated: bytes) -> dict[str, object]:
    mesh_hashes = _mesh_hashes()
    return {
        "schema_version": 1,
        "asset_id": "vega_sharpa_v2",
        "upstream": {
            "path": _SOURCE_URDF.relative_to(_REPO_ROOT).as_posix(),
            "revision": _UPSTREAM_REVISION,
            "urdf_sha256": _sha256(_SOURCE_URDF),
        },
        "generated": {
            "path": _OUTPUT_URDF.relative_to(_REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(generated).hexdigest(),
            "generator": Path(__file__).resolve().relative_to(_REPO_ROOT).as_posix(),
            "generator_sha256": _sha256(Path(__file__).resolve()),
        },
        "active_joints": list(_ACTIVE_JOINTS),
        "frozen_joint_positions": _FROZEN_JOINT_POSITIONS,
        "root_transform": _ROOT_TRANSFORM,
        "meshes": {
            "root": _MESH_ROOT.relative_to(_REPO_ROOT).as_posix(),
            "tree_sha256": _tree_sha256(mesh_hashes),
            "files": mesh_hashes,
        },
    }


def _manifest_bytes(manifest: dict[str, object]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def _check(path: Path, expected: bytes) -> None:
    actual = path.read_bytes() if path.is_file() else None
    if actual != expected:
        raise SystemExit(f"{path.relative_to(_REPO_ROOT)} is stale; regenerate the Vega Sharpa v2 asset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed outputs without modifying them")
    args = parser.parse_args()

    generated = generate_urdf()
    manifest = _manifest_bytes(build_manifest(generated))
    if args.check:
        _check(_OUTPUT_URDF, generated)
        _check(_MANIFEST, manifest)
        return

    _OUTPUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_URDF.write_bytes(generated)
    _MANIFEST.write_bytes(manifest)


if __name__ == "__main__":
    main()
