# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural and kinematic verification for the versioned Vega Sharpa asset."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_V2_ROOT = _REPO_ROOT / "src/flash_chord/assets/urdfs/vega_sharpa/v2"
_SOURCE_PATH = _V2_ROOT / "upstream/vega_sharpa.urdf"
_GENERATED_PATH = _V2_ROOT / "vega_sharpa_58dof.urdf"
_MANIFEST_PATH = _V2_ROOT / "SOURCE.json"
_MESH_ROOT = _REPO_ROOT / "src/flash_chord/assets/meshes/vega_sharpa/v2"
_GENERATOR = _REPO_ROOT / "scripts/assets/generate_vega_sharpa.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xml(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _movable(root: ET.Element) -> tuple[str, ...]:
    return tuple(joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] != "fixed")


def _dfs_movable(root: ET.Element) -> tuple[str, ...]:
    children: dict[str, list[ET.Element]] = {}
    child_links = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children.setdefault(parent, []).append(joint)
        child_links.add(child)
    roots = {link.attrib["name"] for link in root.findall("link")} - child_links
    assert len(roots) == 1

    ordered = []

    def visit(link: str) -> None:
        for joint in children.get(link, ()):
            if joint.attrib["type"] != "fixed":
                ordered.append(joint.attrib["name"])
            visit(joint.find("child").attrib["link"])

    visit(roots.pop())
    return tuple(ordered)


def _vector(element: ET.Element | None, attribute: str, default: tuple[float, float, float]):
    if element is None or attribute not in element.attrib:
        return np.asarray(default, dtype=np.float64)
    value = np.fromstring(element.attrib[attribute], sep=" ", dtype=np.float64)
    assert value.shape == (3,)
    return value


def _rotation_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def _rotation_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    return np.asarray(
        (
            (cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine),
            (y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine),
            (z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement),
        ),
        dtype=np.float64,
    )


def _joint_transform(joint: ET.Element, position: float = 0.0) -> np.ndarray:
    origin = joint.find("origin")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = _vector(origin, "xyz", (0.0, 0.0, 0.0))
    transform[:3, :3] = _rotation_rpy(_vector(origin, "rpy", (0.0, 0.0, 0.0)))
    if joint.attrib["type"] != "fixed":
        transform[:3, :3] @= _rotation_axis_angle(
            _vector(joint.find("axis"), "xyz", (1.0, 0.0, 0.0)),
            position,
        )
    return transform


def _forward_kinematics(
    root: ET.Element,
    joint_position: dict[str, float],
    initial: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    transforms = dict(initial)
    unresolved = list(root.findall("joint"))
    while unresolved:
        remaining = []
        for joint in unresolved:
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            if parent not in transforms:
                remaining.append(joint)
                continue
            transforms[child] = transforms[parent] @ _joint_transform(
                joint,
                joint_position.get(joint.attrib["name"], 0.0),
            )
        assert len(remaining) < len(unresolved), "URDF joint graph is disconnected or cyclic"
        unresolved = remaining
    return transforms


def test_snapshot_hashes_meshes_and_generation_are_reproducible():
    manifest = json.loads(_MANIFEST_PATH.read_text())
    assert manifest["upstream"]["revision"] == "7cd29eecbd1c41bb299100421f7a2068b9fc1bce"
    assert manifest["upstream"]["urdf_sha256"] == _sha256(_SOURCE_PATH)
    assert manifest["generated"]["sha256"] == _sha256(_GENERATED_PATH)
    assert manifest["generated"]["generator_sha256"] == _sha256(_GENERATOR)

    mesh_files = {path.relative_to(_MESH_ROOT).as_posix(): path for path in _MESH_ROOT.rglob("*") if path.is_file()}
    assert len(mesh_files) == 94
    assert set(manifest["meshes"]["files"]) == set(mesh_files)
    assert {name: _sha256(path) for name, path in mesh_files.items()} == manifest["meshes"]["files"]

    runtime_meshes = tuple(
        _GENERATED_PATH.parent / mesh.attrib["filename"] for mesh in _xml(_GENERATED_PATH).iter("mesh")
    )
    assert len(runtime_meshes) == 171
    assert all(path.resolve().is_file() for path in runtime_meshes)

    subprocess.run([sys.executable, str(_GENERATOR), "--check"], cwd=_REPO_ROOT, check=True)


def test_generated_asset_matches_its_current_active_joint_manifest():
    source = _xml(_SOURCE_PATH)
    generated = _xml(_GENERATED_PATH)
    manifest = json.loads(_MANIFEST_PATH.read_text())

    frozen = manifest["frozen_joint_positions"]
    source_active = tuple(name for name in _movable(source) if name not in frozen)
    manifest_active = tuple(manifest["active_joints"])
    assert len(_movable(source)) == 70
    assert len(source_active) == 58
    assert set(_movable(generated)) == set(source_active) == set(manifest_active)
    assert _dfs_movable(generated) == manifest_active

    for name, position in frozen.items():
        source_joint = source.find(f"./joint[@name='{name}']")
        generated_joint = generated.find(f"./joint[@name='{name}']")
        assert source_joint is not None and source_joint.attrib["type"] != "fixed"
        assert generated_joint is not None and generated_joint.attrib["type"] == "fixed"
        assert math.isfinite(position)
        for tag in ("axis", "calibration", "dynamics", "limit", "mimic", "safety_controller"):
            assert generated_joint.find(tag) is None


def test_generated_fixed_tree_matches_upstream_at_frozen_configuration():
    source = _xml(_SOURCE_PATH)
    generated = _xml(_GENERATED_PATH)
    frozen = json.loads(_MANIFEST_PATH.read_text())["frozen_joint_positions"]
    base_offset = generated.find("./joint[@name='base_offset']")
    assert base_offset is not None

    source_fk = _forward_kinematics(source, frozen, {"base": _joint_transform(base_offset)})
    generated_fk = _forward_kinematics(generated, {}, {"root": np.eye(4, dtype=np.float64)})
    source_links = {link.attrib["name"] for link in source.findall("link")}
    assert source_links == set(source_fk)
    assert source_links < set(generated_fk)
    for name in sorted(source_links):
        np.testing.assert_allclose(generated_fk[name], source_fk[name], atol=1.0e-12, rtol=0.0, err_msg=name)

    for name in ("arm_center", "L_ee", "R_ee", "head_l3", "zed_depth_frame", "zed_left_camera", "zed_right_camera"):
        assert name in generated_fk
