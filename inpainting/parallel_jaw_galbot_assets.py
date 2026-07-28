"""Build deterministic Galbot Golf URDFs for offline parallel-jaw rendering.

The public Galbot description contains a fixed-base whole robot, but its leg
and head joints are still actuated.  RoboLab holds those joints at one reviewed
display pose while controlling the two seven-DoF arms and two grippers.  This
module bakes that fixed posture into derived URDFs without changing the pinned
upstream checkout:

* ``galbot_render.urdf`` keeps both arms and articulated mimic grippers;
* ``galbot_ik.urdf`` keeps only the fourteen arm joints actuated.

Mesh paths are rewritten to the declared read-only container mount
``/robot_assets``.  The output manifest fingerprints the upstream repository,
RoboLab definitions, all visual meshes, this converter, and both derived URDFs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np


SCHEMA = "v2d.inpainting.parallel-jaw-robot-bundle/v1"
GALBOT_REVISION = "b311f5ca1acf506e9b7026397e2c74fb2db11df6"
ROBOLAB_REVISION = "8224d5fb8a2a3d21ce445bb198476c1faa4d69e6"
SOURCE_URDF = Path("urdf/galbot_one_golf_fixed_base.urdf")
CONTAINER_ASSET_ROOT = Path("/robot_assets")

FIXED_POSTURE = {
    "leg_joint1": 0.8,
    "leg_joint2": 2.3,
    "leg_joint3": 1.55,
    "leg_joint4": 0.0,
    "leg_joint5": 0.0,
    "head_joint1": 0.0,
    "head_joint2": 0.36,
}
LEFT_ARM_DEFAULT = (
    -0.1535,
    -1.0087,
    -0.0895,
    -1.5743,
    0.2422,
    -0.0009,
    0.9143,
)
RIGHT_ARM_DEFAULT = (
    0.1535,
    1.0087,
    0.0895,
    1.5743,
    -0.2422,
    -0.0009,
    -0.9143,
)
ARM_JOINTS = tuple(
    [f"left_arm_joint{index}" for index in range(1, 8)]
    + [f"right_arm_joint{index}" for index in range(1, 8)]
)
GRIPPER_DRIVERS = ("left_gripper_joint", "right_gripper_joint")
RENDERED_LINK_PREFIXES = (
    "left_arm",
    "right_arm",
    "left_gripper",
    "right_gripper",
    "left_wrist_camera",
    "right_wrist_camera",
)

# Authored linkage values and conversion used by the pinned RoboLab MR.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.703
GRIPPER_KNUCKLE_ANGLE = 1.2465
GRIPPER_INNER_PIVOT_HALF_GAP = 0.026
GRIPPER_FINGER_LINK_LENGTH = 0.045
GRIPPER_PAD_INSET = 0.0062
GRIPPER_MIN_OPENING = max(
    0.0,
    2.0
    * (
        GRIPPER_INNER_PIVOT_HALF_GAP
        - GRIPPER_PAD_INSET
        + GRIPPER_FINGER_LINK_LENGTH
        * math.sin(GRIPPER_KNUCKLE_ANGLE - GRIPPER_CLOSED)
    ),
)
GRIPPER_MAX_OPENING = 2.0 * (
    GRIPPER_INNER_PIVOT_HALF_GAP
    - GRIPPER_PAD_INSET
    + GRIPPER_FINGER_LINK_LENGTH * math.sin(GRIPPER_KNUCKLE_ANGLE)
)


class AssetBuildError(RuntimeError):
    """Raised when pinned inputs or derived assets are inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, *, root: Path | None = None) -> dict[str, object]:
    resolved = path.resolve()
    recorded = (
        resolved.relative_to(root.resolve()).as_posix()
        if root is not None
        else str(resolved)
    )
    return {
        "path": recorded,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise AssetBuildError(
            f"cannot resolve git revision for {root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _parse_vector(text: str | None, *, default: Iterable[float]) -> np.ndarray:
    if text is None:
        return np.asarray(tuple(default), dtype=np.float64)
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise AssetBuildError(f"invalid three-vector {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise AssetBuildError("actuated joint axis is zero")
    x, y, z = axis / norm
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Return fixed-axis XYZ angles for ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""

    matrix = np.asarray(rotation, dtype=np.float64)
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-matrix[0, 1]), float(matrix[1, 1]))
    result = np.array((roll, pitch, yaw), dtype=np.float64)
    if not np.allclose(_rpy_matrix(result), matrix, atol=2e-8, rtol=0.0):
        raise AssetBuildError("failed to represent fixed-joint rotation as URDF RPY")
    return result


def _origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = _parse_vector(
        origin.attrib.get("xyz") if origin is not None else None,
        default=(0.0, 0.0, 0.0),
    )
    rpy = _parse_vector(
        origin.attrib.get("rpy") if origin is not None else None,
        default=(0.0, 0.0, 0.0),
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _format_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _set_origin(joint: ET.Element, transform: np.ndarray) -> None:
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    origin.attrib.clear()
    origin.set("xyz", _format_vector(transform[:3, 3]))
    origin.set("rpy", _format_vector(_matrix_to_rpy(transform[:3, :3])))


def _freeze_joint(joint: ET.Element, value: float) -> None:
    joint_type = joint.attrib.get("type")
    transform = _origin_transform(joint)
    axis_node = joint.find("axis")
    axis = _parse_vector(
        axis_node.attrib.get("xyz") if axis_node is not None else None,
        default=(1.0, 0.0, 0.0),
    )
    motion = np.eye(4, dtype=np.float64)
    if joint_type in {"revolute", "continuous"}:
        motion[:3, :3] = _axis_angle(axis, float(value))
    elif joint_type == "prismatic":
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            raise AssetBuildError(f"joint {joint.attrib.get('name')} has zero axis")
        motion[:3, 3] = axis / norm * float(value)
    elif joint_type == "fixed":
        if not np.isclose(value, 0.0):
            raise AssetBuildError("cannot assign a nonzero value to a fixed joint")
    else:
        raise AssetBuildError(
            f"cannot freeze joint {joint.attrib.get('name')} of type {joint_type!r}"
        )
    _set_origin(joint, transform @ motion)
    joint.set("type", "fixed")
    for child_name in (
        "axis",
        "limit",
        "mimic",
        "dynamics",
        "calibration",
        "safety_controller",
    ):
        child = joint.find(child_name)
        if child is not None:
            joint.remove(child)


def _rewrite_mesh_paths(root: ET.Element, *, source_root: Path) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for mesh in root.findall("./link/visual/geometry/mesh"):
        filename = mesh.attrib.get("filename", "")
        if not filename or filename.startswith(("package://", "file://")):
            raise AssetBuildError(f"unsupported Galbot mesh URI {filename!r}")
        source = (source_root / SOURCE_URDF.parent / filename).resolve()
        try:
            relative = source.relative_to(source_root.resolve())
        except ValueError as exc:
            raise AssetBuildError(f"mesh {source} escapes Galbot source root") from exc
        if not source.is_file() or source.stat().st_size <= 0:
            raise AssetBuildError(f"missing or empty Galbot visual mesh {source}")
        mesh.set("filename", (CONTAINER_ASSET_ROOT / relative).as_posix())
        resolved.append(source)
    if not resolved:
        raise AssetBuildError("Galbot URDF contains no visual meshes")
    return tuple(dict.fromkeys(resolved))


def _strip_non_arm_visuals(root: ET.Element) -> tuple[str, ...]:
    """Keep the study's robot-overlay scope to arms, wrist cameras, and jaws."""

    stripped: list[str] = []
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if name.startswith(RENDERED_LINK_PREFIXES):
            continue
        visuals = list(link.findall("visual"))
        for visual in visuals:
            link.remove(visual)
        if visuals:
            stripped.append(name)
    if not stripped:
        raise AssetBuildError("Galbot arms-only derivation stripped no support visuals")
    return tuple(stripped)


def _link_transforms(root: ET.Element) -> dict[str, np.ndarray]:
    links = {link.attrib["name"] for link in root.findall("link")}
    children: dict[str, tuple[str, np.ndarray]] = {}
    child_links: set[str] = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise AssetBuildError("joint is missing parent or child")
        parent_name, child_name = parent.attrib["link"], child.attrib["link"]
        if child_name in children:
            raise AssetBuildError(f"link {child_name} has multiple parents")
        children[child_name] = (parent_name, _origin_transform(joint))
        child_links.add(child_name)
    roots = sorted(links - child_links)
    if len(roots) != 1:
        raise AssetBuildError(f"URDF must have one root link, got {roots}")
    cache: dict[str, np.ndarray] = {roots[0]: np.eye(4)}

    def resolve(link: str) -> np.ndarray:
        if link in cache:
            return cache[link]
        parent, transform = children[link]
        cache[link] = resolve(parent) @ transform
        return cache[link]

    for link in links:
        resolve(link)
    return cache


def _derived_tree(
    source_tree: ET.ElementTree,
    *,
    source_root: Path,
    keep_grippers: bool,
) -> tuple[ET.ElementTree, tuple[Path, ...], np.ndarray]:
    root = deepcopy(source_tree.getroot())
    if root.tag != "robot":
        raise AssetBuildError("Galbot URDF root is not <robot>")
    root.set(
        "name",
        "galbot_one_golf_parallel_jaw_render"
        if keep_grippers
        else "galbot_one_golf_parallel_jaw_ik",
    )
    joints = {joint.attrib.get("name", ""): joint for joint in root.findall("joint")}
    missing = sorted(set((*FIXED_POSTURE, *ARM_JOINTS, *GRIPPER_DRIVERS)) - set(joints))
    if missing:
        raise AssetBuildError(f"Galbot URDF lacks expected joints {missing}")

    for name, value in FIXED_POSTURE.items():
        _freeze_joint(joints[name], value)
    if not keep_grippers:
        for joint in root.findall("joint"):
            name = joint.attrib.get("name", "")
            if "gripper" in name and joint.attrib.get("type") != "fixed":
                _freeze_joint(joint, 0.0)

    actuated = tuple(
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") not in {"fixed", "floating"}
        and joint.find("mimic") is None
    )
    expected = (*ARM_JOINTS, *GRIPPER_DRIVERS) if keep_grippers else ARM_JOINTS
    if set(actuated) != set(expected) or len(actuated) != len(expected):
        raise AssetBuildError(
            f"derived Galbot actuated joints {actuated} do not match {expected}"
        )

    transforms = _link_transforms(root)
    left_base = transforms["left_arm_base_link"][:3, 3]
    right_base = transforms["right_arm_base_link"][:3, 3]
    # T_A_B maps B-frame coordinates into A. This is T_robot_root_hub:
    # the hub is root-aligned at the midpoint of the two arm bases.
    robot_root_from_hub = np.eye(4, dtype=np.float64)
    robot_root_from_hub[:3, 3] = (left_base + right_base) / 2.0
    _strip_non_arm_visuals(root)
    meshes = _rewrite_mesh_paths(root, source_root=source_root)
    return ET.ElementTree(root), meshes, robot_root_from_hub


def _write_xml(path: Path, tree: ET.ElementTree) -> None:
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_bundle(
    *,
    galbot_root: str | Path,
    robolab_root: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, object]:
    galbot = Path(galbot_root).resolve()
    robolab = Path(robolab_root).resolve()
    destination = Path(output_dir).resolve()
    if _git_revision(galbot) != GALBOT_REVISION:
        raise AssetBuildError(
            f"Galbot checkout must be pinned to {GALBOT_REVISION}"
        )
    if _git_revision(robolab) != ROBOLAB_REVISION:
        raise AssetBuildError(
            f"RoboLab checkout must be pinned to {ROBOLAB_REVISION}"
        )
    source_urdf = galbot / SOURCE_URDF
    if not source_urdf.is_file():
        raise FileNotFoundError(source_urdf)
    definitions = robolab / "robolab/robots/galbot_golf_definitions.py"
    config = robolab / "robolab/robots/galbot_golf.py"
    for source in (definitions, config):
        if not source.is_file():
            raise FileNotFoundError(source)

    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        allowed = {
            "galbot_render.urdf",
            "galbot_ik.urdf",
            "bundle_manifest.json",
        }
        unexpected = sorted(path.name for path in destination.iterdir() if path.name not in allowed)
        if unexpected:
            raise AssetBuildError(
                f"refusing to overwrite bundle with unexpected entries {unexpected}"
            )

    source_tree = ET.parse(source_urdf)
    render_tree, render_meshes, render_hub = _derived_tree(
        source_tree, source_root=galbot, keep_grippers=True
    )
    ik_tree, ik_meshes, ik_hub = _derived_tree(
        source_tree, source_root=galbot, keep_grippers=False
    )
    if not np.allclose(render_hub, ik_hub, atol=1e-12, rtol=0.0):
        raise AssetBuildError("render and IK hub transforms differ")
    if set(render_meshes) != set(ik_meshes):
        raise AssetBuildError("render and IK URDFs reference different visual meshes")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}.candidate-"
    ) as temporary:
        candidate = Path(temporary)
        render_path = candidate / "galbot_render.urdf"
        ik_path = candidate / "galbot_ik.urdf"
        _write_xml(render_path, render_tree)
        _write_xml(ik_path, ik_tree)

        semantic_to_tcp = np.array(
            (
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
            ),
            dtype=np.float64,
        )
        if not np.isclose(np.linalg.det(semantic_to_tcp), 1.0):
            raise AssetBuildError("Galbot semantic-to-TCP rotation is improper")
        source_provenance = {
            "repository": "https://github.com/GalaxyGeneralRobotics/galbot_one_golf_description",
            "revision": GALBOT_REVISION,
            "urdf": _record(source_urdf, root=galbot),
            "visual_meshes": [
                _record(path, root=galbot)
                for path in sorted(
                    render_meshes,
                    key=lambda item: item.relative_to(galbot).as_posix(),
                )
            ],
        }
        robolab_provenance = {
            "merge_request": "https://gitlab-master.nvidia.com/xuningy/robolab/-/merge_requests/62",
            "revision": ROBOLAB_REVISION,
            "definitions": _record(definitions, root=robolab),
            "configuration": _record(config, root=robolab),
        }
        manifest: dict[str, object] = {
            "schema_version": SCHEMA,
            "state": "complete",
            "completed_at": _utc_now(),
            "robot_id": "galbot_one_golf",
            "render_urdf": render_path.name,
            "ik_urdf": ik_path.name,
            "tcp_frames": {
                "left": "left_gripper_tcp_link",
                "right": "right_gripper_tcp_link",
            },
            "T_robot_root_hub": render_hub.tolist(),
            "semantic_target_to_tcp_rotation": semantic_to_tcp.tolist(),
            "arm_joint_names": list(ARM_JOINTS),
            "arm_seed": {
                "left": list(LEFT_ARM_DEFAULT),
                "right": list(RIGHT_ARM_DEFAULT),
            },
            "gripper_mapping": {
                "kind": "galbot_four_bar",
                "joint_names": {
                    "left": ["left_gripper_joint"],
                    "right": ["right_gripper_joint"],
                },
                "params": {
                    "joint_lower_rad": GRIPPER_OPEN,
                    "joint_upper_rad": GRIPPER_CLOSED,
                    "knuckle_angle_rad": GRIPPER_KNUCKLE_ANGLE,
                    "inner_pivot_half_gap_m": GRIPPER_INNER_PIVOT_HALF_GAP,
                    "finger_link_length_m": GRIPPER_FINGER_LINK_LENGTH,
                    "pad_inset_m": GRIPPER_PAD_INSET,
                },
            },
            "fixed_root_posture": {
                "joint_values": {},
                "provenance": {
                    "policy": "baked_into_derived_urdfs",
                    "joint_values": FIXED_POSTURE,
                    "source": "RoboLab MR 62 reviewed display posture",
                },
            },
            "asset_provenance": {
                "source": source_provenance,
                "robolab_example": robolab_provenance,
                "converter": _record(Path(__file__)),
                "container_asset_mount": str(CONTAINER_ASSET_ROOT),
                "visual_scope": {
                    "policy": "arms_wrist_cameras_and_grippers_only",
                    "kept_link_prefixes": list(RENDERED_LINK_PREFIXES),
                    "reason": (
                        "match the established arms-only inpainting overlay and "
                        "avoid rendering the ego robot's torso/head into its own view"
                    ),
                },
                "derived_files": {
                    "render_urdf": _record(render_path, root=candidate),
                    "ik_urdf": _record(ik_path, root=candidate),
                },
                "physical_aperture_range_m": [
                    GRIPPER_MIN_OPENING,
                    GRIPPER_MAX_OPENING,
                ],
            },
        }
        manifest_path = candidate / "bundle_manifest.json"
        _write_json(manifest_path, manifest)

        if destination.exists():
            for name in ("galbot_render.urdf", "galbot_ik.urdf", "bundle_manifest.json"):
                (destination / name).unlink(missing_ok=True)
            destination.rmdir()
        Path(temporary).replace(destination)
    return json.loads((destination / "bundle_manifest.json").read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--galbot-root", required=True, type=Path)
    parser.add_argument("--robolab-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_bundle(
        galbot_root=args.galbot_root,
        robolab_root=args.robolab_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
