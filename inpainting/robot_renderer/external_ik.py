"""Explicit, package-init-free loading of the proven Vega IK/mount modules.

The sibling robotic-grounding checkout contains the implementation we want, but
importing ``robotic_grounding`` itself enters the Isaac package tree.  This
loader executes only ``arm_ik.py`` and ``arm_mount_opt.py`` under a private
synthetic package.  ``arm_mount_opt`` needs one small helper from
``arm_replay.py``; an equivalent local stub is injected because the real module
imports the Isaac-oriented replay package at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
from types import ModuleType

import numpy as np


FLANGES = ("L_arm_l8", "R_arm_l8")

# Copied verbatim from the complete sibling asset configuration
# ``robotic_grounding/assets/vega_arms.py``.  These are physical l8 plate ->
# Sharpa C_MC transforms, not guessed offsets.
HAND_MOUNT = {
    "L_arm_l8": {"xyz": (0.00019, 0.0039, 0.00014), "rpy": (0.0, 0.0, -1.57079)},
    "R_arm_l8": {"xyz": (0.00019, -0.0039, 0.00014), "rpy": (0.0, 0.0, 1.57079)},
}

IK_CONSTRUCTOR_KWARGS = {
    "dt": 0.5,
    "lm_damping": 1e-4,
    "orientation_cost": 0.010,
    "elbow_out_gain": 0.05,
    "elbow_out_margin": 0.06,
    "elbow_out_max_step": 0.002,
    "elbow_out_refinement_iters": 3,
}


class ExternalIKError(RuntimeError):
    """Raised when the isolated external IK sources cannot be loaded."""


@dataclass(frozen=True)
class ExternalIKSources:
    root: Path
    arm_ik: Path
    arm_mount_opt: Path

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "arm_ik": str(self.arm_ik),
            "arm_ik_sha256": _sha256(self.arm_ik),
            "arm_mount_opt": str(self.arm_mount_opt),
            "arm_mount_opt_sha256": _sha256(self.arm_mount_opt),
            "import_mode": "isolated_source_files_no_robotic_grounding_package_init",
            "arm_replay_helper": "local_equivalent_stub_for_build_hand_mount_inverses",
        }


@dataclass(frozen=True)
class ExternalIKModules:
    sources: ExternalIKSources
    arm_ik: ModuleType
    arm_mount_opt: ModuleType


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_external_ik_sources(scene_utils_root: str | Path) -> ExternalIKSources:
    root = Path(scene_utils_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    arm_ik = root / "arm_ik.py"
    arm_mount_opt = root / "arm_mount_opt.py"
    for path in (arm_ik, arm_mount_opt):
        if not path.is_file():
            raise FileNotFoundError(path)
    return ExternalIKSources(root=root, arm_ik=arm_ik, arm_mount_opt=arm_mount_opt)


def _load_source(module_name: str, source: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ExternalIKError(f"could not build import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _build_hand_mount_inverses(hand_mount: dict | None, flanges):
    """Equivalent to the 13-line helper in external ``arm_replay.py``."""

    import pinocchio as pin

    result = {}
    for flange in flanges:
        specification = (hand_mount or {}).get(flange)
        if specification is None:
            result[flange] = pin.SE3.Identity()
        else:
            rotation = pin.rpy.rpyToMatrix(
                *[float(value) for value in specification["rpy"]]
            )
            transform = pin.SE3(
                rotation,
                np.asarray(specification["xyz"], dtype=float),
            )
            result[flange] = transform.inverse()
    return result


def load_external_ik(scene_utils_root: str | Path) -> ExternalIKModules:
    """Load only ``arm_ik.py`` and ``arm_mount_opt.py`` from an explicit path."""

    sources = resolve_external_ik_sources(scene_utils_root)
    token = hashlib.sha256(str(sources.root).encode("utf-8")).hexdigest()[:12]
    package_name = f"_v2d_external_scene_utils_{token}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(sources.root)]
    package.__package__ = package_name
    package.__file__ = str(sources.root)
    sys.modules[package_name] = package

    arm_replay_name = f"{package_name}.arm_replay"
    arm_replay_stub = types.ModuleType(arm_replay_name)
    arm_replay_stub.__package__ = package_name
    arm_replay_stub.__file__ = "<v2d arm_replay helper stub>"
    arm_replay_stub.build_hand_mount_inverses = _build_hand_mount_inverses
    sys.modules[arm_replay_name] = arm_replay_stub

    try:
        arm_ik = _load_source(f"{package_name}.arm_ik", sources.arm_ik)
        arm_mount_opt = _load_source(
            f"{package_name}.arm_mount_opt", sources.arm_mount_opt
        )
    except Exception as exc:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        raise ExternalIKError(
            f"failed to load isolated IK sources from {sources.root}: {exc}"
        ) from exc
    if not hasattr(arm_ik, "ArmIK"):
        raise ExternalIKError(f"{sources.arm_ik} does not define ArmIK")
    if not hasattr(arm_mount_opt, "place_hub_from_wrists"):
        raise ExternalIKError(
            f"{sources.arm_mount_opt} does not define place_hub_from_wrists"
        )
    return ExternalIKModules(
        sources=sources,
        arm_ik=arm_ik,
        arm_mount_opt=arm_mount_opt,
    )
