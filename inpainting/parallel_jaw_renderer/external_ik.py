"""Load the proven Pink-based ``arm_ik.py`` from one explicit source file."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


IK_CONSTRUCTOR_KWARGS = {
    "dt": 0.5,
    "lm_damping": 1e-4,
    "orientation_cost": 0.010,
    # Parallel-jaw embodiments do not share Vega's calibrated elbow frames or
    # shoulder geometry. Disable the Vega-specific null-space preference.
    "elbow_out_gain": 0.0,
    "elbow_out_refinement_iters": 0,
}


class ExternalIKError(RuntimeError):
    """Raised when the isolated Pink IK implementation cannot be loaded."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExternalIK:
    source: Path
    module: ModuleType

    def as_dict(self) -> dict[str, str]:
        return {
            "source": str(self.source),
            "sha256": sha256_file(self.source),
            "import_mode": "isolated_source_file_no_robotic_grounding_package_init",
        }


def resolve_arm_ik_source(scene_utils_root: str | Path) -> Path:
    root = Path(scene_utils_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    source = root / "arm_ik.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def load_arm_ik(scene_utils_root: str | Path) -> ExternalIK:
    source = resolve_arm_ik_source(scene_utils_root)
    token = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
    module_name = f"_v2d_parallel_jaw_arm_ik_{token}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ExternalIKError(f"could not create import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ExternalIKError(
            f"failed to load isolated arm IK from {source}: {exc}"
        ) from exc
    if not hasattr(module, "ArmIK"):
        sys.modules.pop(module_name, None)
        raise ExternalIKError(f"{source} does not define ArmIK")
    return ExternalIK(source=source, module=module)
