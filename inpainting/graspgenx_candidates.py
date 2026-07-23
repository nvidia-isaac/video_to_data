"""Deterministic GraspGenX candidate generation from a metric object mesh.

The adapter deliberately exposes only the sweep-volume-conditioned GraspGenX
path.  A mesh is sampled in its local (object) frame, the sampled cloud is
mean-centered for inference, and returned gripper poses are transformed back
to the original object frame before they are saved.

The NPZ contract is intentionally small and exact:

``object_to_gripper_base``
    ``(N, 4, 4)`` float32 poses of the gripper base in the object mesh frame.

``confidence``
    ``(N,)`` float32 GraspGenX discriminator confidences, sorted descending.

Heavy GraspGenX and torch imports are deferred to the default provider.  Tests
and downstream orchestration can inject an inference provider without needing
a CUDA installation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np


SCHEMA_VERSION = "v2d.inpainting.graspgenx-candidates/v1"
ADAPTER_VERSION = "1.0.0"
SUPPORTED_MESH_SUFFIXES = frozenset({".obj", ".ply", ".glb"})
GRIPPER_TYPE_IDS = {
    "parallel_2f": 0,
    "revolute_2f": 1,
    "revolute_3f": 2,
}
RIGID_ATOL = 1.0e-4


class GraspGenXCandidateError(RuntimeError):
    """Base error for candidate generation failures."""


class CandidateValidationError(ValueError):
    """Raised when inputs or provider outputs violate the artifact contract."""


class ArtifactExistsError(FileExistsError):
    """Raised when a non-overwriting write encounters an existing artifact."""


@dataclass(frozen=True)
class SweepVolume:
    """Sweep-volume-v2 conditioning in the gripper-base frame, in meters."""

    extents_open: tuple[float, float, float]
    offset_open: tuple[float, float, float]
    extents_mid: tuple[float, float, float]
    offset_mid: tuple[float, float, float]
    fingertip_depth: float
    gripper_type: int
    gripper_name: str

    @classmethod
    def create(
        cls,
        *,
        extents_open: Sequence[float],
        offset_open: Sequence[float],
        extents_mid: Sequence[float],
        offset_mid: Sequence[float],
        fingertip_depth: float,
        gripper_type: str | int,
        gripper_name: str,
    ) -> "SweepVolume":
        fields = {
            "extents_open": _vector3(extents_open, "extents_open"),
            "offset_open": _vector3(offset_open, "offset_open"),
            "extents_mid": _vector3(extents_mid, "extents_mid"),
            "offset_mid": _vector3(offset_mid, "offset_mid"),
        }
        for name in ("extents_open", "extents_mid"):
            if not np.all(np.asarray(fields[name]) > 0.0):
                raise CandidateValidationError(
                    f"{name} must contain three strictly positive meter extents; "
                    f"got {list(fields[name])}"
                )

        depth = float(fingertip_depth)
        if not np.isfinite(depth) or depth <= 0.0:
            raise CandidateValidationError(
                "fingertip_depth must be a positive finite value in meters"
            )

        type_id = _gripper_type_id(gripper_type)
        name = str(gripper_name).strip()
        if not name:
            raise CandidateValidationError("gripper_name must not be empty")
        return cls(
            **fields,
            fingertip_depth=depth,
            gripper_type=type_id,
            gripper_name=name,
        )

    def provider_dict(self) -> dict[str, Any]:
        """Return the mapping accepted by ``from_sweep_volume``."""

        return {
            "extents_open": list(self.extents_open),
            "offset_open": list(self.offset_open),
            "extents_mid": list(self.extents_mid),
            "offset_mid": list(self.offset_mid),
            "fingertip_depth": self.fingertip_depth,
            "gripper_type": self.gripper_type,
        }

    def provenance_dict(self) -> dict[str, Any]:
        result = self.provider_dict()
        result["gripper_name"] = self.gripper_name
        result["units"] = "meters"
        result["gripper_type_name"] = next(
            name for name, value in GRIPPER_TYPE_IDS.items() if value == self.gripper_type
        )
        return result


@dataclass(frozen=True)
class CandidateArtifact:
    """Generated candidates and their committed artifact paths."""

    object_to_gripper_base: np.ndarray
    confidence: np.ndarray
    npz_path: Path
    provenance_path: Path
    provenance: Mapping[str, Any]


class InferenceProvider(Protocol):
    """Injectable GraspGenX-compatible inference boundary."""

    def __call__(
        self,
        *,
        point_cloud_object_centered: np.ndarray,
        sweep_volume: Mapping[str, Any],
        graspgenx_root: Path,
        checkpoint_root: Path,
        seed: int,
        num_grasps: int,
    ) -> tuple[Any, Any]:
        """Return centered-frame gripper poses and confidence scores."""


def _vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(f"{name} must contain three numbers") from exc
    if array.shape != (3,):
        raise CandidateValidationError(
            f"{name} must have shape (3,), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise CandidateValidationError(f"{name} must contain only finite values")
    return tuple(float(item) for item in array)


def _gripper_type_id(value: str | int) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized not in GRIPPER_TYPE_IDS:
            raise CandidateValidationError(
                f"unknown gripper_type {value!r}; expected one of "
                f"{sorted(GRIPPER_TYPE_IDS)}"
            )
        return GRIPPER_TYPE_IDS[normalized]
    if isinstance(value, (bool, np.bool_)):
        raise CandidateValidationError("gripper_type must not be boolean")
    try:
        type_id = int(value)
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(
            "gripper_type must be a supported name or integer ID"
        ) from exc
    if type_id not in GRIPPER_TYPE_IDS.values():
        raise CandidateValidationError(
            f"gripper_type ID must be one of {sorted(GRIPPER_TYPE_IDS.values())}, "
            f"got {type_id}"
        )
    return type_id


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _load_metric_mesh(mesh_path: Path) -> tuple[Any, dict[str, Any]]:
    suffix = mesh_path.suffix.lower()
    if suffix not in SUPPORTED_MESH_SUFFIXES:
        raise CandidateValidationError(
            f"unsupported mesh extension {suffix!r}; expected OBJ, PLY, or GLB"
        )
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    try:
        import trimesh
    except ImportError as exc:
        raise GraspGenXCandidateError(
            "trimesh is required to load and sample candidate meshes"
        ) from exc

    try:
        loaded = trimesh.load(str(mesh_path), force="scene", process=False)
    except Exception as exc:  # trimesh uses format-specific exception classes
        raise CandidateValidationError(f"could not load mesh {mesh_path}: {exc}") from exc

    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise CandidateValidationError(
            f"{mesh_path} did not contain triangle-mesh geometry"
        )

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
        raise CandidateValidationError("mesh must contain at least three 3D vertices")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise CandidateValidationError("mesh must contain triangular faces")
    if not np.all(np.isfinite(vertices)):
        raise CandidateValidationError("mesh vertices must be finite")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise CandidateValidationError("mesh contains out-of-range face indices")

    triangles = vertices[faces]
    twice_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    total_area = float(0.5 * twice_area.sum())
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    extents = bounds[1] - bounds[0]
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise CandidateValidationError("mesh has zero or non-finite surface area")
    if not np.all(np.isfinite(extents)) or float(np.max(extents)) <= 0.0:
        raise CandidateValidationError("mesh has zero or non-finite spatial extent")

    return mesh, {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "surface_area_m2": total_area,
        "bounds_m": bounds.tolist(),
        "extents_m": extents.tolist(),
        "units": "meters",
        "format": suffix[1:],
    }


def _sample_surface_deterministically(
    mesh: Any,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Area-sample a triangular mesh using a local PCG64 generator."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    valid = np.isfinite(areas) & (areas > 0.0)
    if not np.any(valid):
        raise CandidateValidationError("mesh has no non-degenerate triangles to sample")
    triangles = triangles[valid]
    areas = areas[valid]

    rng = np.random.Generator(np.random.PCG64(seed))
    cumulative = np.cumsum(areas, dtype=np.float64)
    draws = rng.random(count) * cumulative[-1]
    face_indices = np.searchsorted(cumulative, draws, side="right")
    selected = triangles[face_indices]

    barycentric_draws = rng.random((count, 2))
    root = np.sqrt(barycentric_draws[:, 0])
    weights = np.column_stack(
        (
            1.0 - root,
            root * (1.0 - barycentric_draws[:, 1]),
            root * barycentric_draws[:, 1],
        )
    )
    points = np.einsum("ni,nij->nj", weights, selected)
    if points.shape != (count, 3) or not np.all(np.isfinite(points)):
        raise CandidateValidationError("deterministic mesh sampling produced invalid points")
    return points


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    try:
        result = np.asarray(converted)
    except Exception as exc:
        raise CandidateValidationError(f"{name} could not be converted to numpy") from exc
    if result.dtype.kind not in "fiu":
        raise CandidateValidationError(f"{name} must be a numeric array")
    return result


def validate_candidates(
    object_to_gripper_base: Any,
    confidence: Any,
    *,
    rigid_atol: float = RIGID_ATOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly validate and normalize a candidate array pair."""

    transforms = _as_numpy(
        object_to_gripper_base, name="object_to_gripper_base"
    ).astype(np.float64, copy=False)
    scores = _as_numpy(confidence, name="confidence").astype(np.float64, copy=False)

    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise CandidateValidationError(
            "object_to_gripper_base must have shape (N, 4, 4), "
            f"got {transforms.shape}"
        )
    if transforms.shape[0] == 0:
        raise CandidateValidationError("provider returned no grasp candidates")
    if scores.ndim != 1:
        raise CandidateValidationError(
            f"confidence must have shape (N,), got {scores.shape}"
        )
    if len(scores) != len(transforms):
        raise CandidateValidationError(
            "candidate and confidence counts differ: "
            f"{len(transforms)} != {len(scores)}"
        )
    if not np.all(np.isfinite(transforms)):
        raise CandidateValidationError("object_to_gripper_base contains non-finite values")
    if not np.all(np.isfinite(scores)):
        raise CandidateValidationError("confidence contains non-finite values")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise CandidateValidationError("confidence values must lie in [0, 1]")

    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(
        transforms[:, 3, :], expected_bottom[None, :], atol=rigid_atol, rtol=0.0
    ):
        raise CandidateValidationError(
            "each transform must end in homogeneous row [0, 0, 0, 1]"
        )

    rotations = transforms[:, :3, :3]
    gram = np.einsum("nji,njk->nik", rotations, rotations)
    identities = np.broadcast_to(np.eye(3), gram.shape)
    if not np.allclose(gram, identities, atol=rigid_atol, rtol=0.0):
        raise CandidateValidationError("candidate rotations must be orthonormal")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=rigid_atol, rtol=0.0):
        raise CandidateValidationError(
            "candidate rotations must be proper rotations with determinant +1"
        )

    return transforms.astype(np.float32), scores.astype(np.float32)


def _latest_checkpoint(directory: Path) -> Path:
    epoch_paths = list(directory.glob("epoch_*.pth"))

    def epoch(path: Path) -> tuple[int, str]:
        match = re.fullmatch(r"epoch_(\d+)", path.stem)
        return (int(match.group(1)) if match else -1, path.name)

    candidates = epoch_paths or list(directory.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no .pth checkpoint found in {directory}")
    return max(candidates, key=epoch)


def _validate_default_runtime_paths(
    graspgenx_root: Path,
    checkpoint_root: Path,
) -> None:
    if not (graspgenx_root / "graspgenx" / "grasp_server.py").is_file():
        raise FileNotFoundError(
            f"{graspgenx_root} is not a GraspGenX source checkout"
        )
    for role in ("gen", "dis"):
        role_dir = checkpoint_root / role
        if not role_dir.is_dir():
            raise FileNotFoundError(
                f"checkpoint root must contain {role}/: {checkpoint_root}"
            )
        if not (role_dir / "config.yaml").is_file():
            raise FileNotFoundError(role_dir / "config.yaml")
        _latest_checkpoint(role_dir)


def _checkpoint_provenance(checkpoint_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(checkpoint_root)}
    for role in ("gen", "dis"):
        directory = checkpoint_root / role
        config = directory / "config.yaml"
        role_result: dict[str, Any] = {
            "config": _file_record(config) if config.is_file() else {
                "path": str(config),
                "exists": False,
            }
        }
        try:
            checkpoint = _latest_checkpoint(directory)
        except FileNotFoundError:
            role_result["checkpoint"] = {
                "path": str(directory),
                "exists": False,
            }
        else:
            role_result["checkpoint"] = _file_record(checkpoint)
        result[role] = role_result
    return result


def _git_record(root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"root": str(root)}
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        record.update({"revision": None, "tracked_files_dirty": None})
    else:
        record.update({"revision": revision, "tracked_files_dirty": dirty})
    return record


def _declared_graspgenx_version(root: Path) -> str | None:
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        match = re.search(
            r"(?ms)^\[project\].*?^version\s*=\s*[\"']([^\"']+)[\"']", text
        )
        if match:
            return match.group(1)
    init = root / "graspgenx" / "__init__.py"
    if init.is_file():
        match = re.search(
            r"(?m)^__version__\s*=\s*[\"']([^\"']+)[\"']",
            init.read_text(encoding="utf-8"),
        )
        if match:
            return match.group(1)
    return None


@contextmanager
def _runtime_import_context(
    graspgenx_root: Path,
    checkpoint_root: Path,
) -> Iterator[None]:
    """Prefer the requested checkout and suppress its optional auto-downloads."""

    root_string = str(graspgenx_root)
    inserted = root_string not in sys.path
    if inserted:
        sys.path.insert(0, root_string)

    environment_updates = {
        # The import hook only checks that these roots exist.  Inference below
        # still consumes the explicit checkpoint_root/{gen,dis} paths.
        "GRASPGENX_CHECKPOINT_DIR": str(checkpoint_root),
        "GRASPGENX_GRIPPER_CFG_DIR": str(graspgenx_root),
    }
    previous = {name: os.environ.get(name) for name in environment_updates}
    os.environ.update(environment_updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if inserted:
            try:
                sys.path.remove(root_string)
            except ValueError:
                pass


def _seed_graspgenx(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def _default_inference_provider(
    *,
    point_cloud_object_centered: np.ndarray,
    sweep_volume: Mapping[str, Any],
    graspgenx_root: Path,
    checkpoint_root: Path,
    seed: int,
    num_grasps: int,
) -> tuple[Any, Any]:
    _validate_default_runtime_paths(graspgenx_root, checkpoint_root)
    with _runtime_import_context(graspgenx_root, checkpoint_root):
        try:
            import torch
            from graspgenx.grasp_server import GraspGenXSampler
            from graspgenx.utils.checkpoint_io import load_model_cfg
        except ImportError as exc:
            raise GraspGenXCandidateError(
                "the requested GraspGenX checkout and its Python dependencies "
                "must be installed before GPU inference"
            ) from exc

        imported_source = Path(
            sys.modules[GraspGenXSampler.__module__].__file__
        ).resolve()
        if graspgenx_root not in imported_source.parents:
            raise GraspGenXCandidateError(
                "GraspGenX was already imported from a different checkout: "
                f"{imported_source}"
            )
        if not torch.cuda.is_available():
            raise GraspGenXCandidateError(
                "GraspGenX candidate inference requires a CUDA-capable torch runtime"
            )

        _seed_graspgenx(seed, torch)
        config = load_model_cfg(
            str(checkpoint_root / "gen"),
            str(checkpoint_root / "dis"),
        )
        sampler = GraspGenXSampler.from_sweep_volume(config, dict(sweep_volume))
        return GraspGenXSampler.run_inference(
            point_cloud_object_centered,
            sampler,
            grasp_threshold=-1.0,
            num_grasps=num_grasps,
            topk_num_grasps=num_grasps,
            min_grasps=1,
            max_tries=1,
            remove_outliers=False,
        )


def _provider_record(provider: InferenceProvider) -> dict[str, Any]:
    module = getattr(provider, "__module__", provider.__class__.__module__)
    name = getattr(provider, "__qualname__", provider.__class__.__qualname__)
    result: dict[str, Any] = {"callable": f"{module}.{name}"}
    version = getattr(provider, "__version__", None)
    if version is not None:
        result["version"] = str(version)
    return result


def provenance_path_for(npz_path: str | Path) -> Path:
    """Return the default sibling JSON path for an NPZ candidate artifact."""

    return Path(npz_path).with_suffix(".json")


def _write_temp_npz(
    directory: Path,
    *,
    stem: str,
    transforms: np.ndarray,
    scores: np.ndarray,
) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{stem}.",
        suffix=".tmp",
        dir=directory,
        delete=False,
    )
    path = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(
                handle,
                object_to_gripper_base=transforms,
                confidence=scores,
            )
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _write_temp_json(directory: Path, *, stem: str, payload: Mapping[str, Any]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{stem}.",
        suffix=".tmp",
        dir=directory,
        encoding="utf-8",
        delete=False,
    )
    path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _target_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _assert_targets_available(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if _target_exists(path)]
    if existing:
        raise ArtifactExistsError(
            "refusing to overwrite existing artifact(s): " + ", ".join(existing)
        )


def _unlink_published_link(target: Path, source: Path) -> None:
    """Remove ``target`` only while it still aliases our temporary inode."""

    try:
        target_stat = target.stat()
        source_stat = source.stat()
    except FileNotFoundError:
        return
    if (target_stat.st_dev, target_stat.st_ino) == (
        source_stat.st_dev,
        source_stat.st_ino,
    ):
        target.unlink(missing_ok=True)


def _commit_pair(
    npz_temp: Path,
    json_temp: Path,
    *,
    npz_path: Path,
    json_path: Path,
    overwrite: bool,
) -> None:
    if overwrite:
        os.replace(npz_temp, npz_path)
        try:
            os.replace(json_temp, json_path)
        except Exception:
            # The NPZ remains valid even if an unusual filesystem failure
            # interrupts the explicitly destructive overwrite operation.
            raise
        return

    _assert_targets_available((npz_path, json_path))
    npz_linked = False
    try:
        # Hard links provide an atomic "publish only if absent" primitive.
        # Both temporary files live on the same filesystem as their targets.
        os.link(npz_temp, npz_path)
        npz_linked = True
        os.link(json_temp, json_path)
    except FileExistsError as exc:
        if npz_linked:
            _unlink_published_link(npz_path, npz_temp)
        raise ArtifactExistsError(
            "an output artifact appeared while candidates were being generated"
        ) from exc
    except Exception:
        if npz_linked:
            _unlink_published_link(npz_path, npz_temp)
        raise
    finally:
        npz_temp.unlink(missing_ok=True)
        json_temp.unlink(missing_ok=True)


def generate_graspgenx_candidates(
    mesh_path: str | Path,
    output_npz: str | Path,
    *,
    extents_open: Sequence[float],
    offset_open: Sequence[float],
    extents_mid: Sequence[float],
    offset_mid: Sequence[float],
    fingertip_depth: float,
    gripper_type: str | int,
    gripper_name: str,
    graspgenx_root: str | Path,
    checkpoint_root: str | Path,
    seed: int = 0,
    num_grasps: int = 400,
    top_k: int = 100,
    num_sample_points: int = 3500,
    provenance_output: str | Path | None = None,
    overwrite: bool = False,
    provider: InferenceProvider | None = None,
) -> CandidateArtifact:
    """Generate, validate, rank, and atomically save GraspGenX candidates.

    ``mesh_path`` must already be in metric scale.  The saved transforms are
    gripper-base poses expressed in that mesh's local object frame.  By
    default neither the NPZ nor its sibling JSON may already exist.
    """

    mesh_source = Path(mesh_path).expanduser().resolve()
    output = Path(output_npz).expanduser().resolve()
    provenance_path = (
        Path(provenance_output).expanduser().resolve()
        if provenance_output is not None
        else provenance_path_for(output)
    )
    gx_root = Path(graspgenx_root).expanduser().resolve()
    ckpt_root = Path(checkpoint_root).expanduser().resolve()

    if output.suffix.lower() != ".npz":
        raise CandidateValidationError("output_npz must have a .npz suffix")
    if output == provenance_path:
        raise CandidateValidationError("NPZ and provenance outputs must be different")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise CandidateValidationError("seed must be an integer")
    seed = int(seed)
    if seed < 0 or seed > np.iinfo(np.uint32).max:
        raise CandidateValidationError("seed must lie in [0, 2**32 - 1]")
    if int(num_grasps) != num_grasps or num_grasps <= 0:
        raise CandidateValidationError("num_grasps must be a positive integer")
    if int(top_k) != top_k or top_k <= 0:
        raise CandidateValidationError("top_k must be a positive integer")
    if top_k > num_grasps:
        raise CandidateValidationError("top_k must not exceed num_grasps")
    if int(num_sample_points) != num_sample_points or num_sample_points < 3:
        raise CandidateValidationError("num_sample_points must be an integer >= 3")
    num_grasps = int(num_grasps)
    top_k = int(top_k)
    num_sample_points = int(num_sample_points)

    sweep = SweepVolume.create(
        extents_open=extents_open,
        offset_open=offset_open,
        extents_mid=extents_mid,
        offset_mid=offset_mid,
        fingertip_depth=fingertip_depth,
        gripper_type=gripper_type,
        gripper_name=gripper_name,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        # Fail before mesh loading or expensive GPU inference.
        _assert_targets_available((output, provenance_path))

    mesh, mesh_details = _load_metric_mesh(mesh_source)
    points_object = _sample_surface_deterministically(
        mesh, count=num_sample_points, seed=seed
    )
    point_mean = points_object.mean(axis=0, dtype=np.float64)
    points_centered = (points_object - point_mean).astype(np.float32)
    if not np.allclose(
        points_centered.mean(axis=0, dtype=np.float64),
        0.0,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise CandidateValidationError("failed to center the sampled object cloud")

    inference_provider = provider or _default_inference_provider
    random.seed(seed)
    np.random.seed(seed)
    raw_transforms, raw_scores = inference_provider(
        point_cloud_object_centered=points_centered.copy(),
        sweep_volume=sweep.provider_dict(),
        graspgenx_root=gx_root,
        checkpoint_root=ckpt_root,
        seed=seed,
        num_grasps=num_grasps,
    )
    centered_transforms, scores = validate_candidates(raw_transforms, raw_scores)

    order = np.argsort(-scores, kind="stable")[:top_k]
    centered_transforms = centered_transforms[order]
    scores = scores[order]

    uncenter = np.eye(4, dtype=np.float64)
    uncenter[:3, 3] = point_mean
    object_transforms = np.einsum(
        "ij,njk->nik", uncenter, centered_transforms.astype(np.float64)
    )
    object_transforms, scores = validate_candidates(object_transforms, scores)

    adapter_path = Path(__file__).resolve()
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "adapter": {
            "version": ADAPTER_VERSION,
            "source": _file_record(adapter_path),
        },
        "graspgenx": {
            **_git_record(gx_root),
            "declared_version": _declared_graspgenx_version(gx_root),
            "provider": _provider_record(inference_provider),
        },
        "checkpoints": _checkpoint_provenance(ckpt_root),
        "input": {
            "mesh": _file_record(mesh_source),
            "mesh_details": mesh_details,
        },
        "sampling": {
            "algorithm": "area_weighted_triangles_pcg64/v1",
            "seed": seed,
            "num_surface_points": num_sample_points,
            "sampled_points_object_sha256": _array_sha256(
                points_object.astype("<f8", copy=False)
            ),
            "centered_points_sha256": _array_sha256(
                points_centered.astype("<f4", copy=False)
            ),
            "sample_mean_object_m": point_mean.tolist(),
        },
        "gripper": sweep.provenance_dict(),
        "inference": {
            "requested_num_grasps": num_grasps,
            "requested_top_k": top_k,
            "returned_by_provider": int(len(raw_scores)),
            "saved_candidates": int(len(scores)),
            "seed": seed,
        },
        "contract": {
            "npz_keys": ["object_to_gripper_base", "confidence"],
            "pose_convention": (
                "T_object_gripper_base; maps gripper-base coordinates into "
                "object-mesh coordinates"
            ),
            "pose_dtype": "float32",
            "confidence_dtype": "float32",
            "confidence_order": "descending",
            "rigid_transform_atol": RIGID_ATOL,
        },
    }

    npz_temp: Path | None = None
    json_temp: Path | None = None
    try:
        npz_temp = _write_temp_npz(
            output.parent,
            stem=output.name,
            transforms=object_transforms,
            scores=scores,
        )
        provenance["output"] = {
            "npz": {
                "path": str(output),
                "bytes": npz_temp.stat().st_size,
                "sha256": _sha256_file(npz_temp),
            },
            "provenance_json": {"path": str(provenance_path)},
        }
        json_temp = _write_temp_json(
            provenance_path.parent,
            stem=provenance_path.name,
            payload=provenance,
        )
        _commit_pair(
            npz_temp,
            json_temp,
            npz_path=output,
            json_path=provenance_path,
            overwrite=overwrite,
        )
        npz_temp = None
        json_temp = None
    finally:
        if npz_temp is not None:
            npz_temp.unlink(missing_ok=True)
        if json_temp is not None:
            json_temp.unlink(missing_ok=True)

    return CandidateArtifact(
        object_to_gripper_base=object_transforms,
        confidence=scores,
        npz_path=output,
        provenance_path=provenance_path,
        provenance=provenance,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic, sweep-volume-conditioned GraspGenX "
            "candidates from a metric OBJ, PLY, or GLB object mesh."
        )
    )
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance-output", type=Path)
    parser.add_argument("--graspgenx-root", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        type=Path,
        help="Released checkpoint directory containing gen/ and dis/",
    )
    parser.add_argument("--gripper-name", required=True)
    parser.add_argument(
        "--gripper-type",
        default="parallel_2f",
        choices=sorted(GRIPPER_TYPE_IDS),
    )
    parser.add_argument("--extents-open", nargs=3, type=float, required=True)
    parser.add_argument("--offset-open", nargs=3, type=float, required=True)
    parser.add_argument("--extents-mid", nargs=3, type=float, required=True)
    parser.add_argument("--offset-mid", nargs=3, type=float, required=True)
    parser.add_argument("--fingertip-depth", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-grasps", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--num-sample-points", type=int, default=3500)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    artifact = generate_graspgenx_candidates(
        args.mesh,
        args.output,
        extents_open=args.extents_open,
        offset_open=args.offset_open,
        extents_mid=args.extents_mid,
        offset_mid=args.offset_mid,
        fingertip_depth=args.fingertip_depth,
        gripper_type=args.gripper_type,
        gripper_name=args.gripper_name,
        graspgenx_root=args.graspgenx_root,
        checkpoint_root=args.checkpoint_root,
        seed=args.seed,
        num_grasps=args.num_grasps,
        top_k=args.top_k,
        num_sample_points=args.num_sample_points,
        provenance_output=args.provenance_output,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "candidate_count": len(artifact.confidence),
                "npz": str(artifact.npz_path),
                "provenance": str(artifact.provenance_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
