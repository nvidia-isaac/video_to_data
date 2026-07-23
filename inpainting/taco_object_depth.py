"""Render strict TACO tool/target mask and metric depth artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid

import numpy as np
import pyarrow.parquet as pq

from .contracts import (
    ContractError,
    TACO_OBJECT_RENDER_SCHEMA,
    VideoGeometry,
    validate_depth_array,
    validate_mask_array,
)
from .robot_renderer.transforms import CV_TO_OPENGL, pose_matrix
from .robot_renderer.provenance import file_record, verify_file_record
from .taco_camera import load_taco_camera
from .video_io import probe_video


# Backwards-compatible public alias; the lightweight contract lives in
# contracts.py so generic compositing does not import PyArrow.
OBJECT_RENDER_SCHEMA = TACO_OBJECT_RENDER_SCHEMA
OBJECT_RENDER_PROVENANCE_SCHEMA = "v2d.inpainting.taco-object-render-provenance/v1"
SOURCE_INPUT_NAMES = (
    "source_parquet",
    "source_video",
    "intrinsic",
    "world_to_camera",
)
OBJECT_RENDER_IMPLEMENTATION_FILES = (
    "inpainting/__init__.py",
    "inpainting/contracts.py",
    "inpainting/robot_renderer/__init__.py",
    "inpainting/robot_renderer/container_runner.py",
    "inpainting/robot_renderer/inputs.py",
    "inpainting/robot_renderer/provenance.py",
    "inpainting/robot_renderer/transforms.py",
    "inpainting/taco_camera.py",
    "inpainting/taco_object_depth.py",
    "inpainting/taco_object_depth_container.py",
    "inpainting/video_io.py",
)
REQUIRED_COLUMNS = (
    "fps",
    "object_body_names",
    "object_mesh_paths",
    "object_body_position",
    "object_body_wxyz",
)
TACO_MESH_SCALE = 0.01


class TacoObjectRenderError(RuntimeError):
    """Raised when a TACO object render cannot be produced safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _restore_host_ownership(paths: list[Path]) -> None:
    uid_text = os.environ.get("V2D_RENDER_HOST_UID")
    gid_text = os.environ.get("V2D_RENDER_HOST_GID")
    if uid_text is None and gid_text is None:
        return
    if uid_text is None or gid_text is None:
        raise TacoObjectRenderError(
            "host ownership requires both V2D_RENDER_HOST_UID and V2D_RENDER_HOST_GID"
        )
    try:
        uid, gid = int(uid_text), int(gid_text)
    except ValueError as exc:
        raise TacoObjectRenderError("host UID/GID must be integers") from exc
    if uid < 0 or gid < 0:
        raise TacoObjectRenderError("host UID/GID must be non-negative")
    for path in paths:
        if path.exists():
            os.chown(path, uid, gid)


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(128).startswith(
            b"version https://git-lfs.github.com/spec/v1"
        )


def _resolve_meshes(mesh_root: Path, stored_paths: list[str]) -> tuple[Path, ...]:
    root = mesh_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"TACO mesh root: {root}")
    resolved: list[Path] = []
    for stored in stored_paths:
        if not isinstance(stored, str) or not stored:
            raise ContractError("object_mesh_paths must contain non-empty strings")
        basename = Path(stored).name
        if not basename.endswith("_cm.obj"):
            raise ContractError(
                f"TACO object mesh must use the explicit _cm.obj unit suffix: {stored!r}"
            )
        candidate = (root / basename).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ContractError(
                f"Object mesh escapes explicit mesh root: {stored!r}"
            ) from exc
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Object mesh {stored!r} rerooted to missing {candidate}"
            )
        if candidate.stat().st_size <= 0:
            raise ContractError(f"Object mesh is empty: {candidate}")
        if _is_lfs_pointer(candidate):
            raise ContractError(f"Object mesh is a Git LFS pointer: {candidate}")
        resolved.append(candidate)
    if len(set(resolved)) != len(resolved):
        raise ContractError("TACO tool and target must resolve to distinct meshes")
    return tuple(resolved)


@dataclass(frozen=True)
class TacoObjectInputs:
    sequence_id: str
    source_parquet: Path
    source_video: Path
    intrinsic_path: Path
    world_to_camera_path: Path
    mesh_root: Path
    geometry: VideoGeometry
    body_names: tuple[str, ...]
    stored_mesh_paths: tuple[str, ...]
    mesh_paths: tuple[Path, ...]
    positions_world: np.ndarray
    quaternions_wxyz: np.ndarray
    intrinsic: np.ndarray
    world_to_camera: np.ndarray

    @property
    def frame_count(self) -> int:
        return self.geometry.frame_count

    @property
    def body_count(self) -> int:
        return len(self.body_names)


def _validated_object_arrays(
    row: dict[str, Any], geometry: VideoGeometry, *, sequence_id: str
) -> tuple[str, tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ContractError("Explicit TACO sequence_id must be a non-empty string")
    fps = row.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ContractError("TACO fps must be numeric")
    if not np.isfinite(float(fps)) or abs(float(fps) - geometry.fps) > 1e-3:
        raise ContractError(
            f"TACO parquet FPS {fps!r} does not match video FPS {geometry.fps}"
        )

    body_names_value = row.get("object_body_names")
    if not isinstance(body_names_value, list) or any(
        not isinstance(name, str) or not name for name in body_names_value
    ):
        raise ContractError("object_body_names must be a list of non-empty strings")
    body_names = tuple(body_names_value)
    if body_names != ("tool", "target"):
        raise ContractError(
            f"TACO object body order must be ('tool', 'target'), got {body_names}"
        )

    stored_value = row.get("object_mesh_paths")
    if not isinstance(stored_value, list) or len(stored_value) != len(body_names):
        raise ContractError("object_mesh_paths must contain one path per object body")
    stored_paths = tuple(stored_value)

    positions = np.asarray(row.get("object_body_position"), dtype=np.float64)
    quaternions = np.asarray(row.get("object_body_wxyz"), dtype=np.float64)
    expected_positions = (geometry.frame_count, len(body_names), 3)
    expected_quaternions = (geometry.frame_count, len(body_names), 4)
    if positions.shape != expected_positions:
        raise ContractError(
            f"object_body_position must have shape {expected_positions}, got {positions.shape}"
        )
    if quaternions.shape != expected_quaternions:
        raise ContractError(
            f"object_body_wxyz must have shape {expected_quaternions}, got {quaternions.shape}"
        )
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ContractError("TACO object poses contain non-finite values")
    norms = np.linalg.norm(quaternions, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
        bad = np.unravel_index(int(np.argmax(np.abs(norms - 1.0))), norms.shape)
        raise ContractError(
            f"object_body_wxyz{bad} is not unit length (norm {norms[bad]:.8g})"
        )
    return sequence_id, body_names, stored_paths, positions, quaternions


def load_taco_object_inputs(
    *,
    sequence_id: str,
    parquet: str | Path,
    source_video: str | Path,
    intrinsic_path: str | Path,
    world_to_camera_path: str | Path,
    mesh_root: str | Path,
) -> TacoObjectInputs:
    parquet = Path(parquet).resolve()
    source_video = Path(source_video).resolve()
    intrinsic_path = Path(intrinsic_path).resolve()
    world_to_camera_path = Path(world_to_camera_path).resolve()
    mesh_root = Path(mesh_root).resolve()
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    geometry = probe_video(source_video)
    # Read the physical file directly. The container bind-mounts one parquet at
    # /inputs/motion.parquet, which intentionally removes its Hive partition
    # directory and therefore its virtual sequence_id column.
    table = pq.ParquetFile(parquet).read(columns=list(REQUIRED_COLUMNS))
    if table.num_rows != 1:
        raise ContractError(
            f"Expected exactly one TACO sequence row, got {table.num_rows}"
        )
    row = table.to_pylist()[0]
    sequence_id, names, stored, positions, quaternions = _validated_object_arrays(
        row, geometry, sequence_id=sequence_id
    )
    meshes = _resolve_meshes(mesh_root, list(stored))
    camera = load_taco_camera(
        intrinsic_path,
        world_to_camera_path,
        geometry.frame_count,
        geometry.width,
        geometry.height,
    )
    return TacoObjectInputs(
        sequence_id=sequence_id,
        source_parquet=parquet,
        source_video=source_video,
        intrinsic_path=intrinsic_path,
        world_to_camera_path=world_to_camera_path,
        mesh_root=mesh_root,
        geometry=geometry,
        body_names=names,
        stored_mesh_paths=stored,
        mesh_paths=meshes,
        positions_world=positions,
        quaternions_wxyz=quaternions,
        intrinsic=camera.intrinsic,
        world_to_camera=camera.world_to_camera,
    )


def _load_render_mesh(path: Path, trimesh_module: Any) -> tuple[Any, dict[str, Any]]:
    mesh = trimesh_module.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh_module.Trimesh):
        raise TacoObjectRenderError(f"Expected one triangle mesh in {path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise TacoObjectRenderError(f"Object mesh has invalid vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] < 1:
        raise TacoObjectRenderError(f"Object mesh has invalid faces: {path}")
    if not np.isfinite(vertices).all():
        raise TacoObjectRenderError(f"Object mesh has non-finite vertices: {path}")
    mesh.apply_scale(TACO_MESH_SCALE)
    scaled_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    radius = float(
        np.max(np.linalg.norm(scaled_vertices - scaled_vertices.mean(0), axis=1))
    )
    if not 0.005 <= radius <= 1.0:
        raise TacoObjectRenderError(
            f"Scaled TACO mesh radius {radius:.6g} m is implausible for {path}"
        )
    return mesh, {
        "path": str(path),
        "stored_units": "centimeters",
        "scale_to_meters": TACO_MESH_SCALE,
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "radius_m": radius,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_input_paths(inputs: TacoObjectInputs) -> dict[str, Path]:
    return {
        "source_parquet": inputs.source_parquet,
        "source_video": inputs.source_video,
        "intrinsic": inputs.intrinsic_path,
        "world_to_camera": inputs.world_to_camera_path,
    }


def object_render_source_records(
    repository_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fingerprint every repository source executed by the object-depth stage."""

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    if not root.is_dir():
        raise FileNotFoundError(f"object-render repository root: {root}")
    return [
        file_record(root / relative, recorded_path=relative)
        for relative in OBJECT_RENDER_IMPLEMENTATION_FILES
    ]


def _normalize_recorded_source_paths(
    inputs: TacoObjectInputs,
    recorded_source_paths: dict[str, str] | None,
) -> dict[str, str]:
    actual = _source_input_paths(inputs)
    if recorded_source_paths is None:
        return {name: str(path.resolve()) for name, path in actual.items()}
    if set(recorded_source_paths) != set(SOURCE_INPUT_NAMES):
        raise TacoObjectRenderError(
            "recorded source paths must contain exactly "
            f"{SOURCE_INPUT_NAMES}, got {tuple(sorted(recorded_source_paths))}"
        )
    normalized: dict[str, str] = {}
    for name in SOURCE_INPUT_NAMES:
        value = recorded_source_paths[name]
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise TacoObjectRenderError(
                f"recorded source path {name} must be a non-empty absolute path"
            )
        normalized[name] = value
    return normalized


def _container_identity() -> tuple[str, str]:
    requested_image = os.environ.get("V2D_RENDER_CONTAINER_IMAGE")
    image_id = os.environ.get("V2D_RENDER_CONTAINER_IMAGE_ID")
    if not requested_image:
        raise TacoObjectRenderError(
            "V2D_RENDER_CONTAINER_IMAGE must record the requested image reference"
        )
    if image_id is None or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise TacoObjectRenderError(
            "V2D_RENDER_CONTAINER_IMAGE_ID must be an immutable sha256 image ID"
        )
    return requested_image, image_id


def _source_input_records(
    inputs: TacoObjectInputs,
    recorded_source_paths: dict[str, str] | None,
) -> dict[str, dict[str, Any]]:
    source_paths = _source_input_paths(inputs)
    host_paths = _normalize_recorded_source_paths(inputs, recorded_source_paths)
    records: dict[str, dict[str, Any]] = {}
    for name, source in source_paths.items():
        record = file_record(source)
        records[name] = {
            "container_path": record.pop("path"),
            "host_path": host_paths[name],
            **record,
        }
    return records


def _metadata_base(
    inputs: TacoObjectInputs,
    output_dir: Path,
    run_id: str,
    *,
    recorded_source_paths: dict[str, str] | None,
) -> dict[str, Any]:
    requested_image, image_id = _container_identity()
    return {
        "schema_version": OBJECT_RENDER_SCHEMA,
        "state": "running",
        "run_id": run_id,
        "started_at": _utc_now(),
        "sequence_id": inputs.sequence_id,
        "container_image": requested_image,
        "container_image_id": image_id,
        "host_output_dir": os.environ.get("V2D_RENDER_HOST_OUTPUT_DIR"),
        "geometry": inputs.geometry.as_dict(),
        "source_parquet": str(inputs.source_parquet),
        "source_video": str(inputs.source_video),
        "intrinsic": str(inputs.intrinsic_path),
        "world_to_camera": str(inputs.world_to_camera_path),
        "mesh_root": str(inputs.mesh_root),
        "body_names": list(inputs.body_names),
        "stored_mesh_paths": list(inputs.stored_mesh_paths),
        "coordinate_conventions": {
            "camera": "OpenCV +x right, +y down, +z forward",
            "calibration": "T_camera_world (world-to-camera)",
            "object_pose": "T_world_object",
            "quaternion": "wxyz",
            "depth": "positive metric camera z; invalid +inf",
        },
        "artifacts": {
            "mask": str((output_dir / "object_mask.npy").resolve()),
            "depth": str((output_dir / "object_depth.npy").resolve()),
        },
        "provenance": {
            "schema_version": OBJECT_RENDER_PROVENANCE_SCHEMA,
            "hash_algorithm": "sha256",
            "inputs": _source_input_records(inputs, recorded_source_paths),
            "implementation_sources": object_render_source_records(),
        },
    }


def render_taco_object_depth(
    inputs: TacoObjectInputs,
    output_dir: str | Path,
    *,
    recorded_source_paths: dict[str, str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render the nearest tool/target surface depth for every source frame."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "mask": destination / "object_mask.npy",
        "depth": destination / "object_depth.npy",
        "metadata": destination / "object_render_metadata.json",
    }
    existing = [str(path) for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "object-render outputs already exist; pass --overwrite to replace: "
            + ", ".join(existing)
        )
    run_id = str(uuid.uuid4())
    temporary_paths = {
        "mask": destination / f".object_mask.{run_id}.partial.npy",
        "depth": destination / f".object_depth.{run_id}.partial.npy",
    }
    metadata = _metadata_base(
        inputs,
        destination,
        run_id,
        recorded_source_paths=recorded_source_paths,
    )
    renderer = None
    mask_memmap = None
    depth_memmap = None
    commit_started = False
    try:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        try:
            import pyrender
            import trimesh
        except ImportError as exc:
            raise TacoObjectRenderError(
                "pyrender and trimesh are required; use taco_object_depth_container"
            ) from exc

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        camera = pyrender.IntrinsicsCamera(
            fx=float(inputs.intrinsic[0, 0]),
            fy=float(inputs.intrinsic[1, 1]),
            cx=float(inputs.intrinsic[0, 2]),
            cy=float(inputs.intrinsic[1, 2]),
            znear=0.01,
            zfar=10.0,
        )
        scene.add(camera, pose=np.eye(4))
        nodes = []
        mesh_metadata = []
        for body_name, mesh_path in zip(
            inputs.body_names, inputs.mesh_paths, strict=True
        ):
            mesh, details = _load_render_mesh(mesh_path, trimesh)
            mesh_metadata.append({"body_name": body_name, **details})
            render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            nodes.append(scene.add(render_mesh, pose=np.eye(4), name=body_name))
        metadata["meshes"] = mesh_metadata

        renderer = pyrender.OffscreenRenderer(
            viewport_width=inputs.geometry.width,
            viewport_height=inputs.geometry.height,
        )
        shape = (
            inputs.geometry.frame_count,
            inputs.geometry.height,
            inputs.geometry.width,
        )
        mask_memmap = np.lib.format.open_memmap(
            temporary_paths["mask"], mode="w+", dtype=np.bool_, shape=shape
        )
        depth_memmap = np.lib.format.open_memmap(
            temporary_paths["depth"], mode="w+", dtype=np.float32, shape=shape
        )
        flags = pyrender.RenderFlags.DEPTH_ONLY | pyrender.RenderFlags.SKIP_CULL_FACES
        frame_pixel_counts: list[int] = []
        for frame_index in range(inputs.geometry.frame_count):
            world_to_gl = CV_TO_OPENGL @ inputs.world_to_camera[frame_index]
            for body_index, node in enumerate(nodes):
                world_from_object = pose_matrix(
                    inputs.positions_world[frame_index, body_index],
                    inputs.quaternions_wxyz[frame_index, body_index],
                )
                scene.set_pose(node, pose=world_to_gl @ world_from_object)
            raw_depth = np.asarray(
                renderer.render(scene, flags=flags), dtype=np.float32
            )
            if raw_depth.shape != shape[1:]:
                raise TacoObjectRenderError(
                    f"pyrender returned depth shape {raw_depth.shape}, expected {shape[1:]}"
                )
            frame_mask = np.isfinite(raw_depth) & (raw_depth > 0.0)
            frame_depth = raw_depth.copy()
            frame_depth[~frame_mask] = np.inf
            mask_memmap[frame_index] = frame_mask
            depth_memmap[frame_index] = frame_depth
            frame_pixel_counts.append(int(frame_mask.sum()))

        minimum_pixels = max(
            16,
            int(np.ceil(inputs.geometry.width * inputs.geometry.height * 1e-6)),
        )
        visible_frames = int(
            np.count_nonzero(np.asarray(frame_pixel_counts) >= minimum_pixels)
        )
        required_frames = max(1, int(np.ceil(inputs.geometry.frame_count * 0.10)))
        if visible_frames < required_frames:
            raise TacoObjectRenderError(
                "object render is blank or nearly blank; recheck mesh scale and calibration"
            )
        mask_memmap.flush()
        depth_memmap.flush()
        validate_mask_array(mask_memmap, inputs.geometry)
        validate_depth_array(
            depth_memmap, mask_memmap, inputs.geometry, name="Object depth"
        )
        for name, path in _source_input_paths(inputs).items():
            verify_file_record(
                path,
                metadata["provenance"]["inputs"][name],
                label=f"object render input {name}",
            )
        for mesh_path, mesh_record in zip(
            inputs.mesh_paths, metadata["meshes"], strict=True
        ):
            verify_file_record(
                mesh_path,
                mesh_record,
                label=f"object render mesh {mesh_path.name}",
            )
        if (
            metadata["provenance"]["implementation_sources"]
            != object_render_source_records()
        ):
            raise TacoObjectRenderError(
                "object-depth implementation sources changed during rendering"
            )
        del mask_memmap
        del depth_memmap
        mask_memmap = None
        depth_memmap = None
        renderer.delete()
        renderer = None

        metadata["state"] = "committing"
        metadata["committing_at"] = _utc_now()
        _write_json_atomic(final_paths["metadata"], metadata)
        commit_started = True
        temporary_paths["mask"].replace(final_paths["mask"])
        temporary_paths["depth"].replace(final_paths["depth"])

        final_mask = np.load(final_paths["mask"], mmap_mode="r")
        final_depth = np.load(final_paths["depth"], mmap_mode="r")
        validate_mask_array(final_mask, inputs.geometry)
        validate_depth_array(
            final_depth, final_mask, inputs.geometry, name="Object depth"
        )
        metadata["state"] = "complete"
        metadata["completed_at"] = _utc_now()
        metadata["render_statistics"] = {
            "object_pixel_count": int(sum(frame_pixel_counts)),
            "mean_object_pixels_per_frame": float(np.mean(frame_pixel_counts)),
            "min_object_pixels_per_frame": int(min(frame_pixel_counts)),
            "max_object_pixels_per_frame": int(max(frame_pixel_counts)),
            "visibility_pixel_threshold": minimum_pixels,
            "visible_frame_count": visible_frames,
            "required_visible_frame_count": required_frames,
        }
        metadata["artifact_bytes"] = {
            key: final_paths[key].stat().st_size for key in ("mask", "depth")
        }
        metadata["artifact_sha256"] = {
            key: _sha256(final_paths[key]) for key in ("mask", "depth")
        }
        _write_json_atomic(final_paths["metadata"], metadata)
        _restore_host_ownership(list(final_paths.values()))
        return metadata
    except Exception as exc:
        if mask_memmap is not None:
            mask_memmap.flush()
        if depth_memmap is not None:
            depth_memmap.flush()
        if renderer is not None:
            renderer.delete()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        metadata["state"] = "failed"
        metadata["failed_at"] = _utc_now()
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if commit_started or not final_paths["metadata"].exists():
            _write_json_atomic(final_paths["metadata"], metadata)
            _restore_host_ownership([final_paths["metadata"]])
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--world-to-camera", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-parquet-recorded-path")
    parser.add_argument("--source-video-recorded-path")
    parser.add_argument("--intrinsics-recorded-path")
    parser.add_argument("--world-to-camera-recorded-path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = load_taco_object_inputs(
        sequence_id=args.sequence_id,
        parquet=args.parquet,
        source_video=args.source_video,
        intrinsic_path=args.intrinsics,
        world_to_camera_path=args.world_to_camera,
        mesh_root=args.mesh_root,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "state": "validated",
                    "sequence_id": inputs.sequence_id,
                    "geometry": inputs.geometry.as_dict(),
                    "body_names": list(inputs.body_names),
                    "mesh_paths": [str(path) for path in inputs.mesh_paths],
                },
                indent=2,
            )
        )
        return 0
    recorded_values = {
        "source_parquet": args.source_parquet_recorded_path,
        "source_video": args.source_video_recorded_path,
        "intrinsic": args.intrinsics_recorded_path,
        "world_to_camera": args.world_to_camera_recorded_path,
    }
    provided = [value is not None for value in recorded_values.values()]
    if any(provided) and not all(provided):
        parser.error("all four recorded source paths must be supplied together")
    metadata = render_taco_object_depth(
        inputs,
        args.output_dir,
        recorded_source_paths=(recorded_values if all(provided) else None),
        overwrite=args.overwrite,
    )
    print(
        f"Rendered {inputs.frame_count} object-depth frames -> "
        f"{metadata['artifacts']['depth']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
