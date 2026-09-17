from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import av
import h5py
import joblib
import numpy as np
import trimesh

from behave_data.video_reader import probe_video_frame_count
from lib_mhr import stamp_object_pose_frame_metadata
from prep.mhr_depth_backend import MOGE2_MODEL_ID, MOGE2_MODEL_REVISION, MOGE2_SOURCE_COMMIT
from prep.mhr_export_utils import MHR_CAMERA_NAMES
from prep.mhr_rgb_h5 import encode_rgb_jpeg, set_rgb_jpeg_metadata
from prep.mhr_wild_depth import normalize_wild_depth_backend
from tools.pipeline_timing import PipelineTimer


WILD_EXPORT_SCHEMA = "cari4d.mhr_wild_export.v2"


def _sequence_name(video: Path) -> str:
    suffix = ".0.color.mp4"
    if not video.name.endswith(suffix):
        raise ValueError(f"Wild RGB video must end with {suffix}: {video}")
    return video.name[:-len(suffix)]


def _intrinsics_payload(path: Path) -> dict[str, Any]:
    value = joblib.load(path)
    required = ("fx", "fy", "cx", "cy")
    if not isinstance(value, dict) or any(key not in value for key in required):
        raise ValueError(f"Wild intrinsics file must contain {required}: {path}")
    return value


def _intrinsics_from_payload(value: dict[str, Any], path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    if "H" in value and int(value["H"]) != height or "W" in value and int(value["W"]) != width:
        raise ValueError(f"Wild intrinsics dimensions differ from RGB: intrinsics={(value.get('H'), value.get('W'))}, RGB={(height, width)}")
    K = np.array([[value["fx"], 0.0, value["cx"]], [0.0, value["fy"], value["cy"]], [0.0, 0.0, 1.0]], dtype=np.float32)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"Wild intrinsics are invalid: {K}")
    return K


def _intrinsics(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    return _intrinsics_from_payload(_intrinsics_payload(path), path, image_shape)


def _mask_key(sequence: str, frame_name: str, kind: str) -> str:
    suffix = "person_mask.png" if kind == "human" else "obj_rend_mask.png"
    return f"{sequence}/{frame_name}-k0.{suffix}"


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _validate_existing_wild_export(metadata: dict[str, Any], sequence: str, source_identities: dict[str, dict[str, Any]], depth_backend: str | None, marker: Path) -> None:
    if metadata.get("schema") != WILD_EXPORT_SCHEMA or metadata.get("sequence") != sequence:
        raise ValueError(f"Existing wild export has incompatible identity: {marker}")
    mismatches = [key for key, expected in source_identities.items() if metadata.get(key) != expected]
    if metadata.get("depth_backend") != depth_backend:
        mismatches.append("depth_backend")
    expected_depth_identity = {"depth_model_id": MOGE2_MODEL_ID, "depth_model_revision": MOGE2_MODEL_REVISION, "depth_source_commit": MOGE2_SOURCE_COMMIT}
    mismatches.extend(key for key, expected in expected_depth_identity.items() if metadata.get(key) != expected)
    if mismatches:
        raise ValueError(f"Existing wild export differs in {sorted(set(mismatches))}; use --redo: {marker}")


def _write_object_template(source: Path, output: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(source, force="scene", process=False)
    if not isinstance(loaded, (trimesh.Scene, trimesh.Trimesh)):
        raise TypeError(f"Wild object source did not load as a triangle mesh: {source}")
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    meshes = [mesh for mesh in scene.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
    if not meshes or any(len(mesh.vertices) == 0 or len(mesh.faces) == 0 for mesh in meshes):
        raise ValueError(f"Wild object source contains no usable triangle geometry: {source}")
    combined = scene.dump(concatenate=True)
    if not isinstance(combined, trimesh.Trimesh) or len(combined.vertices) == 0:
        raise ValueError(f"Wild object source could not be combined for oriented-bounds alignment: {source}")
    source_to_aligned, extents = trimesh.bounds.oriented_bounds(combined)
    source_to_aligned = np.asarray(source_to_aligned, dtype=np.float32)
    extents = np.asarray(extents, dtype=np.float32)
    scene.apply_transform(source_to_aligned)
    payload = scene.export(file_type="glb")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"Wild object GLB export is empty: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    check = trimesh.load(temporary, file_type="glb", force="scene", process=False)
    if not isinstance(check, trimesh.Scene) or not any(isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) for mesh in check.geometry.values()):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Wild object GLB round-trip failed: {source}")
    bounds = np.asarray(check.bounds, dtype=np.float32)
    center = bounds.mean(axis=0)
    tolerance = max(float(np.max(extents)) * 1e-5, 1e-6)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.max(np.abs(center)) > tolerance:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Wild aligned object mesh is not centered at its oriented-bounds origin: center={center}, tolerance={tolerance}")
    os.replace(temporary, output)
    return source_to_aligned, extents


def prepare_mhr_wild_export(video: str | Path, mask_h5: str | Path, object_mesh: str | Path, intrinsics_file: str | Path, output_root: str | Path, *, redo: bool = False) -> Path:
    video, mask_h5, object_mesh, intrinsics_file = map(lambda value: Path(value).resolve(), (video, mask_h5, object_mesh, intrinsics_file))
    for path in (video, mask_h5, object_mesh, intrinsics_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    intrinsics_payload = _intrinsics_payload(intrinsics_file)
    depth_backend = normalize_wild_depth_backend(intrinsics_payload.get("depth_backend", ""))
    expected_depth_identity = {"model_id": MOGE2_MODEL_ID, "model_revision": MOGE2_MODEL_REVISION, "source_commit": MOGE2_SOURCE_COMMIT}
    mismatches = [key for key, expected in expected_depth_identity.items() if intrinsics_payload.get(key) != expected]
    if mismatches:
        raise ValueError(f"Wild intrinsics differ from the pinned MoGe 2 identity in {mismatches}: {intrinsics_file}")
    source_identities = {"source_video": _file_identity(video), "source_masks": _file_identity(mask_h5), "source_object_mesh": _file_identity(object_mesh), "source_intrinsics": _file_identity(intrinsics_file)}
    sequence = _sequence_name(video)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / sequence
    marker = output / "wild_export.json"
    if marker.is_file() and not redo:
        metadata = json.loads(marker.read_text())
        _validate_existing_wild_export(metadata, sequence, source_identities, depth_backend, marker)
        return output
    if output.exists() and any(output.iterdir()) and not redo:
        raise FileExistsError(f"Wild export exists without a valid completion marker: {output}")
    output.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video))
    stream = container.streams.video[0]
    expected_frames = int(stream.frames)
    if expected_frames <= 0:
        expected_frames = probe_video_frame_count(video)
    height, width = int(stream.codec_context.height), int(stream.codec_context.width)
    K = _intrinsics_from_payload(intrinsics_payload, intrinsics_file, (height, width))
    camera = MHR_CAMERA_NAMES[0]
    rgb_path = output / "images" / f"{camera}.h5"
    human_path = output / "human_masks" / f"{camera}.h5"
    object_path = output / "object_masks" / f"{camera}.h5"
    for path in (rgb_path, human_path, object_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = [path.with_name(f".{path.name}.{os.getpid()}.tmp") for path in (rgb_path, human_path, object_path)]
    for path in temporary_paths:
        path.unlink(missing_ok=True)

    with h5py.File(mask_h5, "r") as source_masks, h5py.File(temporary_paths[0], "w") as rgb_handle, h5py.File(temporary_paths[1], "w") as human_handle, h5py.File(temporary_paths[2], "w") as object_handle:
        rgb_dataset = rgb_handle.create_dataset("frames", shape=(expected_frames,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
        human_dataset = human_handle.create_dataset("frames", shape=(expected_frames, height, width), dtype=np.uint8, chunks=(1, height, width), compression="lzf", shuffle=True)
        object_dataset = object_handle.create_dataset("frames", shape=(expected_frames, height, width), dtype=np.uint8, chunks=(1, height, width), compression="lzf", shuffle=True)
        set_rgb_jpeg_metadata(rgb_dataset, (height, width, 3))
        for handle in (rgb_handle, human_handle, object_handle):
            handle.attrs["complete"] = False
            handle.attrs["sequence"] = sequence
            handle.attrs["camera_id"] = 0
        observed = 0
        for index, frame in enumerate(container.decode(stream)):
            if index >= expected_frames:
                raise ValueError(f"Wild video decoded more frames than declared: {expected_frames}")
            rgb = frame.to_ndarray(format="rgb24")
            frame_name = f"{index:06d}"
            human_key = _mask_key(sequence, frame_name, "human")
            object_key = _mask_key(sequence, frame_name, "object")
            if human_key not in source_masks or object_key not in source_masks:
                raise KeyError(f"Wild mask H5 is missing frame {frame_name}: {mask_h5}")
            human = np.asarray(source_masks[human_key][()], dtype=bool)
            obj = np.asarray(source_masks[object_key][()], dtype=bool)
            if rgb.shape != (height, width, 3) or human.shape != (height, width) or obj.shape != (height, width):
                raise ValueError(f"Wild frame shape mismatch at {frame_name}: RGB={rgb.shape}, human={human.shape}, object={obj.shape}")
            rgb_dataset[index] = encode_rgb_jpeg(rgb)
            human_dataset[index] = human.astype(np.uint8) * 255
            object_dataset[index] = obj.astype(np.uint8) * 255
            observed += 1
        if observed != expected_frames:
            raise ValueError(f"Wild video frame count mismatch: declared={expected_frames}, decoded={observed}")
        for handle in (rgb_handle, human_handle, object_handle):
            handle.attrs.modify("complete", True)
            handle.flush()
    container.close()
    for temporary, final in zip(temporary_paths, (rgb_path, human_path, object_path)):
        os.replace(temporary, final)

    edex_camera = {"intrinsics": {"focal": [float(K[0, 0]), float(K[1, 1])], "principal": [float(K[0, 2]), float(K[1, 2])]}, "transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]}
    (output / "edex").write_text(json.dumps([{"cameras": [edex_camera]}], indent=2) + "\n")
    aligned_object_path = output / "object_mesh" / "output_aligned.glb"
    source_to_aligned, aligned_extents = _write_object_template(object_mesh, aligned_object_path)
    metadata = {"schema": WILD_EXPORT_SCHEMA, "sequence": sequence, "camera_id": 0, "frame_count": expected_frames, "height": height, "width": width, "intrinsics": K.tolist(), "depth_backend": depth_backend, "depth_camera_policy": intrinsics_payload.get("camera_policy"), "depth_model_id": intrinsics_payload.get("model_id"), "depth_model_revision": intrinsics_payload.get("model_revision"), "depth_source_commit": intrinsics_payload.get("source_commit"), **source_identities, "object_mesh_file": str(aligned_object_path), "source_object_mesh_to_aligned_transform": source_to_aligned.tolist(), "aligned_object_extents_m": aligned_extents.tolist()}
    metadata = stamp_object_pose_frame_metadata(metadata)
    temporary_marker = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary_marker.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_marker, marker)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one-camera in-the-wild RGB, masks, intrinsics, and object mesh as a flat MHR export.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--mask-h5", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--intrinsics-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    with PipelineTimer("wild_export_preparation"):
        print(prepare_mhr_wild_export(args.video, args.mask_h5, args.object_mesh, args.intrinsics_file, args.output_root, redo=args.redo))


if __name__ == "__main__":
    main()
