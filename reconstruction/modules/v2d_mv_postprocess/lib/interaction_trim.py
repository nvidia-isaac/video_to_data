"""Detect the human-object interaction interval and build a trimmed source view."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable

import av
import h5py
import numpy as np
import torch
import trimesh
import yaml

from v2d.common.hdf5_transcode import slice_h5_frames


TRIM_SCHEMA_V1 = "v2d.mv_hoi.interaction_trim.v1"
TRIM_SCHEMA = "v2d.mv_hoi.interaction_trim.v2"
SUPPORTED_TRIM_SCHEMAS = {TRIM_SCHEMA_V1, TRIM_SCHEMA}

MHR_PARAMETER_TEMPORAL_KEYS = {
    "global_rot",
    "body_pose_params",
    "hand_pose_params",
    "scale_params",
    "shape_params",
    "pred_cam_t",
    "pred_keypoints_3d",
    "pred_joint_coords",
    "pred_global_rots",
    "mhr_model_params",
}
MHR_MESH_TEMPORAL_KEYS = {"pred_vertices"}
MHR_MESH_STATIC_KEYS = {"faces"}
SOMA_REQUIRED_TEMPORAL_KEYS = {
    "poses",
    "transl",
    "identity_coeffs",
    "scale_params",
}
SOMA_TEMPORAL_KEYS = SOMA_REQUIRED_TEMPORAL_KEYS | {"bone_length_flexibles"}


@dataclass(frozen=True)
class InteractionTrimConfig:
    """Configuration for stable contact detection and interval retention."""

    enabled: bool = True
    distance_threshold_m: float = 0.10
    pre_contact_padding_seconds: float = 3.0
    post_contact_padding_seconds: float = 3.0
    window_frames: int = 7
    required_under_threshold_frames: int = 5
    no_contact_policy: str = "keep_full"

    def validate(self) -> None:
        if self.distance_threshold_m <= 0:
            raise ValueError("distance_threshold_m must be positive")
        if self.pre_contact_padding_seconds < 0:
            raise ValueError("pre_contact_padding_seconds must be nonnegative")
        if self.post_contact_padding_seconds < 0:
            raise ValueError("post_contact_padding_seconds must be nonnegative")
        if self.window_frames < 1:
            raise ValueError("window_frames must be positive")
        if not 1 <= self.required_under_threshold_frames <= self.window_frames:
            raise ValueError(
                "required_under_threshold_frames must be within window_frames"
            )
        if self.no_contact_policy != "keep_full":
            raise ValueError(
                f"Unsupported no-contact policy: {self.no_contact_policy}"
            )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def select_stable_contact_frame(
    nearest_distances_m: np.ndarray | list[float],
    *,
    distance_threshold_m: float = 0.10,
    window_frames: int = 7,
    required_under_threshold_frames: int = 5,
) -> int | None:
    """Return the first under-threshold frame in the earliest stable window."""
    distances = np.asarray(nearest_distances_m, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError("nearest_distances_m must be one-dimensional")
    if window_frames < 1 or not 1 <= required_under_threshold_frames <= window_frames:
        raise ValueError("Invalid stable-contact window")
    under = np.isfinite(distances) & (distances <= float(distance_threshold_m))
    for start in range(0, max(0, len(under) - window_frames + 1)):
        window = under[start : start + window_frames]
        if int(window.sum()) < required_under_threshold_frames:
            continue
        relative = int(np.flatnonzero(window)[0])
        return start + relative
    return None


def select_stable_contact_frames(
    nearest_distances_m: np.ndarray | list[float],
    *,
    distance_threshold_m: float = 0.10,
    window_frames: int = 7,
    required_under_threshold_frames: int = 5,
) -> tuple[int | None, int | None]:
    """Return first and last close frames belonging to stable contact windows."""
    distances = np.asarray(nearest_distances_m, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError("nearest_distances_m must be one-dimensional")
    first = select_stable_contact_frame(
        distances,
        distance_threshold_m=distance_threshold_m,
        window_frames=window_frames,
        required_under_threshold_frames=required_under_threshold_frames,
    )
    if first is None:
        return None, None
    reverse_first = select_stable_contact_frame(
        distances[::-1],
        distance_threshold_m=distance_threshold_m,
        window_frames=window_frames,
        required_under_threshold_frames=required_under_threshold_frames,
    )
    if reverse_first is None:  # Defensive: forward and reverse must agree.
        raise ValueError("Stable-contact detection is not symmetric")
    return first, len(distances) - 1 - reverse_first


def padded_export_start_frame(
    contact_frame: int | None,
    *,
    source_frame_count: int,
    fps: Fraction | int | float,
    pre_contact_padding_seconds: float = 3.0,
) -> int:
    """Return a valid prefix cut preserving the requested time before contact."""
    if source_frame_count <= 0:
        raise ValueError("source_frame_count must be positive")
    if contact_frame is None:
        return 0
    if not 0 <= int(contact_frame) < source_frame_count:
        raise ValueError("contact_frame is outside the source timeline")
    frame_rate = float(fps)
    if frame_rate <= 0 or pre_contact_padding_seconds < 0:
        raise ValueError("FPS and pre-contact padding must be valid")
    padding_frames = int(round(pre_contact_padding_seconds * frame_rate))
    return max(0, int(contact_frame) - padding_frames)


def padded_export_end_frame(
    contact_frame: int | None,
    *,
    source_frame_count: int,
    fps: Fraction | int | float,
    post_contact_padding_seconds: float = 3.0,
) -> int:
    """Return a half-open suffix cut preserving time after the last contact."""
    if source_frame_count <= 0:
        raise ValueError("source_frame_count must be positive")
    if contact_frame is None:
        return source_frame_count
    if not 0 <= int(contact_frame) < source_frame_count:
        raise ValueError("contact_frame is outside the source timeline")
    frame_rate = float(fps)
    if frame_rate <= 0 or post_contact_padding_seconds < 0:
        raise ValueError("FPS and post-contact padding must be valid")
    padding_frames = int(round(post_contact_padding_seconds * frame_rate))
    return min(source_frame_count, int(contact_frame) + 1 + padding_frames)


def _video_info(path: Path) -> tuple[int, Fraction]:
    count = 0
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream: {path}")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None or rate <= 0:
            raise ValueError(f"Video has no valid frame rate: {path}")
        for _ in container.decode(video=0):
            count += 1
    if count <= 0:
        raise ValueError(f"Video has no frames: {path}")
    return count, Fraction(rate)


def _video_stream_info(path: Path) -> tuple[int | None, Fraction]:
    """Read container-declared frame count and rate without decoding frames."""
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream: {path}")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None or rate <= 0:
            raise ValueError(f"Video has no valid frame rate: {path}")
        declared_count = int(stream.frames) if int(stream.frames or 0) > 0 else None
    return declared_count, Fraction(rate)


def _h5_info(path: Path) -> tuple[int, Fraction | None]:
    with h5py.File(path, "r") as source:
        if "frames" not in source:
            raise ValueError(f"Missing frames dataset: {path}")
        count = int(source["frames"].shape[0])
        raw_rate = source.attrs.get("fps", source.attrs.get("frame_rate"))
    rate = None
    if raw_rate is not None:
        rate = Fraction(str(float(raw_rate))).limit_denominator(1_000_000)
    return count, rate


def resolve_source_timeline(source_root: str | Path) -> tuple[int, Fraction]:
    """Resolve a balanced source frame count and FPS from canonical RGB data."""
    root = Path(source_root)
    images = root / "mv_preprocess" / "images"
    videos = root / "mv_preprocess" / "videos"
    counts: list[tuple[Path, int]] = []
    rates: list[tuple[Path, Fraction]] = []

    for path in sorted(images.glob("*.h5")) if images.is_dir() else []:
        count, rate = _h5_info(path)
        counts.append((path, count))
        if rate is not None:
            rates.append((path, rate))
    if images.is_dir() and not counts:
        for camera_dir in sorted(path for path in images.iterdir() if path.is_dir()):
            frames = sorted(camera_dir.glob("*.png"))
            if frames:
                counts.append((camera_dir, len(frames)))

    video_paths = sorted(videos.glob("*.mp4")) if videos.is_dir() else []
    for path in video_paths:
        declared_count, rate = _video_stream_info(path)
        rates.append((path, rate))
        if counts:
            if declared_count is not None and declared_count != counts[0][1]:
                raise ValueError(
                    f"Video frame count {declared_count} in {path} differs "
                    f"from RGB archive count {counts[0][1]}"
                )
        else:
            count = declared_count
            if count is None:
                count, _ = _video_info(path)
            counts.append((path, count))

    if not counts:
        raise ValueError("Could not resolve a source RGB timeline")
    unique_counts = {count for _, count in counts}
    if len(unique_counts) != 1:
        details = ", ".join(f"{path}={count}" for path, count in counts)
        raise ValueError(f"Source RGB frame counts disagree: {details}")
    frame_count = unique_counts.pop()
    if frame_count <= 0:
        raise ValueError("Source RGB timeline is empty")

    if rates:
        first_rate = rates[0][1]
        disagreement = [
            (path, rate)
            for path, rate in rates
            if abs(float(rate) - float(first_rate)) > 1e-6
        ]
        if disagreement:
            details = ", ".join(f"{path}={float(rate)}" for path, rate in rates)
            raise ValueError(f"Source RGB frame rates disagree: {details}")
        rate = first_rate
    else:
        rate = Fraction(30, 1)
    return frame_count, rate


class _ObjectSurfaceDistance:
    """Unsigned distance query against one static object triangle surface."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        try:
            import open3d as o3d
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Open3D is required to detect the interaction trim point"
            ) from exc
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("Object mesh vertices are invalid")
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError("Object mesh faces are invalid")
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32),
        )
        self._o3d = o3d
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(mesh)

    def minimum_distance(self, points: np.ndarray) -> float:
        points = np.asarray(points, dtype=np.float32)
        values = self._scene.compute_distance(
            self._o3d.core.Tensor(points, dtype=self._o3d.core.Dtype.Float32)
        ).numpy()
        return float(values.min(initial=np.inf))


def compute_human_object_distances(
    human_vertices_world: np.ndarray,
    object_poses_world_from_object: np.ndarray,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    *,
    pose_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute nearest human-vertex to object-surface distance per frame."""
    humans = np.asarray(human_vertices_world)
    poses = np.asarray(object_poses_world_from_object)
    if humans.ndim != 3 or humans.shape[-1] != 3:
        raise ValueError(f"Human vertices have invalid shape {humans.shape}")
    if poses.shape != (len(humans), 4, 4):
        raise ValueError(
            f"Object poses {poses.shape} do not match {len(humans)} human frames"
        )
    valid = (
        np.ones(len(humans), dtype=bool)
        if pose_valid_mask is None
        else np.asarray(pose_valid_mask, dtype=bool)
    )
    if valid.shape != (len(humans),):
        raise ValueError("Object pose validity mask does not match frame count")

    query = _ObjectSurfaceDistance(object_vertices, object_faces)
    distances = np.full(len(humans), np.nan, dtype=np.float64)
    for index in range(len(humans)):
        pose = poses[index]
        vertices = humans[index]
        if (
            not valid[index]
            or not np.isfinite(pose).all()
            or not np.isfinite(vertices).all()
        ):
            continue
        try:
            object_from_world = np.linalg.inv(pose)
        except np.linalg.LinAlgError:
            continue
        vertices_object = (
            vertices @ object_from_world[:3, :3].T
            + object_from_world[:3, 3]
        )
        distances[index] = query.minimum_distance(vertices_object)
    return distances


def detect_interaction_trim(
    source_root: str | Path,
    *,
    config: InteractionTrimConfig | None = None,
) -> dict[str, Any]:
    """Detect the stable interaction interval from canonical trajectories."""
    config = config or InteractionTrimConfig()
    config.validate()
    root = Path(source_root)
    source_count, fps = resolve_source_timeline(root)

    poses_path = root / "foundation_pose" / "poses.npy"
    valid_path = root / "foundation_pose" / "pose_valid_mask.npy"
    human_path = root / "sam3d_body" / "mhr_mesh_mv.pt"
    mesh_path = root / "mv_preprocess" / "object_mesh" / "output_aligned.glb"
    for path in (poses_path, human_path, mesh_path):
        if not path.is_file():
            raise ValueError(f"Missing interaction-trim source artifact: {path}")

    poses = np.load(poses_path, allow_pickle=False)
    valid = (
        np.load(valid_path, allow_pickle=False)
        if valid_path.is_file()
        else None
    )
    human_payload = torch.load(
        human_path, weights_only=False, map_location="cpu",
    )
    if not isinstance(human_payload, dict) or "pred_vertices" not in human_payload:
        raise ValueError("mhr_mesh_mv.pt is missing pred_vertices")
    vertices = human_payload["pred_vertices"]
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    mesh = trimesh.load(mesh_path, process=False, force="mesh")

    lengths = {
        "rgb": source_count,
        "object_poses": int(len(poses)),
        "human_vertices": int(len(vertices)),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Interaction-trim source frame counts disagree: {lengths}")
    if not config.enabled:
        distances = np.full(source_count, np.nan, dtype=np.float64)
        first_contact = None
        last_contact = None
        export_start = 0
        export_end = source_count
        reason = "disabled"
    else:
        distances = compute_human_object_distances(
            vertices,
            poses,
            np.asarray(mesh.vertices),
            np.asarray(mesh.faces),
            pose_valid_mask=valid,
        )
        first_contact, last_contact = select_stable_contact_frames(
            distances,
            distance_threshold_m=config.distance_threshold_m,
            window_frames=config.window_frames,
            required_under_threshold_frames=(
                config.required_under_threshold_frames
            ),
        )
        if first_contact is None:
            export_start = 0
            export_end = source_count
            reason = "no_stable_contact_keep_full"
        else:
            export_start = padded_export_start_frame(
                first_contact,
                source_frame_count=source_count,
                fps=fps,
                pre_contact_padding_seconds=config.pre_contact_padding_seconds,
            )
            export_end = padded_export_end_frame(
                last_contact,
                source_frame_count=source_count,
                fps=fps,
                post_contact_padding_seconds=config.post_contact_padding_seconds,
            )
            reason = "stable_contact"

    finite = distances[np.isfinite(distances)]
    distance_values = [
        None if not math.isfinite(float(value)) else float(value)
        for value in distances
    ]
    manifest: dict[str, Any] = {
        "schema": TRIM_SCHEMA,
        "enabled": bool(config.enabled),
        "reason": reason,
        "config": asdict(config),
        "source_frame_count": source_count,
        "source_fps": {
            "numerator": fps.numerator,
            "denominator": fps.denominator,
            "value": float(fps),
        },
        # Retain the original field as a first-contact compatibility alias.
        "contact_frame_source": first_contact,
        "first_contact_frame_source": first_contact,
        "last_contact_frame_source": last_contact,
        "export_source_start_frame": export_start,
        "export_source_end_frame": export_end,
        "export_frame_count": export_end - export_start,
        "trimmed_prefix_frames": export_start,
        "trimmed_prefix_seconds": export_start / float(fps),
        "trimmed_suffix_frames": source_count - export_end,
        "trimmed_suffix_seconds": (source_count - export_end) / float(fps),
        "export_frame_mapping": {
            "source_start_inclusive": export_start,
            "source_end_exclusive": export_end,
            "export_start": 0,
            "formula": "export_frame = source_frame - source_start_inclusive",
        },
        "distance_summary_m": {
            "valid_frames": int(np.isfinite(distances).sum()),
            "invalid_frames": int((~np.isfinite(distances)).sum()),
            "minimum": float(finite.min()) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
        },
        "nearest_human_object_distance_m": distance_values,
        "nearest_distance_sha256": hashlib.sha256(
            np.asarray(distances, dtype="<f8").tobytes()
        ).hexdigest(),
        "source_artifacts": {
            "object_poses": poses_path.relative_to(root).as_posix(),
            "pose_valid_mask": (
                valid_path.relative_to(root).as_posix()
                if valid_path.is_file() else None
            ),
            "human_mesh": human_path.relative_to(root).as_posix(),
            "aligned_object_mesh": mesh_path.relative_to(root).as_posix(),
        },
    }
    manifest["decision_sha256"] = _canonical_sha256(manifest)
    return manifest


def validate_trim_manifest(manifest: dict[str, Any]) -> tuple[int, int, int]:
    schema = manifest.get("schema")
    if schema not in SUPPORTED_TRIM_SCHEMAS:
        raise ValueError("Unsupported interaction trim manifest")
    source_count = int(manifest.get("source_frame_count", 0))
    start = int(manifest.get("export_source_start_frame", -1))
    end = int(manifest.get("export_source_end_frame", -1))
    output_count = int(manifest.get("export_frame_count", -1))
    valid_interval = 0 <= start < end <= source_count
    if schema == TRIM_SCHEMA_V1:
        valid_interval = valid_interval and end == source_count
    if source_count <= 0 or not valid_interval:
        raise ValueError("Interaction trim source interval is invalid")
    if output_count != end - start:
        raise ValueError("Interaction trim output frame count is inconsistent")
    if "trimmed_prefix_frames" in manifest:
        if int(manifest["trimmed_prefix_frames"]) != start:
            raise ValueError("Interaction trim prefix count is inconsistent")
    if schema == TRIM_SCHEMA:
        required_v2 = {
            "first_contact_frame_source", "last_contact_frame_source",
            "trimmed_suffix_frames", "trimmed_suffix_seconds",
        }
        if not required_v2.issubset(manifest):
            raise ValueError("Interaction trim v2 manifest is incomplete")
        if int(manifest["trimmed_suffix_frames"]) != source_count - end:
            raise ValueError("Interaction trim suffix count is inconsistent")
    decision_sha256 = manifest.get("decision_sha256")
    without_digest = dict(manifest)
    without_digest.pop("decision_sha256", None)
    if decision_sha256 != _canonical_sha256(without_digest):
        raise ValueError("Interaction trim decision digest is inconsistent")
    return start, end, output_count


def write_trim_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    validate_trim_manifest(manifest)
    Path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(
        source.resolve(),
        destination,
        target_is_directory=source.is_dir(),
    )


def _copy_directory_view(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        _symlink(child, destination / child.name)


def _replace_task_directory(
    source_root: Path,
    output_root: Path,
    task_name: str,
    replacements: dict[str, Callable[[Path, Path], None]],
) -> None:
    source = source_root / task_name
    if not source.is_dir():
        return
    destination = output_root / task_name
    _remove_path(destination)
    destination.mkdir(parents=True)
    for child in source.iterdir():
        target = destination / child.name
        replacement = replacements.get(child.name)
        if replacement is None:
            _symlink(child, target)
        else:
            replacement(child, target)


def _slice_png_group(
    source_files: list[Path],
    destination: Path,
    *,
    start: int,
    end: int,
    source_count: int,
) -> None:
    if len(source_files) != source_count:
        raise ValueError(
            f"PNG frame count {len(source_files)} under {source_files[0].parent} "
            f"does not match source timeline {source_count}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for output_index, source_path in enumerate(source_files[start:end]):
        shutil.copy2(
            source_path,
            destination / f"{output_index:06d}{source_path.suffix.lower()}",
        )


def _slice_archive_tree(
    source: Path,
    destination: Path,
    *,
    start: int,
    end: int,
    source_count: int,
) -> None:
    """Slice every H5/PNG frame archive while linking non-frame artifacts."""
    destination.mkdir(parents=True, exist_ok=True)
    direct_pngs = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if direct_pngs:
        _slice_png_group(
            direct_pngs,
            destination,
            start=start,
            end=end,
            source_count=source_count,
        )
    for child in source.iterdir():
        if child in direct_pngs:
            continue
        target = destination / child.name
        if child.is_dir():
            _slice_archive_tree(
                child,
                target,
                start=start,
                end=end,
                source_count=source_count,
            )
        elif child.suffix.lower() in {".h5", ".hdf5"}:
            stats = slice_h5_frames(
                child, target, start_frame=start, end_frame=end,
            )
            if int(stats["source_frames"]) != source_count:
                raise ValueError(
                    f"HDF5 frame count {stats['source_frames']} in {child} "
                    f"does not match source timeline {source_count}"
                )
        else:
            _symlink(child, target)


def trim_video(
    source: str | Path,
    destination: str | Path,
    *,
    start_frame: int,
    end_frame: int,
    expected_source_frames: int,
    crf: int = 17,
    codec_threads: int = 1,
) -> dict[str, Any]:
    """Frame-accurately re-encode one retained H.264 interval."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    decoded_count = 0
    written_count = 0
    try:
        with av.open(str(source)) as input_container:
            if len(input_container.streams.video) != 1:
                raise ValueError(f"Expected one input video stream: {source}")
            input_stream = input_container.streams.video[0]
            rate = input_stream.average_rate or input_stream.guessed_rate
            if rate is None or rate <= 0:
                raise ValueError(f"Video has no valid frame rate: {source}")
            with av.open(str(temporary), "w", format="mp4") as output_container:
                output_stream = output_container.add_stream(
                    "libx264",
                    rate=Fraction(rate),
                    options={"crf": str(crf)},
                )
                output_stream.width = int(input_stream.codec_context.width)
                output_stream.height = int(input_stream.codec_context.height)
                output_stream.pix_fmt = "yuv420p"
                output_stream.codec_context.thread_count = max(
                    1, int(codec_threads)
                )
                for source_index, frame in enumerate(
                    input_container.decode(input_stream)
                ):
                    decoded_count += 1
                    if source_index < start_frame:
                        continue
                    if source_index >= end_frame:
                        continue
                    array = frame.to_ndarray(format="rgb24")
                    output_frame = av.VideoFrame.from_ndarray(
                        array, format="rgb24",
                    )
                    output_frame.pts = written_count
                    output_frame.time_base = Fraction(
                        Fraction(rate).denominator,
                        Fraction(rate).numerator,
                    )
                    for packet in output_stream.encode(output_frame):
                        output_container.mux(packet)
                    written_count += 1
                for packet in output_stream.encode():
                    output_container.mux(packet)
        if decoded_count != expected_source_frames:
            raise ValueError(
                f"Video {source} decoded {decoded_count} frames; expected "
                f"{expected_source_frames}"
            )
        if written_count != end_frame - start_frame:
            raise ValueError(f"Video trim wrote the wrong frame count: {source}")
        verified_count, verified_rate = _video_stream_info(temporary)
        if verified_count is None:
            verified_count, verified_rate = _video_info(temporary)
        if verified_count != written_count:
            raise ValueError(f"Trimmed video verification failed: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source_frames": decoded_count,
        "frames": written_count,
        "fps": float(verified_rate),
    }


def _slice_video_tree(
    source: Path,
    destination: Path,
    *,
    start: int,
    end: int,
    source_count: int,
    max_workers: int = 1,
    codec_threads: int = 1,
) -> None:
    if max_workers < 1 or codec_threads < 1:
        raise ValueError("Video trim worker and codec thread counts must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, Path]] = []
    for child in source.rglob("*"):
        relative = child.relative_to(source)
        target = destination / relative
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif child.suffix.lower() == ".mp4":
            jobs.append((child, target))
        else:
            _symlink(child, target)

    def _trim(item: tuple[Path, Path]) -> dict[str, Any]:
        child, target = item
        return trim_video(
            child,
            target,
            start_frame=start,
            end_frame=end,
            expected_source_frames=source_count,
            codec_threads=codec_threads,
        )

    if len(jobs) < 2 or max_workers == 1:
        for job in jobs:
            _trim(job)
    else:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(jobs)),
            thread_name_prefix="trim-video",
        ) as pool:
            list(pool.map(_trim, jobs))


def _slice_numpy(
    source: Path,
    destination: Path,
    *,
    start: int,
    end: int,
    source_count: int,
) -> None:
    array = np.load(source, allow_pickle=False)
    if array.ndim < 1 or len(array) != source_count:
        raise ValueError(f"Temporal NumPy artifact has incompatible shape: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.save(stream, array[start:end], allow_pickle=False)


def _slice_torch_mapping(
    source: Path,
    destination: Path,
    *,
    start: int,
    end: int,
    source_count: int,
    temporal_keys: set[str],
    static_keys: set[str] | None = None,
) -> None:
    payload = torch.load(source, weights_only=False, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a torch mapping: {source}")
    static_keys = static_keys or set()
    unknown = set(payload) - temporal_keys - static_keys
    output: dict[str, Any] = {}
    for key, value in payload.items():
        if key in temporal_keys:
            if not hasattr(value, "shape") or len(value.shape) < 1:
                raise ValueError(f"Temporal torch value has no frame axis: {key}")
            if int(value.shape[0]) != source_count:
                raise ValueError(
                    f"Temporal torch value {key} has {value.shape[0]} frames; "
                    f"expected {source_count}"
                )
            output[key] = value[start:end]
        else:
            if (
                key in unknown
                and hasattr(value, "shape")
                and len(value.shape) >= 1
                and int(value.shape[0]) == source_count
            ):
                raise ValueError(
                    f"Unknown torch value {key!r} is ambiguously temporal"
                )
            output[key] = value
    missing = temporal_keys - set(payload)
    if missing:
        raise ValueError(f"Temporal torch artifact is missing keys: {sorted(missing)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)


def _slice_soma(
    source: Path,
    destination: Path,
    *,
    start: int,
    end: int,
    source_count: int,
) -> None:
    with np.load(source, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    missing = SOMA_REQUIRED_TEMPORAL_KEYS - set(arrays)
    if missing:
        raise ValueError(f"SOMA artifact is missing temporal keys: {sorted(missing)}")
    output: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key in SOMA_TEMPORAL_KEYS:
            if value.ndim < 1 or int(value.shape[0]) != source_count:
                raise ValueError(
                    f"SOMA temporal value {key} does not match source timeline"
                )
            output[key] = value[start:end]
        else:
            if value.ndim >= 1 and int(value.shape[0]) == source_count:
                raise ValueError(
                    f"Unknown SOMA value {key!r} is ambiguously temporal"
                )
            output[key] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez(stream, **output)


def _trim_metadata(
    source: Path,
    destination: Path,
    *,
    manifest: dict[str, Any],
) -> None:
    payload = yaml.safe_load(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("hoi_metadata.yaml must contain a mapping")
    _, _, output_count = validate_trim_manifest(manifest)
    payload["frame_count"] = output_count
    metadata_keys = (
            "schema",
            "reason",
            "contact_frame_source",
            "first_contact_frame_source",
            "last_contact_frame_source",
            "export_source_start_frame",
            "export_source_end_frame",
            "export_frame_count",
            "trimmed_prefix_frames",
            "trimmed_prefix_seconds",
            "trimmed_suffix_frames",
            "trimmed_suffix_seconds",
            "decision_sha256",
        )
    payload["interaction_trim"] = {
        key: manifest[key] for key in metadata_keys if key in manifest
    }
    destination.write_text(yaml.safe_dump(payload, sort_keys=False))


def _trim_edex(
    source: Path,
    destination: Path,
    *,
    output_count: int,
) -> None:
    payload = json.loads(source.read_text())
    header = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(header, dict):
        raise ValueError("EDEX must contain an object header")
    header["frame_start"] = 0
    header["frame_end"] = output_count
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def prepare_trimmed_source(
    source_root: str | Path,
    output_root: str | Path,
    manifest: dict[str, Any],
    *,
    defer_frame_archives: bool = False,
    max_video_workers: int = 1,
    video_codec_threads: int = 1,
) -> Path:
    """Create a request-local source view containing only retained frames."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    start, end, output_count = validate_trim_manifest(manifest)
    source_count = int(manifest["source_frame_count"])
    if output_root.exists():
        raise ValueError(f"Trimmed source output already exists: {output_root}")
    output_root.mkdir(parents=True)
    _copy_directory_view(source_root, output_root)

    archive = lambda source, destination: _slice_archive_tree(
        source,
        destination,
        start=start,
        end=end,
        source_count=source_count,
    )
    video_tree = lambda source, destination: _slice_video_tree(
        source,
        destination,
        start=start,
        end=end,
        source_count=source_count,
        max_workers=max_video_workers,
        codec_threads=video_codec_threads,
    )

    trim_required = start > 0 or end < source_count
    preprocess_replacements: dict[str, Callable[[Path, Path], None]] = {
        "hoi_metadata.yaml": lambda source, destination: _trim_metadata(
            source, destination, manifest=manifest,
        ),
        "edex": lambda source, destination: _trim_edex(
            source, destination, output_count=output_count,
        ),
    }
    if trim_required:
        preprocess_replacements["videos"] = video_tree
        if not defer_frame_archives:
            preprocess_replacements["images"] = archive
    _replace_task_directory(
        source_root,
        output_root,
        "mv_preprocess",
        preprocess_replacements,
    )
    if not trim_required:
        return output_root

    _replace_task_directory(
        source_root,
        output_root,
        "face_detector",
        (
            {"videos": video_tree}
            if defer_frame_archives
            else {"images": archive, "videos": video_tree}
        ),
    )
    if not defer_frame_archives:
        for task_name in (
            "foundation_stereo",
            "sam2_object_masks",
            "sam2_human_masks",
        ):
            source = source_root / task_name
            if source.is_dir():
                destination = output_root / task_name
                _remove_path(destination)
                archive(source, destination)

    _replace_task_directory(
        source_root,
        output_root,
        "foundation_pose",
        {
            "poses.npy": lambda source, destination: _slice_numpy(
                source,
                destination,
                start=start,
                end=end,
                source_count=source_count,
            ),
            "pose_valid_mask.npy": lambda source, destination: _slice_numpy(
                source,
                destination,
                start=start,
                end=end,
                source_count=source_count,
            ),
        },
    )
    _replace_task_directory(
        source_root,
        output_root,
        "sam3d_body",
        {
            "mhr_params_mv.pt": lambda source, destination: _slice_torch_mapping(
                source,
                destination,
                start=start,
                end=end,
                source_count=source_count,
                temporal_keys=MHR_PARAMETER_TEMPORAL_KEYS,
            ),
            "mhr_mesh_mv.pt": lambda source, destination: _slice_torch_mapping(
                source,
                destination,
                start=start,
                end=end,
                source_count=source_count,
                temporal_keys=MHR_MESH_TEMPORAL_KEYS,
                static_keys=MHR_MESH_STATIC_KEYS,
            ),
        },
    )
    _replace_task_directory(
        source_root,
        output_root,
        "export_soma",
        {
            "soma_params.npz": lambda source, destination: _slice_soma(
                source,
                destination,
                start=start,
                end=end,
                source_count=source_count,
            ),
        },
    )
    _replace_task_directory(
        source_root,
        output_root,
        "render_hoi_overlay",
        {
            "tiled_hoi_overlay.mp4": lambda source, destination: trim_video(
                source,
                destination,
                start_frame=start,
                end_frame=end,
                expected_source_frames=source_count,
                codec_threads=max(
                    1, max_video_workers * video_codec_threads,
                ),
            ),
        },
    )
    return output_root


def trim_failure_segments(
    segments: list[dict[str, Any]],
    *,
    start_frame: int,
    end_frame: int,
) -> list[dict[str, Any]]:
    """Clip source-indexed half-open failure segments to the exported timeline."""
    output = []
    for segment in segments:
        source_start = int(segment["start_frame"])
        source_end = int(segment["end_frame"])
        if source_start < 0 or source_end <= source_start:
            raise ValueError(f"Invalid failure segment: {segment}")
        clipped_start = max(source_start, start_frame)
        clipped_end = min(source_end, end_frame)
        if clipped_end <= clipped_start:
            continue
        adjusted = dict(segment)
        adjusted["source_start_frame"] = source_start
        adjusted["source_end_frame"] = source_end
        adjusted["start_frame"] = clipped_start - start_frame
        adjusted["end_frame"] = clipped_end - start_frame
        output.append(adjusted)
    return output


def merged_interval_coverage(segments: list[dict[str, Any]]) -> int:
    intervals = sorted(
        (int(segment["start_frame"]), int(segment["end_frame"]))
        for segment in segments
    )
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)
