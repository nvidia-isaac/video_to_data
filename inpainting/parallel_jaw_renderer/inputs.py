"""Strict target-trajectory and TACO calibration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .transforms import pose_matrix, validate_transform_batch


TARGET_SCHEMA = "v2d.inpainting.parallel-jaw-target/v1"
TARGET_KEYS = frozenset(
    {
        "schema_version",
        "tracker",
        "coordinate_frame",
        "frame_indices",
        "left_valid",
        "right_valid",
        "left_position",
        "right_position",
        "left_wxyz",
        "right_wxyz",
        "left_aperture_m",
        "right_aperture_m",
    }
)


class InputError(ValueError):
    """Raised when a renderer input would require an implicit convention."""


def _scalar_text(value: object, *, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise InputError(f"{name} must be one scalar string, got {array.shape}")
    result = str(array.reshape(-1)[0])
    if not result:
        raise InputError(f"{name} must not be empty")
    return result


def _load_numeric(
    path: Path, *, preferred_keys: tuple[str, ...], label: str
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        result = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            candidates = [key for key in preferred_keys if key in archive]
            if len(candidates) == 1:
                result = archive[candidates[0]]
            elif len(archive.files) == 1:
                result = archive[archive.files[0]]
            else:
                raise InputError(
                    f"{label} archive needs one array or one of {preferred_keys}; "
                    f"found {archive.files}"
                )
    else:
        try:
            result = np.loadtxt(path)
        except Exception as exc:
            raise InputError(f"could not parse {label} {path}: {exc}") from exc
    return np.asarray(result)


def load_intrinsic(path: str | Path, *, width: int, height: int) -> np.ndarray:
    source = Path(path).resolve()
    intrinsic = np.asarray(
        _load_numeric(source, preferred_keys=("intrinsic", "K"), label="intrinsic"),
        dtype=np.float64,
    )
    if intrinsic.shape == (4,):
        fx, fy, cx, cy = (float(value) for value in intrinsic)
        intrinsic = np.asarray(
            ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
    if intrinsic.shape != (3, 3):
        raise InputError(f"intrinsic must have shape (3,3), got {intrinsic.shape}")
    if not np.isfinite(intrinsic).all():
        raise InputError("intrinsic contains non-finite values")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise InputError("intrinsic focal lengths must be positive")
    if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1e-8, rtol=0.0):
        raise InputError("intrinsic bottom row must be [0,0,1]")
    if not np.isclose(intrinsic[0, 1], 0.0, atol=1e-8) or not np.isclose(
        intrinsic[1, 0], 0.0, atol=1e-8
    ):
        raise InputError("skewed TACO intrinsics are unsupported")
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if not 0.0 <= cx < width or not 0.0 <= cy < height:
        raise InputError(
            f"intrinsic principal point ({cx:.3f},{cy:.3f}) is outside "
            f"the exact source geometry {width}x{height}"
        )
    return intrinsic


def load_world_to_camera(path: str | Path, *, frame_count: int) -> np.ndarray:
    source = Path(path).resolve()
    matrices = _load_numeric(
        source,
        preferred_keys=("world_to_camera",),
        label="world-to-camera calibration",
    )
    try:
        matrices = validate_transform_batch(matrices, name="world_to_camera")
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    if matrices.shape[0] != frame_count:
        raise InputError(
            f"world_to_camera contains {matrices.shape[0]} frames, expected {frame_count}"
        )
    return matrices


def validate_target_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    keys = set(arrays)
    missing = sorted(TARGET_KEYS - keys)
    extra = sorted(keys - TARGET_KEYS)
    if missing or extra:
        raise InputError(
            f"parallel-jaw target keys differ from the v1 contract; "
            f"missing={missing}, extra={extra}"
        )
    schema = _scalar_text(arrays["schema_version"], name="schema_version")
    if schema != TARGET_SCHEMA:
        raise InputError(f"target schema {schema!r} != {TARGET_SCHEMA!r}")
    _scalar_text(arrays["tracker"], name="tracker")
    coordinate_frame = _scalar_text(arrays["coordinate_frame"], name="coordinate_frame")
    if coordinate_frame != "world":
        raise InputError(
            f"coordinate_frame must be exactly 'world', got {coordinate_frame!r}"
        )

    frame_indices = np.asarray(arrays["frame_indices"])
    if frame_indices.ndim != 1 or frame_indices.size == 0:
        raise InputError("frame_indices must be a non-empty vector")
    if not np.issubdtype(frame_indices.dtype, np.integer):
        raise InputError("frame_indices must use an integer dtype")
    expected_indices = np.arange(frame_indices.size, dtype=frame_indices.dtype)
    if not np.array_equal(frame_indices, expected_indices):
        raise InputError(
            "frame_indices must be contiguous source indices 0..N-1; "
            "implicit calibration slicing is forbidden"
        )
    frame_count = int(frame_indices.size)

    for side in ("left", "right"):
        valid_raw = np.asarray(arrays[f"{side}_valid"])
        if valid_raw.shape != (frame_count,):
            raise InputError(
                f"{side}_valid must have shape ({frame_count},), got {valid_raw.shape}"
            )
        if valid_raw.dtype != np.bool_:
            if (
                not np.issubdtype(valid_raw.dtype, np.integer)
                or not np.isin(valid_raw, (0, 1)).all()
            ):
                raise InputError(f"{side}_valid must be boolean or binary integer")
        if not valid_raw.astype(bool).all():
            invalid = int(np.count_nonzero(~valid_raw.astype(bool)))
            raise InputError(
                f"{side}_valid marks {invalid}/{frame_count} frames invalid; "
                "holding or hallucinating poses belongs in the retargeting stage"
            )

        position = np.asarray(arrays[f"{side}_position"])
        quaternion = np.asarray(arrays[f"{side}_wxyz"])
        aperture = np.asarray(arrays[f"{side}_aperture_m"])
        expected_shapes = {
            f"{side}_position": (frame_count, 3),
            f"{side}_wxyz": (frame_count, 4),
            f"{side}_aperture_m": (frame_count,),
        }
        for name, value in (
            (f"{side}_position", position),
            (f"{side}_wxyz", quaternion),
            (f"{side}_aperture_m", aperture),
        ):
            if value.shape != expected_shapes[name]:
                raise InputError(
                    f"{name} must have shape {expected_shapes[name]}, got {value.shape}"
                )
            if not np.issubdtype(value.dtype, np.number):
                raise InputError(f"{name} must be numeric")
            if not np.isfinite(value).all():
                raise InputError(f"{name} contains non-finite values")
        if np.any(aperture < 0.0):
            frame = int(np.argmin(aperture))
            raise InputError(
                f"{side}_aperture_m[{frame}]={float(aperture[frame]):.8g} is negative"
            )
        norms = np.linalg.norm(quaternion.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
            frame = int(np.argmax(np.abs(norms - 1.0)))
            raise InputError(
                f"{side}_wxyz[{frame}] norm is {norms[frame]:.8g}, expected 1"
            )
    return frame_count


def _pose_batch(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    result = np.empty((position.shape[0], 4, 4), dtype=np.float64)
    for frame in range(position.shape[0]):
        result[frame] = pose_matrix(position[frame], quaternion[frame])
    return result


@dataclass(frozen=True)
class VideoGeometry:
    frame_count: int
    width: int
    height: int
    fps: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


@dataclass(frozen=True)
class ParallelJawInputs:
    target_path: Path
    intrinsic_path: Path
    world_to_camera_path: Path
    tracker: str
    geometry: VideoGeometry
    target: dict[str, np.ndarray]
    intrinsic: np.ndarray
    world_to_camera: np.ndarray
    left_world_semantic: np.ndarray
    right_world_semantic: np.ndarray
    preview_source_frame_index: int | None = None

    @property
    def frame_count(self) -> int:
        return self.geometry.frame_count

    def projection_report(self) -> dict[str, dict[str, object]]:
        report: dict[str, dict[str, object]] = {}
        inside_total = 0
        for side, poses in (
            ("left", self.left_world_semantic),
            ("right", self.right_world_semantic),
        ):
            camera = self.world_to_camera @ poses
            xyz = camera[:, :3, 3]
            depth = xyz[:, 2]
            front = depth > 1e-6
            uv = np.full((self.frame_count, 2), np.nan)
            uv[front, 0] = (
                self.intrinsic[0, 0] * xyz[front, 0] / depth[front]
                + self.intrinsic[0, 2]
            )
            uv[front, 1] = (
                self.intrinsic[1, 1] * xyz[front, 1] / depth[front]
                + self.intrinsic[1, 2]
            )
            inside = (
                front
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] < self.geometry.width)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] < self.geometry.height)
            )
            inside_total += int(inside.sum())
            report[side] = {
                "positive_depth_count": int(front.sum()),
                "inside_image_count": int(inside.sum()),
                "depth_m_range": (
                    [float(depth[front].min()), float(depth[front].max())]
                    if front.any()
                    else None
                ),
            }
            if not front.any():
                raise InputError(
                    f"all {side} semantic targets project behind the TACO camera"
                )
        if inside_total == 0:
            raise InputError(
                "no semantic target projects inside the exact source image; "
                "calibration convention or frame alignment is wrong"
            )
        return report


def select_preview_frame(
    inputs: ParallelJawInputs,
    source_frame_index: int,
) -> ParallelJawInputs:
    """Return a one-frame in-memory view for non-production visual QA.

    The immutable source target/calibration files remain the provenance inputs.
    A single frame avoids inventing temporal joint-step semantics between
    disjoint start/middle/end preview frames.
    """

    if (
        isinstance(source_frame_index, bool)
        or not isinstance(source_frame_index, (int, np.integer))
        or not 0 <= int(source_frame_index) < inputs.frame_count
    ):
        raise InputError(
            f"preview frame index {source_frame_index!r} is outside "
            f"[0,{inputs.frame_count})"
        )
    index = int(source_frame_index)
    frame_keys = {
        "frame_indices",
        "left_valid",
        "right_valid",
        "left_position",
        "right_position",
        "left_wxyz",
        "right_wxyz",
        "left_aperture_m",
        "right_aperture_m",
    }
    target = {
        key: (
            np.asarray(value)[[index]].copy()
            if key in frame_keys
            else np.asarray(value).copy()
        )
        for key, value in inputs.target.items()
    }
    result = ParallelJawInputs(
        target_path=inputs.target_path,
        intrinsic_path=inputs.intrinsic_path,
        world_to_camera_path=inputs.world_to_camera_path,
        tracker=inputs.tracker,
        geometry=VideoGeometry(
            frame_count=1,
            width=inputs.geometry.width,
            height=inputs.geometry.height,
            fps=inputs.geometry.fps,
        ),
        target=target,
        intrinsic=inputs.intrinsic.copy(),
        world_to_camera=inputs.world_to_camera[[index]].copy(),
        left_world_semantic=inputs.left_world_semantic[[index]].copy(),
        right_world_semantic=inputs.right_world_semantic[[index]].copy(),
        preview_source_frame_index=index,
    )
    result.projection_report()
    return result


def load_parallel_jaw_inputs(
    *,
    target_path: str | Path,
    intrinsic_path: str | Path,
    world_to_camera_path: str | Path,
    width: int,
    height: int,
    fps: float,
) -> ParallelJawInputs:
    if (
        isinstance(width, bool)
        or not isinstance(width, (int, np.integer))
        or width <= 0
    ):
        raise InputError(f"width must be a positive integer, got {width!r}")
    if (
        isinstance(height, bool)
        or not isinstance(height, (int, np.integer))
        or height <= 0
    ):
        raise InputError(f"height must be a positive integer, got {height!r}")
    if width % 2 or height % 2:
        raise InputError(f"source geometry {width}x{height} must be even for yuv420p")
    if not np.isfinite(fps) or fps <= 0.0:
        raise InputError(f"fps must be positive and finite, got {fps!r}")

    target_source = Path(target_path).resolve()
    if not target_source.is_file():
        raise FileNotFoundError(target_source)
    with np.load(target_source, allow_pickle=False) as archive:
        target = {name: np.asarray(archive[name]) for name in archive.files}
    frame_count = validate_target_arrays(target)
    intrinsic_source = Path(intrinsic_path).resolve()
    world_to_camera_source = Path(world_to_camera_path).resolve()
    intrinsic = load_intrinsic(intrinsic_source, width=int(width), height=int(height))
    world_to_camera = load_world_to_camera(
        world_to_camera_source, frame_count=frame_count
    )
    result = ParallelJawInputs(
        target_path=target_source,
        intrinsic_path=intrinsic_source,
        world_to_camera_path=world_to_camera_source,
        tracker=_scalar_text(target["tracker"], name="tracker"),
        geometry=VideoGeometry(
            frame_count=frame_count,
            width=int(width),
            height=int(height),
            fps=float(fps),
        ),
        target=target,
        intrinsic=intrinsic,
        world_to_camera=world_to_camera,
        left_world_semantic=_pose_batch(
            np.asarray(target["left_position"], dtype=np.float64),
            np.asarray(target["left_wxyz"], dtype=np.float64),
        ),
        right_world_semantic=_pose_batch(
            np.asarray(target["right_position"], dtype=np.float64),
            np.asarray(target["right_wxyz"], dtype=np.float64),
        ),
        preview_source_frame_index=None,
    )
    result.projection_report()
    return result
