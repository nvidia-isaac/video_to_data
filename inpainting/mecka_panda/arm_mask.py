"""Automatic MECKA arm masking from the formal hand-tracking contract.

This is the production version of the automatic mask stage originally proven in
``debug/mecka_arm_pipeline.py``.  It deliberately consumes only the common
``tracking.npz`` contract, the corresponding pinhole intrinsic matrix, and a
source video.  Consequently local-manifest and LeRobot/S3 episodes use exactly
the same path after tracking.

Grounding-DINO supplies complete arm/sleeve boxes on a few sharp seed frames.
The boxes and projected MECKA hand points seed two SAM2 objects, one per hand.
The propagated objects are cleaned, scored against the hand tracks and temporal
continuity, and automatically corrected on failed intervals.  A quality failure
never publishes a complete stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inpainting.adapters.mecka_parallel_jaw import condition_hand
from inpainting.mecka_panda.contracts import (
    artifact,
    load_npz,
    sha256,
    validate_tracking_arrays,
    write_json_atomic,
)
from inpainting.mecka_panda.video_io import Mp4Writer, probe_video

MASK_FILENAME = "arm_mask.npy"
PREVIEW_FILENAME = "mask_preview.mp4"
METADATA_FILENAME = "arm_mask.json"
RUN_SCHEMA = "v2d.inpainting.mecka-arm-mask-run/v1"

CONTAINER_IMAGE_NAMES = {
    "grounding_dino": "v2d_grounding_dino",
    "sam2": "v2d_sam2",
}
GROUNDING_DINO_RUNNER = Path(__file__).with_name("grounding_dino_runner.py")
SAM2_RUNNER = Path(__file__).with_name("sam2_runner.py")
CONTAINER_HELPER_RELATIVE_PATH = Path("modules/v2d_docker/container.py")
_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
HANDS = ("left", "right")
OBJECT_IDS = {"left": 1, "right": 2}
WRIST = 0
PALM = (0, 5, 9, 13, 17)
HAND_POINTS = (0, 4, 8, 12, 16, 20)


class ArmMaskError(RuntimeError):
    """Raised when the automatic mask stage cannot publish a valid artifact."""


class ArmMaskQualityError(ArmMaskError):
    """Raised when all automatic correction attempts fail the quality gate."""


@dataclass(frozen=True)
class ArmMaskConfig:
    """Configuration of conditioning, prompting, quality checks and dilation."""

    conditioning_jump_k: float = 6.0
    conditioning_max_gap: int = 15
    conditioning_smooth_window: int = 11
    conditioning_smooth_poly: int = 2
    working_width: int = 1280
    seed_count: int = 3
    seed_gap: int = 6
    candidate_stride: int = 3
    detector_prompt: str = "human arm . sleeve . forearm . hand"
    detector_box_threshold: float = 0.18
    detector_text_threshold: float = 0.18
    max_retries: int = 2
    max_corrections_per_retry: int = 4
    point_radius: int = 7
    min_hand_coverage: float = 0.67
    min_area_fraction: float = 0.003
    max_area_fraction: float = 0.35
    boundary_margin: int = 12
    max_area_ratio: float = 4.0
    max_centroid_step_fraction: float = 0.28
    max_object_iou: float = 0.25
    min_valid_fraction: float = 0.88
    max_failure_run: int = 8
    mask_dilate_iterations: int = 4

    def __post_init__(self) -> None:
        positive_ints = (
            "conditioning_max_gap",
            "conditioning_smooth_window",
            "working_width",
            "seed_count",
            "seed_gap",
            "candidate_stride",
            "max_corrections_per_retry",
            "point_radius",
            "boundary_margin",
            "max_failure_run",
        )
        if any(int(getattr(self, name)) <= 0 for name in positive_ints):
            raise ValueError(f"{positive_ints} must all be positive")
        if self.max_retries < 0 or self.mask_dilate_iterations < 0:
            raise ValueError("retry and dilation counts must be non-negative")
        if self.conditioning_jump_k <= 0 or self.max_area_ratio < 1:
            raise ValueError("conditioning_jump_k must be >0 and max_area_ratio >=1")
        fractions = (
            self.detector_box_threshold,
            self.detector_text_threshold,
            self.min_hand_coverage,
            self.min_area_fraction,
            self.max_area_fraction,
            self.max_centroid_step_fraction,
            self.max_object_iou,
            self.min_valid_fraction,
        )
        if any(value < 0 or value > 1 for value in fractions):
            raise ValueError("threshold fractions must lie in [0, 1]")
        if self.min_area_fraction >= self.max_area_fraction:
            raise ValueError("min_area_fraction must be smaller than max_area_fraction")
        if not self.detector_prompt.strip():
            raise ValueError("detector_prompt must not be empty")


@dataclass
class _Episode:
    sequence_id: str
    episode_index: int
    source_video: Path
    clip: Path
    output_dir: Path
    width: int
    height: int
    fps: float
    source_start_frame: int
    frame_count: int
    uv: dict[str, np.ndarray]
    present: dict[str, np.ndarray]
    frame_scores: np.ndarray


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_container_images() -> dict[str, dict[str, str]]:
    """Resolve the two mutable model tags to concrete local Docker image IDs."""
    identities: dict[str, dict[str, str]] = {}
    for name, image in CONTAINER_IMAGE_NAMES.items():
        completed = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ArmMaskError(
                f"Cannot resolve required Docker image {image!r}: {detail}"
            )
        image_id = completed.stdout.strip()
        if _IMAGE_ID_PATTERN.fullmatch(image_id) is None:
            raise ArmMaskError(
                f"Docker returned invalid immutable ID for {image!r}: {image_id!r}"
            )
        identities[name] = {"image": image, "image_id": image_id}
    return identities


def _tree_identity(path: Path) -> dict[str, Any]:
    """Return a stable tree digest without exposing files as top-level artifacts."""
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for candidate in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = candidate.relative_to(root).as_posix()
        size = candidate.stat().st_size
        file_hash = sha256(candidate)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        entries.append(
            {
                "path": str(candidate.resolve()),
                "relative_path": relative,
                "bytes": size,
                "size_bytes": size,
                "sha256": file_hash,
            }
        )
    if not entries:
        raise ArmMaskError(f"No model files under {root}")
    return {
        "path": str(root),
        "sha256": digest.hexdigest(),
        "file_count": len(entries),
        "files": entries,
    }


def _without_proxy() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "SOCKS_PROXY",
        "SOCKS5_PROXY",
        "socks_proxy",
        "socks5_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ):
        environment.pop(name, None)
    environment["NO_PROXY"] = "localhost,127.0.0.1"
    environment["no_proxy"] = "localhost,127.0.0.1"
    return environment


def _video_geometry(path: Path) -> tuple[int, int, int, float]:
    geometry = probe_video(path)
    return (
        int(geometry["width"]),
        int(geometry["height"]),
        int(geometry["frame_count"]),
        float(geometry["fps"]),
    )


def _materialize_window(
    source: Path,
    destination: Path,
    *,
    source_start_frame: int,
    frame_count: int,
    working_width: int,
) -> Path:
    """Decode and re-encode an exact frame interval, without timestamp seeking."""
    width, height, source_count, fps = _video_geometry(source)
    stop = source_start_frame + frame_count
    working_height = round(working_width * height / width)
    working_height -= working_height % 2
    if working_height <= 0:
        raise ValueError("working_width produces a non-positive video height")
    if source_start_frame < 0 or frame_count <= 0 or stop > source_count:
        raise ValueError(
            f"Frame window [{source_start_frame}, {stop}) is outside "
            f"{source_count}-frame source {source}"
        )
    if (
        source_start_frame == 0
        and frame_count == source_count
        and (working_width, working_height) == (width, height)
    ):
        return source
    if destination.is_file():
        existing = _video_geometry(destination)
        if existing[:3] == (working_width, working_height, frame_count):
            return destination
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.partial.mp4")
    temporary.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot decode {source}")
    writer: Mp4Writer | None = None
    try:
        for index in range(source_start_frame):
            if not capture.grab():
                raise ArmMaskError(f"{source} ended while skipping frame {index}")
        writer = Mp4Writer(temporary, fps, (working_width, working_height))
        for local_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ArmMaskError(
                    f"{source} ended at window frame {local_index}/{frame_count}"
                )
            if frame.shape[:2] != (working_height, working_width):
                frame = cv2.resize(
                    frame,
                    (working_width, working_height),
                    interpolation=cv2.INTER_AREA,
                )
            writer.write(frame)
        writer.close()
        writer = None
        actual = _video_geometry(temporary)
        if actual[:3] != (working_width, working_height, frame_count):
            raise ArmMaskError(
                f"Exact source window has geometry {actual[:3]}, expected "
                f"{(working_width, working_height, frame_count)}"
            )
        os.replace(temporary, destination)
    finally:
        capture.release()
        if writer is not None:
            with suppress(BrokenPipeError, OSError, RuntimeError):
                writer.close()
        temporary.unlink(missing_ok=True)
    return destination


def _intrinsic(path: Path) -> np.ndarray:
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsic.npy must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("intrinsic focal lengths must be positive")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("intrinsic third row must be [0, 0, 1]")
    return np.asarray(matrix, dtype=np.float64)


def _project_tracks(
    arrays: dict[str, np.ndarray],
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    config: ArmMaskConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    frame_count = validate_tracking_arrays(arrays)
    coordinate_frame = np.asarray(arrays["coordinate_frame"])
    if coordinate_frame.shape != () or str(coordinate_frame.item()) != "camera":
        raise ValueError("Arm masking requires camera-frame tracking")
    coefficients = np.asarray(distortion, dtype=np.float64)
    if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
        raise ValueError("distortion must contain finite k1,k2,p1,p2")
    uv: dict[str, np.ndarray] = {}
    present: dict[str, np.ndarray] = {}
    jumps: dict[str, int] = {}
    for hand in HANDS:
        points = np.asarray(arrays[f"{hand}_joints_3d"], dtype=np.float64)
        valid = np.asarray(arrays[f"{hand}_valid"], dtype=np.bool_).copy()
        valid &= np.isfinite(points).all(axis=(1, 2))
        valid &= np.all(points[:, :, 2] > 1e-4, axis=1)
        conditioned, conditioned_valid, jump_count = condition_hand(
            points,
            valid,
            jump_k=config.conditioning_jump_k,
            max_gap=config.conditioning_max_gap,
            smooth_window=config.conditioning_smooth_window,
            smooth_poly=config.conditioning_smooth_poly,
        )
        projected = np.full((frame_count, 21, 2), np.nan, dtype=np.float64)
        for index in np.flatnonzero(conditioned_valid):
            if np.any(conditioned[index, :, 2] <= 1e-4):
                conditioned_valid[index] = False
                continue
            image_points, _ = cv2.projectPoints(
                conditioned[index],
                np.zeros(3),
                np.zeros(3),
                intrinsic,
                coefficients,
            )
            projected[index] = image_points.reshape(21, 2)
        uv[hand] = projected
        present[hand] = conditioned_valid
        jumps[hand] = int(jump_count)
    return uv, present, jumps


def _frame_blur_scores(clip: Path, frame_count: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(clip))
    values = np.zeros(frame_count, dtype=np.float64)
    try:
        for index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ArmMaskError(f"{clip} ended at frame {index}/{frame_count}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            values[index] = cv2.Laplacian(gray, cv2.CV_64F).var()
    finally:
        capture.release()
    return values


def frame_quality_scores(
    uv: dict[str, np.ndarray],
    present: dict[str, np.ndarray],
    width: int,
    height: int,
    blur: np.ndarray,
) -> np.ndarray:
    """Score shared in-frame, sharp, well-separated hand observations."""
    frame_count = len(next(iter(uv.values())))
    scores = np.full(frame_count, -np.inf, dtype=np.float64)
    positive_blur = blur[blur > 0]
    blur_scale = float(np.percentile(positive_blur, 75)) if positive_blur.size else 1.0
    diagonal = math.hypot(width, height)
    for index in range(frame_count):
        if not all(present[hand][index] for hand in HANDS):
            continue
        points = [uv[hand][index] for hand in HANDS]
        if not all(np.isfinite(value).all() for value in points):
            continue
        all_points = np.vstack(points)
        if (
            (all_points[:, 0] < 0).any()
            or (all_points[:, 0] >= width).any()
            or (all_points[:, 1] < 0).any()
            or (all_points[:, 1] >= height).any()
        ):
            continue
        separation = np.linalg.norm(points[0].mean(axis=0) - points[1].mean(axis=0))
        edge = min(
            all_points[:, 0].min(),
            all_points[:, 1].min(),
            width - 1 - all_points[:, 0].max(),
            height - 1 - all_points[:, 1].max(),
        )
        sharpness = min(float(blur[index]) / max(blur_scale, 1e-6), 2.0)
        scores[index] = (
            0.55 * separation / diagonal
            + 0.30 * max(float(edge), 0.0) / min(width, height)
            + 0.15 * sharpness
        )
    return scores


def select_spaced_frames(
    scores: np.ndarray,
    count: int,
    minimum_gap: int,
    allowed: Iterable[int] | None = None,
) -> list[int]:
    """Select the highest-scoring frames with a minimum temporal separation."""
    allowed_set = None if allowed is None else set(map(int, allowed))
    selected: list[int] = []
    for value in np.argsort(scores)[::-1]:
        index = int(value)
        if not np.isfinite(scores[index]):
            break
        if allowed_set is not None and index not in allowed_set:
            continue
        if all(abs(index - other) >= minimum_gap for other in selected):
            selected.append(index)
        if len(selected) >= count:
            break
    return sorted(selected)


def _decode_frame(clip: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(clip))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ArmMaskError(f"Could not decode frame {index} from {clip}")
    return frame


def _detector_cache_key(config: ArmMaskConfig) -> str:
    return _canonical_hash(
        {
            "prompt": config.detector_prompt,
            "box_threshold": config.detector_box_threshold,
            "text_threshold": config.detector_text_threshold,
        }
    )[:12]


def _run_detector(
    episode: _Episode,
    frame_index: int,
    reconstruction_dir: Path,
    config: ArmMaskConfig,
    image_id: str,
) -> list[dict[str, Any]]:
    detector_dir = episode.output_dir / "detections" / _detector_cache_key(config)
    image_path = detector_dir / f"{frame_index:06d}.png"
    output_path = detector_dir / f"{frame_index:06d}.json"
    if output_path.is_file():
        return json.loads(output_path.read_text(encoding="utf-8"))
    detector_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), _decode_frame(episode.clip, frame_index)):
        raise ArmMaskError(f"Could not write detector frame {image_path}")
    command = [
        str(reconstruction_dir / ".venv" / "bin" / "python"),
        str(GROUNDING_DINO_RUNNER),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
        "--prompt",
        config.detector_prompt,
        "--model_dir",
        str(reconstruction_dir / "data" / "weights" / "grounding_dino"),
        "--box_threshold",
        str(config.detector_box_threshold),
        "--text_threshold",
        str(config.detector_text_threshold),
        "--image_id",
        image_id,
        "--cache_dir",
        str(episode.output_dir / "huggingface_cache"),
    ]
    subprocess.run(
        command,
        cwd=reconstruction_dir,
        check=True,
        env=_without_proxy(),
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(result, list):
        raise ArmMaskError(f"Detector output must be a list: {output_path}")
    return result


def _box_array(box: dict[str, Any]) -> np.ndarray:
    return np.asarray([box["x0"], box["y0"], box["x1"], box["y1"]], dtype=np.float64)


def _box_assignment_cost(
    detection: dict[str, Any],
    points: np.ndarray,
    width: int,
    height: int,
) -> float:
    x0, y0, x1, y1 = _box_array(detection["box"])
    inside = (
        (points[:, 0] >= x0)
        & (points[:, 0] <= x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] <= y1)
    )
    hand_center = points.mean(axis=0)
    box_center = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5])
    distance = np.linalg.norm(hand_center - box_center) / math.hypot(width, height)
    area = max(x1 - x0, 1.0) * max(y1 - y0, 1.0) / (width * height)
    confidence = float(detection.get("confidence", 0.0))
    label = str(detection.get("label", "")).lower()
    hand_only_penalty = 1.8 if "hand" in label and "arm" not in label else 0.0
    arm_bonus = (
        0.8 if any(token in label for token in ("arm", "sleeve", "forearm")) else 0.0
    )
    root_margin = 0.04 * min(width, height)
    reaches_boundary = (
        x0 <= root_margin
        or y0 <= root_margin
        or x1 >= width - root_margin
        or y1 >= height - root_margin
    )
    return (
        2.5 * (0.0 if inside[WRIST] else 1.0)
        + 1.8 * (1.0 - float(inside.mean()))
        + float(distance)
        + 0.25 * area
        + hand_only_penalty
        + (0.0 if reaches_boundary else 1.6)
        - arm_bonus
        - 0.6 * confidence
    )


def assign_boxes_to_hands(
    detections: list[dict[str, Any]],
    uv_by_hand: dict[str, np.ndarray],
    frame_index: int,
    width: int,
    height: int,
) -> dict[str, dict[str, float]]:
    """Jointly assign distinct detector boxes to the tracked hand identities."""
    if not detections:
        return {}
    if len(detections) == 1:
        costs = {
            hand: _box_assignment_cost(
                detections[0], uv_by_hand[hand][frame_index], width, height
            )
            for hand in HANDS
        }
        hand = min(costs, key=costs.get)
        return {hand: detections[0]["box"]} if costs[hand] < 2.8 else {}
    best: tuple[float, tuple[int, ...]] | None = None
    for indices in permutations(range(len(detections)), len(HANDS)):
        cost = sum(
            _box_assignment_cost(
                detections[detection_index],
                uv_by_hand[hand][frame_index],
                width,
                height,
            )
            for hand, detection_index in zip(HANDS, indices, strict=True)
        )
        if best is None or cost < best[0]:
            best = (cost, indices)
    assert best is not None
    assigned: dict[str, dict[str, float]] = {}
    for hand, detection_index in zip(HANDS, best[1], strict=True):
        detection = detections[detection_index]
        if (
            _box_assignment_cost(
                detection, uv_by_hand[hand][frame_index], width, height
            )
            < 2.8
        ):
            assigned[hand] = detection["box"]
    return assigned


def _fallback_arm_box(
    points: np.ndarray, hand: str, width: int, height: int
) -> dict[str, float]:
    wrist = points[WRIST]
    hand_min = points.min(axis=0)
    hand_max = points.max(axis=0)
    palm_span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 35.0)
    side_sign = -1.0 if wrist[0] < width * 0.5 else 1.0
    if abs(wrist[0] - width * 0.5) < width * 0.08:
        side_sign = -1.0 if hand == "left" else 1.0
    root = np.asarray(
        [np.clip(wrist[0] + side_sign * width * 0.18, 0, width - 1), height - 1.0]
    )
    lower = np.minimum(np.minimum(hand_min, wrist), root) - 1.4 * palm_span
    upper = np.maximum(np.maximum(hand_max, wrist), root) + 1.4 * palm_span
    lower = np.maximum(lower, [0.0, 0.0])
    upper = np.minimum(upper, [width - 1.0, height - 1.0])
    return dict(zip(("x0", "y0", "x1", "y1"), map(float, (*lower, *upper))))


def _expand_box_for_hand(
    box: dict[str, Any], points: np.ndarray, width: int, height: int
) -> dict[str, float]:
    values = _box_array(box)
    values[:2] = np.minimum(values[:2], points.min(axis=0) - 8.0)
    values[2:] = np.maximum(values[2:], points.max(axis=0) + 8.0)
    values[[0, 2]] = np.clip(values[[0, 2]], 0, width - 1)
    values[[1, 3]] = np.clip(values[[1, 3]], 0, height - 1)
    return dict(zip(("x0", "y0", "x1", "y1"), map(float, values)))


def _build_prompts(
    episode: _Episode,
    frames_to_prompt: Iterable[int],
    reconstruction_dir: Path,
    config: ArmMaskConfig,
    grounding_dino_image_id: str,
    hands: Iterable[str] = HANDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    requested_hands = tuple(hands)
    for frame_index in sorted(set(map(int, frames_to_prompt))):
        detections = _run_detector(
            episode,
            frame_index,
            reconstruction_dir,
            config,
            grounding_dino_image_id,
        )
        assigned = assign_boxes_to_hands(
            detections, episode.uv, frame_index, episode.width, episode.height
        )
        frame_diagnostics = {
            "detection_count": len(detections),
            "assigned": sorted(assigned),
            "fallback": [],
        }
        for hand in requested_hands:
            if not episode.present[hand][frame_index]:
                continue
            points = episode.uv[hand][frame_index]
            if not np.isfinite(points).all():
                continue
            box = assigned.get(hand)
            if box is None:
                box = _fallback_arm_box(points, hand, episode.width, episode.height)
                frame_diagnostics["fallback"].append(hand)
            box = _expand_box_for_hand(box, points, episode.width, episode.height)
            sparse = points[list(HAND_POINTS)]
            prompts.append(
                {
                    "frame_index": frame_index,
                    "object_id": OBJECT_IDS[hand],
                    "points": [
                        {"x": float(point[0]), "y": float(point[1])} for point in sparse
                    ],
                    "point_labels": [1] * len(sparse),
                    "box": box,
                    "mask_path": None,
                }
            )
        diagnostics[str(frame_index)] = frame_diagnostics
    return prompts, diagnostics


def _write_prompts(
    path: Path, episode: _Episode, prompts: list[dict[str, Any]], attempt: int
) -> None:
    write_json_atomic(
        path,
        {
            "prompts": prompts,
            "metadata": {
                "schema_version": "v2d.inpainting.sam2-prompts/v1",
                "sequence_id": episode.sequence_id,
                "source_video": str(episode.clip),
                "attempt": attempt,
                "object_ids": {str(value): key for key, value in OBJECT_IDS.items()},
            },
        },
    )


def _run_sam2(
    episode: _Episode,
    prompts_path: Path,
    masks_dir: Path,
    reconstruction_dir: Path,
    image_id: str,
) -> None:
    command = [
        str(reconstruction_dir / ".venv" / "bin" / "python"),
        str(SAM2_RUNNER),
        "--video_path",
        str(episode.clip),
        "--prompts_path",
        str(prompts_path),
        "--masks_dir",
        str(masks_dir),
        "--weights_dir",
        str(reconstruction_dir / "data" / "weights" / "sam2"),
        "--image_id",
        image_id,
    ]
    subprocess.run(
        command,
        cwd=reconstruction_dir,
        check=True,
        env=_without_proxy(),
    )


def _points_covered(mask: np.ndarray, points: np.ndarray, radius: int) -> np.ndarray:
    height, width = mask.shape
    covered = np.zeros(len(points), dtype=np.bool_)
    for index, point in enumerate(points):
        if not np.isfinite(point).all():
            continue
        x, y = np.round(point).astype(int)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x0 < x1 and y0 < y1:
            covered[index] = bool(mask[y0:y1, x0:x1].any())
    return covered


def component_supported_by_hand(
    mask: np.ndarray, points: np.ndarray, radius: int
) -> np.ndarray:
    """Keep the component best supported by tracked hand points."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=8
    )
    if count <= 1:
        return mask.astype(bool)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    center = points[finite].mean(axis=0)
    diagonal = math.hypot(*mask.shape)
    best_label = 1
    best_score = -np.inf
    for label in range(1, count):
        component = labels == label
        support = float(_points_covered(component, points, radius).sum())
        distance = np.linalg.norm(centroids[label] - center) / max(diagonal, 1.0)
        area_bonus = min(float(stats[label, cv2.CC_STAT_AREA]) / mask.size, 0.05)
        score = 3.0 * support - float(distance) + area_bonus
        if score > best_score:
            best_label, best_score = label, score
    return labels == best_label


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] <= 0:
        return np.asarray([np.nan, np.nan])
    return np.asarray(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
    )


def _failure_ranges(indices: Iterable[int]) -> list[list[int]]:
    values = sorted(set(map(int, indices)))
    if not values:
        return []
    ranges = [[values[0], values[0]]]
    for value in values[1:]:
        if value == ranges[-1][1] + 1:
            ranges[-1][1] = value
        else:
            ranges.append([value, value])
    return ranges


def _arm_root_evidence(
    mask: np.ndarray, points: np.ndarray, boundary_margin: int
) -> tuple[bool, bool, float]:
    margin = min(boundary_margin, mask.shape[0], mask.shape[1])
    boundary_reach = bool(
        mask[:margin].any()
        or mask[-margin:].any()
        or mask[:, :margin].any()
        or mask[:, -margin:].any()
    )
    ys, xs = np.where(mask)
    wrist = points[WRIST]
    hand_span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    extent_ratio = 0.0
    if len(xs) and np.isfinite(wrist).all():
        extent_ratio = float(
            np.percentile(np.hypot(xs - wrist[0], ys - wrist[1]), 98) / hand_span
        )
    return boundary_reach or extent_ratio >= 2.4, boundary_reach, extent_ratio


def _read_mask(path: Path, width: int, height: int) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return np.zeros((height, width), dtype=np.bool_)
    if raw.shape != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
    return raw > 0


def _evaluate_masks(
    episode: _Episode,
    masks_dir: Path,
    cleaned_dir: Path,
    config: ArmMaskConfig,
) -> dict[str, Any]:
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, list[dict[str, Any]]] = {hand: [] for hand in HANDS}
    diagonal = math.hypot(episode.width, episode.height)
    for hand in HANDS:
        hand_dir = cleaned_dir / str(OBJECT_IDS[hand])
        hand_dir.mkdir(parents=True, exist_ok=True)
        previous_area: float | None = None
        previous_centroid: np.ndarray | None = None
        for index in range(episode.frame_count):
            raw = _read_mask(
                masks_dir / str(OBJECT_IDS[hand]) / f"{index:06d}.png",
                episode.width,
                episode.height,
            )
            points = episode.uv[hand][index]
            cleaned = component_supported_by_hand(raw, points, config.point_radius)
            if not cv2.imwrite(
                str(hand_dir / f"{index:06d}.png"),
                cleaned.astype(np.uint8) * 255,
            ):
                raise ArmMaskError(f"Could not write cleaned mask {hand_dir}")
            coverage_flags = _points_covered(cleaned, points, config.point_radius)
            hand_coverage = float(coverage_flags.mean())
            palm_coverage = float(coverage_flags[list(PALM)].mean())
            area_fraction = float(cleaned.mean())
            centroid = _mask_centroid(cleaned)
            root_supported, boundary_reach, extent_ratio = _arm_root_evidence(
                cleaned, points, config.boundary_margin
            )
            area_ratio = 1.0
            if previous_area is not None and min(previous_area, area_fraction) > 0:
                area_ratio = max(previous_area, area_fraction) / min(
                    previous_area, area_fraction
                )
            centroid_step = 0.0
            if (
                previous_centroid is not None
                and np.isfinite(previous_centroid).all()
                and np.isfinite(centroid).all()
            ):
                centroid_step = float(
                    np.linalg.norm(centroid - previous_centroid) / diagonal
                )
            reasons: list[str] = []
            if episode.present[hand][index]:
                if hand_coverage < config.min_hand_coverage or palm_coverage < 0.6:
                    reasons.append("hand_support")
                if not (
                    config.min_area_fraction
                    <= area_fraction
                    <= config.max_area_fraction
                ):
                    reasons.append("area")
                if not root_supported:
                    reasons.append("arm_root")
                if area_ratio > config.max_area_ratio:
                    reasons.append("area_jump")
                if centroid_step > config.max_centroid_step_fraction:
                    reasons.append("centroid_jump")
            metrics[hand].append(
                {
                    "frame_index": index,
                    "hand_coverage": hand_coverage,
                    "palm_coverage": palm_coverage,
                    "area_fraction": area_fraction,
                    "boundary_reach": boundary_reach,
                    "root_extent_ratio": extent_ratio,
                    "arm_root_supported": root_supported,
                    "area_ratio": area_ratio,
                    "centroid_step_fraction": centroid_step,
                    "reasons": reasons,
                }
            )
            previous_area, previous_centroid = area_fraction, centroid
    overlap_failures: list[int] = []
    for index in range(episode.frame_count):
        left = _read_mask(
            cleaned_dir / str(OBJECT_IDS["left"]) / f"{index:06d}.png",
            episode.width,
            episode.height,
        )
        right = _read_mask(
            cleaned_dir / str(OBJECT_IDS["right"]) / f"{index:06d}.png",
            episode.width,
            episode.height,
        )
        union = left | right
        iou = float((left & right).sum() / max(int(union.sum()), 1))
        if iou > config.max_object_iou:
            overlap_failures.append(index)
            metrics["left"][index]["reasons"].append("object_overlap")
            metrics["right"][index]["reasons"].append("object_overlap")
    summary: dict[str, Any] = {}
    all_passed = True
    for hand in HANDS:
        failures = [item["frame_index"] for item in metrics[hand] if item["reasons"]]
        ranges = _failure_ranges(failures)
        valid_fraction = 1.0 - len(failures) / episode.frame_count
        longest_run = max((stop - start + 1 for start, stop in ranges), default=0)
        passed = (
            valid_fraction >= config.min_valid_fraction
            and longest_run <= config.max_failure_run
        )
        all_passed &= passed
        summary[hand] = {
            "passed": passed,
            "valid_fraction": valid_fraction,
            "failure_count": len(failures),
            "longest_failure_run": longest_run,
            "failure_ranges": ranges,
            "median_hand_coverage": float(
                np.median([item["hand_coverage"] for item in metrics[hand]])
            ),
            "median_area_fraction": float(
                np.median([item["area_fraction"] for item in metrics[hand]])
            ),
        }
    return {
        "passed": bool(all_passed),
        "summary": summary,
        "overlap_failures": overlap_failures,
        "frames": metrics,
    }


def _correction_frames(
    quality: dict[str, Any],
    episode: _Episode,
    config: ArmMaskConfig,
    excluded: dict[str, set[int]],
) -> dict[int, list[str]]:
    selected: dict[int, list[str]] = {}
    for hand in HANDS:
        ranked: list[tuple[float, int]] = []
        for start, stop in quality["summary"][hand]["failure_ranges"]:
            visible = [
                index
                for index in range(start, stop + 1)
                if episode.present[hand][index]
                and index not in excluded.get(hand, set())
            ]
            if not visible:
                continue
            finite = [
                index for index in visible if np.isfinite(episode.frame_scores[index])
            ]
            index = (
                max(finite, key=lambda value: episode.frame_scores[value])
                if finite
                else visible[len(visible) // 2]
            )
            ranked.append((float(episode.frame_scores[index]), index))
        ranked.sort(reverse=True)
        for _, index in ranked[: config.max_corrections_per_retry]:
            selected.setdefault(index, []).append(hand)
    return selected


def _run_automatic_mask(
    episode: _Episode,
    reconstruction_dir: Path,
    config: ArmMaskConfig,
    container_images: dict[str, dict[str, str]],
) -> tuple[Path, dict[str, Any]]:
    seeds = select_spaced_frames(
        episode.frame_scores, config.seed_count, config.seed_gap
    )
    if not seeds:
        raise ArmMaskError(f"{episode.sequence_id}: no valid shared seed frame")
    prompts, diagnostics = _build_prompts(
        episode,
        seeds,
        reconstruction_dir,
        config,
        container_images["grounding_dino"]["image_id"],
    )
    if {int(item["object_id"]) for item in prompts} != set(OBJECT_IDS.values()):
        raise ArmMaskError("Seed prompts did not cover both tracked hands")
    all_diagnostics: dict[str, Any] = {"initial": diagnostics}
    final_quality: dict[str, Any] | None = None
    final_cleaned: Path | None = None
    prompt_hash = ""
    attempt = 0
    for attempt in range(config.max_retries + 1):
        attempt_dir = episode.output_dir / f"attempt_{attempt}"
        prompts_path = attempt_dir / "prompts.json"
        _write_prompts(prompts_path, episode, prompts, attempt)
        prompt_hash = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
        masks_dir = attempt_dir / f"sam2_masks_{prompt_hash[:12]}"
        cleaned_dir = attempt_dir / f"cleaned_masks_{prompt_hash[:12]}"
        _run_sam2(
            episode,
            prompts_path,
            masks_dir,
            reconstruction_dir,
            container_images["sam2"]["image_id"],
        )
        quality = _evaluate_masks(episode, masks_dir, cleaned_dir, config)
        write_json_atomic(attempt_dir / "quality.json", quality)
        final_quality, final_cleaned = quality, cleaned_dir
        if quality["passed"] or attempt >= config.max_retries:
            break
        prompted = {
            hand: {
                int(prompt["frame_index"])
                for prompt in prompts
                if int(prompt["object_id"]) == OBJECT_IDS[hand]
            }
            for hand in HANDS
        }
        corrections = _correction_frames(quality, episode, config, excluded=prompted)
        if not corrections:
            break
        retry_diagnostics: dict[str, Any] = {}
        for frame_index, hands in corrections.items():
            extra, detail = _build_prompts(
                episode,
                [frame_index],
                reconstruction_dir,
                config,
                container_images["grounding_dino"]["image_id"],
                hands=hands,
            )
            prompts.extend(extra)
            retry_diagnostics[str(frame_index)] = {
                "hands": hands,
                "detector": detail.get(str(frame_index), {}),
            }
        all_diagnostics[f"retry_{attempt + 1}"] = retry_diagnostics
    assert final_quality is not None and final_cleaned is not None
    quality_path = episode.output_dir / "quality.json"
    write_json_atomic(quality_path, final_quality)
    reason_counts = {
        hand: {
            reason: sum(
                reason in frame["reasons"] for frame in final_quality["frames"][hand]
            )
            for reason in sorted(
                {
                    reason
                    for frame in final_quality["frames"][hand]
                    for reason in frame["reasons"]
                }
            )
        }
        for hand in HANDS
    }
    result = {
        "passed": final_quality["passed"],
        "seed_frames": seeds,
        "attempts": attempt + 1,
        "prompt_sha256": prompt_hash,
        "quality": final_quality["summary"],
        "quality_reason_counts": reason_counts,
        "prompt_diagnostics": all_diagnostics,
        "quality_diagnostics": artifact(quality_path),
    }
    write_json_atomic(episode.output_dir / "mask_result.json", result)
    if not final_quality["passed"]:
        raise ArmMaskQualityError(
            f"Automatic SAM2 masks failed quality checks; inspect {quality_path}"
        )
    return final_cleaned, result


def _publish_outputs(
    episode: _Episode,
    cleaned_dir: Path,
    mask_path: Path,
    preview_path: Path,
    config: ArmMaskConfig,
    *,
    source_width: int,
    source_height: int,
    source_fps: float,
) -> None:
    """Stream exact bool NPY and full-resolution preview, then atomically replace."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, mask_name = tempfile.mkstemp(
        dir=mask_path.parent, prefix=f".{mask_path.stem}-", suffix=".partial.npy"
    )
    os.close(descriptor)
    temporary_mask = Path(mask_name)
    preview_descriptor, preview_name = tempfile.mkstemp(
        dir=preview_path.parent,
        prefix=f".{preview_path.stem}-",
        suffix=".partial.mp4",
    )
    os.close(preview_descriptor)
    temporary_preview = Path(preview_name)
    temporary_preview.unlink()
    capture = cv2.VideoCapture(str(episode.source_video))
    if not capture.isOpened():
        temporary_mask.unlink(missing_ok=True)
        raise FileNotFoundError(f"Cannot decode {episode.source_video}")
    writer: Mp4Writer | None = None
    mapped: np.memmap | None = None
    try:
        mapped = np.lib.format.open_memmap(
            temporary_mask,
            mode="w+",
            dtype=np.bool_,
            shape=(episode.frame_count, source_height, source_width),
        )
        writer = Mp4Writer(
            temporary_preview,
            source_fps,
            (source_width, source_height),
        )
        for source_index in range(episode.source_start_frame):
            if not capture.grab():
                raise ArmMaskError(
                    f"{episode.source_video} ended while skipping "
                    f"source frame {source_index}"
                )
        kernel = np.ones((3, 3), dtype=np.uint8)
        for index in range(episode.frame_count):
            union = np.zeros((episode.height, episode.width), dtype=np.uint8)
            for hand in HANDS:
                component = _read_mask(
                    cleaned_dir / str(OBJECT_IDS[hand]) / f"{index:06d}.png",
                    episode.width,
                    episode.height,
                )
                union = np.maximum(union, component.astype(np.uint8) * 255)
            if config.mask_dilate_iterations:
                union = cv2.dilate(
                    union,
                    kernel,
                    iterations=config.mask_dilate_iterations,
                )
            if (episode.width, episode.height) != (
                source_width,
                source_height,
            ):
                union = cv2.resize(
                    union,
                    (source_width, source_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = union > 0
            mapped[index] = mask
            ok, frame = capture.read()
            if not ok:
                raise ArmMaskError(
                    f"{episode.source_video} ended at preview frame {index}"
                )
            if frame.shape[:2] != (source_height, source_width):
                raise ArmMaskError(
                    f"Source frame {index} has shape {frame.shape[:2]}, expected "
                    f"{(source_height, source_width)}"
                )
            overlay = np.full_like(frame, (180, 40, 220))
            frame[mask] = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)[mask]
            writer.write(frame)
        mapped.flush()
        del mapped
        mapped = None
        with temporary_mask.open("rb") as stream:
            os.fsync(stream.fileno())
        writer.close()
        writer = None
        capture.release()
        loaded = np.load(temporary_mask, mmap_mode="r", allow_pickle=False)
        if loaded.dtype != np.bool_ or loaded.shape != (
            episode.frame_count,
            source_height,
            source_width,
        ):
            raise ArmMaskError("Published mask has the wrong dtype or shape")
        preview_geometry = _video_geometry(temporary_preview)
        expected = (source_width, source_height, episode.frame_count)
        if preview_geometry[:3] != expected:
            raise ArmMaskError(f"Preview geometry {preview_geometry[:3]} != {expected}")
        if not math.isclose(preview_geometry[3], source_fps, rel_tol=0.0, abs_tol=1e-3):
            raise ArmMaskError(
                f"Preview FPS {preview_geometry[3]} != source FPS {source_fps}"
            )
        os.replace(temporary_mask, mask_path)
        os.replace(temporary_preview, preview_path)
    finally:
        capture.release()
        if mapped is not None:
            mapped.flush()
        if writer is not None:
            with suppress(BrokenPipeError, OSError, RuntimeError):
                writer.close()
        temporary_mask.unlink(missing_ok=True)
        temporary_preview.unlink(missing_ok=True)


def _validate_dependencies(reconstruction_dir: str | Path) -> Path:
    reconstruction = Path(reconstruction_dir).expanduser().resolve()
    python = reconstruction / ".venv" / "bin" / "python"
    required_files = (
        python,
        GROUNDING_DINO_RUNNER,
        SAM2_RUNNER,
        reconstruction / CONTAINER_HELPER_RELATIVE_PATH,
        reconstruction
        / "modules"
        / "v2d_grounding_dino"
        / "docker"
        / "run_image_to_object_bboxes.py",
        reconstruction
        / "data"
        / "weights"
        / "grounding_dino"
        / "groundingdino_swint_ogc.pth",
        reconstruction / "data" / "weights" / "sam2" / "sam2.1_hiera_large.pt",
        reconstruction / "data" / "weights" / "sam2" / "sam2.1_hiera_l.yaml",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing arm-mask dependencies: {missing}")
    if not os.access(python, os.X_OK):
        raise PermissionError(f"Arm-mask Python is not executable: {python}")
    return reconstruction


def preflight(reconstruction_dir: str | Path) -> Path:
    """Validate concrete runners, checkpoints, and both local model images."""
    reconstruction = _validate_dependencies(reconstruction_dir)
    resolve_container_images()
    return reconstruction


def _tracking_metadata(
    path: Path,
    *,
    source_start_frame: int,
    frame_count: int,
    width: int,
    height: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Validate tracking provenance and recover scale-independent distortion."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read tracking metadata {path}") from error
    if metadata.get("state") != "complete":
        raise ValueError("tracking metadata state must be 'complete'")
    window = metadata.get("frame_window")
    if not isinstance(window, dict):
        raise ArmMaskError("tracking metadata is missing frame_window")
    if int(window.get("start", -1)) != source_start_frame:
        raise ValueError(
            "source_start_frame does not match tracking metadata: "
            f"{source_start_frame} != {window.get('start')}"
        )
    if int(window.get("count", -1)) != frame_count:
        raise ValueError(
            "tracking frame count does not match tracking metadata: "
            f"{frame_count} != {window.get('count')}"
        )
    geometry = metadata.get("video_geometry")
    if not isinstance(geometry, dict):
        raise ArmMaskError("tracking metadata is missing video_geometry")
    if (int(geometry.get("width", -1)), int(geometry.get("height", -1))) != (
        width,
        height,
    ):
        raise ValueError(
            "tracking/video geometry mismatch: "
            f"metadata={geometry.get('width')}x{geometry.get('height')}, "
            f"video={width}x{height}"
        )
    values = np.asarray(
        metadata.get("camera_intrinsics_full_resolution"), dtype=np.float64
    )
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError(
            "tracking metadata camera_intrinsics_full_resolution must contain "
            "fx,fy,cx,cy,k1,k2,p1,p2"
        )
    return metadata, values[4:8].copy()


def execute(
    *,
    tracking_path: str | Path,
    tracking_metadata: str | Path,
    intrinsic_path: str | Path,
    source_video: str | Path,
    output_dir: str | Path,
    source_start_frame: int,
    reconstruction_dir: str | Path,
    config: ArmMaskConfig | None = None,
    sequence_id: str | None = None,
    episode_index: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run automatic arm masking and atomically publish mask, preview and marker."""
    resolved_config = config or ArmMaskConfig()
    tracking = Path(tracking_path).expanduser().resolve()
    tracking_metadata_file = Path(tracking_metadata).expanduser().resolve()
    intrinsic_file = Path(intrinsic_path).expanduser().resolve()
    video = Path(source_video).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    reconstruction = _validate_dependencies(reconstruction_dir)
    container_images = resolve_container_images()
    for source in (tracking, tracking_metadata_file, intrinsic_file, video):
        if not source.is_file():
            raise FileNotFoundError(source)
    if source_start_frame < 0:
        raise ValueError("source_start_frame must be non-negative")
    mask_path = output / MASK_FILENAME
    preview_path = output / PREVIEW_FILENAME
    metadata_path = output / METADATA_FILENAME
    existing = [
        path for path in (mask_path, preview_path, metadata_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    arrays = load_npz(tracking)
    frame_count = validate_tracking_arrays(arrays)
    intrinsic = _intrinsic(intrinsic_file)
    width, height, video_frame_count, fps = _video_geometry(video)
    if source_start_frame + frame_count > video_frame_count:
        raise ValueError(
            "Tracking window exceeds source video: "
            f"start={source_start_frame}, count={frame_count}, "
            f"video_frames={video_frame_count}"
        )
    if not (0 <= intrinsic[0, 2] < width and 0 <= intrinsic[1, 2] < height):
        raise ValueError(f"Intrinsic principal point is outside {width}x{height} video")
    _, distortion = _tracking_metadata(
        tracking_metadata_file,
        source_start_frame=source_start_frame,
        frame_count=frame_count,
        width=width,
        height=height,
    )

    config_value = asdict(resolved_config)
    config_hash = _canonical_hash(config_value)
    source_artifacts = {
        "reconstruction_dir": str(reconstruction),
        "tracking": artifact(tracking),
        "tracking_metadata": artifact(tracking_metadata_file),
        "intrinsic": artifact(intrinsic_file),
        "video": artifact(video),
        "implementation": artifact(__file__),
        "runners": {
            "grounding_dino": artifact(GROUNDING_DINO_RUNNER),
            "sam2": artifact(SAM2_RUNNER),
            "container": artifact(reconstruction / CONTAINER_HELPER_RELATIVE_PATH),
        },
        "container_images": container_images,
        "model_weights": {
            name: _tree_identity(reconstruction / "data" / "weights" / name)
            for name in ("grounding_dino", "sam2")
        },
    }
    cache_hash = _canonical_hash(
        {
            "tracking_sha256": source_artifacts["tracking"]["sha256"],
            "tracking_metadata_sha256": source_artifacts["tracking_metadata"]["sha256"],
            "intrinsic_sha256": source_artifacts["intrinsic"]["sha256"],
            "video_sha256": source_artifacts["video"]["sha256"],
            "implementation_sha256": source_artifacts["implementation"]["sha256"],
            "runner_sha256": {
                name: value["sha256"]
                for name, value in source_artifacts["runners"].items()
            },
            "container_image_ids": {
                name: value["image_id"]
                for name, value in source_artifacts["container_images"].items()
            },
            "model_sha256": {
                name: value["sha256"]
                for name, value in source_artifacts["model_weights"].items()
            },
            "source_start_frame": source_start_frame,
            "frame_count": frame_count,
            "config_sha256": config_hash,
        }
    )
    work = output / "_work" / cache_hash[:20]
    clip = _materialize_window(
        video,
        work / "source_window.mp4",
        source_start_frame=source_start_frame,
        frame_count=frame_count,
        working_width=resolved_config.working_width,
    )
    clip_width, clip_height, clip_count, clip_fps = _video_geometry(clip)
    if clip_count != frame_count:
        raise ArmMaskError("Materialized source window changed the frame count")
    scale = np.diag([clip_width / width, clip_height / height, 1.0])
    working_intrinsic = scale @ intrinsic
    uv, present, jump_counts = _project_tracks(
        arrays, working_intrinsic, distortion, resolved_config
    )
    blur = _frame_blur_scores(clip, frame_count)
    scores = frame_quality_scores(uv, present, clip_width, clip_height, blur)
    resolved_sequence = sequence_id or video.stem
    episode = _Episode(
        sequence_id=resolved_sequence,
        episode_index=-1 if episode_index is None else int(episode_index),
        source_video=video,
        clip=clip,
        output_dir=work,
        width=clip_width,
        height=clip_height,
        fps=clip_fps,
        source_start_frame=source_start_frame,
        frame_count=frame_count,
        uv=uv,
        present=present,
        frame_scores=scores,
    )
    cleaned_dir, result = _run_automatic_mask(
        episode,
        reconstruction,
        resolved_config,
        container_images,
    )
    _publish_outputs(
        episode,
        cleaned_dir,
        mask_path,
        preview_path,
        resolved_config,
        source_width=width,
        source_height=height,
        source_fps=fps,
    )
    metadata = {
        "schema_version": RUN_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "sequence_id": resolved_sequence,
        "episode_index": episode_index,
        "frame_window": {
            "source_start": source_start_frame,
            "count": frame_count,
            "mask_start": 0,
        },
        "geometry": {
            "source": {
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "fps": fps,
            },
            "working": {
                "frame_count": frame_count,
                "width": clip_width,
                "height": clip_height,
                "fps": clip_fps,
            },
        },
        "config": config_value,
        "config_sha256": config_hash,
        "conditioning": {
            "jump_counts": jump_counts,
            "valid_counts": {
                hand: int(np.count_nonzero(present[hand])) for hand in HANDS
            },
        },
        "quality": result,
        "source": source_artifacts,
        "output": {
            "mask": artifact(mask_path),
            "preview": artifact(preview_path),
        },
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", required=True, type=Path)
    parser.add_argument("--tracking-metadata", required=True, type=Path)
    parser.add_argument("--intrinsic", required=True, type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-start-frame", type=int, required=True)
    parser.add_argument(
        "--reconstruction-dir", type=Path, default=Path("reconstruction")
    )
    parser.add_argument("--sequence-id")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    metadata = execute(
        tracking_path=args.tracking,
        tracking_metadata=args.tracking_metadata,
        intrinsic_path=args.intrinsic,
        source_video=args.source_video,
        output_dir=args.output_dir,
        source_start_frame=args.source_start_frame,
        reconstruction_dir=args.reconstruction_dir,
        sequence_id=args.sequence_id,
        episode_index=args.episode_index,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
