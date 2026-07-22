"""Evaluate a learned hand track against TACO ground truth without training leakage.

The ground-truth archive is read only by this evaluation stage.  Both tracks
are transformed to the calibrated TACO world frame for metric 3-D and temporal
measurements, then projected through the supplied per-frame world-to-camera
calibration and intrinsic matrix for pixel-space measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ContractError, VideoGeometry, validate_tracking_arrays
from .taco_camera import TacoCamera, load_taco_camera, project_camera_points
from .video_io import probe_video


EVALUATION_SCHEMA = "v2d.inpainting.tracking-evaluation/v1"
LEARNED_TRACKERS = ("phantom", "v2d")
SIDES = ("left", "right")


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ContractError(f"{name} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _summary(values: np.ndarray) -> dict[str, int | float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    if not np.isfinite(values).all():
        raise ContractError("Metric aggregation received non-finite values")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95, method="linear")),
        "max": float(np.max(values)),
    }


def _invalid_gaps(valid: np.ndarray) -> list[dict[str, int]]:
    valid = np.asarray(valid, dtype=np.bool_)
    gaps: list[dict[str, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        if not is_valid and start is None:
            start = index
        if is_valid and start is not None:
            gaps.append(
                {
                    "start_frame": start,
                    "end_frame": index - 1,
                    "length": index - start,
                }
            )
            start = None
    if start is not None:
        gaps.append(
            {
                "start_frame": start,
                "end_frame": int(valid.size) - 1,
                "length": int(valid.size) - start,
            }
        )
    return gaps


def _validity(valid: np.ndarray) -> dict[str, Any]:
    count = int(np.count_nonzero(valid))
    frame_count = int(valid.size)
    return {
        "valid_count": count,
        "invalid_count": frame_count - count,
        "valid_fraction": float(count / frame_count) if frame_count else 0.0,
        "invalid_gaps": _invalid_gaps(valid),
    }


def _transform_points(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Apply frame-aligned homogeneous transforms to ``(N,...,3)`` points."""

    points = np.asarray(points, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    if (
        points.ndim < 2
        or points.shape[0] != transforms.shape[0]
        or points.shape[-1] != 3
    ):
        raise ContractError(
            "Frame-aligned points must have shape (N,...,3) matching transforms"
        )
    rotations = transforms[:, :3, :3]
    translations = transforms[:, :3, 3]
    flattened = points.reshape(points.shape[0], -1, 3)
    transformed = np.einsum("nij,nkj->nki", rotations, flattened)
    transformed += translations[:, None, :]
    return transformed.reshape(points.shape)


def _to_world(
    points: np.ndarray, coordinate_frame: str, camera: TacoCamera
) -> np.ndarray:
    if coordinate_frame == "world":
        return np.asarray(points, dtype=np.float64)
    if coordinate_frame != "camera":
        raise ContractError(f"Unsupported coordinate frame {coordinate_frame!r}")
    camera_to_world = np.linalg.inv(camera.world_to_camera)
    return _transform_points(points, camera_to_world)


def _to_camera(points_world: np.ndarray, camera: TacoCamera) -> np.ndarray:
    return _transform_points(points_world, camera.world_to_camera)


def _temporal_joint_steps(
    joints_world: np.ndarray | None, valid: np.ndarray
) -> dict[str, Any]:
    if joints_world is None:
        return {
            "status": "not_computed",
            "reason": "joints_3d_missing",
        }
    adjacent = np.asarray(valid[:-1] & valid[1:], dtype=np.bool_)
    pair_indices = np.flatnonzero(adjacent)
    if pair_indices.size:
        steps = np.linalg.norm(
            joints_world[1:][adjacent] - joints_world[:-1][adjacent], axis=2
        )
        frame_mean = np.mean(steps, axis=1)
    else:
        steps = np.empty((0, joints_world.shape[1]), dtype=np.float64)
        frame_mean = np.empty(0, dtype=np.float64)
    return {
        "status": "computed",
        "definition": (
            "Euclidean displacement in the calibrated world frame between "
            "consecutive valid frames; gaps are never bridged"
        ),
        "adjacent_frame_pair_count": int(pair_indices.size),
        "adjacent_frame_starts": [int(index) for index in pair_indices],
        "per_joint_step_m": _summary(steps),
        "per_frame_mean_joint_step_m": _summary(frame_mean),
    }


def _not_computed(reason: str) -> dict[str, str]:
    return {"status": "not_computed", "reason": reason}


def _evaluate_side(
    *,
    side: str,
    prediction: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    prediction_frame: str,
    ground_truth_frame: str,
    camera: TacoCamera,
) -> dict[str, Any]:
    prediction_valid = np.asarray(prediction[f"{side}_valid"], dtype=np.bool_)
    ground_truth_valid = np.asarray(ground_truth[f"{side}_valid"], dtype=np.bool_)
    paired_valid = prediction_valid & ground_truth_valid

    prediction_wrist_world = _to_world(
        prediction[f"{side}_wrist_position"], prediction_frame, camera
    )
    ground_truth_wrist_world = _to_world(
        ground_truth[f"{side}_wrist_position"], ground_truth_frame, camera
    )
    wrist_errors = np.linalg.norm(
        prediction_wrist_world[paired_valid] - ground_truth_wrist_world[paired_valid],
        axis=1,
    )

    prediction_joint_key = f"{side}_joints_3d"
    ground_truth_joint_key = f"{side}_joints_3d"
    prediction_joints_world = (
        _to_world(prediction[prediction_joint_key], prediction_frame, camera)
        if prediction_joint_key in prediction
        else None
    )
    ground_truth_joints_world = (
        _to_world(ground_truth[ground_truth_joint_key], ground_truth_frame, camera)
        if ground_truth_joint_key in ground_truth
        else None
    )

    missing_joint_inputs = []
    if prediction_joints_world is None:
        missing_joint_inputs.append(f"prediction.{prediction_joint_key}")
    if ground_truth_joints_world is None:
        missing_joint_inputs.append(f"ground_truth.{ground_truth_joint_key}")
    semantics_compatible = not missing_joint_inputs
    if semantics_compatible:
        assert prediction_joints_world is not None
        assert ground_truth_joints_world is not None
        joint_errors = np.linalg.norm(
            prediction_joints_world[paired_valid]
            - ground_truth_joints_world[paired_valid],
            axis=2,
        )
        per_frame_3d = np.mean(joint_errors, axis=1)
        joint_3d: dict[str, Any] = {
            "status": "computed",
            "joint_error_m": _summary(joint_errors),
            "per_frame_mpjpe_m": _summary(per_frame_3d),
        }

        prediction_camera = _to_camera(prediction_joints_world, camera)
        ground_truth_camera = _to_camera(ground_truth_joints_world, camera)
        prediction_pixels, prediction_in_front = project_camera_points(
            prediction_camera, camera.intrinsic
        )
        ground_truth_pixels, ground_truth_in_front = project_camera_points(
            ground_truth_camera, camera.intrinsic
        )
        joint_projectable = (
            paired_valid[:, None] & prediction_in_front & ground_truth_in_front
        )
        pixel_errors = np.linalg.norm(prediction_pixels - ground_truth_pixels, axis=2)
        projected_errors = pixel_errors[joint_projectable]
        frame_projectable = np.any(joint_projectable, axis=1)
        per_frame_2d = np.asarray(
            [
                np.mean(pixel_errors[index][joint_projectable[index]])
                for index in np.flatnonzero(frame_projectable)
            ],
            dtype=np.float64,
        )
        projected_2d: dict[str, Any] = {
            "status": "computed" if projected_errors.size else "not_computed",
            "definition": (
                "Euclidean pixel error after independently projecting both "
                "3-D MANO-order tracks through supplied TACO K and per-frame "
                "world-to-camera calibration; embedded joints_2d is not used"
            ),
            "jointly_projectable_frame_count": int(np.count_nonzero(frame_projectable)),
            "jointly_projectable_joint_count": int(projected_errors.size),
            "joint_error_px": _summary(projected_errors),
            "per_frame_mpjpe_px": _summary(per_frame_2d),
        }
        if not projected_errors.size:
            projected_2d["reason"] = "no_joint_pair_has_positive_calibrated_depth"
    else:
        reason = "missing MANO-order joint arrays: " + ", ".join(missing_joint_inputs)
        joint_3d = _not_computed(reason)
        projected_2d = _not_computed(reason)

    return {
        "joint_semantics": {
            "compatible": semantics_compatible,
            "basis": (
                "Both archives validate against v2d.inpainting.tracking/v1, "
                "whose joints_3d arrays are defined in MANO order"
                if semantics_compatible
                else "The common tracking contract cannot establish joint semantics without both joints_3d arrays"
            ),
        },
        "validity": {
            "prediction": _validity(prediction_valid),
            "ground_truth": _validity(ground_truth_valid),
            "paired": _validity(paired_valid),
        },
        "metrics": {
            "wrist_3d_error_m": {
                "status": "computed" if wrist_errors.size else "not_computed",
                "definition": (
                    "Euclidean wrist-position error in the calibrated world frame"
                ),
                **_summary(wrist_errors),
                **({} if wrist_errors.size else {"reason": "no_paired_valid_frames"}),
            },
            "joint_3d_mpjpe": joint_3d,
            "projected_2d_mpjpe": projected_2d,
            "temporal_joint_step": {
                "prediction": _temporal_joint_steps(
                    prediction_joints_world, prediction_valid
                ),
                "ground_truth": _temporal_joint_steps(
                    ground_truth_joints_world, ground_truth_valid
                ),
            },
        },
    }


def evaluate_arrays(
    *,
    prediction: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    camera: TacoCamera,
    sequence_id: str,
) -> dict[str, Any]:
    """Return deterministic metrics for already-loaded tracking archives."""

    prediction_count = validate_tracking_arrays(prediction)
    ground_truth_count = validate_tracking_arrays(ground_truth)
    if prediction_count != ground_truth_count:
        raise ContractError(
            f"Prediction has {prediction_count} frames but ground truth has {ground_truth_count}"
        )
    if camera.frame_count != prediction_count:
        raise ContractError(
            f"Camera has {camera.frame_count} frames but tracks have {prediction_count}"
        )
    if not sequence_id:
        raise ValueError("sequence_id must not be empty")

    prediction_tracker = _scalar_text(prediction["tracker"], "prediction.tracker")
    ground_truth_tracker = _scalar_text(ground_truth["tracker"], "ground_truth.tracker")
    if prediction_tracker not in LEARNED_TRACKERS:
        raise ContractError(
            f"Prediction tracker must be one of {LEARNED_TRACKERS}, got {prediction_tracker!r}"
        )
    if ground_truth_tracker != "ground_truth":
        raise ContractError(
            "Reference archive must declare tracker='ground_truth'; "
            f"got {ground_truth_tracker!r}"
        )
    if not np.array_equal(prediction["frame_indices"], ground_truth["frame_indices"]):
        raise ContractError("Prediction and ground-truth frame_indices differ")

    prediction_frame = _scalar_text(
        prediction["coordinate_frame"], "prediction.coordinate_frame"
    )
    ground_truth_frame = _scalar_text(
        ground_truth["coordinate_frame"], "ground_truth.coordinate_frame"
    )
    return {
        "schema_version": EVALUATION_SCHEMA,
        "state": "complete",
        "sequence_id": sequence_id,
        "ground_truth_policy": {
            "usage": "evaluation_only",
            "enters_prediction_or_retargeting": False,
        },
        "frame_alignment": {
            "frame_count": prediction_count,
            "frame_indices": "contiguous_0_through_N_minus_1",
        },
        "evaluation_coordinates": {
            "metric_3d_and_temporal": "calibrated_taco_world",
            "pixel_projection": "opencv_camera_x_right_y_down_z_forward",
            "prediction_input": prediction_frame,
            "ground_truth_input": ground_truth_frame,
        },
        "trackers": {
            "prediction": prediction_tracker,
            "ground_truth": ground_truth_tracker,
        },
        "sides": {
            side: _evaluate_side(
                side=side,
                prediction=prediction,
                ground_truth=ground_truth,
                prediction_frame=prediction_frame,
                ground_truth_frame=ground_truth_frame,
                camera=camera,
            )
            for side in SIDES
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        text = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def evaluate_files(
    *,
    prediction_path: Path,
    ground_truth_path: Path,
    video_path: Path,
    intrinsic_path: Path,
    extrinsic_path: Path,
    output_path: Path,
    sequence_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate, evaluate, fingerprint, and atomically commit one comparison."""

    inputs = {
        "prediction": prediction_path.expanduser().resolve(),
        "ground_truth": ground_truth_path.expanduser().resolve(),
        "video": video_path.expanduser().resolve(),
        "intrinsic": intrinsic_path.expanduser().resolve(),
        "extrinsic": extrinsic_path.expanduser().resolve(),
    }
    output_path = output_path.expanduser().resolve()
    if output_path in inputs.values():
        raise ValueError("Evaluation output must not alias any input")
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} input: {path}")
    geometry: VideoGeometry = probe_video(inputs["video"])
    camera = load_taco_camera(
        inputs["intrinsic"],
        inputs["extrinsic"],
        expected_frames=geometry.frame_count,
        width=geometry.width,
        height=geometry.height,
    )
    with np.load(inputs["prediction"], allow_pickle=False) as prediction_archive:
        prediction = dict(prediction_archive)
    with np.load(inputs["ground_truth"], allow_pickle=False) as ground_truth_archive:
        ground_truth = dict(ground_truth_archive)
    validate_tracking_arrays(prediction, expected_frames=geometry.frame_count)
    validate_tracking_arrays(ground_truth, expected_frames=geometry.frame_count)
    payload = evaluate_arrays(
        prediction=prediction,
        ground_truth=ground_truth,
        camera=camera,
        sequence_id=sequence_id,
    )
    payload["inputs"] = {
        "prediction": _artifact(inputs["prediction"]),
        "ground_truth": _artifact(inputs["ground_truth"]),
        "video": {
            **_artifact(inputs["video"]),
            "geometry": geometry.as_dict(),
        },
        "camera": {
            "intrinsic": {
                **_artifact(inputs["intrinsic"]),
                "matrix": camera.intrinsic.tolist(),
            },
            "world_to_camera": {
                **_artifact(inputs["extrinsic"]),
                "shape": list(camera.world_to_camera.shape),
            },
        },
    }
    _atomic_json(output_path, payload, overwrite=overwrite)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--intrinsic", required=True, type=Path)
    parser.add_argument("--extrinsic", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = evaluate_files(
        prediction_path=args.prediction,
        ground_truth_path=args.ground_truth,
        video_path=args.video,
        intrinsic_path=args.intrinsic,
        extrinsic_path=args.extrinsic,
        output_path=args.output,
        sequence_id=args.sequence_id,
        overwrite=args.overwrite,
    )
    print(
        f"Evaluated {result['trackers']['prediction']} against ground truth "
        f"for {result['sequence_id']} -> {args.output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
