"""Detect and blur faces in every preprocessed multiview RGB stream."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import cv2
from omegaconf import OmegaConf

from v2d.common.video import FrameSource, FrameWriter
from v2d.mv.rig import RigConfig

try:
    from .face_blur import blur_faces
    from .face_detection import FaceDetection, FaceDetector, YuNetFaceDetector
    from .download_weights import MODEL_FILENAME
    from .temporal_filter import filter_detections
except ImportError:  # Support direct imports in lightweight tests.
    from face_blur import blur_faces
    from face_detection import FaceDetection, FaceDetector, YuNetFaceDetector
    from download_weights import MODEL_FILENAME
    from temporal_filter import filter_detections


TEMPORARY_INPUT_FAILURE_EXIT_CODE = 75


class MissingCameraInputsError(FileNotFoundError):
    """A complete multiview input set was not available at task startup."""


@dataclass(frozen=True)
class CameraJob:
    camera_name: str
    rgb_path: str
    image_output_path: str
    video_output_path: str
    detection_output_path: str
    model_path: str
    config: dict


@dataclass(frozen=True)
class CameraResult:
    camera_name: str
    frame_count: int
    detected_count: int
    filtered_count: int
    interpolated_count: int
    frames_without_filtered_faces: int
    longest_no_detection_run: int
    elapsed_seconds: float


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            # tempfile.mkstemp creates mode 0600 files. OSMO's output uploader
            # may run as a different uid than the inference process, so make
            # committed reports readable before publishing them.
            os.fchmod(stream.fileno(), 0o644)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _longest_empty_run(frames: list[list[FaceDetection]]) -> int:
    longest = current = 0
    for detections in frames:
        if detections:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _report_progress(
    camera_name: str,
    stage: str,
    completed_units: int,
    total_units: int,
    next_percent: int,
) -> int:
    """Print each crossed 10% threshold and return the next threshold."""
    if total_units <= 0:
        return next_percent
    completed_percent = completed_units * 100 // total_units
    while next_percent <= 100 and completed_percent >= next_percent:
        print(
            f"Face detector {camera_name}: {next_percent}% ({stage})",
            flush=True,
        )
        next_percent += 10
    return next_percent


def _image_writer(path: Path) -> FrameWriter:
    """Use compressed HDF5 for .h5 outputs and PNG directories otherwise."""
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return FrameWriter.from_path(
            path,
            compression="gzip",
            compression_opts=6,
            shuffle=False,
        )
    return FrameWriter.from_path(path)


def process_camera(
    *,
    camera_name: str,
    rgb_path: Path,
    image_output_path: Path,
    video_output_path: Path,
    detection_output_path: Path,
    detector: FaceDetector,
    config: dict,
) -> CameraResult:
    """Run the two-pass offline detector/filter/writer pipeline for one camera."""
    started = time.perf_counter()
    with FrameSource.from_path(rgb_path) as source:
        stems = list(source.stems)
        frame_count = source.n_frames
        next_detection_percent = 10
        raw_untracked = []
        for index, frame in enumerate(source.iter_frames()):
            raw_untracked.append(detector.detect(frame))
            next_detection_percent = _report_progress(
                camera_name,
                "detecting",
                index + 1,
                frame_count,
                next_detection_percent,
            )

    raw_frames, filtered_frames = filter_detections(
        raw_untracked,
        max_gap_frames=int(config["max_gap_frames"]),
        edge_fill_frames=int(config["edge_fill_frames"]),
        min_iou=float(config["track_min_iou"]),
        max_center_distance=float(config["track_max_center_distance"]),
        median_window=int(config["median_window"]),
        ema_alpha=float(config["ema_alpha"]),
    )

    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    video_output_path.parent.mkdir(parents=True, exist_ok=True)
    with FrameSource.from_path(rgb_path) as source, _image_writer(
        image_output_path
    ) as image_writer, FrameWriter.from_path(
        video_output_path,
        fps=int(config["fps"]),
        crf=int(config["video_crf"]),
    ) as video_writer:
        next_writing_percent = 10
        for index, frame in enumerate(source.iter_frames()):
            blurred = blur_faces(
                frame,
                filtered_frames[index],
                ellipse_scale=float(config["ellipse_scale"]),
                feather_fraction=float(config["feather_fraction"]),
                blur_sigma_fraction=float(config["blur_sigma_fraction"]),
            )
            image_writer.write_frame(blurred, stem=stems[index])
            video_writer.write_frame(blurred)
            next_writing_percent = _report_progress(
                camera_name,
                "writing",
                index + 1,
                frame_count,
                next_writing_percent,
            )

    _atomic_write_json(
        detection_output_path,
        {
            "schema": "v2d.face_detector.detections.v1",
            "camera_name": camera_name,
            "frame_count": len(stems),
            "frames": [
                {
                    "index": index,
                    "stem": stems[index],
                    "raw": [item.to_json() for item in raw_frames[index]],
                    "filtered": [item.to_json() for item in filtered_frames[index]],
                }
                for index in range(len(stems))
            ],
        },
    )
    detected_count = sum(len(frame) for frame in raw_frames)
    filtered_count = sum(len(frame) for frame in filtered_frames)
    interpolated_count = sum(
        item.provenance == "interpolated"
        for frame in filtered_frames
        for item in frame
    )
    return CameraResult(
        camera_name=camera_name,
        frame_count=len(stems),
        detected_count=detected_count,
        filtered_count=filtered_count,
        interpolated_count=interpolated_count,
        frames_without_filtered_faces=sum(not frame for frame in filtered_frames),
        longest_no_detection_run=_longest_empty_run(filtered_frames),
        elapsed_seconds=time.perf_counter() - started,
    )


def _process_camera_worker(job: CameraJob) -> CameraResult:
    cv2.setNumThreads(int(job.config["opencv_threads"]))
    detector = YuNetFaceDetector(
        job.model_path,
        score_threshold=float(job.config["score_threshold"]),
        nms_threshold=float(job.config["nms_threshold"]),
        top_k=int(job.config["top_k"]),
        scales=tuple(float(value) for value in job.config["detection_scales"]),
    )
    return process_camera(
        camera_name=job.camera_name,
        rgb_path=Path(job.rgb_path),
        image_output_path=Path(job.image_output_path),
        video_output_path=Path(job.video_output_path),
        detection_output_path=Path(job.detection_output_path),
        detector=detector,
        config=job.config,
    )


def _resolve_camera_paths(cfg) -> dict[str, Path]:
    """Resolve every camera path declared by the configured multiview rig."""
    rig = RigConfig(str(cfg.rig_config))
    paths: dict[str, Path] = {}
    for camera in rig.get_all_cameras():
        path = Path(cfg.rgb_path_template.format(cam_name=camera.name))
        if not path.exists() and path.suffix in (".h5", ".hdf5"):
            legacy_dir = path.with_suffix("")
            if legacy_dir.is_dir():
                path = legacy_dir
        paths[camera.name] = path
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise MissingCameraInputsError(
            "Missing RGB inputs for configured rig cameras:\n" + details
        )
    return paths


def mv_detect_and_blur_faces(cfg) -> dict:
    model_dir = Path(cfg.model_dir)
    model_path = model_dir / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"YuNet model not found: {model_path}")
    model_sha256 = _sha256_file(model_path)
    expected_model_sha256 = str(cfg.get("model_sha256", "")).strip().lower()
    if expected_model_sha256 and model_sha256 != expected_model_sha256:
        raise ValueError(
            "YuNet model checksum mismatch: "
            f"expected {expected_model_sha256}, got {model_sha256}"
        )
    output_dir = Path(cfg.output_dir)
    config = OmegaConf.to_container(cfg, resolve=True)
    camera_paths = _resolve_camera_paths(cfg)
    jobs = [
        CameraJob(
            camera_name=camera_name,
            rgb_path=str(rgb_path),
            image_output_path=cfg.output_image_path_template.format(
                cam_name=camera_name
            ),
            video_output_path=cfg.output_video_path_template.format(
                cam_name=camera_name
            ),
            detection_output_path=cfg.detection_path_template.format(
                cam_name=camera_name
            ),
            model_path=str(model_path),
            config=config,
        )
        for camera_name, rgb_path in camera_paths.items()
    ]
    results: list[CameraResult] = []
    worker_count = min(max(1, int(cfg.camera_workers)), len(jobs))
    if worker_count == 1:
        results = [_process_camera_worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_process_camera_worker, job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                print(
                    f"Face detector {result.camera_name}: "
                    f"frames={result.frame_count}, detections={result.detected_count}, "
                    f"interpolated={result.interpolated_count}, "
                    f"elapsed={result.elapsed_seconds:.2f}s"
                )
                results.append(result)

    results.sort(key=lambda result: result.camera_name)
    manifest = {
        "schema": "v2d.face_detector.manifest.v1",
        "model": {
            "filename": model_path.name,
            "sha256": model_sha256,
        },
        "config": config,
        "camera_count": len(results),
        "cameras": [asdict(result) for result in results],
    }
    _atomic_write_json(output_dir / "face_detector_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and temporally stabilize faces, then write blurred RGB"
    )
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument(
        "--model_dir",
        required=True,
        help=f"Directory containing {MODEL_FILENAME}",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_path")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).with_suffix(".yaml"))
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    cfg = OmegaConf.merge(
        cfg,
        {
            "rgb_dir": args.rgb_dir,
            "model_dir": args.model_dir,
            "output_dir": args.output_dir,
        },
    )
    try:
        mv_detect_and_blur_faces(cfg)
    except MissingCameraInputsError as exc:
        # EX_TEMPFAIL lets campaign reconciliation distinguish an incomplete
        # upstream handoff from model/configuration/code failures. The
        # orchestrator retries it only when mv_preprocess completed in the
        # same workflow, and still applies its bounded retry policy.
        print(str(exc), file=sys.stderr)
        raise SystemExit(TEMPORARY_INPUT_FAILURE_EXIT_CODE) from exc


if __name__ == "__main__":
    main()
