from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from prep.mhr_depth_backend import MOGE2_MODEL_ID, MOGE2_MODEL_REVISION, MOGE2_SOURCE_COMMIT, normalize_monocular_depth_backend


WILD_DEPTH_SCHEMA = "cari4d.mhr_wild_depth.v1"
DEFAULT_WILD_DEPTH_BACKEND = "moge2"
DEFAULT_MOGE2_BATCH_SIZE = 8


def normalize_wild_depth_backend(value: str) -> str:
    backend = normalize_monocular_depth_backend(value)
    if backend != DEFAULT_WILD_DEPTH_BACKEND:
        raise ValueError(f"Native MHR wild inference requires {DEFAULT_WILD_DEPTH_BACKEND}, got {value!r}")
    return backend


def normalized_to_pixel_intrinsics(value: np.ndarray, height: int, width: int) -> np.ndarray:
    intrinsics = np.asarray(value, dtype=np.float32).copy()
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"Normalized intrinsics must be finite [3,3], got {intrinsics.shape}")
    intrinsics[0] *= float(width)
    intrinsics[1] *= float(height)
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError(f"Pixel intrinsics have invalid focal lengths: {intrinsics}")
    return intrinsics


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_pickle(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class _NpyDepthWriter:
    def __init__(self, path: Path, frame_count: int, height: int, width: int):
        self.path = path
        self.values = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(frame_count, height, width))
        self.index = 0

    def write(self, value: np.ndarray) -> None:
        if self.index >= len(self.values):
            raise ValueError(f"Raw depth writer received more than {len(self.values)} frames")
        self.values[self.index] = value
        self.index += 1

    def close(self) -> None:
        if self.index != len(self.values):
            raise ValueError(f"Raw depth writer received {self.index} of {len(self.values)} frames")
        self.values.flush()
        del self.values

    def abort(self) -> None:
        self.values.flush()
        del self.values


class _VideoDepthWriter:
    def __init__(self, path: Path, frame_count: int, height: int, width: int, fps: float):
        from videoio import Uint16Writer

        self.writer = Uint16Writer(str(path), (width, height), fps=fps)

    def write(self, value: np.ndarray) -> None:
        self.writer.write(value)

    def close(self) -> None:
        self.writer.close()

    def abort(self) -> None:
        self.writer.close()


def _make_depth_writer(path: Path, frame_count: int, height: int, width: int, fps: float) -> _NpyDepthWriter | _VideoDepthWriter:
    if path.suffix.lower() == ".npy":
        return _NpyDepthWriter(path, frame_count, height, width)
    return _VideoDepthWriter(path, frame_count, height, width, fps)


class _MoGe2Backend:
    def __init__(self, device: torch.device, model_id: str, model_revision: str | None, local_files_only: bool):
        from moge.model.v2 import MoGeModel

        kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if model_revision is not None:
            kwargs["revision"] = model_revision
        self.model = MoGeModel.from_pretrained(model_id, **kwargs).to(device).eval()
        self.device = device

    def infer_batch(self, images_rgb: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not images_rgb:
            return []
        shapes = {tuple(np.asarray(image).shape) for image in images_rgb}
        if len(shapes) != 1:
            raise ValueError(f"MoGe 2 batch requires equal RGB shapes, got {sorted(shapes)}")
        height, width = np.asarray(images_rgb[0]).shape[:2]
        images = torch.from_numpy(np.stack(images_rgb, axis=0)).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        with torch.cuda.device(self.device) if self.device.type == "cuda" else torch.inference_mode():
            prediction = self.model.infer(images, resolution_level=9, force_projection=True, apply_mask=True, use_fp16=True)
        depths = prediction["depth"].detach().cpu().numpy().astype(np.float32)
        masks = prediction["mask"].detach().cpu().numpy().astype(bool)
        intrinsics_normalized = prediction["intrinsics"].detach().cpu().numpy().astype(np.float32)
        expected_shape = (len(images_rgb), height, width)
        if depths.shape != expected_shape or masks.shape != expected_shape or intrinsics_normalized.shape != (len(images_rgb), 3, 3):
            raise ValueError(f"MoGe 2 output shape differs: expected={expected_shape}, depth={depths.shape}, mask={masks.shape}, intrinsics={intrinsics_normalized.shape}")
        outputs = []
        for index in range(len(images_rgb)):
            valid = masks[index] & np.isfinite(depths[index]) & (depths[index] > 0.0)
            outputs.append((depths[index], valid, normalized_to_pixel_intrinsics(intrinsics_normalized[index], height, width)))
        return outputs

    def infer(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.infer_batch([image_rgb])[0]


def _load_backend(device: torch.device, local_files_only: bool) -> _MoGe2Backend:
    return _MoGe2Backend(device, MOGE2_MODEL_ID, MOGE2_MODEL_REVISION, local_files_only)


def _resolve_devices(device_name: str, device_names: Sequence[str] | None) -> tuple[torch.device, ...]:
    if device_names is not None:
        values = tuple(str(value) for value in device_names)
        if not values:
            raise ValueError("device_names must contain at least one device")
    elif device_name == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        values = tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    else:
        values = (str(device_name),)
    devices = tuple(torch.device(value) for value in values)
    if len({str(device) for device in devices}) != len(devices):
        raise ValueError(f"MoGe 2 devices must be unique, got {values}")
    if len(devices) > 1 and any(device.type != "cuda" for device in devices):
        raise ValueError(f"Multi-device MoGe 2 inference requires CUDA devices, got {values}")
    return devices


def load_wild_depth_report(path: str | Path, *, expected_depth: str | Path | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if not isinstance(report, dict) or report.get("schema") != WILD_DEPTH_SCHEMA or report.get("verdict") != "PASS":
        raise ValueError(f"Wild depth report is not a completed {WILD_DEPTH_SCHEMA} artifact: {path}")
    normalize_wild_depth_backend(report.get("backend", ""))
    expected_identity = {"modelId": MOGE2_MODEL_ID, "modelRevision": MOGE2_MODEL_REVISION, "sourceCommit": MOGE2_SOURCE_COMMIT}
    mismatches = [key for key, expected in expected_identity.items() if report.get(key) != expected]
    if mismatches:
        raise ValueError(f"Wild depth report differs from the pinned MoGe 2 identity in {mismatches}: {path}")
    if expected_depth is not None:
        expected_path = Path(expected_depth).resolve()
        reported = report.get("rawDepth")
        if not isinstance(reported, dict) or Path(str(reported.get("path", ""))).resolve() != expected_path:
            raise ValueError(f"Wild depth report does not identify {expected_path}")
        identity = _file_identity(expected_path)
        if int(reported.get("size", -1)) != identity["size"] or int(reported.get("mtime_ns", -1)) != identity["mtime_ns"]:
            raise ValueError(f"Wild depth video identity differs from its report: {expected_path}")
    return report


def initialize_mhr_wild_depth(video: str | Path, output_depth: str | Path, output_intrinsics: str | Path, output_report: str | Path, *, expected_frames: int | None = None, device_name: str = "cuda", device_names: Sequence[str] | None = None, batch_size: int = DEFAULT_MOGE2_BATCH_SIZE, local_files_only: bool = False, overwrite: bool = False) -> dict[str, Any]:
    backend = DEFAULT_WILD_DEPTH_BACKEND
    video, output_depth, output_intrinsics, output_report = map(lambda value: Path(value).resolve(), (video, output_depth, output_intrinsics, output_report))
    if not video.is_file():
        raise FileNotFoundError(video)
    outputs = (output_depth, output_intrinsics, output_report)
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Wild depth output already exists: {[str(path) for path in outputs if path.exists()]}")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    if batch_size <= 0:
        raise ValueError(f"MoGe 2 batch size must be positive, got {batch_size}")
    devices = _resolve_devices(device_name, device_names)
    if not torch.cuda.is_available() and any(device.type == "cuda" for device in devices):
        raise RuntimeError("Wild monocular depth initialization requires CUDA")
    inference_started = time.perf_counter()
    resolved_model_id, resolved_model_revision = MOGE2_MODEL_ID, MOGE2_MODEL_REVISION
    estimators = tuple(_load_backend(device, local_files_only) for device in devices)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_count = int(expected_frames) if expected_frames is not None else declared_frames
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise ValueError(f"Wild video has an invalid contract: frames={frame_count}, size={width}x{height}, fps={fps}")
    if expected_frames is not None and declared_frames > 0 and declared_frames != frame_count:
        raise ValueError(f"Wild video declares {declared_frames} frames, expected {frame_count}")
    temporary_depth = output_depth.with_name(f".{output_depth.stem}.{os.getpid()}.tmp{output_depth.suffix}")
    temporary_depth.unlink(missing_ok=True)
    writer: _NpyDepthWriter | _VideoDepthWriter | None = None
    published = False
    intrinsics_px = np.empty((frame_count, 3, 3), dtype=np.float32)
    valid_fractions = np.empty(frame_count, dtype=np.float32)
    depth_min_m = np.empty(frame_count, dtype=np.float32)
    depth_max_m = np.empty(frame_count, dtype=np.float32)
    observed = 0
    try:
        writer = _make_depth_writer(temporary_depth, frame_count, height, width, fps)
        executor = ThreadPoolExecutor(max_workers=len(estimators)) if len(estimators) > 1 else None
        try:
            while observed < frame_count:
                frames_rgb = []
                for _ in range(min(batch_size * len(estimators), frame_count - observed)):
                    readable, frame_bgr = capture.read()
                    if not readable:
                        raise ValueError(f"Wild video ended after {observed + len(frames_rgb)} frames; expected {frame_count}")
                    if frame_bgr.shape != (height, width, 3):
                        raise ValueError(f"Wild video frame {observed + len(frames_rgb)} shape changed to {frame_bgr.shape}")
                    frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                chunks = [frames_rgb[start:start + batch_size] for start in range(0, len(frames_rgb), batch_size)]
                if executor is None:
                    inferred = [estimators[0].infer_batch(chunks[0])]
                else:
                    inferred = [future.result() for future in [executor.submit(estimator.infer_batch, chunk) for estimator, chunk in zip(estimators, chunks)]]
                for depth, valid, intrinsics in [value for chunk in inferred for value in chunk]:
                    if depth.shape != (height, width) or valid.shape != depth.shape or intrinsics.shape != (3, 3):
                        raise ValueError(f"{backend} frame {observed} output shape mismatch: depth={depth.shape}, valid={valid.shape}, intrinsics={intrinsics.shape}")
                    valid_count = int(valid.sum())
                    if valid_count < 100:
                        raise ValueError(f"{backend} frame {observed} has only {valid_count} valid depth pixels")
                    maximum = float(depth[valid].max())
                    if maximum >= np.iinfo(np.uint16).max / 1000.0:
                        raise ValueError(f"{backend} frame {observed} exceeds the uint16 millimetre range: {maximum} m")
                    converted = np.zeros((height, width), dtype=np.uint16)
                    converted[valid] = (depth[valid] * 1000.0).astype(np.uint16)
                    writer.write(converted)
                    intrinsics_px[observed] = intrinsics
                    valid_fractions[observed] = float(valid.mean())
                    depth_min_m[observed] = float(depth[valid].min())
                    depth_max_m[observed] = maximum
                    observed += 1
                if observed == 1 or observed % 10 == 0 or observed == frame_count:
                    print("MHR_WILD_DEPTH_PROGRESS " + json.dumps({"backend": backend, "current": observed, "total": frame_count, "unit": "frames"}, sort_keys=True), flush=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        readable, _ = capture.read()
        if readable:
            raise ValueError(f"Wild video contains more than {frame_count} frames")
        writer.close()
        writer = None
        os.replace(temporary_depth, output_depth)
        first = intrinsics_px[0]
        intrinsics_payload = {"fx": float(first[0, 0]), "fy": float(first[1, 1]), "cx": float(first[0, 2]), "cy": float(first[1, 2]), "focals": intrinsics_px[:, 0, 0].tolist(), "per_frame_intrinsics": intrinsics_px.tolist(), "H": height, "W": width, "depth_backend": backend, "camera_policy": "first_frame_export_per_frame_depth", "model_id": resolved_model_id, "model_revision": resolved_model_revision, "source_commit": MOGE2_SOURCE_COMMIT, "batch_size": int(batch_size), "devices": [str(device) for device in devices]}
        _atomic_pickle(output_intrinsics, intrinsics_payload)
        elapsed_seconds = time.perf_counter() - inference_started
        report = {"schema": WILD_DEPTH_SCHEMA, "backend": backend, "modelId": resolved_model_id, "modelRevision": resolved_model_revision, "sourceCommit": MOGE2_SOURCE_COMMIT, "cameraPolicy": intrinsics_payload["camera_policy"], "batchSize": int(batch_size), "devices": [str(device) for device in devices], "deviceCount": len(devices), "sourceVideo": _file_identity(video), "frames": observed, "height": height, "width": width, "fps": fps, "elapsedSeconds": elapsed_seconds, "framesPerSecond": observed / elapsed_seconds, "firstFrameIntrinsicsPx": first.tolist(), "focalXPx": {"min": float(intrinsics_px[:, 0, 0].min()), "mean": float(intrinsics_px[:, 0, 0].mean()), "max": float(intrinsics_px[:, 0, 0].max())}, "focalYPx": {"min": float(intrinsics_px[:, 1, 1].min()), "mean": float(intrinsics_px[:, 1, 1].mean()), "max": float(intrinsics_px[:, 1, 1].max())}, "validFraction": {"min": float(valid_fractions.min()), "mean": float(valid_fractions.mean()), "max": float(valid_fractions.max())}, "depthM": {"min": float(depth_min_m.min()), "max": float(depth_max_m.max())}, "rawDepth": {**_file_identity(output_depth), "storage": "npy_uint16_mm" if output_depth.suffix.lower() == ".npy" else "uint16_video", "dtype": "uint16_mm", "frames": observed, "height": height, "width": width, "sha256": _sha256(output_depth)}, "intrinsicsFile": _file_identity(output_intrinsics), "verdict": "PASS"}
        _atomic_json(output_report, report)
        published = True
        return report
    finally:
        capture.release()
        if writer is not None:
            writer.abort()
        if not published:
            temporary_depth.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize native MHR wild-video depth with the pinned MoGe 2 model.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-depth", required=True)
    parser.add_argument("--output-intrinsics", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_MOGE2_BATCH_SIZE)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(initialize_mhr_wild_depth(args.video, args.output_depth, args.output_intrinsics, args.output_report, expected_frames=args.expected_frames, device_name=args.device, device_names=args.devices, batch_size=args.batch_size, local_files_only=args.local_files_only, overwrite=args.overwrite), sort_keys=True))


if __name__ == "__main__":
    main()
