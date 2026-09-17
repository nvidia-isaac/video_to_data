"""CARI4D-compatible lossless FFV1 Matroska sidecars."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterator

import av
import h5py
import numpy as np


RGB_SCHEMA = "cari4d.rgb_ffv1_sidecar.v1"
DEPTH_SCHEMA = "cari4d.sensor_depth_ffv1_sidecar.v1"
SUPPORTED_SCHEMAS = {
    RGB_SCHEMA: {
        "kind": "rgb",
        "pixel_format": "bgr0",
        "input_format": "rgb24",
        "gop": 1,
        "dtype": np.dtype("uint8"),
        "channels": 3,
        "source_layout": "legacy-dense",
    },
    DEPTH_SCHEMA: {
        "kind": "depth",
        "pixel_format": "gray16le",
        "input_format": "gray16le",
        "gop": 32,
        "dtype": np.dtype("uint16"),
        "channels": 1,
        "source_layout": "sensor-depth-dense",
    },
}
SCHEMA_BY_KIND = {config["kind"]: schema for schema, config in SUPPORTED_SCHEMAS.items()}
FFV1_LEVEL = 3
FFV1_CODER = 1
FFV1_CONTEXT = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _fraction(value: Any) -> Fraction:
    """Convert persisted numeric FPS values without huge float denominators."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (float, np.floating)):
        return Fraction(str(float(value))).limit_denominator(1_000_000)
    return Fraction(int(value))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_logical_hash(digest, frame: np.ndarray, kind: str) -> None:
    array = np.asarray(frame)
    if kind == "depth":
        array = array.astype(np.dtype("<u2"), copy=False)
    digest.update(np.ascontiguousarray(array).tobytes())


def _source_frames(
    dataset,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> Iterator[np.ndarray]:
    end = int(dataset.shape[0]) if end_frame is None else int(end_frame)
    for index in range(int(start_frame), end):
        yield dataset[index]


def _source_config(dataset, kind: str, frame_encoding: str | None) -> tuple[dict, int, int, int]:
    if kind not in SCHEMA_BY_KIND:
        raise ValueError(f"Unsupported FFV1 kind: {kind}")
    if frame_encoding is not None:
        raise ValueError(
            "CARI4D FFV1 requires a dense HDF5 source; encoded HDF5 frames "
            f"({frame_encoding!r}) cannot be claimed exact to the original dense data"
        )
    config = SUPPORTED_SCHEMAS[SCHEMA_BY_KIND[kind]]
    expected_ndim = 4 if kind == "rgb" else 3
    if dataset.ndim != expected_ndim or dataset.dtype != config["dtype"]:
        expected_shape = "(N, H, W, 3)" if kind == "rgb" else "(N, H, W)"
        raise TypeError(
            f"{kind} FFV1 input must be {config['dtype']} with shape {expected_shape}; "
            f"got dtype={dataset.dtype}, shape={dataset.shape}"
        )
    if kind == "rgb" and dataset.shape[-1] != 3:
        raise TypeError(f"RGB FFV1 input must have three channels; got {dataset.shape}")
    return config, int(dataset.shape[0]), int(dataset.shape[1]), int(dataset.shape[2])


def _measure_random_access(
    sidecar_path: Path,
    frame_pts: list[int],
    keyframe_indices: list[int],
    *,
    kind: str,
) -> dict[str, Any]:
    samples = sorted(
        {
            index
            for index in (0, 1, 31, 32, len(frame_pts) // 2, len(frame_pts) - 1)
            if 0 <= index < len(frame_pts)
        }
    )
    pts_to_index = {pts: index for index, pts in enumerate(frame_pts)}
    timings_ms: list[float] = []
    output_format = "rgb24" if kind == "rgb" else "gray16le"
    for target in samples:
        start = time.perf_counter()
        keyframe = max(index for index in keyframe_indices if index <= target)
        found = False
        with av.open(str(sidecar_path)) as container:
            stream = container.streams.video[0]
            container.seek(
                frame_pts[keyframe], stream=stream, backward=True, any_frame=False
            )
            for frame in container.decode(stream):
                index = pts_to_index.get(int(frame.pts))
                if index == target:
                    frame.to_ndarray(format=output_format)
                    found = True
                    break
                if index is not None and index > target:
                    break
        if not found:
            raise RuntimeError(f"Unable to benchmark FFV1 random frame {target}")
        timings_ms.append((time.perf_counter() - start) * 1000)
    return {
        "random_access_sample_indices": samples,
        "random_access_ms": timings_ms,
        "random_access_p50_ms": float(np.percentile(timings_ms, 50)),
        "random_access_p95_ms": float(np.percentile(timings_ms, 95)),
    }


def read_ffv1_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Read and validate a committed CARI4D FFV1 metadata H5."""
    metadata_path = Path(metadata_path)
    with h5py.File(metadata_path, "r") as metadata:
        schema = _text(metadata.attrs.get("schema", ""))
        if schema not in SUPPORTED_SCHEMAS:
            raise ValueError(f"Not a supported CARI4D FFV1 metadata file: {metadata_path}")
        config = SUPPORTED_SCHEMAS[schema]
        if not bool(metadata.attrs.get("complete", False)):
            raise ValueError("CARI4D FFV1 sidecar metadata is not complete")
        if (
            _text(metadata.attrs.get("container", "")) != "matroska"
            or _text(metadata.attrs.get("encoding", "")) != "ffv1"
            or _text(metadata.attrs.get("encoded_pixel_format", ""))
            != config["pixel_format"]
            or int(metadata.attrs.get("gop", 0)) != config["gop"]
            or int(metadata.attrs.get("ffv1_level", -1)) != FFV1_LEVEL
            or int(metadata.attrs.get("ffv1_coder", -1)) != FFV1_CODER
            or int(metadata.attrs.get("ffv1_context", -1)) != FFV1_CONTEXT
            or int(metadata.attrs.get("ffv1_slicecrc", 0)) != 1
        ):
            raise ValueError("CARI4D FFV1 codec metadata is inconsistent")

        logical_stem = _text(metadata.attrs.get("logical_stem", ""))
        if logical_stem != metadata_path.stem:
            raise ValueError("CARI4D FFV1 logical stem must match the metadata filename")
        sidecar_name = _text(metadata.attrs.get("sidecar_basename", ""))
        sidecar_rel = Path(sidecar_name)
        if sidecar_rel.is_absolute() or len(sidecar_rel.parts) != 1:
            raise ValueError("CARI4D FFV1 sidecar basename must be a sibling filename")
        pattern = re.compile(rf"{re.escape(logical_stem)}\.ffv1\.([0-9a-f]{{64}})\.mkv")
        match = pattern.fullmatch(sidecar_name)
        if match is None:
            raise ValueError("CARI4D FFV1 sidecar basename is not content-addressed")
        sidecar_sha256 = _text(metadata.attrs.get("sidecar_sha256", ""))
        if not _SHA256_RE.fullmatch(sidecar_sha256) or match.group(1) != sidecar_sha256:
            raise ValueError("CARI4D FFV1 sidecar basename digest does not match metadata")
        sidecar_path = metadata_path.parent / sidecar_rel
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"CARI4D FFV1 sidecar not found: {sidecar_path}")
        sidecar_bytes = int(metadata.attrs.get("sidecar_bytes", -1))
        if sidecar_path.stat().st_size != sidecar_bytes:
            raise ValueError("CARI4D FFV1 sidecar size does not match metadata")

        n_frames = int(metadata.attrs.get("n_frames", -1))
        frame_count = int(metadata.attrs.get("frame_count", -1))
        height = int(metadata.attrs.get("height", 0))
        width = int(metadata.attrs.get("width", 0))
        dtype = np.dtype(_text(metadata.attrs.get("source_dtype", "")))
        frame_rate = _fraction(metadata.attrs.get("frame_rate", 0))
        if (
            n_frames <= 0
            or frame_count != n_frames
            or height <= 0
            or width <= 0
            or dtype != config["dtype"]
            or frame_rate <= 0
            or not bool(metadata.attrs.get("exact_to_original_dense", False))
        ):
            raise ValueError("CARI4D FFV1 source metadata is inconsistent")
        if config["kind"] == "rgb" and int(metadata.attrs.get("channels", 0)) != 3:
            raise ValueError("CARI4D RGB metadata must declare three channels")
        source_logical_sha256 = _text(metadata.attrs.get("source_logical_sha256", ""))
        if not _SHA256_RE.fullmatch(source_logical_sha256):
            raise ValueError("CARI4D FFV1 source logical digest is invalid")

        try:
            stems = json.loads(_text(metadata.attrs["stems"]))
            frame_pts = metadata["frame_pts"][:].astype(np.int64).tolist()
            keyframes = metadata["keyframe_indices"][:].astype(np.int64).tolist()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("CARI4D FFV1 frame metadata is invalid") from exc
        if (
            not isinstance(stems, list)
            or len(stems) != n_frames
            or not all(isinstance(stem, str) for stem in stems)
            or len(frame_pts) != n_frames
            or frame_pts != sorted(set(frame_pts))
            or (frame_pts and frame_pts[0] != 0)
            or keyframes != list(range(0, n_frames, config["gop"]))
        ):
            raise ValueError("CARI4D FFV1 frame indexes are inconsistent")

        return {
            "metadata_path": metadata_path,
            "schema": schema,
            "sidecar_path": sidecar_path,
            "sidecar_basename": sidecar_name,
            "sidecar_sha256": sidecar_sha256,
            "sidecar_bytes": sidecar_bytes,
            "source_logical_sha256": source_logical_sha256,
            "kind": config["kind"],
            "pixel_format": config["pixel_format"],
            "dtype": dtype,
            "n_frames": n_frames,
            "height": height,
            "width": width,
            "channels": config["channels"],
            "gop": config["gop"],
            "fps_num": frame_rate.numerator,
            "fps_den": frame_rate.denominator,
            "stems": stems,
            "frame_pts": frame_pts,
            "keyframe_indices": keyframes,
        }


def is_ffv1_sidecar_h5(path: str | Path) -> bool:
    try:
        with h5py.File(path, "r") as metadata:
            return _text(metadata.attrs.get("schema", "")) in SUPPORTED_SCHEMAS
    except (OSError, ValueError):
        return False


def verify_ffv1_sidecar(
    metadata_path: str | Path,
    *,
    verify_decoded_frames: bool = False,
) -> dict[str, Any]:
    """Verify committed MKV bytes and optionally decode every frame."""
    info = read_ffv1_metadata(metadata_path)
    actual_sidecar_hash = _hash_file(info["sidecar_path"])
    if actual_sidecar_hash != info["sidecar_sha256"]:
        raise ValueError("CARI4D FFV1 sidecar SHA-256 does not match metadata")
    result = {
        "sidecar_sha256": actual_sidecar_hash,
        "decoded_logical_sha256": None,
        "decoded_frames": None,
    }
    if verify_decoded_frames:
        digest = hashlib.sha256()
        decoded_count = 0
        output_format = "rgb24" if info["kind"] == "rgb" else "gray16le"
        with av.open(str(info["sidecar_path"])) as container:
            for frame in container.decode(video=0):
                array = frame.to_ndarray(format=output_format)
                _update_logical_hash(digest, array, info["kind"])
                decoded_count += 1
        if decoded_count != info["n_frames"]:
            raise ValueError("CARI4D FFV1 decoded frame count does not match metadata")
        result["decoded_logical_sha256"] = digest.hexdigest()
        result["decoded_frames"] = decoded_count
    return result


def transcode_h5_to_ffv1_sidecar(
    source_path: str | Path,
    metadata_path: str | Path,
    *,
    kind: str,
    fps: int | Fraction | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    reindex_stems: bool = False,
    measure_random_access: bool = True,
) -> dict[str, Any]:
    """Stream a dense HDF5 frame interval into an exact CARI4D FFV1 pair."""
    source_path = Path(source_path)
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    logical_stem = metadata_path.stem
    temp_sidecar = metadata_path.parent / f".{logical_stem}.ffv1.tmp.mkv"
    temp_metadata = metadata_path.parent / f".{metadata_path.name}.tmp"
    temp_sidecar.unlink(missing_ok=True)
    temp_metadata.unlink(missing_ok=True)

    encode_start = time.perf_counter()
    source_hash = hashlib.sha256()
    final_sidecar: Path | None = None
    try:
        with h5py.File(source_path, "r") as source:
            if "frames" not in source:
                raise ValueError(f"Missing frames dataset in {source_path}")
            dataset = source["frames"]
            source_encoding = source.attrs.get("frame_encoding")
            source_encoding = _text(source_encoding) if source_encoding is not None else None
            config, source_frames, height, width = _source_config(
                dataset, kind, source_encoding
            )
            source_start = int(start_frame)
            source_end = source_frames if end_frame is None else int(end_frame)
            if (
                source_start < 0
                or source_end <= source_start
                or source_end > source_frames
            ):
                raise ValueError(
                    f"Invalid FFV1 frame range [{source_start}, {source_end}) "
                    f"for {source_frames} source frames"
                )
            n_frames = source_end - source_start
            if n_frames == 0:
                raise ValueError("Cannot encode an empty CARI4D FFV1 sidecar")
            stems_attr = source.attrs.get("stems")
            stems = (
                json.loads(_text(stems_attr))
                if stems_attr is not None
                else [f"{index:06d}" for index in range(n_frames)]
            )
            if len(stems) != source_frames or not all(
                isinstance(stem, str) for stem in stems
            ):
                raise ValueError("Source stems do not match frame count")
            stems = stems[source_start:source_end]
            if reindex_stems:
                stems = [f"{index:06d}" for index in range(n_frames)]
            source_fps = source.attrs.get("fps")
            rate = _fraction(fps or source_fps or 30)
            if rate <= 0:
                raise ValueError("FFV1 frame rate must be positive")

            with av.open(str(temp_sidecar), "w", format="matroska") as output:
                stream = output.add_stream(
                    "ffv1",
                    rate=rate,
                    options={
                        "level": str(FFV1_LEVEL),
                        "coder": str(FFV1_CODER),
                        "context": str(FFV1_CONTEXT),
                        "slicecrc": "1",
                    },
                )
                stream.width = width
                stream.height = height
                stream.pix_fmt = config["pixel_format"]
                stream.codec_context.gop_size = config["gop"]
                for index, array in enumerate(_source_frames(
                    dataset,
                    start_frame=source_start,
                    end_frame=source_end,
                )):
                    _update_logical_hash(source_hash, array, kind)
                    frame = av.VideoFrame.from_ndarray(
                        array, format=config["input_format"]
                    )
                    frame.pts = index
                    frame.time_base = Fraction(rate.denominator, rate.numerator)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)

            encode_seconds = time.perf_counter() - encode_start
            decode_start = time.perf_counter()
            decoded_hash = hashlib.sha256()
            frame_pts: list[int] = []
            keyframe_indices: list[int] = []
            with av.open(str(temp_sidecar)) as encoded:
                video_stream = encoded.streams.video[0]
                if video_stream.codec_context.name != "ffv1":
                    raise ValueError("Encoded Matroska stream is not FFV1")
                if video_stream.codec_context.pix_fmt != config["pixel_format"]:
                    raise ValueError("Encoded FFV1 pixel format changed")
                for index, frame in enumerate(encoded.decode(video=0)):
                    decoded = frame.to_ndarray(format=config["input_format"])
                    _update_logical_hash(decoded_hash, decoded, kind)
                    if frame.pts is None:
                        raise ValueError("Encoded FFV1 frame has no presentation timestamp")
                    frame_pts.append(int(frame.pts))
                    if frame.key_frame:
                        keyframe_indices.append(index)
            decode_seconds = time.perf_counter() - decode_start
            if len(frame_pts) != n_frames:
                raise ValueError(
                    f"FFV1 frame count changed from {n_frames} to {len(frame_pts)}"
                )
            if decoded_hash.digest() != source_hash.digest():
                raise ValueError("CARI4D FFV1 exact round-trip verification failed")
            expected_keyframes = list(range(0, n_frames, config["gop"]))
            if keyframe_indices != expected_keyframes:
                raise ValueError(
                    f"FFV1 keyframes {keyframe_indices} do not match {expected_keyframes}"
                )

            random_access_stats = (
                _measure_random_access(
                    temp_sidecar,
                    frame_pts,
                    keyframe_indices,
                    kind=kind,
                )
                if measure_random_access
                else {}
            )
            sidecar_sha256 = _hash_file(temp_sidecar)
            final_sidecar = metadata_path.parent / (
                f"{logical_stem}.ffv1.{sidecar_sha256}.mkv"
            )
            os.replace(temp_sidecar, final_sidecar)

            frame_rate_value: int | float = (
                rate.numerator if rate.denominator == 1 else float(rate)
            )
            attrs: dict[str, Any] = {
                "complete": True,
                "container": "matroska",
                "encoded_pixel_format": config["pixel_format"],
                "encoding": "ffv1",
                "exact_to_original_dense": True,
                "ffv1_coder": FFV1_CODER,
                "ffv1_context": FFV1_CONTEXT,
                "ffv1_level": FFV1_LEVEL,
                "ffv1_slicecrc": 1,
                "frame_count": n_frames,
                "frame_rate": frame_rate_value,
                "gop": config["gop"],
                "height": height,
                "logical_stem": logical_stem,
                "n_frames": n_frames,
                "schema": SCHEMA_BY_KIND[kind],
                "sidecar_basename": final_sidecar.name,
                "sidecar_bytes": final_sidecar.stat().st_size,
                "sidecar_sha256": sidecar_sha256,
                "source_dtype": str(config["dtype"]),
                "source_h5_bytes": source_path.stat().st_size,
                "source_layout": config["source_layout"],
                "source_logical_sha256": source_hash.hexdigest(),
                "stems": json.dumps(stems),
                "width": width,
            }
            if kind == "rgb":
                attrs["channels"] = 3
            with h5py.File(temp_metadata, "w") as metadata:
                metadata.attrs.update(attrs)
                metadata.create_dataset(
                    "frame_pts", data=np.asarray(frame_pts, dtype=np.int64)
                )
                metadata.create_dataset(
                    "keyframe_indices",
                    data=np.asarray(keyframe_indices, dtype=np.int64),
                )
            os.replace(temp_metadata, metadata_path)
    finally:
        temp_sidecar.unlink(missing_ok=True)
        temp_metadata.unlink(missing_ok=True)

    if final_sidecar is None:
        raise RuntimeError("CARI4D FFV1 sidecar was not committed")
    metadata_bytes = metadata_path.stat().st_size
    sidecar_bytes = final_sidecar.stat().st_size
    return {
        "kind": f"{kind}_ffv1_sidecar",
        "source_frames": source_frames,
        "source_start_frame": source_start,
        "source_end_frame": source_end,
        "frames": n_frames,
        "width": width,
        "height": height,
        "gop": config["gop"],
        "pixel_format": config["pixel_format"],
        "source_bytes": source_path.stat().st_size,
        "metadata_bytes": metadata_bytes,
        "metadata_filename": metadata_path.name,
        "sidecar_bytes": sidecar_bytes,
        "sidecar_filename": final_sidecar.name,
        "sidecar_path": str(final_sidecar),
        "output_bytes": metadata_bytes + sidecar_bytes,
        "source_logical_sha256": source_hash.hexdigest(),
        "decoded_logical_sha256": decoded_hash.hexdigest(),
        "sidecar_sha256": sidecar_sha256,
        "exact_round_trip": True,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "sequential_decode_fps": n_frames / max(decode_seconds, 1e-9),
        **random_access_stats,
    }
