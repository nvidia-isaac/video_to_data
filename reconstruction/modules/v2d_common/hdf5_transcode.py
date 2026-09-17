"""Streaming HDF5 transcoders used by export and storage migration."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, JpegImagePlugin


JPEG_ENCODING = "jpeg"
JPEG_QUALITY = 100
JPEG_SUBSAMPLING = "4:4:4"


def _attr_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def encode_rgb_jpeg(frame: np.ndarray) -> bytes:
    """Encode one RGB uint8 frame using the exported-dataset JPEG settings."""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise TypeError(
            "JPEG HDF5 encoding requires an RGB uint8 array with shape (H, W, 3); "
            f"got dtype={frame.dtype}, shape={frame.shape}"
        )
    buffer = BytesIO()
    Image.fromarray(frame, mode="RGB").save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,
        optimize=True,
    )
    return buffer.getvalue()


def decode_rgb_jpeg(payload: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Decode one JPEG byte payload to an RGB uint8 array."""
    if isinstance(payload, np.ndarray):
        payload = payload.tobytes()
    try:
        with Image.open(BytesIO(bytes(payload))) as image:
            decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Invalid JPEG frame payload: {exc}") from exc
    return decoded


def is_jpeg_hdf5(path: str | Path) -> bool:
    with h5py.File(path, "r") as h5_file:
        return _attr_text(h5_file.attrs.get("frame_encoding")) == JPEG_ENCODING


def has_hdf5_filters(
    path: str | Path,
    *,
    compression: str,
    compression_opts: Any,
    shuffle: bool,
) -> bool:
    with h5py.File(path, "r") as h5_file:
        dataset = h5_file["frames"]
        return (
            dataset.compression == compression
            and dataset.compression_opts == compression_opts
            and dataset.shuffle is shuffle
        )


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _normalize_frame_range(
    frame_count: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[int, int]:
    start = int(start_frame)
    end = frame_count if end_frame is None else int(end_frame)
    if start < 0 or end <= start or end > frame_count:
        raise ValueError(
            f"Invalid frame range [{start}, {end}) for {frame_count} frames"
        )
    return start, end


def _reindexed_stems(frame_count: int) -> str:
    return json.dumps([f"{index:06d}" for index in range(frame_count)])


def slice_h5_frames(
    source_path: str | Path,
    output_path: str | Path,
    *,
    start_frame: int,
    end_frame: int | None = None,
    verify_frames: bool = True,
) -> dict[str, Any]:
    """Copy a dense frame interval while preserving the source HDF5 filters.

    The retained frames are byte-identical after decoding, while their stems
    and frame-count metadata are rebased to an exported timeline beginning at
    zero.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as src, h5py.File(output_path, "w") as dst:
        if "frames" not in src:
            raise ValueError(f"Missing 'frames' dataset in {source_path}")
        source_dataset = src["frames"]
        if source_dataset.ndim < 1:
            raise TypeError("Frame slicing requires an HDF5 frames dataset")
        source_count = int(source_dataset.shape[0])
        start, end = _normalize_frame_range(
            source_count, start_frame, end_frame,
        )
        output_count = end - start
        output_shape = (output_count, *source_dataset.shape[1:])

        create_options: dict[str, Any] = {}
        if source_dataset.chunks is not None:
            create_options["chunks"] = (
                min(int(source_dataset.chunks[0]), output_count),
                *source_dataset.chunks[1:],
            )
        if source_dataset.compression is not None:
            create_options["compression"] = source_dataset.compression
            create_options["compression_opts"] = source_dataset.compression_opts
        if source_dataset.shuffle:
            create_options["shuffle"] = True
        if source_dataset.fletcher32:
            create_options["fletcher32"] = True
        if source_dataset.scaleoffset is not None:
            create_options["scaleoffset"] = source_dataset.scaleoffset

        output_dataset = dst.create_dataset(
            "frames",
            shape=output_shape,
            dtype=source_dataset.dtype,
            **create_options,
        )
        _copy_attrs(src, dst)
        _copy_attrs(source_dataset, output_dataset)
        for output_index, source_index in enumerate(range(start, end)):
            output_dataset[output_index] = source_dataset[source_index]

        dst.attrs["stems"] = _reindexed_stems(output_count)
        dst.attrs["n_frames"] = output_count
        if "frame_count" in dst.attrs:
            dst.attrs["frame_count"] = output_count
        if "frame_start" in dst.attrs:
            dst.attrs["frame_start"] = 0
        if "frame_end" in dst.attrs:
            dst.attrs["frame_end"] = output_count

    if verify_frames:
        with h5py.File(source_path, "r") as src, h5py.File(output_path, "r") as dst:
            for output_index, source_index in enumerate(range(start, end)):
                source_frame = np.asarray(src["frames"][source_index])
                output_frame = np.asarray(dst["frames"][output_index])
                if source_frame.dtype != output_frame.dtype or not np.array_equal(
                    source_frame, output_frame,
                ):
                    raise ValueError(
                        f"HDF5 frame slicing changed source frame {source_index}"
                    )

    return {
        "kind": "sliced_hdf5",
        "source_frames": source_count,
        "start_frame": start,
        "end_frame": end,
        "frames": output_count,
        "source_bytes": source_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
    }


def transcode_rgb_h5_to_jpeg_h5(
    source_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Stream a dense RGB HDF5 file into variable-length JPEG payloads."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    sample_count = 0
    max_absolute_error = 0

    with h5py.File(source_path, "r") as src, h5py.File(output_path, "w") as dst:
        if "frames" not in src:
            raise ValueError(f"Missing 'frames' dataset in {source_path}")
        source_dataset = src["frames"]
        if source_dataset.ndim != 4 or source_dataset.dtype != np.uint8:
            raise TypeError(
                "RGB export expects dense uint8 frames with shape (N, H, W, 3); "
                f"got dtype={source_dataset.dtype}, shape={source_dataset.shape}"
            )
        if source_dataset.shape[-1] != 3:
            raise TypeError(f"RGB export requires 3 channels; got {source_dataset.shape[-1]}")

        n_frames, height, width, _ = source_dataset.shape
        jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        output_dataset = dst.create_dataset(
            "frames",
            shape=(n_frames,),
            dtype=jpeg_dtype,
            chunks=(1,),
        )
        _copy_attrs(src, dst)
        _copy_attrs(source_dataset, output_dataset)

        for index in range(n_frames):
            source_frame = source_dataset[index]
            payload = encode_rgb_jpeg(source_frame)
            decoded = decode_rgb_jpeg(payload)
            if decoded.shape != source_frame.shape:
                raise ValueError(
                    f"JPEG frame {index} shape changed from {source_frame.shape} "
                    f"to {decoded.shape}"
                )
            with Image.open(BytesIO(payload)) as encoded_image:
                if JpegImagePlugin.get_sampling(encoded_image) != 0:
                    raise ValueError(f"JPEG frame {index} is not 4:4:4 subsampled")
            output_dataset[index] = np.frombuffer(payload, dtype=np.uint8)

            difference = decoded.astype(np.int16) - source_frame.astype(np.int16)
            absolute = np.abs(difference)
            squared_error_sum += float(np.square(difference.astype(np.float64)).sum())
            absolute_error_sum += float(absolute.sum())
            sample_count += int(source_frame.size)
            max_absolute_error = max(max_absolute_error, int(absolute.max(initial=0)))

        stems = src.attrs.get("stems")
        if stems is not None and len(json.loads(_attr_text(stems))) != n_frames:
            raise ValueError("The stems attribute does not match the RGB frame count")
        dst.attrs["n_frames"] = n_frames
        dst.attrs["width"] = width
        dst.attrs["height"] = height
        dst.attrs["frame_encoding"] = JPEG_ENCODING
        dst.attrs["jpeg_quality"] = JPEG_QUALITY
        dst.attrs["jpeg_subsampling"] = JPEG_SUBSAMPLING
        dst.attrs["jpeg_optimized"] = True
        dst.attrs["color_space"] = "RGB"

    mse = squared_error_sum / sample_count if sample_count else 0.0
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10((255.0 ** 2) / mse)
    return {
        "kind": "rgb_jpeg",
        "frames": n_frames,
        "width": width,
        "height": height,
        "source_bytes": source_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
        "psnr_db": None if math.isinf(psnr) else psnr,
        "mean_absolute_error": absolute_error_sum / sample_count if sample_count else 0.0,
        "max_absolute_error": max_absolute_error,
    }


def transcode_h5_lossless(
    source_path: str | Path,
    output_path: str | Path,
    *,
    compression: str = "gzip",
    compression_opts: Any = 6,
    shuffle: bool = True,
) -> dict[str, Any]:
    """Rewrite a frame HDF5 file with new lossless filters and verify content."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()

    with h5py.File(source_path, "r") as src, h5py.File(output_path, "w") as dst:
        if "frames" not in src:
            raise ValueError(f"Missing 'frames' dataset in {source_path}")
        source_dataset = src["frames"]
        n_frames = int(source_dataset.shape[0])
        if source_dataset.ndim < 2:
            raise TypeError("Lossless frame transcoding requires a dense HDF5 dataset")
        output_dataset = dst.create_dataset(
            "frames",
            shape=source_dataset.shape,
            dtype=source_dataset.dtype,
            chunks=source_dataset.chunks or (1, *source_dataset.shape[1:]),
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
        )
        _copy_attrs(src, dst)
        _copy_attrs(source_dataset, output_dataset)
        for index in range(n_frames):
            frame = source_dataset[index]
            source_hash.update(np.ascontiguousarray(frame).tobytes())
            output_dataset[index] = frame

    output_hash = hashlib.sha256()
    with h5py.File(output_path, "r") as dst:
        output_dataset = dst["frames"]
        for index in range(output_dataset.shape[0]):
            output_hash.update(np.ascontiguousarray(output_dataset[index]).tobytes())
        if output_hash.digest() != source_hash.digest():
            raise ValueError("Lossless HDF5 verification failed: frame content changed")

    return {
        "kind": "lossless_hdf5",
        "frames": n_frames,
        "source_bytes": source_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
        "sha256": output_hash.hexdigest(),
        "compression": compression,
        "compression_opts": compression_opts,
        "shuffle": shuffle,
    }
