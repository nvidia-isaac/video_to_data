from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from prep.mhr_ffv1_sidecar import RGB as RGB_FFV1_KIND, ffv1_sidecar_kind, read_ffv1_frame, validate_ffv1_metadata


RGB_DATASET_NAME = "frames"
RGB_JPEG_ENCODING = "jpeg"
RGB_JPEG_FORMAT = "JPEG"
RGB_JPEG_QUALITY = 100
RGB_JPEG_SUBSAMPLING = "4:4:4"
RGB_JPEG_SUBSAMPLING_VALUE = 0
RGB_JPEG_OPTIMIZE = True
RGB_COLORSPACE = "RGB"
RGB_CHANNELS = 3
RGB_SOURCE_DTYPE = "uint8"
RGB_LEGACY_DENSE_LAYOUT = "legacy-dense"
RGB_JPEG_VLEN_LAYOUT = "jpeg-vlen"
RGB_FFV1_SIDECAR_LAYOUT = "ffv1-sidecar"
RGB_JPEG_METADATA_KEYS = frozenset(("encoding", "format", "quality", "subsampling", "optimize", "colorspace", "height", "width", "channels", "source_dtype"))


def _payload_bytes(payload: Any) -> bytes:
    value = np.asarray(payload)
    if value.dtype != np.dtype("uint8") or value.ndim != 1:
        raise TypeError(f"JPEG payload must be a one-dimensional uint8 array, got shape={value.shape}, dtype={value.dtype}")
    if value.size == 0:
        raise ValueError("JPEG payload is empty")
    return value.tobytes()


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame)
    if value.dtype != np.dtype("uint8"):
        raise TypeError(f"RGB frame must have dtype uint8, got {value.dtype}")
    if value.ndim != 3 or value.shape[2] != RGB_CHANNELS or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"RGB frame must have shape (height, width, 3) with positive dimensions, got {value.shape}")
    return value


def encode_rgb_jpeg(frame: np.ndarray) -> np.ndarray:
    from PIL import Image

    value = _validate_frame(frame)
    output = BytesIO()
    Image.fromarray(value).save(output, format=RGB_JPEG_FORMAT, quality=RGB_JPEG_QUALITY, subsampling=RGB_JPEG_SUBSAMPLING_VALUE, optimize=RGB_JPEG_OPTIMIZE)
    return np.frombuffer(output.getvalue(), dtype=np.uint8).copy()


def _open_rgb_jpeg(payload: Any) -> Any:
    from PIL import Image

    image = Image.open(BytesIO(_payload_bytes(payload)))
    if image.format != RGB_JPEG_FORMAT:
        image.close()
        raise ValueError(f"RGB payload must be JPEG, got {image.format}")
    if image.mode != RGB_COLORSPACE:
        mode = image.mode
        image.close()
        raise ValueError(f"RGB JPEG must use {RGB_COLORSPACE} colorspace, got {mode}")
    return image


def decode_rgb_jpeg(payload: Any) -> np.ndarray:
    with _open_rgb_jpeg(payload) as image:
        image.load()
        value = np.asarray(image)
        if value.dtype != np.dtype("uint8") or value.ndim != 3 or value.shape[2] != RGB_CHANNELS:
            raise TypeError(f"RGB JPEG decoded to unsupported shape={value.shape}, dtype={value.dtype}")
        return value.copy()


def validate_rgb_jpeg(payload: Any, expected_shape: tuple[int, int, int] | None = None) -> np.ndarray:
    """Validate JPEG-declared properties; encoder optimization has no reliable bitstream marker and is recorded in HDF5 metadata."""
    with _open_rgb_jpeg(payload) as image:
        quantization = getattr(image, "quantization", None)
        if not isinstance(quantization, dict) or set(quantization) != {0, 1} or any(len(table) != 64 or any(int(value) != 1 for value in table) for table in quantization.values()):
            raise ValueError("RGB JPEG quantization tables do not match Pillow quality=100")
        layers = getattr(image, "layer", None)
        if layers != [(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)]:
            raise ValueError(f"RGB JPEG component sampling is not 4:4:4: {layers}")
        image.load()
        value = np.asarray(image)
        if value.dtype != np.dtype("uint8") or value.ndim != 3 or value.shape[2] != RGB_CHANNELS:
            raise TypeError(f"RGB JPEG decoded to unsupported shape={value.shape}, dtype={value.dtype}")
        if expected_shape is not None and tuple(value.shape) != tuple(expected_shape):
            raise ValueError(f"RGB JPEG decoded shape differs: expected={tuple(expected_shape)}, actual={value.shape}")
        return value.copy()


def _metadata_values(frame_shape: tuple[int, int, int]) -> dict[str, Any]:
    shape = tuple(int(value) for value in frame_shape)
    if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] != RGB_CHANNELS:
        raise ValueError(f"RGB logical frame shape must be (height, width, 3), got {shape}")
    return {"encoding": RGB_JPEG_ENCODING, "format": RGB_JPEG_FORMAT, "quality": RGB_JPEG_QUALITY, "subsampling": RGB_JPEG_SUBSAMPLING, "optimize": RGB_JPEG_OPTIMIZE, "colorspace": RGB_COLORSPACE, "height": shape[0], "width": shape[1], "channels": RGB_CHANNELS, "source_dtype": RGB_SOURCE_DTYPE}


def _text_attr(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _metadata_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return _text_attr(actual) == expected
    if isinstance(expected, bool):
        return isinstance(actual, (bool, np.bool_)) and bool(actual) == expected
    return isinstance(actual, (int, np.integer)) and int(actual) == expected


def _ffv1_metadata_path(source: str | Path | h5py.Group | h5py.Dataset) -> Path | None:
    if isinstance(source, h5py.Dataset):
        return None
    if isinstance(source, h5py.Group):
        kind = ffv1_sidecar_kind(source.file)
        path = Path(source.file.filename)
    else:
        path = Path(source)
        kind = ffv1_sidecar_kind(path)
    if kind is None:
        return None
    if kind != RGB_FFV1_KIND:
        raise ValueError(f"H5 source is FFV1 metadata for {kind}, not RGB: {path}")
    return path


def set_rgb_jpeg_metadata(dataset: h5py.Dataset, frame_shape: tuple[int, int, int]) -> None:
    if dataset.ndim != 1 or h5py.check_dtype(vlen=dataset.dtype) != np.dtype("uint8"):
        raise TypeError(f"JPEG RGB dataset must be one-dimensional variable-length uint8, got shape={dataset.shape}, dtype={dataset.dtype}")
    for key, expected in _metadata_values(frame_shape).items():
        if key in dataset.attrs:
            if not _metadata_equal(dataset.attrs[key], expected):
                raise ValueError(f"Existing RGB dataset attribute {key!r} conflicts with required JPEG metadata: existing={dataset.attrs[key]!r}, required={expected!r}")
            continue
        dataset.attrs[key] = expected


def validate_rgb_jpeg_metadata(dataset: h5py.Dataset) -> tuple[int, int, int]:
    missing = sorted(RGB_JPEG_METADATA_KEYS - set(dataset.attrs))
    if missing:
        raise ValueError(f"JPEG RGB dataset metadata is missing: {missing}")
    shape = (int(dataset.attrs["height"]), int(dataset.attrs["width"]), int(dataset.attrs["channels"]))
    for key, expected in _metadata_values(shape).items():
        if not _metadata_equal(dataset.attrs[key], expected):
            raise ValueError(f"JPEG RGB dataset metadata {key!r} is invalid: expected={expected!r}, actual={dataset.attrs[key]!r}")
    return shape


@contextmanager
def _rgb_dataset(source: str | Path | h5py.Group | h5py.Dataset) -> Iterator[h5py.Dataset]:
    if isinstance(source, h5py.Dataset):
        yield source
        return
    if isinstance(source, h5py.Group):
        if RGB_DATASET_NAME not in source:
            raise KeyError(f"HDF5 group {source.name} is missing /{RGB_DATASET_NAME}")
        dataset = source[RGB_DATASET_NAME]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"/{RGB_DATASET_NAME} must be a dataset")
        yield dataset
        return
    path = Path(source)
    with h5py.File(path, "r") as handle:
        with _rgb_dataset(handle) as dataset:
            yield dataset


def rgb_h5_layout(source: str | Path | h5py.Group | h5py.Dataset) -> str:
    if _ffv1_metadata_path(source) is not None:
        return RGB_FFV1_SIDECAR_LAYOUT
    with _rgb_dataset(source) as dataset:
        if dataset.dtype == np.dtype("uint8") and dataset.ndim == 4:
            if dataset.shape[1] <= 0 or dataset.shape[2] <= 0 or dataset.shape[3] != RGB_CHANNELS:
                raise ValueError(f"Legacy RGB dataset must have shape (T, H, W, 3) with positive spatial dimensions, got {dataset.shape}")
            return RGB_LEGACY_DENSE_LAYOUT
        if dataset.ndim == 1 and h5py.check_dtype(vlen=dataset.dtype) == np.dtype("uint8"):
            validate_rgb_jpeg_metadata(dataset)
            return RGB_JPEG_VLEN_LAYOUT
        if dataset.ndim == 4:
            raise TypeError(f"Legacy RGB dataset must have dtype uint8, got {dataset.dtype}")
        raise ValueError(f"Unsupported RGB HDF5 layout: shape={dataset.shape}, dtype={dataset.dtype}")


def rgb_h5_frame_count(source: str | Path | h5py.Group | h5py.Dataset) -> int:
    metadata_path = _ffv1_metadata_path(source)
    if metadata_path is not None:
        return int(validate_ffv1_metadata(metadata_path, expected_kind=RGB_FFV1_KIND)["frame_count"])
    with _rgb_dataset(source) as dataset:
        rgb_h5_layout(dataset)
        return int(dataset.shape[0])


def rgb_h5_frame_shape(source: str | Path | h5py.Group | h5py.Dataset) -> tuple[int, int, int]:
    metadata_path = _ffv1_metadata_path(source)
    if metadata_path is not None:
        metadata = validate_ffv1_metadata(metadata_path, expected_kind=RGB_FFV1_KIND)
        return int(metadata["height"]), int(metadata["width"]), 3
    with _rgb_dataset(source) as dataset:
        layout = rgb_h5_layout(dataset)
        return tuple(int(value) for value in dataset.shape[1:]) if layout == RGB_LEGACY_DENSE_LAYOUT else validate_rgb_jpeg_metadata(dataset)


def read_rgb_h5_frame(source: str | Path | h5py.Group | h5py.Dataset, index: int) -> np.ndarray:
    metadata_path = _ffv1_metadata_path(source)
    if metadata_path is not None:
        return read_ffv1_frame(metadata_path, index)
    with _rgb_dataset(source) as dataset:
        layout = rgb_h5_layout(dataset)
        if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < dataset.shape[0]:
            raise IndexError(f"RGB frame index {index!r} is outside [0, {dataset.shape[0]})")
        if layout == RGB_LEGACY_DENSE_LAYOUT:
            value = np.asarray(dataset[int(index)])
            if value.dtype != np.dtype("uint8") or tuple(value.shape) != tuple(dataset.shape[1:]):
                raise TypeError(f"Legacy RGB frame decoded to unsupported shape={value.shape}, dtype={value.dtype}")
            return value
        return validate_rgb_jpeg(dataset[int(index)], expected_shape=validate_rgb_jpeg_metadata(dataset))
