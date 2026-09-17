from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from prep.mhr_ffv1_sidecar import SENSOR_DEPTH, ffv1_sidecar_kind, read_ffv1_frame, validate_ffv1_metadata


SENSOR_DEPTH_DATASET_NAME = "frames"
SENSOR_DEPTH_DENSE_LAYOUT = "sensor-depth-dense"
SENSOR_DEPTH_FFV1_SIDECAR_LAYOUT = "ffv1-sidecar"


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
    if kind != SENSOR_DEPTH:
        raise ValueError(f"H5 source is FFV1 metadata for {kind}, not sensor depth: {path}")
    return path


@contextmanager
def _dense_dataset(source: str | Path | h5py.Group | h5py.Dataset) -> Iterator[h5py.Dataset]:
    if isinstance(source, h5py.Dataset):
        yield source
        return
    if isinstance(source, h5py.Group):
        if SENSOR_DEPTH_DATASET_NAME not in source:
            raise KeyError(f"HDF5 group {source.name} is missing /{SENSOR_DEPTH_DATASET_NAME}")
        dataset = source[SENSOR_DEPTH_DATASET_NAME]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"/{SENSOR_DEPTH_DATASET_NAME} must be a dataset")
        yield dataset
        return
    with h5py.File(source, "r") as handle:
        with _dense_dataset(handle) as dataset:
            yield dataset


def sensor_depth_h5_layout(source: str | Path | h5py.Group | h5py.Dataset) -> str:
    if _ffv1_metadata_path(source) is not None:
        return SENSOR_DEPTH_FFV1_SIDECAR_LAYOUT
    with _dense_dataset(source) as dataset:
        if dataset.dtype != np.dtype("uint16") or dataset.ndim != 3 or min(int(value) for value in dataset.shape) <= 0:
            raise TypeError(f"Dense sensor-depth dataset must have shape (T,H,W), dtype uint16, got {dataset.shape}/{dataset.dtype}")
        return SENSOR_DEPTH_DENSE_LAYOUT


def sensor_depth_h5_frame_count(source: str | Path | h5py.Group | h5py.Dataset) -> int:
    metadata_path = _ffv1_metadata_path(source)
    if metadata_path is not None:
        return int(validate_ffv1_metadata(metadata_path, expected_kind=SENSOR_DEPTH)["frame_count"])
    with _dense_dataset(source) as dataset:
        sensor_depth_h5_layout(dataset)
        return int(dataset.shape[0])


def sensor_depth_h5_frame_shape(source: str | Path | h5py.Group | h5py.Dataset) -> tuple[int, int]:
    metadata_path = _ffv1_metadata_path(source)
    if metadata_path is not None:
        metadata = validate_ffv1_metadata(metadata_path, expected_kind=SENSOR_DEPTH)
        return int(metadata["height"]), int(metadata["width"])
    with _dense_dataset(source) as dataset:
        sensor_depth_h5_layout(dataset)
        return int(dataset.shape[1]), int(dataset.shape[2])


def read_sensor_depth_h5_frame(source: str | Path | h5py.Group | h5py.Dataset, index: int) -> np.ndarray:
    metadata_path = _ffv1_metadata_path(source)
    if metadata_path is not None:
        return read_ffv1_frame(metadata_path, index)
    with _dense_dataset(source) as dataset:
        sensor_depth_h5_layout(dataset)
        if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < dataset.shape[0]:
            raise IndexError(f"Sensor-depth frame index {index!r} is outside [0, {dataset.shape[0]})")
        value = np.asarray(dataset[int(index)])
        if value.dtype != np.dtype("uint16") or value.shape != tuple(dataset.shape[1:]):
            raise TypeError(f"Sensor-depth frame decoded to unsupported shape={value.shape}, dtype={value.dtype}")
        return value
