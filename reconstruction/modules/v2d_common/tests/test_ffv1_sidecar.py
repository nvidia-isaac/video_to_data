import json
import re
import sys
from pathlib import Path

import av
import h5py
import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from ffv1_sidecar import (
    DEPTH_SCHEMA,
    RGB_SCHEMA,
    is_ffv1_sidecar_h5,
    read_ffv1_metadata,
    transcode_h5_to_ffv1_sidecar,
    verify_ffv1_sidecar,
)
from hdf5_transcode import transcode_rgb_h5_to_jpeg_h5
from video import FrameSource, FrameWriter


def _write_source(path: Path, frames: np.ndarray) -> list[str]:
    stems = [f"frame_{index:03d}" for index in range(len(frames))]
    with FrameWriter.from_path(path) as writer:
        for stem, frame in zip(stems, frames):
            writer.write_frame(frame, stem)
    with h5py.File(path, "a") as h5_file:
        h5_file.attrs["custom_root"] = "not-copied"
        h5_file["frames"].attrs["custom_dataset"] = 17
    return stems


COMMON_REFERENCE_ATTRS = {
    "complete",
    "container",
    "encoded_pixel_format",
    "encoding",
    "exact_to_original_dense",
    "ffv1_coder",
    "ffv1_context",
    "ffv1_level",
    "ffv1_slicecrc",
    "frame_count",
    "frame_rate",
    "gop",
    "height",
    "logical_stem",
    "n_frames",
    "schema",
    "sidecar_basename",
    "sidecar_bytes",
    "sidecar_sha256",
    "source_dtype",
    "source_h5_bytes",
    "source_layout",
    "source_logical_sha256",
    "stems",
    "width",
}


@pytest.mark.parametrize(
    ("kind", "shape", "dtype", "gop", "pixel_format", "schema"),
    [
        ("rgb", (35, 18, 24, 3), np.uint8, 1, "bgr0", RGB_SCHEMA),
        ("depth", (35, 18, 24), np.uint16, 32, "gray16le", DEPTH_SCHEMA),
    ],
)
def test_cari4d_ffv1_exact_round_trip_and_reader(
    tmp_path: Path,
    kind: str,
    shape: tuple[int, ...],
    dtype,
    gop: int,
    pixel_format: str,
    schema: str,
):
    rng = np.random.default_rng(9)
    high = 256 if dtype is np.uint8 else 65536
    frames = rng.integers(0, high, size=shape, dtype=dtype)
    source_path = tmp_path / f"{kind}_source.h5"
    metadata_path = tmp_path / f"{kind}.h5"
    stems = _write_source(source_path, frames)

    stats = transcode_h5_to_ffv1_sidecar(source_path, metadata_path, kind=kind)
    sidecar_path = Path(stats["sidecar_path"])

    assert stats["exact_round_trip"] is True
    assert stats["gop"] == gop
    assert stats["output_bytes"] == metadata_path.stat().st_size + sidecar_path.stat().st_size
    assert re.fullmatch(
        rf"{kind}\.ffv1\.[0-9a-f]{{64}}\.mkv", sidecar_path.name
    )
    assert sidecar_path.name == stats["sidecar_filename"]
    assert stats["source_logical_sha256"] == stats["decoded_logical_sha256"]

    info = read_ffv1_metadata(metadata_path)
    assert info["schema"] == schema
    assert info["pixel_format"] == pixel_format
    assert info["keyframe_indices"] == list(range(0, len(frames), gop))
    assert info["stems"] == stems
    assert info["frame_pts"][:3] == [0, 33, 67]
    assert info["dtype"] == np.dtype(dtype)
    assert info["sidecar_path"] == sidecar_path
    with h5py.File(metadata_path, "r") as metadata:
        expected_attrs = COMMON_REFERENCE_ATTRS | ({"channels"} if kind == "rgb" else set())
        assert set(metadata.attrs) == expected_attrs
        assert set(metadata) == {"frame_pts", "keyframe_indices"}
        assert metadata.attrs["schema"] == schema
        assert metadata.attrs["ffv1_level"] == 3
        assert metadata.attrs["ffv1_coder"] == 1
        assert metadata.attrs["ffv1_context"] == 1
        assert metadata.attrs["ffv1_slicecrc"] == 1
        assert bool(metadata.attrs["complete"]) is True
        assert json.loads(metadata.attrs["stems"]) == stems
        assert "source_metadata" not in metadata
    with av.open(str(sidecar_path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "ffv1"
        assert stream.codec_context.pix_fmt == pixel_format

    with FrameSource.from_path(metadata_path) as reader:
        assert reader.stems == stems
        assert reader.image_size == (shape[2], shape[1])
        assert np.array_equal(np.stack(list(reader.iter_frames())), frames)
        for index in (0, 1, 31, 32, len(frames) // 2, len(frames) - 1):
            assert np.array_equal(reader[index], frames[index])
        verified = reader.verify_integrity(verify_decoded_frames=True)
        assert verified["decoded_frames"] == len(frames)
        assert verified["decoded_logical_sha256"] == stats["source_logical_sha256"]

    with FrameSource.from_path(metadata_path, frames_slice=slice(1, 34, 3)) as reader:
        assert np.array_equal(np.stack(list(reader.iter_frames())), frames[1:34:3])
        assert [start for start, _ in reader.iter_batches(4)] == [0, 4, 8]


def test_cari4d_metadata_rejects_orphan_and_traversal(tmp_path: Path):
    source = tmp_path / "source.h5"
    metadata = tmp_path / "camera.h5"
    _write_source(source, np.zeros((2, 8, 10, 3), dtype=np.uint8))
    transcode_h5_to_ffv1_sidecar(source, metadata, kind="rgb")
    sidecar = read_ffv1_metadata(metadata)["sidecar_path"]
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="sidecar not found"):
        FrameSource.from_path(metadata)

    with h5py.File(metadata, "a") as h5_file:
        h5_file.attrs["sidecar_basename"] = "../escape.mkv"
    with pytest.raises(ValueError, match="sibling filename"):
        read_ffv1_metadata(metadata)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("complete", False, "not complete"),
        ("schema", "ffv1_matroska_sidecar", "Not a supported"),
        ("ffv1_coder", 2, "codec metadata"),
    ],
)
def test_cari4d_metadata_rejects_incomplete_or_unsupported_values(
    tmp_path: Path, attribute: str, value, message: str
):
    source = tmp_path / "source.h5"
    metadata = tmp_path / "camera.h5"
    _write_source(source, np.zeros((2, 8, 10, 3), dtype=np.uint8))
    transcode_h5_to_ffv1_sidecar(source, metadata, kind="rgb")
    with h5py.File(metadata, "a") as h5_file:
        h5_file.attrs[attribute] = value
    with pytest.raises(ValueError, match=message):
        read_ffv1_metadata(metadata)


def test_cari4d_metadata_rejects_bad_content_address_and_size(tmp_path: Path):
    source = tmp_path / "source.h5"
    metadata = tmp_path / "camera.h5"
    _write_source(source, np.zeros((2, 8, 10), dtype=np.uint16))
    transcode_h5_to_ffv1_sidecar(source, metadata, kind="depth")
    with h5py.File(metadata, "a") as h5_file:
        h5_file.attrs["sidecar_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        read_ffv1_metadata(metadata)

    with h5py.File(metadata, "a") as h5_file:
        basename = h5_file.attrs["sidecar_basename"]
        h5_file.attrs["sidecar_sha256"] = basename.split(".")[-2]
        h5_file.attrs["sidecar_bytes"] += 1
    with pytest.raises(ValueError, match="size"):
        read_ffv1_metadata(metadata)


def test_cari4d_integrity_detects_same_size_corruption(tmp_path: Path):
    source = tmp_path / "source.h5"
    metadata = tmp_path / "camera.h5"
    frames = np.arange(3 * 8 * 10, dtype=np.uint16).reshape(3, 8, 10)
    _write_source(source, frames)
    transcode_h5_to_ffv1_sidecar(source, metadata, kind="depth")
    sidecar = read_ffv1_metadata(metadata)["sidecar_path"]
    with sidecar.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        verify_ffv1_sidecar(metadata)


def test_cari4d_ffv1_rejects_jpeg_hdf5_source(tmp_path: Path):
    dense = tmp_path / "dense.h5"
    jpeg = tmp_path / "jpeg.h5"
    metadata = tmp_path / "camera.h5"
    frames = np.random.default_rng(11).integers(
        0, 256, size=(3, 16, 20, 3), dtype=np.uint8
    )
    _write_source(dense, frames)
    transcode_rgb_h5_to_jpeg_h5(dense, jpeg)

    with pytest.raises(ValueError, match="dense HDF5 source"):
        transcode_h5_to_ffv1_sidecar(jpeg, metadata, kind="rgb")


def test_cari4d_ffv1_encodes_selected_interval_and_reindexes_stems(
    tmp_path: Path,
):
    source = tmp_path / "source.h5"
    metadata = tmp_path / "camera.h5"
    frames = np.random.default_rng(31).integers(
        0, 256, size=(8, 12, 16, 3), dtype=np.uint8,
    )
    _write_source(source, frames)

    stats = transcode_h5_to_ffv1_sidecar(
        source,
        metadata,
        kind="rgb",
        start_frame=2,
        end_frame=7,
        reindex_stems=True,
        measure_random_access=False,
    )

    assert stats["source_frames"] == 8
    assert stats["source_start_frame"] == 2
    assert stats["source_end_frame"] == 7
    assert stats["frames"] == 5
    assert "random_access_ms" not in stats
    info = read_ffv1_metadata(metadata)
    assert info["stems"] == [f"{index:06d}" for index in range(5)]
    with FrameSource.from_path(metadata) as reader:
        assert np.array_equal(
            np.stack(list(reader.iter_frames())),
            frames[2:7],
        )
        assert reader.verify_integrity(
            verify_decoded_frames=True,
        )["decoded_frames"] == 5


def test_experimental_schema_is_not_detected(tmp_path: Path):
    metadata = tmp_path / "camera.h5"
    with h5py.File(metadata, "w") as h5_file:
        h5_file.attrs["storage_schema"] = "ffv1_matroska_sidecar"
        h5_file.attrs["schema_version"] = 1
    assert not is_ffv1_sidecar_h5(metadata)
