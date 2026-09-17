import json
import sys
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from video import FrameSource, FrameWriter, pack_directory_to_h5
from hdf5_transcode import (
    decode_rgb_jpeg,
    slice_h5_frames,
    transcode_h5_lossless,
    transcode_rgb_h5_to_jpeg_h5,
)


@pytest.mark.parametrize(
    ("dtype", "compression_opts", "shuffle"),
    [
        (np.uint8, 1, False),
        (np.uint8, 6, False),
        (np.uint16, 6, True),
    ],
)
def test_hdf5_writer_round_trip(
    tmp_path: Path,
    dtype,
    compression_opts: int,
    shuffle: bool,
):
    rng = np.random.default_rng(7)
    high = 256 if dtype is np.uint8 else 65536
    frames = rng.integers(0, high, size=(3, 8, 10, 3), dtype=dtype)
    stems = ["000010", "000020", "custom"]
    output = tmp_path / "frames.h5"

    kwargs = {}
    if compression_opts != 1 or shuffle:
        kwargs = {
            "compression": "gzip",
            "compression_opts": compression_opts,
            "shuffle": shuffle,
        }
    with FrameWriter.from_path(output, **kwargs) as writer:
        for frame, stem in zip(frames, stems):
            writer.write_frame(frame, stem=stem)

    with h5py.File(output, "r") as h5_file:
        dataset = h5_file["frames"]
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == compression_opts
        assert dataset.shuffle is shuffle
        assert dataset.chunks == (1, *frames.shape[1:])
        assert dataset.dtype == frames.dtype
        assert np.array_equal(dataset[:], frames)
        assert json.loads(h5_file.attrs["stems"]) == stems
        assert h5_file.attrs["n_frames"] == len(frames)
        assert h5_file.attrs["height"] == frames.shape[1]
        assert h5_file.attrs["width"] == frames.shape[2]

    with FrameSource.from_path(output) as source:
        assert source.stems == stems
        assert source.n_frames == len(frames)
        assert source.dtype == frames.dtype
        for idx, expected in enumerate(frames):
            assert np.array_equal(source[idx], expected)
        assert np.array_equal(np.stack(list(source.iter_frames())), frames)


def test_pack_directory_uses_explicit_rgb_compression(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    frames = [
        np.full((6, 7, 3), fill_value=value, dtype=np.uint8)
        for value in (12, 34)
    ]
    for idx, frame in enumerate(frames):
        iio.imwrite(image_dir / f"{idx:06d}.png", frame)

    output = tmp_path / "rgb.h5"
    pack_directory_to_h5(
        image_dir,
        output,
        show_progress=False,
        compression="gzip",
        compression_opts=6,
        shuffle=False,
    )

    with h5py.File(output, "r") as h5_file:
        dataset = h5_file["frames"]
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 6
        assert dataset.shuffle is False
        assert np.array_equal(dataset[:], np.stack(frames))


def test_pack_directory_selects_interval_and_reindexes_stems(tmp_path: Path):
    image_dir = tmp_path / "masks"
    image_dir.mkdir()
    frames = [
        np.full((6, 7), fill_value=value, dtype=np.uint8)
        for value in range(6)
    ]
    for idx, frame in enumerate(frames):
        iio.imwrite(image_dir / f"source-{idx:03d}.png", frame)

    output = tmp_path / "masks.h5"
    pack_directory_to_h5(
        image_dir,
        output,
        show_progress=False,
        compression="gzip",
        compression_opts=1,
        shuffle=False,
        start_frame=2,
        end_frame=5,
        reindex_stems=True,
    )

    with h5py.File(output, "r") as h5_file:
        dataset = h5_file["frames"]
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 1
        assert dataset.shuffle is False
        assert json.loads(h5_file.attrs["stems"]) == [
            "000000", "000001", "000002",
        ]
        assert np.array_equal(dataset[:], np.stack(frames[2:5]))


def test_non_hdf5_writer_rejects_custom_hdf5_options(tmp_path: Path):
    with pytest.raises(ValueError, match="only be used with"):
        FrameWriter.from_path(
            tmp_path / "pngs",
            compression="gzip",
            compression_opts=6,
        )


def test_png_writer_is_unchanged_with_default_options(tmp_path: Path):
    output = tmp_path / "pngs"
    frame = np.arange(30, dtype=np.uint16).reshape(5, 6)
    with FrameWriter.from_path(output) as writer:
        writer.write_frame(frame, stem="depth")

    assert np.array_equal(iio.imread(output / "depth.png"), frame)


def test_slice_h5_frames_preserves_filters_and_reindexes_stems(tmp_path: Path):
    source = tmp_path / "source.h5"
    output = tmp_path / "output.h5"
    frames = np.arange(8 * 5 * 6, dtype=np.uint16).reshape(8, 5, 6)
    with h5py.File(source, "w") as h5_file:
        dataset = h5_file.create_dataset(
            "frames",
            data=frames,
            chunks=(1, 5, 6),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        h5_file.attrs["stems"] = json.dumps(
            [f"source-{index}" for index in range(8)]
        )
        h5_file.attrs["n_frames"] = 8
        h5_file.attrs["frame_count"] = 8
        dataset.attrs["units"] = "inverse_depth_uint16"

    stats = slice_h5_frames(
        source, output, start_frame=3, end_frame=7,
    )

    assert stats["source_frames"] == 8
    assert stats["frames"] == 4
    with h5py.File(output, "r") as h5_file:
        dataset = h5_file["frames"]
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 6
        assert dataset.shuffle
        assert dataset.attrs["units"] == "inverse_depth_uint16"
        assert h5_file.attrs["n_frames"] == 4
        assert h5_file.attrs["frame_count"] == 4
        assert json.loads(h5_file.attrs["stems"]) == [
            "000000", "000001", "000002", "000003",
        ]
        assert np.array_equal(dataset[:], frames[3:7])


def test_missing_h5_path_falls_back_to_legacy_image_directory(tmp_path: Path):
    legacy_dir = tmp_path / "camera"
    legacy_dir.mkdir()
    frames = [
        np.full((5, 6, 3), value, dtype=np.uint8)
        for value in (12, 34)
    ]
    for index, frame in enumerate(frames):
        iio.imwrite(legacy_dir / f"{index:06d}.png", frame)

    with FrameSource.from_path(tmp_path / "camera.h5") as source:
        assert source.path == legacy_dir
        assert source.stems == ["000000", "000001"]
        assert np.array_equal(np.stack(list(source.iter_frames())), frames)


def test_jpeg_hdf5_transcode_and_dual_format_reader(tmp_path: Path):
    y, x = np.mgrid[:32, :48]
    frames = np.stack(
        [
            np.stack((x * 5, y * 7, (x + y) * 3), axis=-1),
            np.stack((255 - x * 5, y * 7, x * 2), axis=-1),
        ]
    ).clip(0, 255).astype(np.uint8)
    source_path = tmp_path / "dense.h5"
    output_path = tmp_path / "jpeg.h5"
    stems = ["000003", "000009"]
    with FrameWriter.from_path(source_path) as writer:
        for frame, stem in zip(frames, stems):
            writer.write_frame(frame, stem=stem)

    stats = transcode_rgb_h5_to_jpeg_h5(source_path, output_path)

    assert stats["frames"] == 2
    assert stats["psnr_db"] > 30
    with h5py.File(output_path, "r") as h5_file:
        dataset = h5_file["frames"]
        assert dataset.ndim == 1
        assert dataset.compression is None
        assert h5_file.attrs["frame_encoding"] == "jpeg"
        assert h5_file.attrs["jpeg_quality"] == 100
        assert h5_file.attrs["jpeg_subsampling"] == "4:4:4"
        assert bool(h5_file.attrs["jpeg_optimized"]) is True
        assert decode_rgb_jpeg(dataset[0]).shape == frames[0].shape

    with FrameSource.from_path(output_path) as source:
        assert source.image_size == (48, 32)
        assert source.dtype == np.dtype(np.uint8)
        assert source.stems == stems
        decoded = list(source.iter_frames())
        assert len(decoded) == 2
        assert all(frame.dtype == np.uint8 for frame in decoded)
        assert all(frame.shape == (32, 48, 3) for frame in decoded)

    with FrameSource.from_path(output_path, frames_slice=slice(1, 2)) as source:
        assert source.stems == ["000009"]
        assert np.array_equal(source[0], decoded[1])


def test_lossless_hdf5_transcode_preserves_depth(tmp_path: Path):
    frames = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
    source_path = tmp_path / "depth_source.h5"
    output_path = tmp_path / "depth_output.h5"
    with FrameWriter.from_path(source_path) as writer:
        for index, frame in enumerate(frames):
            writer.write_frame(frame, stem=f"{index:06d}")

    transcode_h5_lossless(source_path, output_path)

    with h5py.File(output_path, "r") as h5_file:
        dataset = h5_file["frames"]
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 6
        assert dataset.shuffle is True
        assert np.array_equal(dataset[:], frames)
        assert json.loads(h5_file.attrs["stems"]) == ["000000", "000001", "000002"]


def test_jpeg_hdf5_reader_rejects_corrupt_payload(tmp_path: Path):
    path = tmp_path / "corrupt.h5"
    with h5py.File(path, "w") as h5_file:
        dataset = h5_file.create_dataset(
            "frames",
            shape=(1,),
            dtype=h5py.vlen_dtype(np.dtype("uint8")),
        )
        dataset[0] = np.array([1, 2, 3], dtype=np.uint8)
        h5_file.attrs["frame_encoding"] = "jpeg"
        h5_file.attrs["width"] = 10
        h5_file.attrs["height"] = 10

    with FrameSource.from_path(path) as source:
        with pytest.raises(ValueError, match="Invalid JPEG"):
            _ = source[0]
