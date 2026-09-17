import sys
import json
import threading
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pytest

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import export_sequence
from v2d.common.video import FrameSource


@pytest.mark.parametrize(
    ("out_sub", "shape", "dtype"),
    [
        ("images", (3, 12, 16, 3), np.uint8),
        ("depth", (3, 12, 16), np.uint16),
    ],
)
def test_ffv1_materializes_legacy_png_directory_losslessly(
    tmp_path: Path,
    out_sub: str,
    shape: tuple[int, ...],
    dtype,
):
    source = tmp_path / "frames"
    source.mkdir()
    high = 256 if dtype is np.uint8 else 65536
    frames = np.random.default_rng(17).integers(0, high, size=shape, dtype=dtype)
    for index, frame in enumerate(frames):
        iio.imwrite(source / f"{index:06d}.png", frame)
    destination = tmp_path / "out" / "camera.h5"

    changed, present = export_sequence._materialize_png_directory_as_ffv1(
        source, destination, out_sub,
    )

    assert changed and present
    metadata = export_sequence.read_ffv1_metadata(destination)
    assert metadata["stems"] == ["000000", "000001", "000002"]
    assert metadata["kind"] == ("rgb" if out_sub == "images" else "depth")
    with FrameSource.from_path(destination) as reader:
        assert np.array_equal(np.stack(list(reader.iter_frames())), frames)
        verification = reader.verify_integrity(verify_decoded_frames=True)
    assert verification["decoded_logical_sha256"] == metadata["source_logical_sha256"]


def test_cli_uses_only_anonymized_rgb_flag():
    parser = export_sequence._build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--include_anonymized_rgb" in option_strings
    assert "--include_blurred_rgb" not in option_strings
    assert "--max_camera_workers" in option_strings


def test_camera_jobs_run_with_bounded_parallelism():
    entered = 0
    peak = 0
    lock = threading.Lock()
    all_workers_entered = threading.Event()

    def worker(value):
        nonlocal entered, peak
        with lock:
            entered += 1
            peak = max(peak, entered)
            if entered == 4:
                all_workers_entered.set()
        assert all_workers_entered.wait(timeout=2)
        with lock:
            entered -= 1
        return value * 2

    assert export_sequence._run_camera_jobs(
        list(range(8)), worker, max_camera_workers=4,
    ) == [value * 2 for value in range(8)]
    assert peak == 4


def test_export_sequence_rejects_invalid_camera_worker_count(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="max_camera_workers"):
        export_sequence.export_sequence(
            source_dir=str(source),
            output_dir=str(tmp_path / "out"),
            max_camera_workers=0,
        )


def test_ffv1_materialization_is_default_and_produces_pair(tmp_path: Path):
    source = tmp_path / "rgb.h5"
    destination = tmp_path / "out" / "camera.h5"
    frames = np.arange(3 * 12 * 16 * 3, dtype=np.uint8).reshape(3, 12, 16, 3)
    with h5py.File(source, "w") as h5_file:
        h5_file.create_dataset("frames", data=frames)
        h5_file.attrs["stems"] = json.dumps(["a", "b", "c"])

    changed, present = export_sequence._materialize_h5(
        source,
        destination,
        "images",
    )

    assert changed and present
    assert destination.is_file()
    info = export_sequence.read_ffv1_metadata(destination)
    assert info["sidecar_path"].is_file()
    assert info["sidecar_path"].name.startswith("camera.ffv1.")
    assert export_sequence._target_h5_matches(
        destination, "images", rgb_storage="ffv1_sidecar"
    )
    info["sidecar_path"].unlink()
    assert not export_sequence._target_h5_matches(destination, "images")


def test_ffv1_materializes_source_interval_without_intermediate_h5(
    tmp_path: Path,
):
    source = tmp_path / "rgb.h5"
    destination = tmp_path / "out" / "camera.h5"
    frames = np.random.default_rng(29).integers(
        0, 256, size=(8, 12, 16, 3), dtype=np.uint8,
    )
    with h5py.File(source, "w") as h5_file:
        h5_file.create_dataset("frames", data=frames)
        h5_file.attrs["stems"] = json.dumps(
            [f"source-{index}" for index in range(8)]
        )

    changed, present = export_sequence._materialize_h5(
        source,
        destination,
        "images",
        start_frame=2,
        end_frame=7,
    )

    assert changed and present
    info = export_sequence.read_ffv1_metadata(destination)
    assert info["stems"] == [f"{index:06d}" for index in range(5)]
    with FrameSource.from_path(destination) as reader:
        assert np.array_equal(
            np.stack(list(reader.iter_frames())),
            frames[2:7],
        )


def test_legacy_png_masks_are_packed_as_trimmed_gzip_h5(tmp_path: Path):
    source = tmp_path / "camera" / "0"
    source.mkdir(parents=True)
    frames = [
        np.full((8, 10), fill_value=value, dtype=np.uint8)
        for value in range(7)
    ]
    for index, frame in enumerate(frames):
        iio.imwrite(source / f"{index:06d}.png", frame)
    destination = tmp_path / "out" / "camera.h5"

    changed, present = export_sequence._materialize_png_mask_directory(
        source,
        destination,
        start_frame=2,
        end_frame=6,
    )

    assert changed and present
    with h5py.File(destination, "r") as h5_file:
        dataset = h5_file["frames"]
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 1
        assert dataset.shuffle is False
        assert json.loads(h5_file.attrs["stems"]) == [
            "000000", "000001", "000002", "000003",
        ]
        assert np.array_equal(dataset[:], np.stack(frames[2:6]))


def test_nested_legacy_mask_camera_directory_is_discovered(tmp_path: Path):
    source = tmp_path / "masks"
    expected = []
    for camera in export_sequence.LEFT_CAMERAS:
        directory = source / camera / "0"
        directory.mkdir(parents=True)
        iio.imwrite(directory / "000000.png", np.zeros((2, 3), dtype=np.uint8))
        expected.append((directory, f"{camera}.h5"))

    assert export_sequence._find_png_camera_dirs_local(
        source, "object_masks", export_sequence._is_left_camera_path,
    ) == expected


def test_default_depth_materialization_produces_ffv1_pair(tmp_path: Path):
    source = tmp_path / "depth.h5"
    destination = tmp_path / "out" / "camera.h5"
    frames = np.arange(3 * 12 * 16, dtype=np.uint16).reshape(3, 12, 16)
    with h5py.File(source, "w") as h5_file:
        h5_file.create_dataset("frames", data=frames)
        h5_file.attrs["stems"] = json.dumps(["a", "b", "c"])

    changed, present = export_sequence._materialize_h5(
        source, destination, "depth"
    )

    assert changed and present
    info = export_sequence.read_ffv1_metadata(destination)
    assert info["kind"] == "depth"
    assert info["gop"] == 32
    assert info["sidecar_path"].is_file()


def test_anonymized_rgb_uses_rgb_ffv1_materialization(tmp_path: Path):
    source = tmp_path / "anonymized.h5"
    destination = tmp_path / "out" / "camera.h5"
    frames = np.arange(3 * 12 * 16 * 3, dtype=np.uint8).reshape(3, 12, 16, 3)
    with h5py.File(source, "w") as h5_file:
        h5_file.create_dataset("frames", data=frames)
        h5_file.attrs["stems"] = json.dumps(["a", "b", "c"])

    changed, present = export_sequence._materialize_h5(
        source, destination, "images_anonymized"
    )

    assert changed and present
    info = export_sequence.read_ffv1_metadata(destination)
    assert info["kind"] == "rgb"
    assert info["sidecar_path"].is_file()


def test_anonymized_rgb_uses_rgb_jpeg_h5_materialization(tmp_path: Path):
    source = tmp_path / "anonymized.h5"
    destination = tmp_path / "out" / "camera.h5"
    frames = np.arange(3 * 12 * 16 * 3, dtype=np.uint8).reshape(3, 12, 16, 3)
    with h5py.File(source, "w") as h5_file:
        h5_file.create_dataset("frames", data=frames)
        h5_file.attrs["stems"] = json.dumps(["a", "b", "c"])

    changed, present = export_sequence._materialize_h5(
        source,
        destination,
        "images_anonymized",
        rgb_storage="jpeg_h5",
    )

    assert changed and present
    assert export_sequence._target_h5_matches(
        destination, "images_anonymized", rgb_storage="jpeg_h5"
    )
    with h5py.File(destination, "r") as h5_file:
        assert h5_file.attrs["frame_encoding"] == "jpeg"
        assert h5_file.attrs["n_frames"] == 3


def test_anonymized_rgb_is_required_only_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [
            (
                "face_detector/images",
                "images_anonymized",
                "h5_or_dir",
                export_sequence._is_rgb_camera_path,
                None,
                None,
            )
        ],
    )
    source = tmp_path / "source"
    source.mkdir()

    export_sequence.export_sequence(
        source_dir=str(source), output_dir=str(tmp_path / "disabled")
    )
    with pytest.raises(export_sequence.MissingRequiredExportDataError) as exc_info:
        export_sequence.export_sequence(
            source_dir=str(source),
            output_dir=str(tmp_path / "enabled"),
            include_anonymized_rgb=True,
        )
    assert "images_anonymized" in str(exc_info.value)


def _write_h5(path: Path, frames: np.ndarray, *, compression_opts=1, shuffle=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "frames",
            data=frames,
            chunks=(1, *frames.shape[1:]),
            compression="gzip",
            compression_opts=compression_opts,
            shuffle=shuffle,
        )
        h5_file.attrs["stems"] = json.dumps(
            [f"{index:06d}" for index in range(len(frames))]
        )
        h5_file.attrs["n_frames"] = len(frames)
        h5_file.attrs["height"] = frames.shape[1]
        h5_file.attrs["width"] = frames.shape[2]


def test_export_sequence_fails_when_required_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [("mv_preprocess/edex", "edex", "file", None, None, None)],
    )

    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(export_sequence.MissingRequiredExportDataError) as exc_info:
        export_sequence.export_sequence(
            source_dir=str(source),
            output_dir=str(tmp_path / "out"),
            rgb_storage="jpeg_h5",
        )

    message = str(exc_info.value)
    assert "refusing partial export" in message
    assert "edex: missing source file" in message


def test_export_sequence_fails_when_required_camera_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [
            (
                "mv_preprocess/images",
                "images",
                "h5_or_dir",
                export_sequence._is_rgb_camera_path,
                None,
                None,
            )
        ],
    )

    source = tmp_path / "source"
    image_dir = source / "mv_preprocess" / "images"
    cam_dir = image_dir / "front_stereo_camera_left"
    cam_dir.mkdir(parents=True)
    (cam_dir / "000001.png").write_bytes(b"image")

    with pytest.raises(export_sequence.MissingRequiredExportDataError) as exc_info:
        export_sequence.export_sequence(
            source_dir=str(source),
            output_dir=str(tmp_path / "out"),
            rgb_storage="jpeg_h5",
        )

    message = str(exc_info.value)
    assert "images: missing cameras" in message
    assert "back_stereo_camera_left" in message
    assert "left_stereo_camera_left" in message
    assert "right_stereo_camera_left" in message
    assert "front_stereo_camera_right" in message
    assert "back_stereo_camera_right" in message
    assert "left_stereo_camera_right" in message
    assert "right_stereo_camera_right" in message


def test_export_sequence_fails_when_camera_file_counts_are_uneven(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [
            (
                "foundation_stereo",
                "depth",
                "h5_or_dir",
                None,
                export_sequence._remap_depth,
                ("*/depth.h5", "{cam}.h5"),
            )
        ],
    )

    source = tmp_path / "source"
    for cam in export_sequence.LEFT_CAMERAS:
        depth_dir = source / "foundation_stereo" / cam / "depth"
        depth_dir.mkdir(parents=True)
        (depth_dir / "000001.png").write_bytes(b"depth")
    extra_depth_dir = (
        source / "foundation_stereo" / "front_stereo_camera_left" / "depth"
    )
    (extra_depth_dir / "000002.png").write_bytes(b"depth")

    with pytest.raises(export_sequence.MissingRequiredExportDataError) as exc_info:
        export_sequence.export_sequence(
            source_dir=str(source),
            output_dir=str(tmp_path / "out"),
            depth_storage="gzip_h5",
        )

    assert "depth: uneven camera file counts" in str(exc_info.value)


def test_rgb_validation_uses_eight_cameras_but_depth_uses_left_cameras():
    rgb_paths = [f"{cam}.h5" for cam in export_sequence.RGB_CAMERAS]
    rgb_paths.append("front_stereo_camera_left_extra.h5")
    missing = []
    export_sequence._record_required_group(
        missing, "images", "source/images", rgb_paths
    )
    assert missing and "images: uneven camera file counts" in missing[0]

    missing = []
    export_sequence._record_required_group(
        missing,
        "depth",
        "source/depth",
        [f"{cam}.h5" for cam in export_sequence.LEFT_CAMERAS],
    )
    assert missing == []


def test_export_sequence_allows_complete_required_camera_data(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [
            (
                "mv_preprocess/images",
                "images",
                "h5_or_dir",
                export_sequence._is_rgb_camera_path,
                None,
                None,
            )
        ],
    )

    source = tmp_path / "source"
    for cam in export_sequence.RGB_CAMERAS:
        cam_dir = source / "mv_preprocess" / "images" / cam
        cam_dir.mkdir(parents=True)
        (cam_dir / "000001.png").write_bytes(b"image")
        (cam_dir / "000002.png").write_bytes(b"image")

    export_sequence.export_sequence(
        source_dir=str(source),
        output_dir=str(tmp_path / "out"),
        rgb_storage="jpeg_h5",
    )

    assert (
        tmp_path
        / "out"
        / "images"
        / "front_stereo_camera_left"
        / "000001.png"
    ).is_file()


def test_export_all_rgb_cameras_as_ffv1_pairs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [
            (
                "mv_preprocess/images",
                "images",
                "h5_or_dir",
                export_sequence._is_rgb_camera_path,
                None,
                None,
            ),
            (
                "face_detector/images",
                "images_anonymized",
                "h5_or_dir",
                export_sequence._is_rgb_camera_path,
                None,
                None,
            ),
        ],
    )
    source = tmp_path / "source"
    frames = np.arange(2 * 8 * 10 * 3, dtype=np.uint8).reshape(2, 8, 10, 3)
    for cam in export_sequence.RGB_CAMERAS:
        _write_h5(source / "mv_preprocess" / "images" / f"{cam}.h5", frames)
        _write_h5(source / "face_detector" / "images" / f"{cam}.h5", frames)

    output = tmp_path / "out"
    export_sequence.export_sequence(
        source_dir=str(source),
        output_dir=str(output),
        include_anonymized_rgb=True,
    )

    for out_sub in ("images", "images_anonymized"):
        for cam in export_sequence.RGB_CAMERAS:
            metadata = output / out_sub / f"{cam}.h5"
            info = export_sequence.read_ffv1_metadata(metadata)
            assert info["kind"] == "rgb"
            assert info["gop"] == 1
            assert info["pixel_format"] == "bgr0"
            assert info["sidecar_path"].is_file()


def test_export_copies_all_original_and_anonymized_videos(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [
            (
                "mv_preprocess/videos",
                "videos",
                "dir",
                export_sequence._is_rgb_camera_video,
                None,
                None,
            ),
            (
                "face_detector/videos",
                "videos_anonymized",
                "dir",
                export_sequence._is_rgb_camera_video,
                None,
                None,
            ),
        ],
    )
    source = tmp_path / "source"
    for cam in export_sequence.RGB_CAMERAS:
        for source_sub in ("mv_preprocess/videos", "face_detector/videos"):
            path = source / source_sub / f"{cam}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"video:{cam}".encode())

    output = tmp_path / "out"
    export_sequence.export_sequence(
        source_dir=str(source),
        output_dir=str(output),
        include_anonymized_rgb=True,
    )

    for out_sub in ("videos", "videos_anonymized"):
        assert sorted(path.stem for path in (output / out_sub).glob("*.mp4")) == sorted(
            export_sequence.RGB_CAMERAS
        )


def test_export_rejects_missing_right_camera_video(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [
            (
                "mv_preprocess/videos",
                "videos",
                "dir",
                export_sequence._is_rgb_camera_video,
                None,
                None,
            )
        ],
    )
    source = tmp_path / "source"
    missing_cam = "right_stereo_camera_right"
    for cam in export_sequence.RGB_CAMERAS:
        if cam == missing_cam:
            continue
        path = source / "mv_preprocess" / "videos" / f"{cam}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")

    with pytest.raises(export_sequence.MissingRequiredExportDataError) as exc_info:
        export_sequence.export_sequence(
            source_dir=str(source), output_dir=str(tmp_path / "out")
        )

    assert f"videos: missing cameras {missing_cam}" in str(exc_info.value)


def test_final_only_omits_anonymized_rgb_even_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [
            (
                "face_detector/images",
                "images_anonymized",
                "h5_or_dir",
                export_sequence._is_rgb_camera_path,
                None,
                None,
            )
        ],
    )
    source = tmp_path / "source"
    source.mkdir()

    export_sequence.export_sequence(
        source_dir=str(source),
        output_dir=str(tmp_path / "out"),
        include_anonymized_rgb=True,
        final_only=True,
    )

    assert not (tmp_path / "out" / "images_anonymized").exists()


def test_export_transcodes_rgb_h5(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [("mv_preprocess/images", "images", "h5_or_dir", None, None, None)],
    )
    source = tmp_path / "source"
    frames = np.full((2, 16, 20, 3), 90, dtype=np.uint8)
    for cam in export_sequence.RGB_CAMERAS:
        _write_h5(source / "mv_preprocess" / "images" / f"{cam}.h5", frames)

    output = tmp_path / "out"
    export_sequence.export_sequence(
        source_dir=str(source),
        output_dir=str(output),
        rgb_storage="jpeg_h5",
    )

    for cam in export_sequence.RGB_CAMERAS:
        with h5py.File(output / "images" / f"{cam}.h5", "r") as h5_file:
            assert h5_file.attrs["frame_encoding"] == "jpeg"
            assert h5_file["frames"].ndim == 1


def test_export_rewrites_depth_filters_losslessly(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [
            (
                "foundation_stereo",
                "depth",
                "h5_or_dir",
                None,
                None,
                ("*/depth.h5", "{cam}.h5"),
            )
        ],
    )
    source = tmp_path / "source"
    frames = np.arange(2 * 8 * 9, dtype=np.uint16).reshape(2, 8, 9)
    for cam in export_sequence.LEFT_CAMERAS:
        _write_h5(source / "foundation_stereo" / cam / "depth.h5", frames)

    output = tmp_path / "out"
    export_sequence.export_sequence(
        source_dir=str(source),
        output_dir=str(output),
        depth_storage="gzip_h5",
    )

    for cam in export_sequence.LEFT_CAMERAS:
        with h5py.File(output / "depth" / f"{cam}.h5", "r") as h5_file:
            dataset = h5_file["frames"]
            assert dataset.compression_opts == 6
            assert dataset.shuffle is True
            assert np.array_equal(dataset[:], frames)
