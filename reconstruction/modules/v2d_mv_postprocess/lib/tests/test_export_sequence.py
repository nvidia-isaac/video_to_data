import sys
from pathlib import Path

import pytest

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import export_sequence


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
                export_sequence._is_left_camera_path,
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
        )

    message = str(exc_info.value)
    assert "images: missing cameras" in message
    assert "back_stereo_camera_left" in message
    assert "left_stereo_camera_left" in message
    assert "right_stereo_camera_left" in message


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
        )

    assert "depth: uneven camera file counts" in str(exc_info.value)


def test_export_sequence_allows_complete_required_camera_data(monkeypatch, tmp_path):
    monkeypatch.setattr(
        export_sequence,
        "_DATA_MAP",
        [
            (
                "mv_preprocess/images",
                "images",
                "h5_or_dir",
                export_sequence._is_left_camera_path,
                None,
                None,
            )
        ],
    )

    source = tmp_path / "source"
    for cam in export_sequence.LEFT_CAMERAS:
        cam_dir = source / "mv_preprocess" / "images" / cam
        cam_dir.mkdir(parents=True)
        (cam_dir / "000001.png").write_bytes(b"image")
        (cam_dir / "000002.png").write_bytes(b"image")

    export_sequence.export_sequence(
        source_dir=str(source),
        output_dir=str(tmp_path / "out"),
    )

    assert (
        tmp_path
        / "out"
        / "images"
        / "front_stereo_camera_left"
        / "000001.png"
    ).is_file()
