import json
import sys
from pathlib import Path

import h5py
import numpy as np

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import migrate_export_sequence as migration


def _write_h5(path: Path, frames: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "frames",
            data=frames,
            chunks=(1, *frames.shape[1:]),
            compression="gzip",
            compression_opts=1,
        )
        h5_file.attrs["stems"] = json.dumps(
            [f"{index:06d}" for index in range(len(frames))]
        )


def test_migrate_export_sequence_stages_only_rgb_and_depth(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "staged"
    rgb = np.full((2, 16, 20, 3), 120, dtype=np.uint8)
    depth = np.arange(2 * 8 * 9, dtype=np.uint16).reshape(2, 8, 9)
    _write_h5(source / "images" / "front.h5", rgb)
    _write_h5(source / "depth" / "front.h5", depth)
    _write_h5(source / "object_masks" / "front.h5", np.zeros((2, 8, 9), np.uint8))

    manifest = migration.migrate_export_sequence(source, output)

    assert manifest["status"] == "complete"
    assert not (output / "object_masks").exists()
    with h5py.File(output / "images" / "front.h5", "r") as h5_file:
        assert h5_file.attrs["frame_encoding"] == "jpeg"
    with h5py.File(output / "depth" / "front.h5", "r") as h5_file:
        dataset = h5_file["frames"]
        assert dataset.compression_opts == 6
        assert dataset.shuffle is True
        assert np.array_equal(dataset[:], depth)


def test_migrate_export_sequence_ffv1_stages_pairs_and_leaves_masks(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "staged"
    rgb = np.arange(3 * 16 * 20 * 3, dtype=np.uint8).reshape(3, 16, 20, 3)
    anonymized = np.flip(rgb, axis=2).copy()
    depth = np.arange(3 * 8 * 10, dtype=np.uint16).reshape(3, 8, 10)
    _write_h5(source / "images" / "front.h5", rgb)
    _write_h5(source / "images_anonymized" / "front.h5", anonymized)
    _write_h5(source / "depth" / "front.h5", depth)
    _write_h5(source / "object_masks" / "front.h5", np.zeros((3, 8, 10), np.uint8))

    manifest = migration.migrate_export_sequence(
        source, output, storage_format="ffv1"
    )

    assert manifest["status"] == "complete"
    assert manifest["storage_format"] == "ffv1"
    assert not (output / "object_masks").exists()
    for kind in ("images", "images_anonymized", "depth"):
        assert (output / kind / "front.h5").is_file()
    records = {item["kind_dir"]: item for item in manifest["files"]}
    assert records["images"]["gop"] == 1
    assert records["images"]["pixel_format"] == "bgr0"
    assert records["images_anonymized"]["gop"] == 1
    assert records["images_anonymized"]["pixel_format"] == "bgr0"
    assert records["depth"]["gop"] == 32
    assert records["depth"]["pixel_format"] == "gray16le"
    for item in records.values():
        sidecar = output / item["kind_dir"] / item["sidecar_filename"]
        assert sidecar.is_file()
        assert item["sidecar_sha256"] in sidecar.name
    assert all(item["exact_round_trip"] for item in manifest["files"])
