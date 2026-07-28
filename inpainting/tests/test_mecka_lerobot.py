"""CPU tests for the LeRobot v3 reader, using a synthetic local shard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from inpainting.adapters.mecka_lerobot import intrinsic_matrix
from inpainting.mecka_panda import lerobot_source as ls

VIDEO_PREFIX = f"videos/{ls.VIDEO_KEY}/"
INTRINSICS = [752.5, 753.0, 961.8, 553.2, 0.05, -0.03, 0.0, 0.0]


def _write_shard(root: Path, lengths: tuple[int, ...]) -> None:
    """Write a minimal v3 shard with one row group per episode."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": len(lengths),
        "total_chunks": 1,
        "chunks_size": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "episodes_path": "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    episodes: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    start = 0
    elapsed = 0.0
    for index, length in enumerate(lengths):
        duration = length / 30.0
        episodes.append(
            {
                "episode_index": index,
                "tasks": ["demo"],
                "length": length,
                "duration": duration,
                "task_id": f"task_{index}",
                "task_description": f"description {index}",
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": start,
                "dataset_to_index": start + length,
                f"{VIDEO_PREFIX}chunk_index": 0,
                f"{VIDEO_PREFIX}file_index": 0,
                f"{VIDEO_PREFIX}from_timestamp": elapsed,
                f"{VIDEO_PREFIX}to_timestamp": elapsed + duration,
                "camera_intrinsics": INTRINSICS,
            }
        )
        for offset in range(length):
            frames.append(
                {
                    "episode_index": index,
                    "frame_index": offset,
                    "observation.state.camera_rotation": [0.0, 0.0, 0.0, 1.0],
                }
            )
        start += length
        elapsed += duration

    pq.write_table(
        pa.Table.from_pylist(episodes),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    # One row group per episode, matching how the real shards are written.
    writer = pq.ParquetWriter(
        root / "data" / "chunk-000" / "file-000.parquet",
        pa.Table.from_pylist(frames).schema,
    )
    offset = 0
    for length in lengths:
        writer.write_table(pa.Table.from_pylist(frames[offset : offset + length]))
        offset += length
    writer.close()


def test_reads_only_the_requested_episode_rows(tmp_path: Path) -> None:
    _write_shard(tmp_path, (4, 6, 5))
    source = ls.open_source(tmp_path)
    info = ls.load_info(source)

    record = ls.find_episode(source, info, 1)
    assert (record.row_start, record.row_stop) == (4, 10)
    assert record.length == 6
    assert record.camera_intrinsics == tuple(INTRINSICS)

    table = ls.read_episode_table(source, info, record, columns=["frame_index"])
    assert table.num_rows == 6
    assert table.column("frame_index").to_pylist() == list(range(6))


def test_defaults_to_the_lowest_episode(tmp_path: Path) -> None:
    _write_shard(tmp_path, (4, 6))
    source = ls.open_source(tmp_path)
    info = ls.load_info(source)
    assert ls.find_episode(source, info).episode_index == 0


def test_missing_episode_is_an_error(tmp_path: Path) -> None:
    _write_shard(tmp_path, (4,))
    source = ls.open_source(tmp_path)
    info = ls.load_info(source)
    with pytest.raises(ls.LeRobotSourceError, match="not present"):
        ls.find_episode(source, info, 7)


def test_unknown_columns_fail_instead_of_being_dropped(tmp_path: Path) -> None:
    _write_shard(tmp_path, (4,))
    source = ls.open_source(tmp_path)
    info = ls.load_info(source)
    record = ls.find_episode(source, info, 0)
    with pytest.raises(ls.LeRobotSourceError, match="are not in"):
        ls.read_episode_table(source, info, record, columns=["frame_index", "absent"])


def test_unsupported_codebase_version_is_rejected(tmp_path: Path) -> None:
    _write_shard(tmp_path, (4,))
    path = tmp_path / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["codebase_version"] = "v2.1"
    path.write_text(json.dumps(info), encoding="utf-8")
    source = ls.open_source(tmp_path)
    with pytest.raises(ls.LeRobotSourceError, match="Unsupported codebase_version"):
        ls.load_info(source)


def test_intrinsic_matrix_scales_to_decoded_resolution() -> None:
    full = intrinsic_matrix(INTRINSICS, width=1920, height=1080)
    np.testing.assert_allclose(full[0, 0], INTRINSICS[0])
    np.testing.assert_allclose(full[1, 2], INTRINSICS[3])

    half = intrinsic_matrix(INTRINSICS, width=960, height=540)
    np.testing.assert_allclose(half[0, 0], INTRINSICS[0] / 2)
    np.testing.assert_allclose(half[1, 1], INTRINSICS[1] / 2)
    np.testing.assert_allclose(half[0, 2], INTRINSICS[2] / 2)


def test_intrinsic_matrix_rejects_wrong_arity() -> None:
    with pytest.raises(ValueError, match="fx,fy,cx,cy"):
        intrinsic_matrix([1.0, 2.0, 3.0], width=1920, height=1080)


def test_presign_url_is_signed_and_scoped() -> None:
    credentials = ls.Credentials(
        access_key="AKIAEXAMPLE",
        secret_key="secret",
        endpoint="https://storage.googleapis.com",
        region="us-east4",
    )
    url = ls.presign_url(credentials, "bucket", "a/b/file.mp4", expires_seconds=900)
    assert url.startswith("https://storage.googleapis.com/bucket/a/b/file.mp4?")
    assert "X-Amz-Signature=" in url
    assert "X-Amz-Expires=900" in url
    assert "us-east4%2Fs3%2Faws4_request" in url
    assert credentials.secret_key not in url
