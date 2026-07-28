"""CPU tests for the LeRobot v3 reader, using a synthetic local shard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from inpainting.adapters.mecka_lerobot import (
    expected_video_geometry,
    intrinsic_matrix,
    validate_video_geometry,
)
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


def _write_multi_file_shard(root: Path, files: tuple[tuple[int, ...], ...]) -> None:
    """Write a v3 shard whose episodes are spread over several data files.

    Row indices in `meta/episodes` accumulate across the whole shard, so the
    second data file onwards starts at a non-zero `dataset_from_index`.
    """
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": sum(len(group) for group in files),
        "chunks_size": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "episodes_path": "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    episodes: list[dict[str, object]] = []
    index = 0
    start = 0
    elapsed = 0.0
    for chunk, lengths in enumerate(files):
        directory = root / "data" / f"chunk-{chunk:03d}"
        directory.mkdir(parents=True)
        frames: list[dict[str, object]] = []
        for length in lengths:
            duration = length / 30.0
            episodes.append(
                {
                    "episode_index": index,
                    "length": length,
                    "duration": duration,
                    "task_id": f"task_{index}",
                    "task_description": f"description {index}",
                    "data/chunk_index": chunk,
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
            frames.extend(
                {"episode_index": index, "frame_index": offset}
                for offset in range(length)
            )
            index += 1
            start += length
            elapsed += duration
        pq.write_table(pa.Table.from_pylist(frames), directory / "file-000.parquet")

    pq.write_table(
        pa.Table.from_pylist(episodes),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )


def test_reads_episodes_from_data_files_after_the_first(tmp_path: Path) -> None:
    """Global row indices must be rebased onto the file that actually holds them."""
    _write_multi_file_shard(tmp_path, ((4, 6), (5, 7)))
    source = ls.open_source(tmp_path)
    info = ls.load_info(source)

    third = ls.find_episode(source, info, 2)
    assert (third.data_chunk_index, third.row_start, third.row_stop) == (1, 10, 15)
    assert ls.data_file_row_origin(source, info, third) == 10

    table = ls.read_episode_table(source, info, third, columns=["episode_index"])
    assert table.column("episode_index").to_pylist() == [2] * 5

    fourth = ls.find_episode(source, info, 3)
    table = ls.read_episode_table(source, info, fourth, columns=["episode_index"])
    assert table.column("episode_index").to_pylist() == [3] * 7


def _write_container(root: Path, shards: tuple[str, ...]) -> None:
    """Write several independent shard roots side by side, as Cosmos3 publishes."""
    for name in shards:
        _write_shard(root / name, (4, 6))


def test_shard_root_resolves_to_itself(tmp_path: Path) -> None:
    _write_shard(tmp_path, (4,))
    assert ls.resolve_shard(tmp_path) == str(tmp_path)


def test_container_without_a_selector_takes_the_first_shard(tmp_path: Path) -> None:
    _write_container(tmp_path, ("shard_00", "shard_01"))
    assert ls.resolve_shard(tmp_path) == f"{tmp_path}/shard_00"


def test_container_selects_a_shard_by_index_or_name(tmp_path: Path) -> None:
    _write_container(tmp_path, ("shard_00", "shard_01"))
    expected = f"{tmp_path}/shard_01"
    assert ls.resolve_shard(tmp_path, shard=1) == expected
    assert ls.resolve_shard(tmp_path, shard="1") == expected
    assert ls.resolve_shard(tmp_path, shard="shard_01") == expected


def test_shard_index_is_zero_padded_only_below_one_hundred(tmp_path: Path) -> None:
    _write_container(tmp_path, ("shard_07", "shard_123"))
    assert ls.resolve_shard(tmp_path, shard=7) == f"{tmp_path}/shard_07"
    assert ls.resolve_shard(tmp_path, shard=123) == f"{tmp_path}/shard_123"


def test_episode_indices_restart_in_every_shard(tmp_path: Path) -> None:
    """Episode 0 is a different episode in each shard, so the shard must be named."""
    _write_shard(tmp_path / "shard_00", (4, 6))
    _write_shard(tmp_path / "shard_01", (9, 6))
    lengths = []
    for shard in (0, 1):
        source = ls.open_source(ls.resolve_shard(tmp_path, shard=shard))
        info = ls.load_info(source)
        lengths.append(ls.find_episode(source, info, 0).length)
    assert lengths == [4, 9]


def test_missing_shard_is_an_error(tmp_path: Path) -> None:
    _write_container(tmp_path, ("shard_00",))
    with pytest.raises(ls.LeRobotSourceError, match="No LeRobot shard at"):
        ls.resolve_shard(tmp_path, shard=4)


def test_selector_on_a_shard_root_is_an_error(tmp_path: Path) -> None:
    _write_shard(tmp_path, (4,))
    with pytest.raises(ls.LeRobotSourceError, match="cannot select"):
        ls.resolve_shard(tmp_path, shard=1)


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


def _video_info(
    shape: list[int], *, names: list[str] | None = None, fps: float = 30.0
) -> dict[str, object]:
    feature: dict[str, object] = {"shape": shape}
    if names is not None:
        feature["names"] = names
    return {"fps": fps, "features": {ls.VIDEO_KEY: feature}}


def test_expected_video_geometry_accepts_named_hwc_and_chw_shapes() -> None:
    hwc = _video_info([1080, 1920, 3], names=["height", "width", "channel"])
    chw = _video_info([3, 1080, 1920], names=["channel", "height", "width"])
    assert expected_video_geometry(hwc) == (1920, 1080, 30.0)
    assert expected_video_geometry(chw) == (1920, 1080, 30.0)


def test_expected_video_geometry_infers_unnamed_channel_axis() -> None:
    assert expected_video_geometry(_video_info([1080, 1920, 3])) == (
        1920,
        1080,
        30.0,
    )
    assert expected_video_geometry(_video_info([3, 1080, 1920])) == (
        1920,
        1080,
        30.0,
    )


def test_expected_video_geometry_rejects_ambiguous_shape() -> None:
    with pytest.raises(ValueError, match="Cannot infer channel axis"):
        expected_video_geometry(_video_info([3, 1080, 4]))


def test_validate_video_geometry_accepts_fps_within_tolerance() -> None:
    info = _video_info([1080, 1920, 3], names=["height", "width", "channel"], fps=30.0)
    validate_video_geometry(info, width=1920, height=1080, fps=30.0009)


@pytest.mark.parametrize(
    ("width", "height", "fps"),
    [(1919, 1080, 30.0), (1920, 1079, 30.0), (1920, 1080, 30.002)],
)
def test_validate_video_geometry_rejects_metadata_mismatch(
    width: int, height: int, fps: float
) -> None:
    info = _video_info([1080, 1920, 3], names=["height", "width", "channel"], fps=30.0)
    with pytest.raises(ValueError, match="does not match LeRobot info.json"):
        validate_video_geometry(info, width=width, height=height, fps=fps)


def _explicit_credentials() -> ls.Credentials:
    return ls.Credentials(
        access_key="AKIAEXAMPLE",
        secret_key="secret",
        endpoint="https://storage.googleapis.com",
        region="us-east4",
    )


def test_open_source_temporarily_isolates_unset_aws_profile_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = ("AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE")
    for name in variables:
        monkeypatch.delenv(name, raising=False)

    observed: dict[str, str | None] = {}
    sentinel = object()

    def fake_s3_filesystem(**_: object) -> object:
        observed.update({name: ls.os.environ.get(name) for name in variables})
        return sentinel

    monkeypatch.setattr(ls.pafs, "S3FileSystem", fake_s3_filesystem)
    source = ls.open_source(
        "s3://example-bucket/dataset", credentials=_explicit_credentials()
    )

    assert source.filesystem is sentinel
    assert observed == {name: ls.os.devnull for name in variables}
    assert all(name not in ls.os.environ for name in variables)


def test_open_source_preserves_explicit_aws_profile_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = {
        "AWS_SHARED_CREDENTIALS_FILE": "/explicit/credentials",
        "AWS_CONFIG_FILE": "/explicit/config",
    }
    for name, value in configured.items():
        monkeypatch.setenv(name, value)

    observed: dict[str, str | None] = {}

    def fake_s3_filesystem(**_: object) -> object:
        observed.update({name: ls.os.environ.get(name) for name in configured})
        return object()

    monkeypatch.setattr(ls.pafs, "S3FileSystem", fake_s3_filesystem)
    ls.open_source("s3://example-bucket/dataset", credentials=_explicit_credentials())

    assert observed == configured
    assert {name: ls.os.environ[name] for name in configured} == configured


def test_open_source_restores_profile_env_when_s3_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = ("AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE")
    for name in variables:
        monkeypatch.delenv(name, raising=False)

    def fail_s3_filesystem(**_: object) -> object:
        assert all(ls.os.environ[name] == ls.os.devnull for name in variables)
        raise RuntimeError("S3 initialization failed")

    monkeypatch.setattr(ls.pafs, "S3FileSystem", fail_s3_filesystem)
    with pytest.raises(RuntimeError, match="S3 initialization failed"):
        ls.open_source(
            "s3://example-bucket/dataset", credentials=_explicit_credentials()
        )

    assert all(name not in ls.os.environ for name in variables)


def test_presign_url_is_signed_and_scoped() -> None:
    credentials = _explicit_credentials()
    url = ls.presign_url(credentials, "bucket", "a/b/file.mp4", expires_seconds=900)
    assert url.startswith("https://storage.googleapis.com/bucket/a/b/file.mp4?")
    assert "X-Amz-Signature=" in url
    assert "X-Amz-Expires=900" in url
    assert "us-east4%2Fs3%2Faws4_request" in url
    assert credentials.secret_key not in url
