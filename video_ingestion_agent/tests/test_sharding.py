# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Tests for batch-ingestion sharding utilities.

Focus: ``aggregate_worker_progress``, which the webapp Ingest tab uses to count
processed videos from ``progress_worker_*.jsonl``. It must dedup by ``video_id``
so a stale/duplicated/orphaned record never inflates the count above the number
of distinct videos actually processed.
"""

import json
from pathlib import Path

from video_ingestion_agent.utils.sharding import aggregate_worker_progress


def _write_progress(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _rec(video_id: str, *, status: str = "success", n_clips: int = 3) -> dict:
    return {
        "video": f"data/test_videos/{video_id}.mp4",
        "video_id": video_id,
        "status": status,
        "elapsed_s": 49.7,
        "n_clips": n_clips,
        "error": None,
    }


def test_empty_dir_returns_no_records(tmp_path):
    assert aggregate_worker_progress(tmp_path) == {}


def test_counts_distinct_videos(tmp_path):
    _write_progress(
        tmp_path / "progress_worker_0.jsonl",
        [_rec("test_1"), _rec("test_2")],
    )
    records = aggregate_worker_progress(tmp_path)
    assert set(records) == {"test_1", "test_2"}
    success = sum(1 for r in records.values() if r["status"] == "success")
    assert success == 2
    assert sum(r["n_clips"] for r in records.values()) == 6


def test_stale_and_new_lines_for_same_video_count_once(tmp_path):
    """Reproduces the reported bug: a truncate/recreate race left a file
    containing a stale record plus the two real records of this run. A
    line-summing reader reported 3 videos / 9 clips; deduping by video_id
    must report 2 distinct videos / 6 clips."""
    _write_progress(
        tmp_path / "progress_worker_0.jsonl",
        [_rec("test_1"), _rec("test_2"), _rec("test_2")],  # test_2 appears twice
    )
    records = aggregate_worker_progress(tmp_path)
    assert set(records) == {"test_1", "test_2"}
    success = sum(1 for r in records.values() if r["status"] == "success")
    assert success == 2
    assert sum(r["n_clips"] for r in records.values()) == 6


def test_latest_record_wins_for_duplicate_video_id(tmp_path):
    """A re-processed video appends a second line; the latest status wins."""
    _write_progress(
        tmp_path / "progress_worker_0.jsonl",
        [
            _rec("test_1", status="error", n_clips=0),
            _rec("test_1", status="success", n_clips=4),
        ],
    )
    records = aggregate_worker_progress(tmp_path)
    assert records["test_1"]["status"] == "success"
    assert records["test_1"]["n_clips"] == 4


def test_orphan_worker_file_from_higher_shard_count_deduped(tmp_path):
    """A prior run used 2 shards; this run used 1. The orphan
    progress_worker_1.jsonl must not double-count a video already in
    progress_worker_0.jsonl."""
    _write_progress(
        tmp_path / "progress_worker_0.jsonl",
        [_rec("test_1"), _rec("test_2")],
    )
    _write_progress(
        tmp_path / "progress_worker_1.jsonl",
        [_rec("test_2")],  # orphan duplicate of test_2 from the old 2-shard run
    )
    records = aggregate_worker_progress(tmp_path)
    assert set(records) == {"test_1", "test_2"}


def test_ignores_blank_and_malformed_lines(tmp_path):
    path = tmp_path / "progress_worker_0.jsonl"
    path.write_text(
        json.dumps(_rec("test_1"))
        + "\n\n"  # blank line
        + "{not valid json\n"  # malformed
        + json.dumps(_rec("test_2"))
        + "\n"
    )
    records = aggregate_worker_progress(tmp_path)
    assert set(records) == {"test_1", "test_2"}


def test_video_id_falls_back_to_stem(tmp_path):
    _write_progress(
        tmp_path / "progress_worker_0.jsonl",
        [{"video": "data/test_videos/clip_a.mp4", "status": "success", "n_clips": 1}],
    )
    records = aggregate_worker_progress(tmp_path)
    assert set(records) == {"clip_a"}


def test_records_carry_worker_id(tmp_path):
    _write_progress(tmp_path / "progress_worker_3.jsonl", [_rec("test_1")])
    records = aggregate_worker_progress(tmp_path)
    assert records["test_1"]["worker_id"] == 3


def test_non_integer_worker_suffix_skipped(tmp_path):
    _write_progress(tmp_path / "progress_worker_main.jsonl", [_rec("test_1")])
    _write_progress(tmp_path / "progress_worker_0.jsonl", [_rec("test_2")])
    records = aggregate_worker_progress(tmp_path)
    assert set(records) == {"test_2"}
