# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for strict single-reference Parquet discovery and row projection."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flash_chord.data.parquet import read_parquet_row, resolve_parquet


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data), path)


def test_read_projected_row_recovers_partition_values(tmp_path):
    path = tmp_path / "sequence_id=demo%20sequence" / "robot_name=vega" / "data.parquet"
    _write(path, {"keep": [[1, 2]], "omit": [b"large payload"]})

    result = read_parquet_row(path.parent, columns=("keep", "sequence_id", "robot_name"))

    assert result.path == path
    assert result.cells == {
        "keep": [1, 2],
        "robot_name": "vega",
        "sequence_id": "demo sequence",
    }


def test_resolve_parquet_rejects_zero_or_multiple_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .parquet"):
        resolve_parquet(tmp_path)

    _write(tmp_path / "first.parquet", {"value": [1]})
    _write(tmp_path / "nested" / "second.parquet", {"value": [2]})
    with pytest.raises(ValueError, match="exactly one"):
        resolve_parquet(tmp_path)


def test_read_parquet_row_requires_exactly_one_row(tmp_path):
    path = tmp_path / "data.parquet"
    _write(path, {"value": [1, 2]})
    with pytest.raises(ValueError, match="exactly one row"):
        read_parquet_row(path)


def test_read_parquet_row_rejects_physical_partition_disagreement(tmp_path):
    path = tmp_path / "robot_name=vega" / "data.parquet"
    _write(path, {"robot_name": ["other"]})
    with pytest.raises(ValueError, match="disagrees with partition"):
        read_parquet_row(path)


def test_resolve_parquet_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_parquet(tmp_path / "missing")
