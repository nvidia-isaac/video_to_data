# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict single-motion Parquet discovery and projected row loading."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pyarrow.parquet as pq


@dataclass(frozen=True)
class ParquetRow:
    """One physical Parquet row plus Hive partition metadata from its path."""

    path: Path
    cells: Mapping[str, Any]


def resolve_parquet(path: str | Path) -> Path:
    """Resolve a file or a directory containing exactly one Parquet file."""
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.exists():
        raise FileNotFoundError(f"reference path does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"reference path must be a file or directory: {resolved}")

    files = sorted(candidate for candidate in resolved.rglob("*.parquet") if candidate.is_file())
    if not files:
        raise FileNotFoundError(f"no .parquet file under reference directory: {resolved}")
    if len(files) != 1:
        raise ValueError(f"reference directory must contain exactly one .parquet file, found {len(files)}: {resolved}")
    return files[0]


def read_parquet_row(path: str | Path, columns: Iterable[str] | None = None) -> ParquetRow:
    """Read one row without materializing unrequested columns or inferring a dataset schema."""
    parquet_path = resolve_parquet(path)
    parquet = pq.ParquetFile(parquet_path)
    if parquet.metadata.num_rows != 1:
        raise ValueError(
            f"reference parquet must contain exactly one row, got {parquet.metadata.num_rows}: {parquet_path}"
        )

    physical_names = tuple(parquet.schema_arrow.names)
    requested = None if columns is None else tuple(dict.fromkeys(columns))
    projected = physical_names if requested is None else tuple(name for name in requested if name in physical_names)
    table = parquet.read(columns=list(projected))
    cells = {name: table[name][0].as_py() for name in projected}

    partitions = _partition_values(parquet_path)
    for name, value in partitions.items():
        if name in cells and cells[name] is not None and str(cells[name]) != value:
            raise ValueError(
                f"physical column {name!r}={cells[name]!r} disagrees with partition value {value!r}: {parquet_path}"
            )
        if requested is None or name in requested:
            cells.setdefault(name, value)
    return ParquetRow(path=parquet_path, cells=cells)


def _partition_values(parquet_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for parent in parquet_path.parents:
        if "=" not in parent.name:
            continue
        key, encoded_value = parent.name.split("=", 1)
        if not key:
            continue
        value = unquote(encoded_value)
        if key in values and values[key] != value:
            raise ValueError(
                f"conflicting {key!r} partition values {values[key]!r} and {value!r}: {parquet_path}"
            )
        values[key] = value
    return values
