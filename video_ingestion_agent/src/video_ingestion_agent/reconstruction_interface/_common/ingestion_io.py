# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read `video_ingestion_agent` output surfaces as a list of `IngestedSegment`s.

Two equivalent sources:
  - `runs/<run>/clips_final.jsonl` — one ClipContext per line
  - `outputs/graph.db.action_segments` joined to `video_metadata` for the path
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestedSegment:
    segment_id: str
    video_path: Path
    start_t: float
    end_t: float
    object_label: str
    action_label: str


def read_segments_from_clips_jsonl(path: Path) -> list[IngestedSegment]:
    out: list[IngestedSegment] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(
                IngestedSegment(
                    segment_id=row["clip_id"],
                    video_path=Path(row["video_path"]),
                    start_t=float(row["start_t"]),
                    end_t=float(row["end_t"]),
                    object_label=row.get("object", ""),
                    action_label=row.get("action", ""),
                )
            )
    return out


def read_segments_from_graph_db(db_path: Path) -> list[IngestedSegment]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT s.id, v.video_path, s.start_t, s.end_t,
                   COALESCE(s.primary_object_id, ''), s.action_type
            FROM action_segments s
            JOIN video_metadata v ON v.id = s.video_id
            ORDER BY v.id, s.start_t
            """
        ).fetchall()
    finally:
        conn.close()

    return [
        IngestedSegment(
            segment_id=f"segment_{row[0]:06d}",
            video_path=Path(row[1]),
            start_t=float(row[2]),
            end_t=float(row[3]),
            object_label=row[4],
            action_label=row[5],
        )
        for row in rows
    ]
