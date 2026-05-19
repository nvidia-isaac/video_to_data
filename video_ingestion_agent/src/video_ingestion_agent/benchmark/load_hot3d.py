#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ground-truth loader for HOT3D contact-segmentation benchmark.

The HOT3D ground truth is pre-generated externally and stored as one
JSONL file with one row per per-object manipulation event. Schema:

    {"clip_id": "P0001_4bf4e21a__obj0_ev0000",
     "source_sequence_id": "P0001_4bf4e21a",
     "video_path": "<absolute path to MP4>",
     "object_uid": "96945373046044",
     "object_name": "food_vegetables",
     "hands_involved": ["right"],
     "start_t": 10.766667, "end_t": 14.366667,
     "contact_start_t": 11.566667, "contact_end_t": 14.366667,
     "fps": 30.0,
     ...}

The companion evaluator is ``evaluate_hot3d.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HotGTSegment:
    """A single HOT3D ground-truth manipulation event.

    Mirrors the EPIC ``GTSegment`` interface where it overlaps so the temporal
    IoU helpers from ``evaluate.py`` can be reused unchanged. Verb / noun /
    class fields are intentionally absent — HOT3D doesn't ship verbs, and we
    benchmark on temporal alignment + object identity only.
    """

    clip_id: str
    video_id: str  # source sequence id (matches the agent's per-video grouping)
    video_path: str
    object_uid: str
    object_name: str
    hands_involved: list[str]
    start_t: float
    end_t: float
    contact_start_t: float
    contact_end_t: float
    fps: float
    # Convenience copies for matching
    start_frame: int = 0
    end_frame: int = 0
    contact_start_frame: int = 0
    contact_end_frame: int = 0
    object_body_idx: int = -1
    event_idx: int = -1

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t

    @property
    def contact_duration(self) -> float:
        return self.contact_end_t - self.contact_start_t


class Hot3dGT:
    """Load and query HOT3D ground-truth segments from JSONL.

    Attributes:
        ground_truth_path: Path to the JSONL ground-truth file.
        segments: All parsed ``HotGTSegment`` objects.
        video_ids: Set of source sequence ids represented.
    """

    def __init__(
        self,
        ground_truth_path: str | Path,
        video_filter: list[str] | None = None,
    ):
        self.ground_truth_path = Path(ground_truth_path)
        self.segments: list[HotGTSegment] = []
        self._by_video: dict[str, list[HotGTSegment]] = {}

        keep: set[str] | None = set(video_filter) if video_filter else None
        with self.ground_truth_path.open() as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                vid = row.get("source_sequence_id") or row.get("video_id")
                if keep is not None and vid not in keep:
                    continue
                seg = HotGTSegment(
                    clip_id=row["clip_id"],
                    video_id=vid,
                    video_path=row.get("video_path", ""),
                    object_uid=row.get("object_uid", ""),
                    object_name=row.get("object_name", ""),
                    hands_involved=list(row.get("hands_involved", [])),
                    start_t=float(row["start_t"]),
                    end_t=float(row["end_t"]),
                    contact_start_t=float(row.get("contact_start_t", row["start_t"])),
                    contact_end_t=float(row.get("contact_end_t", row["end_t"])),
                    fps=float(row.get("fps", 0.0)),
                    start_frame=int(row.get("start_frame", 0)),
                    end_frame=int(row.get("end_frame", 0)),
                    contact_start_frame=int(row.get("contact_start_frame", 0)),
                    contact_end_frame=int(row.get("contact_end_frame", 0)),
                    object_body_idx=int(row.get("object_body_idx", -1)),
                    event_idx=int(row.get("event_idx", -1)),
                )
                self.segments.append(seg)
                self._by_video.setdefault(vid, []).append(seg)

        for vid in self._by_video:
            self._by_video[vid].sort(key=lambda s: s.start_t)

        self.video_ids = set(self._by_video.keys())
        logger.info(
            "Loaded %d HOT3D GT segments across %d videos from %s",
            len(self.segments),
            len(self.video_ids),
            self.ground_truth_path,
        )

    def get_segments_for_video(self, video_id: str) -> list[HotGTSegment]:
        return self._by_video.get(video_id, [])

    def get_all_video_ids(self) -> list[str]:
        return sorted(self.video_ids)

    def video_path(self, video_id: str) -> str | None:
        segs = self._by_video.get(video_id)
        return segs[0].video_path if segs else None

    def summary(self) -> dict:
        if not self.segments:
            return {"total_segments": 0, "total_videos": 0}
        durations = [s.duration for s in self.segments]
        contact_durations = [s.contact_duration for s in self.segments]
        return {
            "total_segments": len(self.segments),
            "total_videos": len(self.video_ids),
            "avg_segments_per_video": len(self.segments) / max(len(self.video_ids), 1),
            "duration": {
                "min": min(durations),
                "median": sorted(durations)[len(durations) // 2],
                "mean": sum(durations) / len(durations),
                "max": max(durations),
            },
            "contact_duration": {
                "min": min(contact_durations),
                "mean": sum(contact_durations) / len(contact_durations),
                "max": max(contact_durations),
            },
            "unique_objects": len({s.object_name for s in self.segments}),
            "hand_breakdown": {
                "right_only": sum(1 for s in self.segments if s.hands_involved == ["right"]),
                "left_only": sum(1 for s in self.segments if s.hands_involved == ["left"]),
                "both": sum(1 for s in self.segments if set(s.hands_involved) == {"right", "left"}),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a HOT3D GT JSONL.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--video-filter",
        nargs="*",
        default=None,
        help="Optional list of source_sequence_ids to keep.",
    )
    args = parser.parse_args()
    gt = Hot3dGT(args.ground_truth, video_filter=args.video_filter)
    print(json.dumps(gt.summary(), indent=2))


if __name__ == "__main__":
    main()
