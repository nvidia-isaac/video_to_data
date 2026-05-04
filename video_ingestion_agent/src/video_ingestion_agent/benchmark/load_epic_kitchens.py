#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ground truth loader for EPIC-KITCHENS-100 annotations.

Parses EPIC_100_validation.csv and class vocabulary files into standardized
GT segment structures for benchmarking the video_ingestion_agent segmentation pipeline.

Usage:
    # As a module
    from video_ingestion_agent.benchmark.load_epic_kitchens import EpicKitchensGT
    gt = EpicKitchensGT("data/benchmark/epic_kitchens/annotations")
    segments = gt.get_segments_for_video("P01_01")

    # As a script (prints summary)
    python -m video_ingestion_agent.benchmark.load_epic_kitchens \\
        --annotations-dir data/benchmark/epic_kitchens/annotations \\
        --videos-dir data/benchmark/epic_kitchens/videos
"""

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GTSegment:
    """A single ground truth action segment from EPIC-KITCHENS-100."""

    narration_id: str
    participant_id: str
    video_id: str
    start_t: float  # seconds
    end_t: float  # seconds
    start_frame: int
    stop_frame: int
    verb: str
    verb_class: int
    noun: str
    noun_class: int
    narration: str
    all_nouns: list[str] = field(default_factory=list)
    all_noun_classes: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def timestamp_to_seconds(ts: str) -> float:
    """
    Convert EPIC-KITCHENS timestamp to seconds.

    Supports formats:
        HH:MM:SS.mmm  (e.g., "00:01:23.456")
        HH:MM:SS      (e.g., "00:01:23")
    """
    parts = ts.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {ts}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------


class EpicKitchensGT:
    """
    Load and query EPIC-KITCHENS-100 ground truth annotations.

    Attributes:
        annotations_dir: Path to cloned epic-kitchens-100-annotations repo
        segments: List of all parsed GTSegment objects
        verb_classes: Dict mapping verb_class (int) -> verb label (str)
        noun_classes: Dict mapping noun_class (int) -> noun label (str)
        verb_instances: Dict mapping verb_class (int) -> list of synonym strings
        noun_instances: Dict mapping noun_class (int) -> list of instance strings
        video_ids: Set of all video IDs in the loaded split
    """

    def __init__(
        self,
        annotations_dir: str | Path,
        split: str = "validation",
        video_filter: list[str] | None = None,
    ):
        """
        Initialize the GT loader.

        Args:
            annotations_dir: Path to the annotations directory
                (cloned epic-kitchens-100-annotations repo)
            split: Which split to load ("train", "validation", or "test")
            video_filter: Optional list of video IDs to keep.
                If None, all videos in the split are loaded.
        """
        self.annotations_dir = Path(annotations_dir)
        self.split = split

        # Load class vocabularies (labels + synonym instances)
        self.verb_classes, self.verb_instances = self._load_class_vocab("EPIC_100_verb_classes.csv")
        self.noun_classes, self.noun_instances = self._load_class_vocab("EPIC_100_noun_classes.csv")

        # Load segments
        self.segments = self._load_segments(video_filter)

        # Index by video
        self._by_video: dict[str, list[GTSegment]] = {}
        for seg in self.segments:
            self._by_video.setdefault(seg.video_id, []).append(seg)

        # Sort each video's segments by start time
        for vid in self._by_video:
            self._by_video[vid].sort(key=lambda s: s.start_t)

        self.video_ids = set(self._by_video.keys())

        logger.info(
            f"Loaded {len(self.segments)} GT segments across "
            f"{len(self.video_ids)} videos ({split} split)"
        )
        logger.info(
            f"Verb classes: {len(self.verb_classes)}, Noun classes: {len(self.noun_classes)}"
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_segments_for_video(self, video_id: str) -> list[GTSegment]:
        """Get all GT segments for a given video, sorted by start time."""
        return self._by_video.get(video_id, [])

    def get_all_video_ids(self) -> list[str]:
        """Get sorted list of all video IDs."""
        return sorted(self.video_ids)

    def get_verb_label(self, verb_class: int) -> str:
        """Look up verb label by class ID."""
        return self.verb_classes.get(verb_class, f"unknown_verb_{verb_class}")

    def get_noun_label(self, noun_class: int) -> str:
        """Look up noun label by class ID."""
        return self.noun_classes.get(noun_class, f"unknown_noun_{noun_class}")

    def filter_to_videos(self, video_ids: list[str]) -> "EpicKitchensGT":
        """Return a new EpicKitchensGT filtered to only the specified videos."""
        filtered = EpicKitchensGT.__new__(EpicKitchensGT)
        filtered.annotations_dir = self.annotations_dir
        filtered.split = self.split
        filtered.verb_classes = self.verb_classes
        filtered.noun_classes = self.noun_classes
        filtered.verb_instances = self.verb_instances
        filtered.noun_instances = self.noun_instances
        filtered.segments = [s for s in self.segments if s.video_id in set(video_ids)]
        filtered._by_video = {
            vid: segs for vid, segs in self._by_video.items() if vid in set(video_ids)
        }
        filtered.video_ids = set(filtered._by_video.keys())
        return filtered

    def summary(self) -> dict:
        """Return a summary dict of the loaded annotations."""
        durations = [s.duration for s in self.segments]
        return {
            "split": self.split,
            "total_segments": len(self.segments),
            "total_videos": len(self.video_ids),
            "participants": sorted({s.participant_id for s in self.segments}),
            "avg_segments_per_video": len(self.segments) / max(len(self.video_ids), 1),
            "avg_segment_duration_s": sum(durations) / max(len(durations), 1),
            "min_segment_duration_s": min(durations) if durations else 0,
            "max_segment_duration_s": max(durations) if durations else 0,
            "unique_verbs": len({s.verb_class for s in self.segments}),
            "unique_nouns": len({s.noun_class for s in self.segments}),
        }

    # -----------------------------------------------------------------------
    # Internal loaders
    # -----------------------------------------------------------------------

    def _load_class_vocab(self, filename: str) -> tuple[dict[int, str], dict[int, list[str]]]:
        """
        Load verb or noun class vocabulary CSV.

        Expected CSV columns: id, key, instances, category
        - ``id``: integer class index
        - ``key``: canonical label (e.g. "take")
        - ``instances``: Python-style list of synonyms/variants
          (e.g. "['grab', 'pick-up', 'fetch', ...]")

        Returns:
            Tuple of (classes, instances) where
            - classes maps class_id -> canonical label
            - instances maps class_id -> list of synonym strings
        """
        csv_path = self.annotations_dir / filename
        if not csv_path.exists():
            logger.warning(f"Class vocabulary file not found: {csv_path}")
            return {}, {}

        classes: dict[int, str] = {}
        instances: dict[int, list[str]] = {}
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # The CSV has columns: id, key, instances
                # 'id' is the class index, 'key' is the label
                try:
                    class_id = int(row.get("id", row.get("class_key", -1)))
                    label = row.get("key", row.get("class", ""))
                    classes[class_id] = label

                    # Parse the instances (synonyms) column
                    instances_str = row.get("instances", "[]")
                    instances[class_id] = _parse_list_field(instances_str)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping malformed row in {filename}: {row} ({e})")

        total_syns = sum(len(v) for v in instances.values())
        logger.info(
            f"Loaded {len(classes)} classes from {filename} ({total_syns} total synonym instances)"
        )
        return classes, instances

    def _load_segments(self, video_filter: list[str] | None) -> list[GTSegment]:
        """Load action segments from the annotations CSV."""
        csv_filename = f"EPIC_100_{self.split}.csv"
        csv_path = self.annotations_dir / csv_filename

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Annotations CSV not found: {csv_path}\n"
                f"Make sure you've cloned epic-kitchens-100-annotations into {self.annotations_dir}"
            )

        video_filter_set = set(video_filter) if video_filter else None
        segments = []

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row["video_id"]

                # Filter if requested
                if video_filter_set and video_id not in video_filter_set:
                    continue

                try:
                    # Parse all_nouns (stored as a Python list string like "['pan', 'oil']")
                    all_nouns_str = row.get("all_nouns", "[]")
                    all_nouns = _parse_list_field(all_nouns_str)

                    all_noun_classes_str = row.get("all_noun_classes", "[]")
                    all_noun_classes = [int(x) for x in _parse_list_field(all_noun_classes_str)]

                    segment = GTSegment(
                        narration_id=row["narration_id"],
                        participant_id=row["participant_id"],
                        video_id=video_id,
                        start_t=timestamp_to_seconds(row["start_timestamp"]),
                        end_t=timestamp_to_seconds(row["stop_timestamp"]),
                        start_frame=int(row["start_frame"]),
                        stop_frame=int(row["stop_frame"]),
                        verb=row["verb"],
                        verb_class=int(row["verb_class"]),
                        noun=row["noun"],
                        noun_class=int(row["noun_class"]),
                        narration=row["narration"],
                        all_nouns=all_nouns,
                        all_noun_classes=all_noun_classes,
                    )
                    segments.append(segment)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping malformed row: {row.get('narration_id', '?')} ({e})")

        return segments


def _parse_list_field(s: str) -> list[str]:
    """
    Parse a Python-style list string from CSV.

    Examples:
        "['pan', 'oil']" -> ['pan', 'oil']
        "[0, 1]" -> ['0', '1']
        "[]" -> []
    """
    s = s.strip()
    if s in ("[]", ""):
        return []

    # Remove brackets
    s = s.strip("[]")
    items = []
    for item in s.split(","):
        item = item.strip().strip("'\"")
        if item:
            items.append(item)
    return items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for inspecting GT annotations."""
    parser = argparse.ArgumentParser(
        description="Load and inspect EPIC-KITCHENS-100 ground truth annotations"
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/benchmark/epic_kitchens/annotations",
        help="Path to cloned epic-kitchens-100-annotations repo",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="data/benchmark/epic_kitchens/videos",
        help="Path to downloaded EPIC-KITCHENS videos (for filtering)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to load",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Show segments for a specific video ID",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Determine video filter from downloaded videos
    videos_dir = Path(args.videos_dir)
    video_filter = None
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.MP4")) + list(videos_dir.glob("*.mp4"))
        if video_files:
            video_filter = [f.stem for f in video_files]
            logger.info(f"Found {len(video_filter)} downloaded videos: {video_filter}")

    # Load GT
    try:
        gt = EpicKitchensGT(
            annotations_dir=args.annotations_dir,
            split=args.split,
            video_filter=video_filter,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Print summary
    summary = gt.summary()
    print("\n" + "=" * 60)
    print(f"EPIC-KITCHENS-100 Ground Truth Summary ({args.split} split)")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        elif isinstance(value, list) and len(value) > 10:
            print(f"  {key}: [{', '.join(value[:5])}, ... +{len(value) - 5} more]")
        else:
            print(f"  {key}: {value}")

    # Show specific video
    if args.video_id:
        segments = gt.get_segments_for_video(args.video_id)
        print(f"\n{'=' * 60}")
        print(f"Segments for {args.video_id} ({len(segments)} segments):")
        print(f"{'=' * 60}")
        for seg in segments:
            print(
                f"  [{seg.start_t:7.2f}s - {seg.end_t:7.2f}s] "
                f"({seg.duration:.1f}s) "
                f"{seg.verb} {seg.noun} | {seg.narration}"
            )
    else:
        # Show per-video counts
        print(f"\n{'=' * 60}")
        print("Per-video segment counts:")
        print(f"{'=' * 60}")
        for vid in gt.get_all_video_ids():
            segs = gt.get_segments_for_video(vid)
            total_dur = sum(s.duration for s in segs)
            print(f"  {vid}: {len(segs):3d} segments, {total_dur:.1f}s total action time")


if __name__ == "__main__":
    main()
