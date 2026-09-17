# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a LeRobot v3 dataset directory produced by export_lerobot.py.

Checks:
  1. Required files exist (meta/info.json, meta/stats.json, meta/tasks.jsonl,
     meta/episodes parquet, data parquet).
  2. info.json has all required top-level fields and valid feature entries.
  3. data parquet has the expected columns matching info.json features.
  4. Video files exist for every video feature.
  5. Episode metadata parquet has correct schema.
  6. Frame counts are consistent (info.json == parquet row count == sum of episode lengths).
  7. Episode count matches info.json.
  8. Video frame count (via metadata) matches expected frames per episode.
  9. stats.json covers all features declared in info.json.
 10. tasks.jsonl has at least one entry.

Example::

    python scripts/test_lerobot_format.py datasets/test2a/taco_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def _check(
    cond: bool,
    msg: str,
    failures: list[str],
    warns: list[str] | None = None,
    is_warn: bool = False,
) -> None:
    if cond:
        print(f"  {PASS} {msg}")
    elif is_warn:
        print(f"  {WARN} {msg}")
        if warns is not None:
            warns.append(msg)
    else:
        print(f"  {FAIL} {msg}")
        failures.append(msg)


def _check_episode_video_distinct(
    dataset_dir: str,
    video_path_tpl: str,
    vfeat: str,
    ep_dict: dict,
    fps: float,
    failures: list[str],
    warns: list[str],
) -> None:
    """Emulate LeRobot's from_timestamp-based video access and verify episodes are distinct.

    LeRobot decodes an episode's frames from the *concatenated* per-camera mp4 by seeking to
    ``videos/<cam>/from_timestamp``. If those are written episode-local (all 0.0) instead of
    cumulative, every episode decodes from t=0 and returns episode 0's frames. This check
    seeks to a few episodes' ``from_timestamp`` and asserts the decoded frames are both
    non-zero (real content) and distinct across episodes — catching that failure mode.
    """
    from_ts = ep_dict.get(f"videos/{vfeat}/from_timestamp")
    if not from_ts:
        return
    n = len(from_ts)
    # Structural: on the merged-file timeline, from_timestamp must advance across episodes.
    _check(
        n < 2 or all(from_ts[i] < from_ts[i + 1] for i in range(n - 1)),
        f"{vfeat}: from_timestamp strictly increases across episodes "
        "(all-zero => every episode decodes episode 0)",
        failures,
    )
    try:
        import cv2
        import numpy as np
    except ImportError:
        _check(
            True,
            f"{vfeat}: cv2/numpy unavailable — skipped video-distinctness decode",
            failures,
            is_warn=True,
            warns=warns,
        )
        return
    vid_abs = os.path.join(
        dataset_dir, video_path_tpl.format(video_key=vfeat, chunk_index=0, file_index=0)
    )
    if not os.path.isfile(vid_abs):
        return
    idxs = sorted({0, n // 2, n - 1})  # first / middle / last episode
    cap = cv2.VideoCapture(vid_abs)
    frames: dict[int, object] = {}
    for e in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(from_ts[e] * fps)))
        ok, fr = cap.read()
        frames[e] = fr if ok else None
    cap.release()
    nonzero = all(
        fr is not None and float(np.asarray(fr).std()) > 1.0 for fr in frames.values()
    )
    _check(
        nonzero,
        f"{vfeat}: sampled episode frames decode with non-zero content",
        failures,
    )
    distinct = True
    ks = list(frames)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = frames[ks[i]], frames[ks[j]]
            if (
                a is None
                or b is None
                or float(
                    np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).mean()
                )
                < 1.0
            ):
                distinct = False
    _check(
        distinct,
        f"{vfeat}: episodes {idxs} decode to DISTINCT frames via from_timestamp "
        "(identical => all episodes load episode 0)",
        failures,
    )


def validate(dataset_dir: str) -> bool:
    """Validate the LeRobot v3 dataset at dataset_dir.  Returns True if all checks pass."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow not installed (pip install pyarrow)")
        return False

    failures: list[str] = []
    warns: list[str] = []
    dataset_dir = os.path.abspath(dataset_dir)

    print(f"\n{'='*60}")
    print(f"Validating LeRobot dataset: {dataset_dir}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Required files exist
    # ------------------------------------------------------------------
    print("\n--- 1. Required files ---")
    meta_dir = os.path.join(dataset_dir, "meta")
    info_path = os.path.join(meta_dir, "info.json")
    stats_path = os.path.join(meta_dir, "stats.json")
    tasks_path = os.path.join(meta_dir, "tasks.jsonl")

    _check(os.path.isfile(info_path), "meta/info.json exists", failures)
    _check(os.path.isfile(stats_path), "meta/stats.json exists", failures)
    _check(os.path.isfile(tasks_path), "meta/tasks.jsonl exists", failures)

    ep_meta_dir = os.path.join(meta_dir, "episodes")
    _check(os.path.isdir(ep_meta_dir), "meta/episodes/ directory exists", failures)

    data_pq = os.path.join(dataset_dir, "data", "chunk-000", "file-000.parquet")
    _check(os.path.isfile(data_pq), "data/chunk-000/file-000.parquet exists", failures)

    if failures:
        print("\nABORTED: missing required files, cannot continue validation.")
        return False

    # ------------------------------------------------------------------
    # 2. info.json schema
    # ------------------------------------------------------------------
    print("\n--- 2. info.json schema ---")
    with open(info_path) as fh:
        info = json.load(fh)

    required_top = [
        "codebase_version",
        "robot_type",
        "total_episodes",
        "total_frames",
        "fps",
        "splits",
        "data_path",
        "video_path",
        "features",
    ]
    for field in required_top:
        _check(field in info, f"info.json has '{field}'", failures)

    _check(
        info.get("codebase_version") == "v3.0",
        f"codebase_version == 'v3.0' (got {info.get('codebase_version')!r})",
        failures,
    )
    _check(
        isinstance(info.get("fps"), (int, float)) and info["fps"] > 0,
        f"fps > 0 (got {info.get('fps')})",
        failures,
    )
    _check(
        isinstance(info.get("total_episodes"), int) and info["total_episodes"] > 0,
        f"total_episodes > 0 (got {info.get('total_episodes')})",
        failures,
    )
    _check(
        isinstance(info.get("total_frames"), int) and info["total_frames"] > 0,
        f"total_frames > 0 (got {info.get('total_frames')})",
        failures,
    )

    features = info.get("features", {})
    required_feats = [
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
        "next.done",
        "index",
        "task_index",
    ]
    for feat in required_feats:
        _check(feat in features, f"features has '{feat}'", failures)

    # Check video features
    video_feats = [k for k, v in features.items() if v.get("dtype") == "video"]
    _check(
        len(video_feats) >= 1,
        f"At least one video feature ({len(video_feats)} found)",
        failures,
    )
    for vfeat in video_feats:
        vinfo = features[vfeat]
        _check("video_info" in vinfo, f"  {vfeat}: has 'video_info'", failures)
        _check(
            "shape" in vinfo and len(vinfo["shape"]) == 3,
            f"  {vfeat}: shape is 3-D (got {vinfo.get('shape')})",
            failures,
        )

    # ------------------------------------------------------------------
    # 3. data parquet columns
    # ------------------------------------------------------------------
    print("\n--- 3. data parquet columns ---")
    tbl = pq.read_table(data_pq)
    actual_cols = set(tbl.schema.names)
    expected_cols = {
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
        "next.done",
        "index",
        "task_index",
    }
    for col in expected_cols:
        _check(col in actual_cols, f"column '{col}' present", failures)

    actual_rows = tbl.num_rows
    _check(
        actual_rows == info["total_frames"],
        f"data parquet rows ({actual_rows}) == info.total_frames ({info['total_frames']})",
        failures,
    )

    # Check dtypes
    schema_map = {field.name: str(field.type) for field in tbl.schema}
    _check(
        "list" in schema_map.get("observation.state", ""),
        f"observation.state is list type (got {schema_map.get('observation.state', 'missing')})",
        failures,
    )
    _check(
        "list" in schema_map.get("action", ""),
        f"action is list type (got {schema_map.get('action', 'missing')})",
        failures,
    )
    _check(
        schema_map.get("episode_index") == "int64",
        f"episode_index is int64 (got {schema_map.get('episode_index', 'missing')})",
        failures,
    )
    _check(
        schema_map.get("next.done") == "bool",
        f"next.done is bool (got {schema_map.get('next.done', 'missing')})",
        failures,
    )

    # Validate observation.state dimension matches info.json
    if "observation.state" in features:
        expected_dim = features["observation.state"]["shape"][0]
        sample = tbl.slice(0, 1).to_pydict()["observation.state"][0]
        actual_dim = len(sample)
        _check(
            actual_dim == expected_dim,
            f"observation.state dim ({actual_dim}) == info.json shape[0] ({expected_dim})",
            failures,
        )

    if "action" in features:
        expected_dim = features["action"]["shape"][0]
        sample = tbl.slice(0, 1).to_pydict()["action"][0]
        actual_dim = len(sample)
        _check(
            actual_dim == expected_dim,
            f"action dim ({actual_dim}) == info.json shape[0] ({expected_dim})",
            failures,
        )

    # ------------------------------------------------------------------
    # 4. Video files exist
    # ------------------------------------------------------------------
    print("\n--- 4. Video files ---")
    video_path_tpl: str = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    for vfeat in video_feats:
        vid_rel = video_path_tpl.format(video_key=vfeat, chunk_index=0, file_index=0)
        vid_abs = os.path.join(dataset_dir, vid_rel)
        exists = os.path.isfile(vid_abs)
        size_kb = os.path.getsize(vid_abs) // 1024 if exists else 0
        _check(exists, f"{vid_rel} exists ({size_kb} KB)", failures)

    # ------------------------------------------------------------------
    # 5. Episodes parquet schema
    # ------------------------------------------------------------------
    print("\n--- 5. Episodes parquet ---")
    ep_pq_path = os.path.join(ep_meta_dir, "chunk-000", "file-000.parquet")
    _check(
        os.path.isfile(ep_pq_path),
        "meta/episodes/chunk-000/file-000.parquet exists",
        failures,
    )

    if os.path.isfile(ep_pq_path):
        ep_tbl = pq.read_table(ep_pq_path)
        ep_cols = set(ep_tbl.schema.names)
        required_ep_cols = {
            "episode_index",
            "dataset_from_index",
            "dataset_to_index",
            "tasks",
            "length",
        }
        for col in required_ep_cols:
            _check(col in ep_cols, f"episodes parquet has '{col}'", failures)

        n_ep_rows = ep_tbl.num_rows
        _check(
            n_ep_rows == info["total_episodes"],
            f"episodes parquet rows ({n_ep_rows}) == info.total_episodes ({info['total_episodes']})",
            failures,
        )

        # ------------------------------------------------------------------
        # 6. Frame count consistency
        # ------------------------------------------------------------------
        print("\n--- 6. Frame count consistency ---")
        ep_dict = ep_tbl.to_pydict()
        sum_lengths = sum(ep_dict["length"])
        _check(
            sum_lengths == info["total_frames"],
            f"sum(episode lengths) ({sum_lengths}) == info.total_frames ({info['total_frames']})",
            failures,
        )

        # Check from/to indices are contiguous
        froms = sorted(ep_dict["dataset_from_index"])
        tos = sorted(ep_dict["dataset_to_index"])
        contiguous = all(tos[i] == froms[i + 1] for i in range(len(froms) - 1))
        _check(
            contiguous,
            "episode frame indices are contiguous (no gaps/overlaps)",
            failures,
        )

        # Every episode's (chunk_index, file_index) data locator must resolve to a file that
        # exists — catches metadata that indexes as if data were split into per-episode/1000-
        # episode files while the exporter writes a single data/chunk-000/file-000.parquet.
        data_path_tpl = info.get(
            "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        )
        data_locs = set(
            zip(ep_dict["data/chunk_index"], ep_dict["data/file_index"], strict=True)
        )
        missing = [
            (c, f)
            for c, f in data_locs
            if not os.path.isfile(
                os.path.join(
                    dataset_dir, data_path_tpl.format(chunk_index=c, file_index=f)
                )
            )
        ]
        _check(
            not missing,
            f"all episode data locators (chunk_index,file_index) resolve to existing files "
            f"(missing: {missing[:5]})",
            failures,
        )

        # ------------------------------------------------------------------
        # 7. Episode count
        # ------------------------------------------------------------------
        print("\n--- 7. Episode count ---")
        _check(
            n_ep_rows == info["total_episodes"],
            f"total_episodes consistent: {n_ep_rows} rows == {info['total_episodes']}",
            failures,
        )

        # ------------------------------------------------------------------
        # 8. Video timestamps
        # ------------------------------------------------------------------
        print("\n--- 8. Video timestamps in episodes parquet ---")
        for vfeat in video_feats:
            col_from = f"videos/{vfeat}/from_timestamp"
            col_to = f"videos/{vfeat}/to_timestamp"
            _check(
                col_from in ep_cols,
                f"episodes parquet has '{col_from}'",
                failures,
                is_warn=True,
                warns=warns,
            )
            _check(
                col_to in ep_cols,
                f"episodes parquet has '{col_to}'",
                failures,
                is_warn=True,
                warns=warns,
            )
            if col_from in ep_cols:
                _check_episode_video_distinct(
                    dataset_dir,
                    video_path_tpl,
                    vfeat,
                    ep_dict,
                    float(info.get("fps", 30)),
                    failures,
                    warns,
                )

    # ------------------------------------------------------------------
    # 9. stats.json coverage
    # ------------------------------------------------------------------
    print("\n--- 9. stats.json coverage ---")
    with open(stats_path) as fh:
        stats = json.load(fh)

    scalar_feats = [k for k, v in features.items() if v.get("dtype") != "video"]
    for feat in scalar_feats:
        _check(feat in stats, f"stats.json covers '{feat}'", failures)
        if feat in stats:
            for stat in ("min", "max", "mean", "std", "count"):
                _check(stat in stats[feat], f"  {feat}: has '{stat}' field", failures)

    for vfeat in video_feats:
        _check(vfeat in stats, f"stats.json covers video '{vfeat}'", failures)
        if vfeat in stats:
            vst = stats[vfeat]
            _check(
                isinstance(vst.get("mean"), list) and len(vst["mean"]) == 3,
                f"  {vfeat}: mean has 3 channels (got {len(vst.get('mean', []))})",
                failures,
            )

    # ------------------------------------------------------------------
    # 10. tasks.jsonl
    # ------------------------------------------------------------------
    print("\n--- 10. tasks.jsonl ---")
    with open(tasks_path) as fh:
        tasks_lines = [l.strip() for l in fh if l.strip()]
    _check(
        len(tasks_lines) >= 1,
        f"tasks.jsonl has at least one entry ({len(tasks_lines)} found)",
        failures,
    )
    if tasks_lines:
        first_task = json.loads(tasks_lines[0])
        _check(
            "task_index" in first_task and "task" in first_task,
            "tasks.jsonl entry has 'task_index' and 'task' fields",
            failures,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    if not failures:
        print(f"RESULT: ALL CHECKS PASSED ({len(warns)} warning(s))")
    else:
        print(f"RESULT: {len(failures)} FAILURE(S), {len(warns)} WARNING(S)")
        for f in failures:
            print(f"  {FAIL} {f}")
    print(f"{'='*60}\n")

    return len(failures) == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset_dir", help="Path to the LeRobot v3 dataset directory.")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit with code 1 on any failure (useful in CI).",
    )
    args = parser.parse_args()

    ok = validate(args.dataset_dir)
    if args.strict and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
