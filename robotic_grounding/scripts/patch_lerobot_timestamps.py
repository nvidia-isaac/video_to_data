# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patch episode video timestamps + data-file locators in existing LeRobot v3 datasets.

Older exports (before the export_lerobot.py fix) had two metadata bugs vs. the actual
single-file layout (all episodes concatenated into one data parquet + one mp4 per camera):
  1. ``videos/<cam>/from_timestamp``/``to_timestamp`` were **episode-local** (every episode
     starts at 0.0) instead of cumulative on the merged-file timeline, so a standard LeRobot
     loader decodes episode 0's frames for every episode.
  2. ``data/file_index`` was ``episode_index`` (0,1,2,…) instead of ``0``, pointing every
     episode past the first at a data/chunk-000/file-<NNN>.parquet that doesn't exist.

This rewrites the timestamps (from each episode's ``length`` + fps) and sets
``data/chunk_index``/``data/file_index`` to 0. Metadata-only (videos and data parquets are
untouched) and idempotent.

Usage (from repo root):
  python scripts/patch_lerobot_timestamps.py datasets/Taco_Datagen_DR100
  python scripts/patch_lerobot_timestamps.py datasets/Taco_Datagen_DR100/<task>  # single task
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq


def _task_dirs(root: str) -> list[str]:
    """Return the LeRobot task dirs under ``root`` (or ``[root]`` if it is one itself)."""
    if os.path.isfile(os.path.join(root, "meta", "info.json")):
        return [root]
    return sorted(
        os.path.dirname(os.path.dirname(p))
        for p in glob.glob(os.path.join(root, "*", "meta", "info.json"))
    )


def patch_task(task_dir: str) -> str:
    """Rewrite cumulative from/to_timestamps for every episodes parquet in one task dir."""
    with open(os.path.join(task_dir, "meta", "info.json")) as fh:
        fps = float(json.load(fh).get("fps", 30))
    ep_parquets = glob.glob(
        os.path.join(task_dir, "meta", "episodes", "**", "*.parquet"), recursive=True
    )
    if not ep_parquets:
        return f"SKIP {os.path.basename(task_dir)} (no episodes parquet)"
    patched = 0
    for path in ep_parquets:
        t = pq.read_table(path)
        cols = set(t.schema.names)
        cams = [
            m.group(1)
            for c in t.schema.names
            if (m := re.match(r"videos/(.+)/from_timestamp$", c))
        ]
        if not cams:
            continue
        epi = t.column("episode_index").to_pylist()
        length = dict(zip(epi, t.column("length").to_pylist(), strict=True))
        # cumulative frame offset by ascending episode_index (video is written in that order)
        offset, acc = {}, 0
        for e in sorted(length):
            offset[e] = acc
            acc += length[e]
        for cam in cams:
            fcol, tcol = f"videos/{cam}/from_timestamp", f"videos/{cam}/to_timestamp"
            new_from = [offset[e] / fps for e in epi]
            new_to = [(offset[e] + length[e]) / fps for e in epi]
            for name, vals in ((fcol, new_from), (tcol, new_to)):
                if name not in cols:
                    continue
                idx = t.schema.get_field_index(name)
                t = t.set_column(
                    idx, name, pa.array(vals, type=t.schema.field(name).type)
                )
        # Single concatenated data file -> data locators are 0/0 for every episode.
        # Older exports wrote data/file_index = episode_index, pointing at missing files.
        n_rows = len(epi)
        for name in ("data/chunk_index", "data/file_index"):
            if name in cols:
                idx = t.schema.get_field_index(name)
                t = t.set_column(
                    idx, name, pa.array([0] * n_rows, type=t.schema.field(name).type)
                )
        pq.write_table(t, path)
        patched += 1
    return (
        f"OK   {os.path.basename(task_dir)} ({patched} episodes parquet, fps={fps:g})"
    )


def main() -> None:
    """CLI entry: patch every task dir under the given root."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "root", help="Dataset root (holding task dirs) or a single task dir."
    )
    args = parser.parse_args()
    dirs = _task_dirs(args.root)
    if not dirs:
        print(f"No LeRobot task dirs found under {args.root}")
        return
    ok = 0
    for d in dirs:
        msg = patch_task(d)
        print(msg, flush=True)
        ok += msg.startswith("OK")
    print(f"\nDONE {ok}/{len(dirs)} task dirs patched")


if __name__ == "__main__":
    main()
