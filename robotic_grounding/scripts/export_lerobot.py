# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a RecorderManager HDF5 dataset to LeRobot v3 format.

LeRobot v3 layout produced::

    <output_dir>/
      meta/
        info.json
        stats.json
        tasks.jsonl
        episodes/chunk-000/file-000.parquet
      data/chunk-000/
        file-000.parquet
      videos/observation.images.<cam>/chunk-000/
        file-000.mp4

Example::

    python scripts/export_lerobot.py \\
        --hdf5 datasets/test2a/taco_smoke.hdf5 \\
        --output_dir datasets/test2a/taco_smoke_lerobot \\
        --fps 30 \\
        --task "robot manipulation"
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
import pyarrow as pa

_CHUNK_SIZE = 1000  # episodes per chunk (v3 convention)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _scalar_stats(arr: np.ndarray) -> dict:
    """Stats dict for a 2-D array (n_frames, dim).  Values are plain lists."""
    a = arr.astype(np.float64)
    return {
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "count": [int(len(a))],
    }


def _scalar_stats_1d(arr: np.ndarray) -> dict:
    """Stats for a 1-D array (timestamps, indices, etc.).  Wraps scalars in [...]."""
    a = arr.astype(np.float64)
    return {
        "min": [float(a.min())],
        "max": [float(a.max())],
        "mean": [float(a.mean())],
        "std": [float(a.std())],
        "count": [int(len(a))],
    }


def _bool_stats(arr: np.ndarray) -> dict:
    a = arr.astype(np.float64)
    return {
        "min": [bool(arr.min())],
        "max": [bool(arr.max())],
        "mean": [float(a.mean())],
        "std": [float(a.std())],
        "count": [int(len(arr))],
    }


def _image_stats(frames_uint8: np.ndarray, sample_every: int = 10) -> dict:
    """Per-channel image stats in LeRobot shape [[[ val ]]] per channel.

    frames_uint8: (N, H, W, 3) uint8, values in [0, 255].
    LeRobot normalises images to [0, 1] before computing stats.
    """
    sampled = frames_uint8[::sample_every].astype(np.float32) / 255.0  # (M, H, W, 3)
    C = sampled.shape[-1]
    chan_min, chan_max, chan_mean, chan_std = [], [], [], []
    for c in range(C):
        ch = sampled[..., c]
        chan_min.append([[float(ch.min())]])
        chan_max.append([[float(ch.max())]])
        chan_mean.append([[float(ch.mean())]])
        chan_std.append([[float(ch.std())]])
    return {
        "min": chan_min,
        "max": chan_max,
        "mean": chan_mean,
        "std": chan_std,
        "count": [int(len(sampled))],
    }


# ---------------------------------------------------------------------------
# Episodes parquet helper
# ---------------------------------------------------------------------------


def _build_episodes_table(
    episodes_meta: list[dict], camera_names: list[str]
) -> pa.Table:

    n = len(episodes_meta)

    def col(name, vals, pa_type):
        return pa.chunked_array([pa.array(vals, type=pa_type)])

    # Base columns
    arrays: dict[str, pa.ChunkedArray] = {}
    arrays["episode_index"] = col(
        "episode_index", [m["episode_index"] for m in episodes_meta], pa.int64()
    )
    arrays["data/chunk_index"] = col(
        "data/chunk_index", [m["chunk_index"] for m in episodes_meta], pa.int64()
    )
    arrays["data/file_index"] = col(
        "data/file_index", [m["file_index"] for m in episodes_meta], pa.int64()
    )
    arrays["dataset_from_index"] = col(
        "dataset_from_index", [m["from_index"] for m in episodes_meta], pa.int64()
    )
    arrays["dataset_to_index"] = col(
        "dataset_to_index", [m["to_index"] for m in episodes_meta], pa.int64()
    )

    for cam in camera_names:
        vkey = f"observation.images.{cam}"
        arrays[f"videos/{vkey}/chunk_index"] = col(
            f"videos/{vkey}/chunk_index", [0] * n, pa.int64()
        )
        arrays[f"videos/{vkey}/file_index"] = col(
            f"videos/{vkey}/file_index", [0] * n, pa.int64()
        )
        arrays[f"videos/{vkey}/from_timestamp"] = col(
            f"videos/{vkey}/from_timestamp",
            [m["from_ts"] for m in episodes_meta],
            pa.float64(),
        )
        arrays[f"videos/{vkey}/to_timestamp"] = col(
            f"videos/{vkey}/to_timestamp",
            [m["to_ts"] for m in episodes_meta],
            pa.float64(),
        )

    arrays["tasks"] = pa.chunked_array([pa.array([[m["task"]] for m in episodes_meta])])
    arrays["length"] = col("length", [m["length"] for m in episodes_meta], pa.int64())

    # Per-episode quality metrics (carried from the HDF5; enables filtering/QA in LeRobot).
    arrays["completion_ratio"] = col(
        "completion_ratio",
        [m.get("completion_ratio", float("nan")) for m in episodes_meta],
        pa.float64(),
    )
    arrays["full_completion"] = col(
        "full_completion",
        [bool(m.get("full_completion", False)) for m in episodes_meta],
        pa.bool_(),
    )
    arrays["success"] = col(
        "success", [bool(m.get("success", False)) for m in episodes_meta], pa.bool_()
    )

    # Per-episode stats for scalar features
    scalar_feats = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    bool_feats = ["next.done"]

    for feat in scalar_feats:
        key = f"ep_{feat.replace('.', '_').replace('/', '_')}_stats"
        for stat in ("min", "max", "mean", "std"):
            arrays[f"stats/{feat}/{stat}"] = pa.chunked_array(
                [pa.array([m["stats"][feat][stat] for m in episodes_meta])]
            )
        arrays[f"stats/{feat}/count"] = pa.chunked_array(
            [pa.array([m["stats"][feat]["count"] for m in episodes_meta])]
        )

    for feat in bool_feats:
        for stat in ("min", "max"):
            arrays[f"stats/{feat}/{stat}"] = pa.chunked_array(
                [pa.array([m["stats"][feat][stat] for m in episodes_meta])]
            )
        for stat in ("mean", "std"):
            arrays[f"stats/{feat}/{stat}"] = pa.chunked_array(
                [pa.array([[float(m["stats"][feat][stat][0])] for m in episodes_meta])]
            )
        arrays[f"stats/{feat}/count"] = pa.chunked_array(
            [pa.array([m["stats"][feat]["count"] for m in episodes_meta])]
        )

    # Per-episode stats for image features
    for cam in camera_names:
        vkey = f"observation.images.{cam}"
        for stat in ("min", "max", "mean", "std"):
            arrays[f"stats/{vkey}/{stat}"] = pa.chunked_array(
                [pa.array([m["stats"][vkey][stat] for m in episodes_meta])]
            )
        arrays[f"stats/{vkey}/count"] = pa.chunked_array(
            [pa.array([m["stats"][vkey]["count"] for m in episodes_meta])]
        )

    # Self-referential meta pointer (all in chunk-000/file-000)
    arrays["meta/episodes/chunk_index"] = col(
        "meta/episodes/chunk_index", [0] * n, pa.int64()
    )
    arrays["meta/episodes/file_index"] = col(
        "meta/episodes/file_index", [0] * n, pa.int64()
    )

    return pa.table(arrays)


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def hdf5_to_lerobot(
    hdf5_path: str,
    output_dir: str,
    fps: int = 30,
    task_name: str = "robot manipulation",
    robot_type: str = "sharpa_wave",
    video_codec: str = "libx264",
    video_pix_fmt: str = "yuv420p",
    video_crf: int = 23,
    image_sample_every: int = 10,
) -> None:
    """Convert a RecorderManager HDF5 file to LeRobot v3 format.

    Args:
        hdf5_path: Path to the RecorderManager HDF5 file.
        output_dir: Root directory for the LeRobot dataset (created if absent).
        fps: Control / camera frame rate for timestamps and video encoding.
        task_name: Natural-language task description (written to tasks.jsonl).
        robot_type: Robot identifier written to info.json.
        video_codec: ffmpeg codec for video encoding (default libx264).
        video_pix_fmt: Pixel format for video (default yuv420p).
        video_crf: Constant rate factor for libx264 (lower = better quality).
        image_sample_every: Stride for image stats sampling (every Nth frame).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for LeRobot export: pip install pyarrow"
        ) from e

    hdf5_path = os.path.abspath(hdf5_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Pass 1: read HDF5, write videos, collect frame-level arrays
    # -----------------------------------------------------------------------
    with h5py.File(hdf5_path, "r") as f:
        episode_names = sorted(f["data"].keys())
        n_episodes = len(episode_names)
        if n_episodes == 0:
            raise ValueError(f"No episodes found in {hdf5_path}")

        first_ep = f["data"][episode_names[0]]

        # Discover cameras (only record RGB here)
        camera_names: list[str] = []
        if "camera" in first_ep:
            for cam, grp in first_ep["camera"].items():
                if "rgb" in grp:
                    camera_names.append(cam)
        camera_names = sorted(camera_names)

        obs_dim = int(first_ep["obs"].shape[1])
        action_dim = int(first_ep["actions"].shape[1])

        cam_hw: dict[str, tuple[int, int]] = {}
        for cam in camera_names:
            rgb = first_ep[f"camera/{cam}/rgb"]
            cam_hw[cam] = (int(rgb.shape[1]), int(rgb.shape[2]))  # (H, W)

        print(
            f"[lerobot] {n_episodes} episodes | obs_dim={obs_dim} action_dim={action_dim} "
            f"cameras={camera_names} fps={fps}"
        )

        # Open video writers
        video_writers: dict[str, Any] = {}
        for cam in camera_names:
            vkey = f"observation.images.{cam}"
            vid_dir = os.path.join(output_dir, "videos", vkey, "chunk-000")
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(vid_dir, "file-000.mp4")
            writer = imageio.get_writer(
                vid_path,
                fps=fps,
                codec=video_codec,
                pixelformat=video_pix_fmt,
                macro_block_size=None,
                output_params=["-crf", str(video_crf)],
            )
            video_writers[cam] = writer

        # Per-frame accumulators
        all_obs: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_episode_indices: list[np.ndarray] = []
        all_frame_indices: list[np.ndarray] = []
        all_timestamps: list[np.ndarray] = []
        all_next_done: list[np.ndarray] = []
        all_index: list[np.ndarray] = []
        task_index_list: list[np.ndarray] = []

        episodes_meta: list[dict] = []
        global_frame_idx = 0

        for ep_idx, ep_name in enumerate(episode_names):
            ep = f["data"][ep_name]
            T = int(ep["obs"].shape[0])

            obs_arr = ep["obs"][:].astype(np.float32)  # (T, obs_dim)
            action_arr = ep["actions"][:].astype(np.float32)  # (T, action_dim)
            ts_arr = (np.arange(T, dtype=np.float32)) / fps
            done_arr = np.zeros(T, dtype=bool)
            done_arr[-1] = True
            ep_idx_arr = np.full(T, ep_idx, dtype=np.int64)
            fi_arr = np.arange(T, dtype=np.int64)
            gi_arr = np.arange(global_frame_idx, global_frame_idx + T, dtype=np.int64)
            ti_arr = np.zeros(T, dtype=np.int64)

            all_obs.append(obs_arr)
            all_actions.append(action_arr)
            all_episode_indices.append(ep_idx_arr)
            all_frame_indices.append(fi_arr)
            all_timestamps.append(ts_arr)
            all_next_done.append(done_arr)
            all_index.append(gi_arr)
            task_index_list.append(ti_arr)

            # Per-episode stats (scalar features)
            ep_meta: dict = {
                "episode_index": ep_idx,
                # All episodes are written to one data/video file (chunk-000/file-000);
                # we don't split into 1000-episode chunks. So the data-file locators are
                # 0/0 for every episode (like the video/meta locators), and the row range
                # is given by dataset_from_index/dataset_to_index. If real chunking is ever
                # added here, these must track the actual split.
                "chunk_index": 0,
                "file_index": 0,
                "from_index": global_frame_idx,
                "to_index": global_frame_idx + T,
                # Video from/to timestamps locate this episode inside the *concatenated*
                # per-camera mp4, so they must be cumulative on the merged-file timeline
                # (NOT episode-local — that would make every episode decode from t=0 and
                # return episode 0's frames). global_frame_idx is the running frame offset
                # before this episode; all episodes share one file here (< chunk size).
                "from_ts": global_frame_idx / fps,
                "to_ts": (global_frame_idx + T) / fps,
                "length": T,
                "task": task_name,
                # Per-episode metrics carried over from the HDF5 episode attrs (written by
                # record_dataset's completion reporter). NaN/False if absent.
                "completion_ratio": float(
                    ep.attrs.get("completion_ratio", float("nan"))
                ),
                "full_completion": bool(ep.attrs.get("full_completion", False)),
                "success": bool(ep.attrs.get("success", False)),
                "stats": {},
            }

            ep_meta["stats"]["observation.state"] = _scalar_stats(obs_arr)
            ep_meta["stats"]["action"] = _scalar_stats(action_arr)
            ep_meta["stats"]["timestamp"] = _scalar_stats_1d(ts_arr)
            ep_meta["stats"]["frame_index"] = _scalar_stats_1d(
                fi_arr.astype(np.float64)
            )
            ep_meta["stats"]["episode_index"] = _scalar_stats_1d(
                ep_idx_arr.astype(np.float64)
            )
            ep_meta["stats"]["index"] = _scalar_stats_1d(gi_arr.astype(np.float64))
            ep_meta["stats"]["task_index"] = {
                "min": [0],
                "max": [0],
                "mean": [0.0],
                "std": [0.0],
                "count": [T],
            }
            ep_meta["stats"]["next.done"] = _bool_stats(done_arr)

            # Per-episode image stats + write video frames
            for cam in camera_names:
                rgb_arr = ep[f"camera/{cam}/rgb"][:]  # (T, H, W, 3) uint8
                vkey = f"observation.images.{cam}"
                ep_meta["stats"][vkey] = _image_stats(
                    rgb_arr, sample_every=image_sample_every
                )
                for frame in rgb_arr:
                    video_writers[cam].append_data(frame[..., :3])  # ensure 3-channel

            global_frame_idx += T
            episodes_meta.append(ep_meta)

            if (ep_idx + 1) % 10 == 0 or ep_idx + 1 == n_episodes:
                print(f"  processed {ep_idx + 1}/{n_episodes} episodes", flush=True)

    # Close video writers
    for writer in video_writers.values():
        writer.close()

    total_frames = global_frame_idx
    print(f"[lerobot] Total frames: {total_frames}")

    # -----------------------------------------------------------------------
    # Write data parquet
    # -----------------------------------------------------------------------
    obs_all = np.concatenate(all_obs, axis=0)  # (total_frames, obs_dim)
    action_all = np.concatenate(all_actions, axis=0)
    ep_idx_all = np.concatenate(all_episode_indices, axis=0)
    fi_all = np.concatenate(all_frame_indices, axis=0)
    ts_all = np.concatenate(all_timestamps, axis=0)
    done_all = np.concatenate(all_next_done, axis=0)
    gi_all = np.concatenate(all_index, axis=0)
    ti_all = np.concatenate(task_index_list, axis=0)

    data_dir = os.path.join(output_dir, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)

    data_table = pa.table(
        {
            "observation.state": pa.array(
                obs_all.tolist(), type=pa.list_(pa.float32())
            ),
            "action": pa.array(action_all.tolist(), type=pa.list_(pa.float32())),
            "episode_index": pa.array(ep_idx_all, type=pa.int64()),
            "frame_index": pa.array(fi_all, type=pa.int64()),
            "timestamp": pa.array(ts_all, type=pa.float32()),
            "next.done": pa.array(done_all, type=pa.bool_()),
            "index": pa.array(gi_all, type=pa.int64()),
            "task_index": pa.array(ti_all, type=pa.int64()),
        }
    )
    pq.write_table(data_table, os.path.join(data_dir, "file-000.parquet"))
    print(f"[lerobot] Written data parquet ({total_frames} rows)")

    # -----------------------------------------------------------------------
    # Compute global stats
    # -----------------------------------------------------------------------
    global_stats: dict = {}
    global_stats["observation.state"] = _scalar_stats(obs_all)
    global_stats["action"] = _scalar_stats(action_all)
    global_stats["timestamp"] = _scalar_stats_1d(ts_all)
    global_stats["frame_index"] = _scalar_stats_1d(fi_all.astype(np.float64))
    global_stats["episode_index"] = _scalar_stats_1d(ep_idx_all.astype(np.float64))
    global_stats["index"] = _scalar_stats_1d(gi_all.astype(np.float64))
    global_stats["task_index"] = {
        "min": [0],
        "max": [0],
        "mean": [0.0],
        "std": [0.0],
        "count": [total_frames],
    }
    global_stats["next.done"] = _bool_stats(done_all)

    # Aggregate image stats across episodes (average of per-episode means/stds)
    for cam in camera_names:
        vkey = f"observation.images.{cam}"
        ep_stats = [m["stats"][vkey] for m in episodes_meta]
        C = 3
        chan_min = [[[min(s["min"][c][0][0] for s in ep_stats)]] for c in range(C)]
        chan_max = [[[max(s["max"][c][0][0] for s in ep_stats)]] for c in range(C)]
        # Simple mean-of-means (approx; good enough for normalization)
        chan_mean = [
            [[float(np.mean([s["mean"][c][0][0] for s in ep_stats]))]] for c in range(C)
        ]
        chan_std = [
            [[float(np.mean([s["std"][c][0][0] for s in ep_stats]))]] for c in range(C)
        ]
        count_total = sum(s["count"][0] for s in ep_stats)
        global_stats[vkey] = {
            "min": chan_min,
            "max": chan_max,
            "mean": chan_mean,
            "std": chan_std,
            "count": [count_total],
        }

    # -----------------------------------------------------------------------
    # Write meta files
    # -----------------------------------------------------------------------
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # stats.json
    with open(os.path.join(meta_dir, "stats.json"), "w") as fh:
        json.dump(global_stats, fh, indent=2)

    # tasks.jsonl
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as fh:
        fh.write(json.dumps({"task_index": 0, "task": task_name}) + "\n")

    # info.json
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": [obs_dim],
            "names": None,
            "fps": fps,
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": None,
            "fps": fps,
        },
    }
    for cam in camera_names:
        vkey = f"observation.images.{cam}"
        H, W = cam_hw[cam]
        features[vkey] = {
            "dtype": "video",
            "shape": [H, W, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": video_codec,
                "video.pix_fmt": video_pix_fmt,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    for feat in (
        "episode_index",
        "frame_index",
        "timestamp",
        "next.done",
        "index",
        "task_index",
    ):
        dtype_map = {"next.done": "bool", "timestamp": "float32"}
        features[feat] = {
            "dtype": dtype_map.get(feat, "int64"),
            "shape": [1],
            "names": None,
            "fps": fps,
        }

    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": _CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "data_files_size_in_mb": round(
            os.path.getsize(os.path.join(data_dir, "file-000.parquet")) / 1e6, 1
        ),
        "video_files_size_in_mb": sum(
            round(
                os.path.getsize(
                    os.path.join(
                        output_dir,
                        "videos",
                        f"observation.images.{cam}",
                        "chunk-000",
                        "file-000.mp4",
                    )
                )
                / 1e6,
                1,
            )
            for cam in camera_names
        ),
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    # episodes parquet
    ep_pq_dir = os.path.join(meta_dir, "episodes", "chunk-000")
    os.makedirs(ep_pq_dir, exist_ok=True)
    ep_table = _build_episodes_table(episodes_meta, camera_names)
    pq.write_table(ep_table, os.path.join(ep_pq_dir, "file-000.parquet"))

    print(f"[lerobot] Dataset written to {output_dir}")
    print(f"  episodes: {n_episodes}, frames: {total_frames}, cameras: {camera_names}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hdf5", required=True, help="Input HDF5 path.")
    parser.add_argument(
        "--output_dir", required=True, help="Output LeRobot dataset directory."
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument(
        "--task", default="robot manipulation", help="Task description."
    )
    parser.add_argument(
        "--robot_type", default="sharpa_wave", help="Robot type identifier."
    )
    parser.add_argument(
        "--video_codec", default="libx264", help="Video codec (default libx264)."
    )
    parser.add_argument(
        "--video_crf", type=int, default=23, help="CRF quality (lower=better)."
    )
    args = parser.parse_args()

    hdf5_to_lerobot(
        hdf5_path=args.hdf5,
        output_dir=args.output_dir,
        fps=args.fps,
        task_name=args.task,
        robot_type=args.robot_type,
        video_codec=args.video_codec,
        video_crf=args.video_crf,
    )


if __name__ == "__main__":
    main()
