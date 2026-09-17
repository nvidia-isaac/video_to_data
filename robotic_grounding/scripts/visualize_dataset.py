# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay recorded camera observations from a dataset and write MP4s.

Supports two formats:
- **HDF5** (RecorderManager): ``--dataset path/to/file.hdf5``
- **LeRobot v3** (directory): ``--dataset path/to/lerobot_dir/``

For HDF5 datasets the full RGB/depth/segmentation grid is shown.
For LeRobot datasets the RGB video panels are decoded from MP4 and tiled.

Both modes write one MP4 per episode to ``<output_dir>/<episode>.mp4`` using
the same tiled grid (rows = modality, cols = camera).

HDF5 dataset layout (per episode group ``data/<demo_i>``):
    camera/<sensor_name>/<data_type>   # (T, H, W, C)

LeRobot v3 dataset layout:
    meta/info.json
    meta/episodes/chunk-000/file-000.parquet
    videos/observation.images.<cam>/chunk-000/file-000.mp4

Example:
    python scripts/visualize_dataset.py \\
        --dataset datasets/test2a/taco_smoke.hdf5 \\
        --output_dir datasets/test2a/videos \\
        --data_types rgb depth seg

    python scripts/visualize_dataset.py \\
        --dataset datasets/test2a/taco_smoke \\
        --output_dir datasets/test2a/lerobot_videos
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

# Friendly aliases -> the camera data_type keys written by the HDF5 recorder.
DATA_TYPE_ALIASES = {
    "rgb": "rgb",
    "depth": "distance_to_image_plane",
    "distance_to_image_plane": "distance_to_image_plane",
    "seg": "instance_id_segmentation_fast",
    "segmentation": "instance_id_segmentation_fast",
    "instance_id_segmentation_fast": "instance_id_segmentation_fast",
}


def _to_uint8_rgb(data_type: str, frame: np.ndarray) -> np.ndarray:
    """Convert one (H, W, C) frame of an arbitrary data type to an (H, W, 3) uint8 image."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[..., None]

    if data_type == "rgb":
        img = arr[..., :3].astype(np.uint8)
    elif data_type == "distance_to_image_plane":
        # Depth: replace non-finite (background/sky) with 0, normalize finite range -> gray.
        depth = arr[..., 0].astype(np.float32)
        finite = np.isfinite(depth)
        depth = np.where(finite, depth, 0.0)
        if finite.any():
            hi = float(np.percentile(depth[finite], 99)) or 1.0
        else:
            hi = 1.0
        gray = np.clip(depth / hi, 0.0, 1.0)
        gray = (gray * 255).astype(np.uint8)
        img = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[..., ::-1]  # BGR->RGB
    else:
        # Segmentation / integer id maps: deterministic color per id (0 = black background).
        ids = arr[..., 0].astype(np.int64)
        uniq = np.unique(ids)
        lut = {}
        for uid in uniq:
            if uid == 0:
                lut[uid] = (0, 0, 0)
            else:
                h = (int(uid) * 2654435761) & 0xFFFFFFFF
                lut[uid] = ((h >> 16) & 255, (h >> 8) & 255, h & 255)
        img = np.zeros((*ids.shape, 3), dtype=np.uint8)
        for uid, color in lut.items():
            img[ids == uid] = color
    return np.ascontiguousarray(img)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a small text label in the top-left corner (with a dark backing for contrast)."""
    out = img.copy()
    cv2.rectangle(
        out, (0, 0), (min(len(text) * 9 + 8, out.shape[1]), 20), (0, 0, 0), -1
    )
    cv2.putText(
        out,
        text,
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _resolve_data_types(requested: list[str]) -> list[str]:
    resolved = []
    for r in requested:
        key = DATA_TYPE_ALIASES.get(r, r)
        if key not in resolved:
            resolved.append(key)
    return resolved


def _discover_cameras_hdf5(ep) -> dict[str, list[str]]:
    """Return {sensor_name: [data_type, ...]} for an HDF5 episode group."""
    cams: dict[str, list[str]] = {}
    if "camera" not in ep:
        return cams
    for sensor_name, sensor_grp in ep["camera"].items():
        cams[sensor_name] = list(sensor_grp.keys())
    return cams


def _tile(
    panels: list[np.ndarray | None], n_rows: int, n_cols: int, cell_hw: tuple[int, int]
) -> np.ndarray:
    """Compose a row-major grid of equal-sized panels; None -> black cell."""
    ch, cw = cell_hw
    grid = np.zeros((n_rows * ch, n_cols * cw, 3), dtype=np.uint8)
    for idx, panel in enumerate(panels):
        if panel is None:
            continue
        r, c = divmod(idx, n_cols)
        if panel.shape[:2] != (ch, cw):
            panel = cv2.resize(panel, (cw, ch), interpolation=cv2.INTER_NEAREST)
        grid[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw] = panel
    return grid


# ---------------------------------------------------------------------------
# HDF5 mode
# ---------------------------------------------------------------------------


def visualize_episode_hdf5(
    ep, data_types: list[str], out_path: str, fps: int, max_frames: int | None
) -> bool:
    cams = _discover_cameras_hdf5(ep)
    if not cams:
        print(f"  [skip] {os.path.basename(out_path)}: no camera data in episode")
        return False

    sensors = sorted(cams.keys())
    present = {dt for dts in cams.values() for dt in dts}
    dtypes = [dt for dt in data_types if dt in present]
    if not dtypes:
        print(
            f"  [skip] {os.path.basename(out_path)}: requested data types not present "
            f"(have: {sorted(present)})"
        )
        return False

    n_rows, n_cols = len(dtypes), len(sensors)

    ref_key = f"camera/{sensors[0]}/{cams[sensors[0]][0]}"
    n_frames = ep[ref_key].shape[0]
    ref_shape = ep[ref_key].shape[1:3]
    cell_hw = (int(ref_shape[0]), int(ref_shape[1]))
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
    try:
        for t in range(n_frames):
            panels: list[np.ndarray | None] = []
            for dt in dtypes:
                for sensor in sensors:
                    if dt in cams[sensor]:
                        frame = ep[f"camera/{sensor}/{dt}"][t]
                        img = _to_uint8_rgb(dt, frame)
                        panels.append(_label(img, f"{sensor}:{dt}"))
                    else:
                        panels.append(None)
            writer.append_data(_tile(panels, n_rows, n_cols, cell_hw))
    finally:
        writer.close()

    success = bool(ep.attrs.get("success", False))
    print(
        f"  [ok] {os.path.basename(out_path)}: {n_frames} frames, grid {n_rows}x{n_cols} "
        f"({n_cols} cam(s) x {n_rows} modality), success={success}"
    )
    return True


def run_hdf5(
    dataset_path: str,
    out_dir: str,
    data_types: list[str],
    episodes: list[str] | None,
    fps: int,
    max_frames: int | None,
) -> None:
    import h5py

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise SystemExit(
                f"'{dataset_path}' has no 'data' group — not a RecorderManager dataset."
            )
        all_eps = list(f["data"].keys())

        if episodes is None:
            selected = all_eps
        else:
            selected = []
            for e in episodes:
                if e in all_eps:
                    selected.append(e)
                elif e.isdigit() and int(e) < len(all_eps):
                    selected.append(all_eps[int(e)])
                else:
                    print(f"  [warn] episode '{e}' not found; available: {all_eps}")

        print(
            f"[INFO] HDF5 {dataset_path}: {len(all_eps)} episode(s); rendering {len(selected)} "
            f"-> {out_dir} (modalities: {data_types})"
        )
        n_ok = 0
        for name in selected:
            out_path = os.path.join(out_dir, f"{name}.mp4")
            n_ok += visualize_episode_hdf5(
                f["data"][name], data_types, out_path, fps, max_frames
            )
    print(f"[INFO] Done. Wrote {n_ok} video(s) to {out_dir}")


# ---------------------------------------------------------------------------
# LeRobot v3 mode
# ---------------------------------------------------------------------------


def _read_episodes_parquet(meta_dir: str) -> list[dict]:
    """Return list of episode dicts from meta/episodes parquet chunks."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "pyarrow required for LeRobot datasets: pip install pyarrow"
        ) from e

    ep_dir = os.path.join(meta_dir, "episodes")
    rows = []
    for chunk_dir in sorted(os.listdir(ep_dir)):
        chunk_path = os.path.join(ep_dir, chunk_dir)
        if not os.path.isdir(chunk_path):
            continue
        for fname in sorted(os.listdir(chunk_path)):
            if not fname.endswith(".parquet"):
                continue
            tbl = pq.read_table(os.path.join(chunk_path, fname))
            d = tbl.to_pydict()
            n = len(d["episode_index"])
            for i in range(n):
                rows.append({k: v[i] for k, v in d.items()})
    rows.sort(key=lambda r: r["episode_index"])
    return rows


def visualize_episode_lerobot(
    ep_row: dict,
    video_readers: dict[str, Any],  # cam_name -> imageio reader
    cam_names: list[str],
    out_path: str,
    fps: int,
    max_frames: int | None,
) -> bool:
    """Render one LeRobot episode from opened video readers."""
    from_idx = int(ep_row["dataset_from_index"])
    length = int(ep_row["length"])
    if max_frames is not None:
        length = min(length, max_frames)

    n_rows, n_cols = 1, len(cam_names)
    # Peek at first frame to get cell size
    cell_hw = None
    for cam in cam_names:
        try:
            video_readers[cam].set_image_index(from_idx)
            frame = video_readers[cam].get_next_data()
            cell_hw = (frame.shape[0], frame.shape[1])
            break
        except Exception:
            continue
    if cell_hw is None:
        print(
            f"  [skip] episode {ep_row['episode_index']}: could not read any video frame"
        )
        return False

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
    try:
        for t in range(length):
            global_t = from_idx + t
            panels: list[np.ndarray | None] = []
            for cam in cam_names:
                try:
                    video_readers[cam].set_image_index(global_t)
                    frame = video_readers[cam].get_next_data()
                    img = np.asarray(frame)[..., :3].astype(np.uint8)
                    panels.append(_label(img, f"{cam}:rgb"))
                except Exception:
                    panels.append(None)
            writer.append_data(_tile(panels, n_rows, n_cols, cell_hw))
    finally:
        writer.close()

    task = ep_row.get("tasks", ["?"])
    if isinstance(task, list):
        task = task[0] if task else "?"
    print(
        f"  [ok] episode_{ep_row['episode_index']:04d}.mp4: {length} frames, "
        f"grid 1x{n_cols}, task='{task}'"
    )
    return True


def run_lerobot(
    dataset_dir: str,
    out_dir: str,
    episodes: list[str] | None,
    fps: int,
    max_frames: int | None,
) -> None:
    meta_dir = os.path.join(dataset_dir, "meta")
    info_path = os.path.join(meta_dir, "info.json")
    if not os.path.isfile(info_path):
        raise SystemExit(
            f"'{dataset_dir}' is not a LeRobot dataset (missing meta/info.json)."
        )

    with open(info_path) as fh:
        info = json.load(fh)

    # Discover RGB video features
    cam_names: list[str] = []
    for feat_key, feat in info.get("features", {}).items():
        if feat.get("dtype") == "video" and feat_key.startswith("observation.images."):
            cam_names.append(feat_key[len("observation.images.") :])
    cam_names = sorted(cam_names)
    if not cam_names:
        raise SystemExit(
            f"No 'observation.images.*' video features found in {info_path}"
        )

    video_path_tpl: str = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )

    # Load episode metadata
    all_ep_rows = _read_episodes_parquet(meta_dir)
    n_eps = len(all_ep_rows)

    if episodes is None:
        selected = all_ep_rows
    else:
        ep_by_idx = {str(r["episode_index"]): r for r in all_ep_rows}
        selected = []
        for e in episodes:
            if e in ep_by_idx:
                selected.append(ep_by_idx[e])
            elif e.isdigit() and int(e) < n_eps:
                selected.append(all_ep_rows[int(e)])
            else:
                print(f"  [warn] episode '{e}' not found")

    print(
        f"[INFO] LeRobot {dataset_dir}: {n_eps} episode(s); rendering {len(selected)} "
        f"-> {out_dir} (cameras: {cam_names})"
    )

    # Open one reader per camera (shared across episodes — we seek per episode)
    video_readers: dict[str, Any] = {}
    for cam in cam_names:
        vkey = f"observation.images.{cam}"
        vid_rel = video_path_tpl.format(video_key=vkey, chunk_index=0, file_index=0)
        vid_abs = os.path.join(dataset_dir, vid_rel)
        if not os.path.isfile(vid_abs):
            print(f"  [warn] video file not found: {vid_abs} — skipping camera '{cam}'")
            continue
        video_readers[cam] = imageio.get_reader(vid_abs)

    active_cams = [c for c in cam_names if c in video_readers]
    if not active_cams:
        raise SystemExit("No readable video files found.")

    os.makedirs(out_dir, exist_ok=True)
    n_ok = 0
    for ep_row in selected:
        ep_idx = ep_row["episode_index"]
        out_path = os.path.join(out_dir, f"episode_{ep_idx:04d}.mp4")
        n_ok += visualize_episode_lerobot(
            ep_row, video_readers, active_cams, out_path, fps, max_frames
        )

    for reader in video_readers.values():
        reader.close()

    print(f"[INFO] Done. Wrote {n_ok} video(s) to {out_dir}")


# ---------------------------------------------------------------------------
# Auto-detect format and dispatch
# ---------------------------------------------------------------------------


def _detect_format(dataset_path: str) -> str:
    """Return 'hdf5' or 'lerobot'."""
    if os.path.isfile(dataset_path) and dataset_path.endswith(".hdf5"):
        return "hdf5"
    if os.path.isdir(dataset_path) and os.path.isfile(
        os.path.join(dataset_path, "meta", "info.json")
    ):
        return "lerobot"
    # Fallback: if it's a file, assume HDF5; if a dir, assume LeRobot
    if os.path.isfile(dataset_path):
        return "hdf5"
    if os.path.isdir(dataset_path):
        return "lerobot"
    raise SystemExit(
        f"Cannot detect format for '{dataset_path}' — path does not exist."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to HDF5 file or LeRobot dataset directory (auto-detected).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write MP4s (default: <dataset_dir>/videos).",
    )
    parser.add_argument(
        "--data_types",
        nargs="+",
        default=["rgb"],
        help="HDF5 modalities to show: rgb depth seg (default: rgb). "
        "Ignored for LeRobot (always shows rgb).",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        help="Episode names/indices to render (default: all).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
    parser.add_argument(
        "--max_frames", type=int, default=None, help="Cap frames per episode."
    )
    args = parser.parse_args()

    fmt = _detect_format(args.dataset)
    dataset_abs = os.path.abspath(args.dataset)
    base_dir = (
        dataset_abs if os.path.isdir(dataset_abs) else os.path.dirname(dataset_abs)
    )
    out_dir = args.output_dir or os.path.join(base_dir, "videos")

    if fmt == "hdf5":
        data_types = _resolve_data_types(args.data_types)
        run_hdf5(
            dataset_abs, out_dir, data_types, args.episodes, args.fps, args.max_frames
        )
    else:
        print("[INFO] Detected LeRobot v3 dataset (ignoring --data_types; always rgb)")
        run_lerobot(dataset_abs, out_dir, args.episodes, args.fps, args.max_frames)


if __name__ == "__main__":
    main()
