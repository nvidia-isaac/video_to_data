# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a reconstructed sequence into a flat training-ready layout.

Supports two modes:
  - Remote: download from CSS via boto3 (swift:// URL or bare S3 path).
  - Local: copy from a local directory (e.g. OSMO-mounted task outputs).

The mode is auto-detected: if the source path is an existing local directory,
local copy is used; otherwise it's treated as a remote S3 path.

Usage (remote):
    python -m v2d.mv.postprocess.lib.export_sequence \
        --swift_output_base swift://storage.example.com/AUTH_.../data_output/<seq> \
        --output_dir /local/path/to/sequence

Usage (local):
    python -m v2d.mv.postprocess.lib.export_sequence \
        --swift_output_base /osmo/data/input/0 \
        --output_dir /osmo/data/output
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import h5py
from botocore.config import Config
from tqdm import tqdm
from v2d.common.object_storage import parse_storage_url, s3_client_kwargs

from v2d.common.hdf5_transcode import (
    has_hdf5_filters,
    is_jpeg_hdf5,
    slice_h5_frames,
    transcode_h5_lossless,
    transcode_rgb_h5_to_jpeg_h5,
)
from v2d.common.ffv1_sidecar import (
    read_ffv1_metadata,
    transcode_h5_to_ffv1_sidecar,
)
from v2d.common.video import pack_directory_to_h5

DEFAULT_DOWNLOAD_WORKERS = os.cpu_count() or 8
DEFAULT_CAMERA_WORKERS = min(4, os.cpu_count() or 4)


LEFT_CAMERAS = [
    "front_stereo_camera_left",
    "back_stereo_camera_left",
    "left_stereo_camera_left",
    "right_stereo_camera_left",
]

RIGHT_CAMERAS = [
    "front_stereo_camera_right",
    "back_stereo_camera_right",
    "left_stereo_camera_right",
    "right_stereo_camera_right",
]

RGB_CAMERAS = LEFT_CAMERAS + RIGHT_CAMERAS

_REQUIRED_CAMERAS_BY_OUTPUT = {
    "images": RGB_CAMERAS,
    "images_anonymized": RGB_CAMERAS,
    "videos": RGB_CAMERAS,
    "videos_anonymized": RGB_CAMERAS,
    "depth": LEFT_CAMERAS,
    "object_masks": LEFT_CAMERAS,
    "human_masks": LEFT_CAMERAS,
}

_RGB_H5_OUTPUTS = {"images", "images_anonymized"}
_MASK_H5_OUTPUTS = {"object_masks", "human_masks"}
_ANONYMIZED_OUTPUTS = {"images_anonymized", "videos_anonymized"}


class MissingRequiredExportDataError(FileNotFoundError):
    """Raised when a required export input is missing or incomplete."""


def _run_camera_jobs(items, worker, max_camera_workers: int):
    """Run independent per-camera transforms concurrently in stable order."""
    if max_camera_workers < 1:
        raise ValueError("max_camera_workers must be at least 1")
    if len(items) < 2 or max_camera_workers == 1:
        return [worker(item) for item in items]
    with ThreadPoolExecutor(
        max_workers=min(max_camera_workers, len(items)),
        thread_name_prefix="export-camera",
    ) as pool:
        return list(pool.map(worker, items))


def _get_s3_client(remote_url: str | None = None, *, endpoint_url: str | None = None):
    endpoint = parse_storage_url(remote_url)[0] if remote_url else endpoint_url
    return boto3.client(
        "s3", **s3_client_kwargs(endpoint), config=Config(connect_timeout=10),
    )


def _parse_swift_url(url: str) -> tuple[str, str]:
    """Compatibility alias accepting s3://, swift:// and bare bucket paths."""
    _, bucket, prefix = parse_storage_url(url)
    return bucket, prefix


def _list_objects(client, bucket: str, prefix: str) -> list[dict]:
    """List all objects under a prefix."""
    paginator = client.get_paginator("list_objects_v2")
    folder_prefix = prefix.rstrip("/") + "/"
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
        objects.extend(page.get("Contents", []))
    return objects


def _download_file(
    client,
    bucket: str,
    key: str,
    dest: Path,
    dry_run: bool = False,
    missing_ok: bool = False,
) -> tuple[bool, bool]:
    """Download one file.

    Returns (downloaded, source_present). Already-existing destinations count as
    present but not downloaded.
    """
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        remote_size = head["ContentLength"]
    except client.exceptions.ClientError:
        if not missing_ok:
            print(f"  WARNING: key not found: {key}")
        return False, False

    if remote_size <= 0:
        if not missing_ok:
            print(f"  WARNING: key is empty: {key}")
        return False, False

    if dest.exists() and dest.stat().st_size == remote_size:
        return False, True

    if dry_run:
        print(f"  [dry-run] would download: {key}")
        return True, True

    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))
    return True, True


def _download_prefix(
    client,
    bucket: str,
    css_prefix: str,
    local_dir: Path,
    remap_fn=None,
    filter_fn=None,
    dry_run: bool = False,
    label: str = "",
    max_workers: int = DEFAULT_DOWNLOAD_WORKERS,
) -> tuple[int, int, list[str]]:
    """Download all objects under css_prefix into local_dir.

    Uses a thread pool for parallel downloads. Each thread gets its own
    boto3 client since they are not thread-safe.

    Args:
        remap_fn: Optional function (rel_path) -> new_rel_path to remap file paths.
        filter_fn: Optional function (rel_path) -> bool to filter files.
        label: Human-readable label for progress bar.
        max_workers: Number of parallel download threads.

    Returns (downloaded_count, skipped_count, output_relative_paths).
    """
    objects = _list_objects(client, bucket, css_prefix)
    folder_prefix = css_prefix.rstrip("/") + "/"

    # Pre-filter and resolve destinations
    work_items: list[tuple[str, int, Path]] = []  # (key, remote_size, dest)
    for obj in objects:
        rel = obj["Key"][len(folder_prefix):]
        if not rel:
            continue
        if filter_fn and not filter_fn(rel):
            continue
        if remap_fn:
            rel = remap_fn(rel)
            if rel is None:
                continue
        if obj["Size"] <= 0:
            continue
        work_items.append((obj["Key"], obj["Size"], local_dir / rel))

    downloaded = 0
    skipped = 0
    rel_paths = [str(dest.relative_to(local_dir)) for _, _, dest in work_items]
    desc = f"  {label}" if label else "  downloading"

    # Separate into skip vs actual download
    to_download: list[tuple[str, Path]] = []
    for key, remote_size, dest in work_items:
        if dest.exists() and dest.stat().st_size == remote_size:
            skipped += 1
        else:
            to_download.append((key, dest))

    if dry_run:
        for key, dest in to_download:
            downloaded += 1
        return len(to_download), skipped, rel_paths

    # Thread-local boto3 clients
    _local = threading.local()

    def _get_thread_client():
        if not hasattr(_local, "client"):
            _local.client = _get_s3_client(endpoint_url=client.meta.endpoint_url)
        return _local.client

    def _do_download(item: tuple[str, Path]) -> None:
        key, dest = item
        dest.parent.mkdir(parents=True, exist_ok=True)
        _get_thread_client().download_file(bucket, key, str(dest))

    pbar = tqdm(total=len(to_download), desc=desc, unit="file")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_do_download, item): item for item in to_download}
        for future in as_completed(futures):
            future.result()
            downloaded += 1
            pbar.update(1)
    pbar.close()

    return downloaded, skipped, rel_paths


def _copy_file(
    src: Path,
    dest: Path,
    dry_run: bool = False,
    missing_ok: bool = False,
) -> tuple[bool, bool]:
    """Copy one file.

    Returns (copied, source_present). Already-existing destinations count as
    present but not copied.
    """
    if not src.is_file():
        if not missing_ok:
            print(f"  WARNING: source not found: {src}")
        return False, False

    src_size = src.stat().st_size
    if src_size <= 0:
        if not missing_ok:
            print(f"  WARNING: source is empty: {src}")
        return False, False

    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return False, True
    if dry_run:
        return True, True
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True, True


def _storage_mode(out_sub: str, rgb_storage: str, depth_storage: str) -> str:
    return rgb_storage if out_sub in _RGB_H5_OUTPUTS else depth_storage


def _target_h5_matches(
    dest: Path,
    out_sub: str,
    rgb_storage: str = "ffv1_sidecar",
    depth_storage: str = "ffv1_sidecar",
    expected_frames: int | None = None,
) -> bool:
    if not dest.is_file():
        return False
    try:
        if _storage_mode(out_sub, rgb_storage, depth_storage) == "ffv1_sidecar":
            info = read_ffv1_metadata(dest)
            return (
                expected_frames is None
                or int(info["n_frames"]) == int(expected_frames)
            )
        if out_sub in _RGB_H5_OUTPUTS:
            return is_jpeg_hdf5(dest)
        if out_sub == "depth":
            return has_hdf5_filters(
                dest,
                compression="gzip",
                compression_opts=6,
                shuffle=True,
            )
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return False


def _materialize_h5(
    src: Path,
    dest: Path,
    out_sub: str,
    dry_run: bool = False,
    rgb_storage: str = "ffv1_sidecar",
    depth_storage: str = "ffv1_sidecar",
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[bool, bool]:
    """Copy or transcode one HDF5 export artifact."""
    if not src.is_file() or src.stat().st_size <= 0:
        return False, False
    with h5py.File(src, "r") as source:
        if "frames" not in source:
            return False, False
        source_count = int(source["frames"].shape[0])
    start = int(start_frame)
    end = source_count if end_frame is None else int(end_frame)
    if start < 0 or end <= start or end > source_count:
        raise ValueError(
            f"Invalid export frame range [{start}, {end}) for "
            f"{source_count} frames in {src}"
        )
    output_count = end - start

    if out_sub in _MASK_H5_OUTPUTS:
        if start == 0 and end == source_count:
            return _copy_file(src, dest, dry_run)
        if dry_run:
            return True, True
        dest.parent.mkdir(parents=True, exist_ok=True)
        slice_h5_frames(
            src,
            dest,
            start_frame=start,
            end_frame=end,
            verify_frames=False,
        )
        return True, True
    if out_sub not in _RGB_H5_OUTPUTS | {"depth"}:
        return _copy_file(src, dest, dry_run)
    storage_mode = _storage_mode(out_sub, rgb_storage, depth_storage)
    if _target_h5_matches(
        dest,
        out_sub,
        rgb_storage,
        depth_storage,
        expected_frames=output_count,
    ):
        return False, True
    if dry_run:
        return True, True

    if storage_mode == "ffv1_sidecar":
        stats = transcode_h5_to_ffv1_sidecar(
            src,
            dest,
            kind="rgb" if out_sub in _RGB_H5_OUTPUTS else "depth",
            start_frame=start,
            end_frame=end,
            reindex_stems=(start != 0 or end != source_count),
            measure_random_access=False,
        )
        stats["source"] = str(src)
        stats["destination"] = str(dest)
        stats["sidecar_destination"] = stats["sidecar_path"]
        stats["size_ratio"] = stats["output_bytes"] / max(1, stats["source_bytes"])
        print("  transform: " + json.dumps(stats, sort_keys=True))
        return True, True

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_name(f".{dest.name}.tmp")
    temp_dest.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{dest.stem}.source-interval-", dir=dest.parent,
    ) as temporary:
        materialize_source = src
        if start != 0 or end != source_count:
            materialize_source = Path(temporary) / src.name
            slice_h5_frames(
                src,
                materialize_source,
                start_frame=start,
                end_frame=end,
                verify_frames=False,
            )
        try:
            if out_sub in _RGB_H5_OUTPUTS:
                if is_jpeg_hdf5(materialize_source):
                    shutil.copy2(materialize_source, temp_dest)
                    stats = {
                        "kind": "rgb_jpeg_copy",
                        "source_bytes": materialize_source.stat().st_size,
                        "output_bytes": temp_dest.stat().st_size,
                    }
                else:
                    stats = transcode_rgb_h5_to_jpeg_h5(
                        materialize_source, temp_dest,
                    )
            else:
                if has_hdf5_filters(
                    materialize_source,
                    compression="gzip",
                    compression_opts=6,
                    shuffle=True,
                ):
                    shutil.copy2(materialize_source, temp_dest)
                    stats = {
                        "kind": "depth_copy",
                        "source_bytes": materialize_source.stat().st_size,
                        "output_bytes": temp_dest.stat().st_size,
                    }
                else:
                    stats = transcode_h5_lossless(
                        materialize_source, temp_dest,
                    )
            os.replace(temp_dest, dest)
        finally:
            temp_dest.unlink(missing_ok=True)

    stats["source"] = str(src)
    stats["destination"] = str(dest)
    source_bytes = max(1, int(stats["source_bytes"]))
    stats["size_ratio"] = float(stats["output_bytes"]) / source_bytes
    print("  transform: " + json.dumps(stats, sort_keys=True))
    return True, True


def _materialize_png_directory_as_ffv1(
    src: Path,
    dest: Path,
    out_sub: str,
    dry_run: bool = False,
    rgb_storage: str = "ffv1_sidecar",
    depth_storage: str = "ffv1_sidecar",
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[bool, bool]:
    """Losslessly transcode one legacy per-frame PNG directory to FFV1."""
    if (
        not src.is_dir()
        or not any(src.glob("*.png"))
        or _storage_mode(out_sub, rgb_storage, depth_storage) != "ffv1_sidecar"
    ):
        return False, False
    png_count = len(list(src.glob("*.png")))
    start = int(start_frame)
    end = png_count if end_frame is None else int(end_frame)
    if start < 0 or end <= start or end > png_count:
        raise ValueError(
            f"Invalid export frame range [{start}, {end}) for "
            f"{png_count} PNG frames in {src}"
        )
    if _target_h5_matches(
        dest,
        out_sub,
        rgb_storage,
        depth_storage,
        expected_frames=end - start,
    ):
        return False, True
    if dry_run:
        return True, True
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{dest.stem}.png-frames-", dir=dest.parent,
    ) as temporary:
        dense_h5 = Path(temporary) / f"{dest.stem}.h5"
        pack_directory_to_h5(
            src,
            dense_h5,
            show_progress=False,
            compression=None,
            compression_opts=None,
            shuffle=False,
            start_frame=start,
            end_frame=end,
            reindex_stems=(start != 0 or end != png_count),
        )
        changed, present = _materialize_h5(
            dense_h5,
            dest,
            out_sub,
            rgb_storage=rgb_storage,
            depth_storage=depth_storage,
        )
    if changed:
        print(f"  source PNG directory: {src}")
    return changed, present


def _materialize_png_mask_directory(
    src: Path,
    dest: Path,
    *,
    dry_run: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[bool, bool]:
    """Pack one retained legacy PNG mask interval into gzip-1 HDF5."""
    png_files = sorted(src.glob("*.png")) if src.is_dir() else []
    if not png_files:
        return False, False
    start = int(start_frame)
    end = len(png_files) if end_frame is None else int(end_frame)
    if start < 0 or end <= start or end > len(png_files):
        raise ValueError(
            f"Invalid mask frame range [{start}, {end}) for "
            f"{len(png_files)} frames in {src}"
        )
    if dry_run:
        return True, True
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_name(f".{dest.name}.tmp")
    temp_dest.unlink(missing_ok=True)
    try:
        pack_directory_to_h5(
            src,
            temp_dest,
            show_progress=False,
            compression="gzip",
            compression_opts=1,
            shuffle=False,
            start_frame=start,
            end_frame=end,
            reindex_stems=(start != 0 or end != len(png_files)),
        )
        os.replace(temp_dest, dest)
    finally:
        temp_dest.unlink(missing_ok=True)
    return True, True


def _download_and_materialize_h5(
    client,
    bucket: str,
    key: str,
    dest: Path,
    out_sub: str,
    dry_run: bool,
    rgb_storage: str = "ffv1_sidecar",
    depth_storage: str = "ffv1_sidecar",
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[bool, bool]:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        if head["ContentLength"] <= 0:
            return False, False
    except client.exceptions.ClientError:
        return False, False
    if _target_h5_matches(
        dest,
        out_sub,
        rgb_storage,
        depth_storage,
        expected_frames=(
            None if end_frame is None else int(end_frame) - int(start_frame)
        ),
    ):
        return False, True
    if dry_run:
        return True, True

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_source = dest.with_name(f".{dest.name}.download.tmp")
    temp_source.unlink(missing_ok=True)
    try:
        client.download_file(bucket, key, str(temp_source))
        return _materialize_h5(
            temp_source,
            dest,
            out_sub,
            rgb_storage=rgb_storage,
            depth_storage=depth_storage,
            start_frame=start_frame,
            end_frame=end_frame,
        )
    finally:
        temp_source.unlink(missing_ok=True)


def _copy_prefix(
    src_dir: Path,
    local_dir: Path,
    remap_fn=None,
    filter_fn=None,
    dry_run: bool = False,
    label: str = "",
) -> tuple[int, int, list[str]]:
    """Copy files from a local source directory with the same filter/remap logic."""
    if not src_dir.exists():
        print(f"  WARNING: source dir not found: {src_dir}")
        return 0, 0, []

    all_files = sorted(
        f for f in src_dir.rglob("*") if f.is_file() and f.stat().st_size > 0
    )

    candidates: list[tuple[Path, Path]] = []
    for f in all_files:
        rel = str(f.relative_to(src_dir))
        if filter_fn and not filter_fn(rel):
            continue
        if remap_fn:
            rel = remap_fn(rel)
            if rel is None:
                continue
        candidates.append((f, local_dir / rel))

    downloaded = 0
    skipped = 0
    rel_paths = [str(dest.relative_to(local_dir)) for _, dest in candidates]
    desc = f"  {label}" if label else "  copying"

    for src, dest in tqdm(candidates, desc=desc, unit="file"):
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        if dry_run:
            downloaded += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        downloaded += 1

    return downloaded, skipped, rel_paths


def _is_left_camera_path(rel: str) -> bool:
    """Check if a relative path belongs to a left camera."""
    first_component = rel.split("/")[0]
    return first_component in LEFT_CAMERAS


def _is_rgb_camera_path(rel: str) -> bool:
    """Check if a relative path belongs to any camera in the stereo rig."""
    first_component = rel.split("/")[0]
    return first_component in RGB_CAMERAS


def _strip_mask_object_id(rel: str) -> str | None:
    """Remap mask paths: {cam}/0/{frame}.png -> {cam}/{frame}.png"""
    parts = rel.split("/")
    if len(parts) >= 3 and parts[1] == "0":
        return "/".join([parts[0]] + parts[2:])
    return rel


def _remap_depth(rel: str) -> str | None:
    """Flatten depth paths: {cam}/depth/{frame}.png -> {cam}/{frame}.png"""
    parts = rel.split("/")
    if len(parts) < 3 or parts[0] not in LEFT_CAMERAS or parts[1] != "depth":
        return None
    return "/".join([parts[0]] + parts[2:])


def _is_rgb_camera_video(rel: str) -> bool:
    return Path(rel).stem in RGB_CAMERAS


# Data mapping: (css_subpath, output_subpath, type, filter_fn, remap_fn, h5_layout)
# type: "file" for required single files, "optional_file" for optional single files,
#       "dir" for directory prefixes,
#       "h5_or_dir" for data that may be packed as .h5 files or PNG dirs
# h5_layout (h5_or_dir entries): None for top-level "*.h5" with original filename;
#       otherwise (glob, name_template). The glob is relative to css_subpath and
#       uses '*' to match a single path segment. The name_template formats the
#       output filename via {cam} (parent dir name of the matched h5) and {stem}
#       (h5 file stem). E.g. ("*/depth.h5", "{cam}.h5") finds <src>/<cam>/depth.h5
#       and writes it as <out>/<cam>.h5.
DEFAULT_H5_LAYOUT = ("*.h5", "{stem}.h5")
_DATA_MAP = [
    ("render_hoi_overlay/tiled_hoi_overlay.mp4", "tiled_hoi_overlay.mp4", "file", None, None, None),
    ("mv_preprocess/edex",               "edex",              "file", None, None, None),
    ("mv_preprocess/hoi_metadata.yaml",   "hoi_metadata.yaml", "file", None, None, None),
    ("mv_preprocess/images",              "images",            "h5_or_dir", _is_rgb_camera_path, None, None),
    ("mv_preprocess/videos",              "videos",            "dir",  _is_rgb_camera_video, None, None),
    ("face_detector/images",              "images_anonymized", "h5_or_dir", _is_rgb_camera_path, None, None),
    ("face_detector/videos",              "videos_anonymized", "dir",  _is_rgb_camera_video, None, None),
    ("mv_preprocess/object_mesh",         "object_mesh",       "dir",  lambda rel: rel != "output.glb", None, None),
    ("foundation_stereo",                 "depth",             "h5_or_dir", None, _remap_depth, ("*/depth.h5", "{cam}.h5")),
    ("sam2_object_masks",                 "object_masks",      "h5_or_dir", _is_left_camera_path, _strip_mask_object_id, ("*/*.h5", "{cam}.h5")),
    ("sam2_human_masks",                  "human_masks",       "h5_or_dir", _is_left_camera_path, _strip_mask_object_id, ("*/*.h5", "{cam}.h5")),
    ("foundation_pose/poses.npy",         "poses.npy",         "file", None, None, None),
    ("foundation_pose/pose_valid_mask.npy", "pose_valid_mask.npy", "optional_file", None, None, None),
    ("sam3d_body/mhr_params_mv.pt",       "mhr_params_mv.pt",  "file", None, None, None),
    ("sam3d_body/mhr_mesh_mv.pt",         "mhr_mesh_mv.pt",    "file", None, None, None),
    ("export_soma/soma_params.npz",       "soma_params.npz",   "file", None, None, None),
    ("estimate_ground_plane/ground_plane.json", "ground_plane.json", "file", None, None, None),
]

# Subset for --final_only: human/object trajectories, ground plane, object mesh
# (incl. symmetry), edex, and the tiled overlay video for QA.
_FINAL_OUTPUT_SUBPATHS = {
    "tiled_hoi_overlay.mp4",
    "edex",
    "object_mesh",
    "poses.npy",
    "pose_valid_mask.npy",
    "mhr_params_mv.pt",
    "soma_params.npz",
    "ground_plane.json",
}


def _h5_stem_matches_filter(stem: str, filter_fn) -> bool:
    """Apply a directory-based filter_fn to an h5 stem.

    Handles stems like 'front_stereo_camera_left' and also multi-object
    stems like 'front_stereo_camera_left_0' by checking if the stem starts
    with any accepted camera name.
    """
    if filter_fn is None:
        return True
    if filter_fn(stem):
        return True
    for cam in RGB_CAMERAS:
        if stem.startswith(cam) and filter_fn(cam):
            return True
    return False


def _h5_cam(rel: Path) -> str:
    """Camera identifier for an h5 file. Top-level uses stem; nested uses parent dir name."""
    return rel.parent.name if rel.parent.parts else rel.stem


def _matches_path_glob(rel: str, glob_pat: str) -> bool:
    """Match a forward-slash relative path against a glob; '*' matches one segment (no '/')."""
    r = rel.split("/")
    p = glob_pat.split("/")
    if len(r) != len(p):
        return False
    return all(fnmatch.fnmatchcase(rp, pp) for rp, pp in zip(r, p))


def _camera_from_export_rel(rel: str, cameras: list[str]) -> str | None:
    first = rel.split("/", 1)[0]
    stem = Path(first).stem
    for cam in cameras:
        if first == cam or stem == cam or stem.startswith(f"{cam}_"):
            return cam
    return None


def _camera_counts(rel_paths: list[str], cameras: list[str]) -> dict[str, int]:
    counts = {cam: 0 for cam in cameras}
    for rel in rel_paths:
        cam = _camera_from_export_rel(rel, cameras)
        if cam is not None:
            counts[cam] += 1
    return counts


def _record_required_group(
    missing: list[str],
    out_sub: str,
    source_label: str,
    rel_paths: list[str],
) -> None:
    if not rel_paths:
        missing.append(f"{out_sub}: no files found under {source_label}")
        return

    required_cameras = _REQUIRED_CAMERAS_BY_OUTPUT.get(out_sub)
    if required_cameras is None:
        return

    counts = _camera_counts(rel_paths, required_cameras)
    missing_cameras = [cam for cam in required_cameras if counts[cam] == 0]
    if missing_cameras:
        missing.append(
            f"{out_sub}: missing cameras {', '.join(missing_cameras)} "
            f"under {source_label}"
        )
        return

    unique_counts = sorted(set(counts.values()))
    if len(unique_counts) > 1:
        counts_text = ", ".join(f"{cam}={counts[cam]}" for cam in required_cameras)
        missing.append(
            f"{out_sub}: uneven camera file counts under {source_label} "
            f"({counts_text})"
        )


def _raise_if_missing_required(missing: list[str]) -> None:
    if not missing:
        return
    details = "\n".join(f"  - {item}" for item in missing)
    raise MissingRequiredExportDataError(
        "Missing required export data; refusing partial export:\n" + details
    )


def _find_h5_files_local(
    src_dir: Path,
    filter_fn=None,
    h5_layout: tuple[str, str] | None = None,
) -> list[tuple[Path, str]]:
    """Find .h5 files in src_dir per h5_layout. Returns list of (abs_path, output_name)."""
    glob_pat, name_template = h5_layout or DEFAULT_H5_LAYOUT
    if not src_dir.exists():
        return []
    results: list[tuple[Path, str]] = []
    for f in sorted(src_dir.glob(glob_pat)):
        rel = f.relative_to(src_dir)
        cam = _h5_cam(rel)
        if not _h5_stem_matches_filter(cam, filter_fn):
            continue
        out_name = name_template.format(cam=cam, stem=f.stem)
        results.append((f, out_name))
    return results


def _find_png_camera_dirs_local(
    src_dir: Path,
    out_sub: str,
    filter_fn=None,
) -> list[tuple[Path, str]]:
    """Find legacy per-camera PNG directories for RGB, depth, or masks."""
    cameras = RGB_CAMERAS if out_sub in _RGB_H5_OUTPUTS else LEFT_CAMERAS
    results: list[tuple[Path, str]] = []
    for camera in cameras:
        if not _h5_stem_matches_filter(camera, filter_fn):
            continue
        if out_sub in _RGB_H5_OUTPUTS:
            camera_dir = src_dir / camera
        elif out_sub == "depth":
            camera_dir = src_dir / camera / "depth"
        else:
            nested = src_dir / camera / "0"
            camera_dir = nested if nested.is_dir() else src_dir / camera
        if camera_dir.is_dir() and any(camera_dir.glob("*.png")):
            results.append((camera_dir, f"{camera}.h5"))
    return results


def export_sequence(
    output_dir: str,
    swift_output_base: str | None = None,
    source_dir: str | None = None,
    dry_run: bool = False,
    max_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    max_camera_workers: int = DEFAULT_CAMERA_WORKERS,
    final_only: bool = False,
    include_anonymized_rgb: bool = False,
    rgb_storage: str = "ffv1_sidecar",
    depth_storage: str = "ffv1_sidecar",
    source_start_frame: int = 0,
    source_end_frame: int | None = None,
) -> None:
    """Export a sequence to a flat local directory structure.

    Exactly one of swift_output_base or source_dir must be provided.

    Args:
        output_dir: Local directory to write the exported data.
        swift_output_base: Swift URL or bare S3 path for remote download.
        source_dir: Local directory path for local copy.
        dry_run: If True, list files without downloading/copying.
        max_workers: Parallel download threads (remote mode only).
        max_camera_workers: Parallel per-camera archive transforms.
        final_only: If True, export only final outputs (trajectories, ground
            plane, object mesh, edex, tiled overlay) — skips intermediate
            depth/mask/image/video data.
        include_anonymized_rgb: Export and require face-anonymized RGB artifacts.
        source_start_frame: Inclusive source-frame offset for temporal archives.
        source_end_frame: Exclusive source-frame offset; defaults to source end.
    """
    if (swift_output_base is None) == (source_dir is None):
        raise ValueError("Exactly one of swift_output_base or source_dir must be provided")
    if rgb_storage not in {"jpeg_h5", "ffv1_sidecar"}:
        raise ValueError(f"Unsupported RGB storage mode: {rgb_storage}")
    if depth_storage not in {"gzip_h5", "ffv1_sidecar"}:
        raise ValueError(f"Unsupported depth storage mode: {depth_storage}")
    if max_camera_workers < 1:
        raise ValueError("max_camera_workers must be at least 1")
    if source_start_frame < 0:
        raise ValueError("source_start_frame must be non-negative")
    if (
        source_end_frame is not None
        and int(source_end_frame) <= int(source_start_frame)
    ):
        raise ValueError("source_end_frame must exceed source_start_frame")

    output = Path(output_dir)
    is_local = source_dir is not None
    source_label = source_dir if is_local else swift_output_base

    if final_only:
        data_map = [e for e in _DATA_MAP if e[1] in _FINAL_OUTPUT_SUBPATHS]
    else:
        data_map = [
            entry
            for entry in _DATA_MAP
            if include_anonymized_rgb or entry[1] not in _ANONYMIZED_OUTPUTS
        ]

    total_copied = 0
    total_skipped = 0
    missing_required: list[str] = []

    def _report(label: str, dl: int, sk: int):
        nonlocal total_copied, total_skipped
        total_copied += dl
        total_skipped += sk
        verb = "copied" if is_local else "downloaded"
        status = f"{verb}={dl} skipped={sk}" if not dry_run else f"would {verb}={dl}"
        print(f"  {label}: {status}")

    mode = "local" if is_local else "remote"
    print(f"Exporting from {source_label} (mode={mode}, final_only={final_only})")
    print(f"  -> {output_dir}")
    print(f"  camera workers: {max_camera_workers}")
    if not is_local:
        print(f"  workers: {max_workers}")
    print()

    if is_local:
        _export_local(
            Path(source_dir),
            output,
            dry_run,
            _report,
            data_map,
            missing_required,
            rgb_storage,
            depth_storage,
            max_camera_workers,
            source_start_frame,
            source_end_frame,
        )
    else:
        _export_remote(
            swift_output_base,
            output,
            dry_run,
            max_workers,
            _report,
            data_map,
            missing_required,
            rgb_storage,
            depth_storage,
            max_camera_workers,
            source_start_frame,
            source_end_frame,
        )

    verb = "copied" if is_local else "downloaded"
    print(f"\nTotal: {verb}={total_copied} skipped={total_skipped}")
    _raise_if_missing_required(missing_required)


def _export_local(
    source: Path,
    output: Path,
    dry_run: bool,
    report,
    data_map: list,
    missing_required: list[str],
    rgb_storage: str,
    depth_storage: str,
    max_camera_workers: int,
    source_start_frame: int,
    source_end_frame: int | None,
) -> None:
    """Copy from a local directory (e.g. OSMO-mounted inputs)."""
    for css_sub, out_sub, entry_type, filter_fn, remap_fn, h5_layout in data_map:
        src_path = source / css_sub
        if entry_type == "file":
            did, present = _copy_file(src_path, output / out_sub, dry_run)
            if not present:
                missing_required.append(f"{out_sub}: missing source file {src_path}")
            report(out_sub, int(did), int(present and not did))
        elif entry_type == "optional_file":
            did, present = _copy_file(
                src_path,
                output / out_sub,
                dry_run,
                missing_ok=True,
            )
            if did or present:
                report(out_sub, int(did), int(present and not did))
        elif entry_type == "h5_or_dir":
            h5_files = _find_h5_files_local(src_path, filter_fn, h5_layout)
            if h5_files:
                dl_total, sk_total = 0, 0
                rel_paths: list[str] = []

                def _materialize_camera(item):
                    h5_src, h5_name = item
                    return _materialize_h5(
                        h5_src,
                        output / out_sub / h5_name,
                        out_sub,
                        dry_run,
                        rgb_storage,
                        depth_storage,
                        source_start_frame,
                        source_end_frame,
                    )
                results = _run_camera_jobs(
                    h5_files, _materialize_camera, max_camera_workers,
                )
                for (h5_src, h5_name), (did, present) in zip(
                    h5_files, results, strict=True,
                ):
                    if not present:
                        missing_required.append(f"{out_sub}: missing source file {h5_src}")
                        continue
                    rel_paths.append(h5_name)
                    if did:
                        dl_total += 1
                    else:
                        sk_total += 1
                _record_required_group(missing_required, out_sub, str(src_path), rel_paths)
                report(out_sub, dl_total, sk_total)
            else:
                if out_sub in _MASK_H5_OUTPUTS:
                    png_dirs = _find_png_camera_dirs_local(
                        src_path, out_sub, filter_fn,
                    )
                    dl_total, sk_total = 0, 0
                    rel_paths: list[str] = []

                    def _materialize_mask_camera(item):
                        png_dir, h5_name = item
                        return _materialize_png_mask_directory(
                            png_dir,
                            output / out_sub / h5_name,
                            dry_run=dry_run,
                            start_frame=source_start_frame,
                            end_frame=source_end_frame,
                        )

                    results = _run_camera_jobs(
                        png_dirs, _materialize_mask_camera, max_camera_workers,
                    )
                    for (png_dir, h5_name), (did, present) in zip(
                        png_dirs, results, strict=True,
                    ):
                        if not present:
                            missing_required.append(
                                f"{out_sub}: invalid PNG source directory {png_dir}"
                            )
                            continue
                        rel_paths.append(h5_name)
                        dl_total += int(did)
                        sk_total += int(not did)
                    _record_required_group(
                        missing_required, out_sub, str(src_path), rel_paths,
                    )
                    report(out_sub, dl_total, sk_total)
                    continue
                if (
                    out_sub in _RGB_H5_OUTPUTS | {"depth"}
                    and _storage_mode(out_sub, rgb_storage, depth_storage)
                    == "ffv1_sidecar"
                ):
                    png_dirs = _find_png_camera_dirs_local(
                        src_path, out_sub, filter_fn,
                    )
                    dl_total, sk_total = 0, 0
                    rel_paths: list[str] = []

                    def _materialize_png_camera(item):
                        png_dir, h5_name = item
                        return _materialize_png_directory_as_ffv1(
                            png_dir,
                            output / out_sub / h5_name,
                            out_sub,
                            dry_run,
                            rgb_storage,
                            depth_storage,
                            source_start_frame,
                            source_end_frame,
                        )
                    results = _run_camera_jobs(
                        png_dirs, _materialize_png_camera, max_camera_workers,
                    )
                    for (png_dir, h5_name), (did, present) in zip(
                        png_dirs, results, strict=True,
                    ):
                        if not present:
                            missing_required.append(
                                f"{out_sub}: invalid PNG source directory {png_dir}"
                            )
                            continue
                        rel_paths.append(h5_name)
                        if did:
                            dl_total += 1
                        else:
                            sk_total += 1
                    _record_required_group(
                        missing_required, out_sub, str(src_path), rel_paths,
                    )
                    report(out_sub, dl_total, sk_total)
                    continue
                print(f"Copying {out_sub} (dir)...")
                dl, sk, rel_paths = _copy_prefix(
                    src_path, output / out_sub,
                    remap_fn=remap_fn, filter_fn=filter_fn,
                    dry_run=dry_run, label=out_sub,
                )
                _record_required_group(missing_required, out_sub, str(src_path), rel_paths)
                report(out_sub, dl, sk)
        else:
            print(f"Copying {out_sub}...")
            dl, sk, rel_paths = _copy_prefix(
                src_path, output / out_sub,
                remap_fn=remap_fn, filter_fn=filter_fn,
                dry_run=dry_run, label=out_sub,
            )
            _record_required_group(missing_required, out_sub, str(src_path), rel_paths)
            report(out_sub, dl, sk)


def _has_h5_remote(
    client,
    bucket: str,
    prefix: str,
    filter_fn=None,
    h5_layout: tuple[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Find .h5 keys under prefix matching h5_layout. Returns (key, output_name) pairs."""
    glob_pat, name_template = h5_layout or DEFAULT_H5_LAYOUT
    objects = _list_objects(client, bucket, prefix)
    base = prefix.rstrip("/") + "/"
    results: list[tuple[str, str]] = []
    for obj in objects:
        key = obj["Key"]
        if obj.get("Size", 0) <= 0:
            continue
        if not key.startswith(base):
            continue
        rel_str = key[len(base):]
        if not _matches_path_glob(rel_str, glob_pat):
            continue
        rel = Path(rel_str)
        cam = _h5_cam(rel)
        if not _h5_stem_matches_filter(cam, filter_fn):
            continue
        out_name = name_template.format(cam=cam, stem=rel.stem)
        results.append((key, out_name))
    return results


def _export_remote(
    swift_output_base: str,
    output: Path,
    dry_run: bool,
    max_workers: int,
    report,
    data_map: list,
    missing_required: list[str],
    rgb_storage: str,
    depth_storage: str,
    max_camera_workers: int,
    source_start_frame: int,
    source_end_frame: int | None,
) -> None:
    """Download from CSS via boto3."""
    client = _get_s3_client(swift_output_base)
    bucket, base_prefix = _parse_swift_url(swift_output_base)

    for css_sub, out_sub, entry_type, filter_fn, remap_fn, h5_layout in data_map:
        if entry_type == "file":
            key = f"{base_prefix}/{css_sub}"
            did, present = _download_file(
                client,
                bucket,
                key,
                output / out_sub,
                dry_run,
            )
            if not present:
                missing_required.append(f"{out_sub}: missing key s3://{bucket}/{key}")
            report(out_sub, int(did), int(present and not did))
        elif entry_type == "optional_file":
            key = f"{base_prefix}/{css_sub}"
            did, present = _download_file(
                client,
                bucket,
                key,
                output / out_sub,
                dry_run,
                missing_ok=True,
            )
            if did or present:
                report(out_sub, int(did), int(present and not did))
        elif entry_type == "h5_or_dir":
            css_prefix = f"{base_prefix}/{css_sub}"
            h5_keys = _has_h5_remote(client, bucket, css_prefix, filter_fn, h5_layout)
            if h5_keys:
                dl_total, sk_total = 0, 0
                rel_paths: list[str] = []

                def _download_camera(item):
                    key, h5_name = item
                    return _download_and_materialize_h5(
                        client,
                        bucket,
                        key,
                        output / out_sub / h5_name,
                        out_sub,
                        dry_run,
                        rgb_storage,
                        depth_storage,
                        source_start_frame,
                        source_end_frame,
                    )
                results = _run_camera_jobs(
                    h5_keys, _download_camera, max_camera_workers,
                )
                for (key, h5_name), (did, present) in zip(
                    h5_keys, results, strict=True,
                ):
                    if not present:
                        missing_required.append(
                            f"{out_sub}: missing key s3://{bucket}/{key}"
                        )
                        continue
                    rel_paths.append(h5_name)
                    if did:
                        dl_total += 1
                    else:
                        sk_total += 1
                _record_required_group(
                    missing_required,
                    out_sub,
                    f"s3://{bucket}/{css_prefix}",
                    rel_paths,
                )
                report(out_sub, dl_total, sk_total)
            else:
                if (
                    out_sub in _RGB_H5_OUTPUTS | {"depth"}
                    and _storage_mode(out_sub, rgb_storage, depth_storage)
                    == "ffv1_sidecar"
                ):
                    missing_required.append(
                        f"{out_sub}: FFV1 export requires HDF5 source frames under "
                        f"s3://{bucket}/{css_prefix}"
                    )
                    continue
                print(f"Downloading {out_sub} (dir)...")
                dl, sk, rel_paths = _download_prefix(
                    client, bucket, css_prefix,
                    output / out_sub,
                    filter_fn=filter_fn, remap_fn=remap_fn,
                    dry_run=dry_run, label=out_sub,
                    max_workers=max_workers,
                )
                _record_required_group(
                    missing_required,
                    out_sub,
                    f"s3://{bucket}/{css_prefix}",
                    rel_paths,
                )
                report(out_sub, dl, sk)
        else:
            print(f"Downloading {out_sub}...")
            css_prefix = f"{base_prefix}/{css_sub}"
            dl, sk, rel_paths = _download_prefix(
                client, bucket,
                css_prefix,
                output / out_sub,
                filter_fn=filter_fn, remap_fn=remap_fn,
                dry_run=dry_run, label=out_sub,
                max_workers=max_workers,
            )
            _record_required_group(
                missing_required,
                out_sub,
                f"s3://{bucket}/{css_prefix}",
                rel_paths,
            )
            report(out_sub, dl, sk)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a reconstructed sequence to a flat training layout"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--swift_output_base", type=str,
        help="Swift URL for remote download "
             "(e.g. swift://storage.example.com/AUTH_.../data_output/<seq>)",
    )
    source.add_argument(
        "--source_dir", type=str,
        help="Local directory containing OSMO task outputs",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Local directory to write the exported data",
    )
    parser.add_argument("--dry_run", action="store_true", help="List files without downloading/copying")
    parser.add_argument("--max_workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS,
                        help=f"Parallel download threads for remote mode (default: {DEFAULT_DOWNLOAD_WORKERS})")
    parser.add_argument(
        "--max_camera_workers",
        type=int,
        default=DEFAULT_CAMERA_WORKERS,
        help=(
            "Parallel per-camera archive transforms "
            f"(default: {DEFAULT_CAMERA_WORKERS})"
        ),
    )
    parser.add_argument("--final_only", action="store_true",
                        help="Export only final outputs (trajectories, ground plane, object mesh, "
                             "edex, tiled overlay); skip depth/masks/images/videos.")
    parser.add_argument(
        "--include_anonymized_rgb",
        action="store_true",
        help="Export and require face_detector images/videos as anonymized siblings.",
    )
    parser.add_argument(
        "--rgb_storage",
        choices=("jpeg_h5", "ffv1_sidecar"),
        default="ffv1_sidecar",
    )
    parser.add_argument(
        "--depth_storage",
        choices=("gzip_h5", "ffv1_sidecar"),
        default="ffv1_sidecar",
    )
    parser.add_argument("--source-start-frame", type=int, default=0)
    parser.add_argument("--source-end-frame", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    export_sequence(
        output_dir=args.output_dir,
        swift_output_base=args.swift_output_base,
        source_dir=args.source_dir,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
        max_camera_workers=args.max_camera_workers,
        final_only=args.final_only,
        include_anonymized_rgb=args.include_anonymized_rgb,
        rgb_storage=args.rgb_storage,
        depth_storage=args.depth_storage,
        source_start_frame=args.source_start_frame,
        source_end_frame=args.source_end_frame,
    )


if __name__ == "__main__":
    main()
