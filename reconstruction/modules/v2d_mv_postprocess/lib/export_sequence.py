"""Export a reconstructed sequence into a flat training-ready layout.

Supports two modes:
  - Remote: download from CSS via boto3 (swift:// URL or bare S3 path).
  - Local: copy from a local directory (e.g. OSMO-mounted task outputs).

The mode is auto-detected: if the source path is an existing local directory,
local copy is used; otherwise it's treated as a remote S3 path.

Usage (remote):
    python -m v2d.mv.postprocess.lib.export_sequence \
        --swift_output_base swift://pdx.s8k.io/AUTH_.../data_output/<seq> \
        --output_dir /local/path/to/sequence

Usage (local):
    python -m v2d.mv.postprocess.lib.export_sequence \
        --swift_output_base /osmo/data/input/0 \
        --output_dir /osmo/data/output
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from tqdm import tqdm

DEFAULT_DOWNLOAD_WORKERS = os.cpu_count() or 8

ENDPOINT_URL = os.environ.get("CSS_ENDPOINT_URL", "https://pdx.s8k.io")
ACCESS_KEY = os.environ.get("CSS_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("CSS_SECRET_KEY", "")
REGION = os.environ.get("CSS_REGION", "us-east-1")

LEFT_CAMERAS = [
    "front_stereo_camera_left",
    "back_stereo_camera_left",
    "left_stereo_camera_left",
    "right_stereo_camera_left",
]


def _get_s3_client():
    if not ACCESS_KEY or not SECRET_KEY:
        print(
            "Error: Set CSS_ACCESS_KEY and CSS_SECRET_KEY environment variables.\n"
            "  source reconstruction/scripts/setup_css_env.sh",
            file=sys.stderr,
        )
        sys.exit(1)
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
        config=Config(connect_timeout=10),
    )


def _parse_swift_url(url: str) -> tuple[str, str]:
    """Return (bucket, prefix) from a swift:// URL or a bare bucket/path."""
    if url.startswith("swift://"):
        stripped = url.replace("swift://", "").rstrip("/")
        parts = stripped.split("/", 3)
        bucket = parts[2] if len(parts) > 2 else ""
        prefix = parts[3] if len(parts) > 3 else ""
        if not bucket:
            print("Error: swift:// URL must include a container/bucket.", file=sys.stderr)
            sys.exit(1)
        return bucket, prefix

    stripped = url.strip("/")
    parts = stripped.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if not bucket:
        print("Error: remote path must not be empty.", file=sys.stderr)
        sys.exit(1)
    return bucket, prefix


def _list_objects(client, bucket: str, prefix: str) -> list[dict]:
    """List all objects under a prefix."""
    paginator = client.get_paginator("list_objects_v2")
    folder_prefix = prefix.rstrip("/") + "/"
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
        objects.extend(page.get("Contents", []))
    return objects


def _download_file(client, bucket: str, key: str, dest: Path, dry_run: bool = False) -> bool:
    """Download a single file, skipping if already exists with same size. Returns True if downloaded."""
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        remote_size = head["ContentLength"]
    except client.exceptions.ClientError:
        print(f"  WARNING: key not found: {key}")
        return False

    if dest.exists() and dest.stat().st_size == remote_size:
        return False

    if dry_run:
        print(f"  [dry-run] would download: {key}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))
    return True


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
) -> tuple[int, int]:
    """Download all objects under css_prefix into local_dir.

    Uses a thread pool for parallel downloads. Each thread gets its own
    boto3 client since they are not thread-safe.

    Args:
        remap_fn: Optional function (rel_path) -> new_rel_path to remap file paths.
        filter_fn: Optional function (rel_path) -> bool to filter files.
        label: Human-readable label for progress bar.
        max_workers: Number of parallel download threads.

    Returns (downloaded_count, skipped_count).
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
        work_items.append((obj["Key"], obj["Size"], local_dir / rel))

    downloaded = 0
    skipped = 0
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
        return len(to_download), skipped

    # Thread-local boto3 clients
    _local = threading.local()

    def _get_thread_client():
        if not hasattr(_local, "client"):
            _local.client = _get_s3_client()
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

    return downloaded, skipped


def _copy_file(src: Path, dest: Path, dry_run: bool = False) -> bool:
    """Copy a single local file, skipping if already exists with same size. Returns True if copied."""
    if not src.exists():
        print(f"  WARNING: source not found: {src}")
        return False
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return False
    if dry_run:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _copy_prefix(
    src_dir: Path,
    local_dir: Path,
    remap_fn=None,
    filter_fn=None,
    dry_run: bool = False,
    label: str = "",
) -> tuple[int, int]:
    """Copy files from a local source directory with the same filter/remap logic."""
    if not src_dir.exists():
        print(f"  WARNING: source dir not found: {src_dir}")
        return 0, 0

    all_files = sorted(f for f in src_dir.rglob("*") if f.is_file())

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

    return downloaded, skipped


def _is_left_camera_path(rel: str) -> bool:
    """Check if a relative path belongs to a left camera."""
    first_component = rel.split("/")[0]
    return first_component in LEFT_CAMERAS


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


def _is_left_camera_video(rel: str) -> bool:
    return Path(rel).stem in LEFT_CAMERAS


# Data mapping: (css_subpath, output_subpath, type, filter_fn, remap_fn, h5_layout)
# type: "file" for single files, "dir" for directory prefixes,
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
    ("mv_preprocess/images",              "images",            "h5_or_dir", _is_left_camera_path, None, None),
    ("mv_preprocess/videos",              "videos",            "dir",  _is_left_camera_video, None, None),
    ("mv_preprocess/object_mesh",         "object_mesh",       "dir",  lambda rel: rel != "output.glb", None, None),
    ("foundation_stereo",                 "depth",             "h5_or_dir", None, _remap_depth, ("*/depth.h5", "{cam}.h5")),
    ("sam2_object_masks",                 "object_masks",      "h5_or_dir", _is_left_camera_path, _strip_mask_object_id, ("*/*.h5", "{cam}.h5")),
    ("sam2_human_masks",                  "human_masks",       "h5_or_dir", _is_left_camera_path, _strip_mask_object_id, ("*/*.h5", "{cam}.h5")),
    ("foundation_pose/poses.npy",         "poses.npy",         "file", None, None, None),
    ("sam3d_body/mhr_params_mv.pt",       "mhr_params_mv.pt",  "file", None, None, None),
    ("sam3d_body/mhr_mesh_mv.pt",         "mhr_mesh_mv.pt",    "file", None, None, None),
    ("export_soma/soma_params.npz",       "soma_params.npz",   "file", None, None, None),
    ("estimate_ground_plane/ground_plane.json", "ground_plane.json", "file", None, None, None),
]


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
    for cam in LEFT_CAMERAS:
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


def export_sequence(
    output_dir: str,
    swift_output_base: str | None = None,
    source_dir: str | None = None,
    dry_run: bool = False,
    max_workers: int = DEFAULT_DOWNLOAD_WORKERS,
) -> None:
    """Export a sequence to a flat local directory structure.

    Exactly one of swift_output_base or source_dir must be provided.

    Args:
        output_dir: Local directory to write the exported data.
        swift_output_base: Swift URL or bare S3 path for remote download.
        source_dir: Local directory path for local copy.
        dry_run: If True, list files without downloading/copying.
        max_workers: Parallel download threads (remote mode only).
    """
    if (swift_output_base is None) == (source_dir is None):
        raise ValueError("Exactly one of swift_output_base or source_dir must be provided")

    output = Path(output_dir)
    is_local = source_dir is not None
    source_label = source_dir if is_local else swift_output_base

    total_copied = 0
    total_skipped = 0

    def _report(label: str, dl: int, sk: int):
        nonlocal total_copied, total_skipped
        total_copied += dl
        total_skipped += sk
        verb = "copied" if is_local else "downloaded"
        status = f"{verb}={dl} skipped={sk}" if not dry_run else f"would {verb}={dl}"
        print(f"  {label}: {status}")

    mode = "local" if is_local else "remote"
    print(f"Exporting from {source_label} (mode={mode})")
    print(f"  -> {output_dir}")
    if not is_local:
        print(f"  workers: {max_workers}")
    print()

    if is_local:
        _export_local(Path(source_dir), output, dry_run, _report)
    else:
        _export_remote(swift_output_base, output, dry_run, max_workers, _report)

    verb = "copied" if is_local else "downloaded"
    print(f"\nTotal: {verb}={total_copied} skipped={total_skipped}")


def _export_local(
    source: Path,
    output: Path,
    dry_run: bool,
    report,
) -> None:
    """Copy from a local directory (e.g. OSMO-mounted inputs)."""
    for css_sub, out_sub, entry_type, filter_fn, remap_fn, h5_layout in _DATA_MAP:
        src_path = source / css_sub
        if entry_type == "file":
            did = _copy_file(src_path, output / out_sub, dry_run)
            report(out_sub, int(did), int(not did))
        elif entry_type == "h5_or_dir":
            h5_files = _find_h5_files_local(src_path, filter_fn, h5_layout)
            if h5_files:
                dl_total, sk_total = 0, 0
                for h5_src, h5_name in h5_files:
                    did = _copy_file(h5_src, output / out_sub / h5_name, dry_run)
                    if did:
                        dl_total += 1
                    else:
                        sk_total += 1
                report(out_sub, dl_total, sk_total)
            else:
                print(f"Copying {out_sub} (dir)...")
                dl, sk = _copy_prefix(
                    src_path, output / out_sub,
                    remap_fn=remap_fn, filter_fn=filter_fn,
                    dry_run=dry_run, label=out_sub,
                )
                report(out_sub, dl, sk)
        else:
            print(f"Copying {out_sub}...")
            dl, sk = _copy_prefix(
                src_path, output / out_sub,
                remap_fn=remap_fn, filter_fn=filter_fn,
                dry_run=dry_run, label=out_sub,
            )
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
) -> None:
    """Download from CSS via boto3."""
    client = _get_s3_client()
    bucket, base_prefix = _parse_swift_url(swift_output_base)

    for css_sub, out_sub, entry_type, filter_fn, remap_fn, h5_layout in _DATA_MAP:
        if entry_type == "file":
            key = f"{base_prefix}/{css_sub}"
            did = _download_file(client, bucket, key, output / out_sub, dry_run)
            report(out_sub, int(did), int(not did))
        elif entry_type == "h5_or_dir":
            css_prefix = f"{base_prefix}/{css_sub}"
            h5_keys = _has_h5_remote(client, bucket, css_prefix, filter_fn, h5_layout)
            if h5_keys:
                dl_total, sk_total = 0, 0
                for key, h5_name in h5_keys:
                    did = _download_file(client, bucket, key, output / out_sub / h5_name, dry_run)
                    if did:
                        dl_total += 1
                    else:
                        sk_total += 1
                report(out_sub, dl_total, sk_total)
            else:
                print(f"Downloading {out_sub} (dir)...")
                dl, sk = _download_prefix(
                    client, bucket, css_prefix,
                    output / out_sub,
                    filter_fn=filter_fn, remap_fn=remap_fn,
                    dry_run=dry_run, label=out_sub,
                    max_workers=max_workers,
                )
                report(out_sub, dl, sk)
        else:
            print(f"Downloading {out_sub}...")
            dl, sk = _download_prefix(
                client, bucket,
                f"{base_prefix}/{css_sub}",
                output / out_sub,
                filter_fn=filter_fn, remap_fn=remap_fn,
                dry_run=dry_run, label=out_sub,
                max_workers=max_workers,
            )
            report(out_sub, dl, sk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a reconstructed sequence to a flat training layout"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--swift_output_base", type=str,
        help="Swift URL for remote download "
             "(e.g. swift://pdx.s8k.io/AUTH_.../data_output/<seq>)",
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
    args = parser.parse_args()

    export_sequence(
        output_dir=args.output_dir,
        swift_output_base=args.swift_output_base,
        source_dir=args.source_dir,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
    )
