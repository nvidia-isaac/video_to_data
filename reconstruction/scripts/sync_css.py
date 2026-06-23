#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download or upload files/folders to CSS (NVIDIA PDX storage).

Operates on swift:// URLs or bucket-relative paths used by the reconstruction
pipelines.  Skips files that already exist at the destination with the same
size.

Prerequisites:
  - boto3: pip install boto3
  - CSS credentials configured via environment variables:
      source reconstruction/scripts/setup_css_env.sh

Usage:
  # Download a folder
  python reconstruction/scripts/sync_css.py download \
      swift://pdx.s8k.io/AUTH_team-isaac/recordings/v2d/multiview/sc_office_4exo_1/data/seq_001 \
      /tmp/seq_001

  # Upload a folder
  python reconstruction/scripts/sync_css.py upload \
      /tmp/seq_001 \
      swift://pdx.s8k.io/AUTH_team-isaac/recordings/v2d/multiview/sc_office_4exo_1/data_output/seq_001

  # Download a single file
  python reconstruction/scripts/sync_css.py download \
      swift://pdx.s8k.io/AUTH_team-isaac/recordings/v2d/mesh/tall_bar_stool/einstar/mesh.obj \
      /tmp/mesh.obj

  # List remote directory contents
  python reconstruction/scripts/sync_css.py ls \
      swift://pdx.s8k.io/AUTH_team-isaac/recordings/v2d/multiview/sc_office_4exo_1/data/seq_001

  # Dry-run to see what would be transferred
  python reconstruction/scripts/sync_css.py download \
      swift://pdx.s8k.io/AUTH_team-isaac/recordings/v2d/multiview/sc_office_4exo_1/data/seq_001 \
      /tmp/seq_001 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

ENDPOINT_URL = os.environ.get("CSS_ENDPOINT_URL", "https://pdx.s8k.io")
ACCESS_KEY = os.environ.get("CSS_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("CSS_SECRET_KEY", "")
REGION = os.environ.get("CSS_REGION", "us-east-1")  # Ignored by CSS, location is inferred from the endpoint URL


# ── S3 helpers ─────────────────────────────────────────────────────────


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


def _env_host() -> str:
    """Derive the host from CSS_ENDPOINT_URL (e.g. 'pdx.s8k.io')."""
    return ENDPOINT_URL.replace("https://", "").replace("http://", "").rstrip("/")


def _env_account() -> str:
    """Derive the Swift account from CSS_ACCESS_KEY.

    e.g. 'v2p:AUTH_team-isaac' -> 'AUTH_team-isaac'
    """
    if ":" in ACCESS_KEY:
        return ACCESS_KEY.split(":", 1)[1]
    return ""


def _parse_swift_url(url: str) -> tuple[str, str]:
    """Return (bucket, prefix) from a swift:// URL or a bare bucket/path.

    Swift URLs have the form:
        swift://pdx.s8k.io/AUTH_team-isaac/recordings/v2d/mesh/...
                ^host       ^account       ^bucket    ^prefix...

    The account is handled by credentials and is skipped.  The S3 bucket
    is the Swift container (``recordings`` above), and everything after it
    is the object key prefix.

    Bare paths are also accepted:
        recordings/v2d/mesh/...   ->  bucket='recordings', prefix='v2d/mesh/...'

    Validates that the host and account match the env vars when present.
    """
    if url.startswith("swift://"):
        stripped = url.replace("swift://", "").rstrip("/")
        # host / account / bucket / prefix...
        parts = stripped.split("/", 3)
        host = parts[0]
        account = parts[1] if len(parts) > 1 else ""
        bucket = parts[2] if len(parts) > 2 else ""
        prefix = parts[3] if len(parts) > 3 else ""

        expected_host = _env_host()
        expected_account = _env_account()
        if expected_host and host != expected_host:
            print(
                f"Warning: URL host '{host}' does not match "
                f"CSS_ENDPOINT_URL '{expected_host}'",
                file=sys.stderr,
            )
        if expected_account and account != expected_account:
            print(
                f"Warning: URL account '{account}' does not match "
                f"CSS_ACCESS_KEY account '{expected_account}'",
                file=sys.stderr,
            )
        if not bucket:
            print("Error: swift:// URL must include a container/bucket.", file=sys.stderr)
            sys.exit(1)
        return bucket, prefix

    # Bare path — first component is the bucket (container)
    stripped = url.strip("/")
    parts = stripped.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if not bucket:
        print("Error: remote path must not be empty.", file=sys.stderr)
        sys.exit(1)
    return bucket, prefix


# ── Download ───────────────────────────────────────────────────────────


def download(
    client,
    bucket: str,
    prefix: str,
    local_path: Path,
    dry_run: bool = False,
) -> None:
    """Download objects under *prefix* to *local_path*.

    If the remote path is a single object, downloads it directly.
    If it's a prefix (folder), downloads all objects beneath it,
    preserving the directory structure.
    """
    # Ensure prefix doesn't end with / for the initial existence check,
    # but try both as a key (file) and as a prefix (folder).
    paginator = client.get_paginator("list_objects_v2")
    folder_prefix = prefix.rstrip("/") + "/"

    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
        objects.extend(page.get("Contents", []))

    if not objects:
        # Try as a single-file key
        try:
            head = client.head_object(Bucket=bucket, Key=prefix)
            objects = [{"Key": prefix, "Size": head["ContentLength"]}]
            folder_prefix = None
        except client.exceptions.ClientError:
            print(f"No objects found at s3://{bucket}/{prefix}")
            return

    downloaded = 0
    skipped = 0

    for obj in objects:
        key = obj["Key"]
        remote_size = obj["Size"]

        if folder_prefix:
            rel = key[len(folder_prefix):]
            if not rel:
                continue
            dest = local_path / rel
        else:
            dest = local_path

        if dest.exists() and dest.stat().st_size == remote_size:
            skipped += 1
            continue

        if dry_run:
            print(f"  [dry-run] would download: {key} ({remote_size} bytes)")
            downloaded += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading: {key} ({remote_size} bytes)")
        client.download_file(bucket, key, str(dest))
        downloaded += 1

    print(f"  done: {downloaded} downloaded, {skipped} skipped (same size)")


# ── Upload ─────────────────────────────────────────────────────────────


def upload(
    client,
    bucket: str,
    prefix: str,
    local_path: Path,
    dry_run: bool = False,
) -> None:
    """Upload *local_path* (file or directory) to *prefix* in the bucket.

    If *local_path* is a directory, all files beneath it are uploaded,
    preserving relative paths under *prefix*.
    """
    if local_path.is_file():
        files = [(local_path, prefix)]
    elif local_path.is_dir():
        dest_prefix = prefix.rstrip("/")
        files = []
        for f in sorted(local_path.rglob("*")):
            if f.is_file():
                rel = f.relative_to(local_path)
                files.append((f, f"{dest_prefix}/{rel}"))
    else:
        print(f"Local path does not exist: {local_path}", file=sys.stderr)
        sys.exit(1)

    uploaded = 0
    skipped = 0

    for local_file, key in files:
        local_size = local_file.stat().st_size

        try:
            head = client.head_object(Bucket=bucket, Key=key)
            if head["ContentLength"] == local_size:
                skipped += 1
                continue
        except client.exceptions.ClientError:
            pass

        if dry_run:
            print(f"  [dry-run] would upload: {local_file} -> {key} ({local_size} bytes)")
            uploaded += 1
            continue

        print(f"  uploading: {local_file} -> s3://{bucket}/{key} ({local_size} bytes)")
        client.upload_file(str(local_file), bucket, key)
        uploaded += 1

    print(f"  done: {uploaded} uploaded, {skipped} skipped (same size)")


# ── List ───────────────────────────────────────────────────────────────


def ls(
    client,
    bucket: str,
    prefix: str,
    recursive: bool = False,
    limit: int | None = None,
) -> None:
    """List objects or immediate subdirectories under *prefix*."""
    if not prefix.endswith("/") and prefix:
        prefix += "/"

    def _fmt_mtime(dt) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else " " * 19

    if recursive:
        paginator = client.get_paginator("list_objects_v2")
        total = 0
        total_size = 0
        truncated = False
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix):]
                if rel:
                    size = obj["Size"]
                    mtime = _fmt_mtime(obj.get("LastModified"))
                    total_size += size
                    total += 1
                    if limit and total > limit:
                        truncated = True
                        break
                    print(f"  {mtime}  {size:>12}  {rel}")
            if truncated:
                break
        msg = f"\n  {total:,} objects"
        if truncated:
            msg += f"+ (limited to {limit})"
        msg += f", {total_size:,} bytes shown"
        print(msg)
    else:
        paginator = client.get_paginator("list_objects_v2")
        dirs: list[str] = []
        files: list[tuple[str, int, str]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"][len(prefix):].rstrip("/")
                if name:
                    dirs.append(name)
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix):]
                if rel:
                    files.append((rel, obj["Size"], _fmt_mtime(obj.get("LastModified"))))

        shown = 0
        for d in sorted(dirs):
            shown += 1
            if limit and shown > limit:
                print(f"  ... truncated at {limit} entries")
                break
            print(f"  {' ' * 19}  {'DIR':>12}  {d}/")
        for name, size, mtime in sorted(files):
            shown += 1
            if limit and shown > limit:
                print(f"  ... truncated at {limit} entries")
                break
            print(f"  {mtime}  {size:>12}  {name}")
        print(f"\n  {len(dirs)} directories, {len(files)} files")


# ── Delete ─────────────────────────────────────────────────────────────


def delete(
    client,
    bucket: str,
    prefix: str,
    dry_run: bool = False,
) -> None:
    """Delete a single object or all objects under *prefix* recursively."""
    paginator = client.get_paginator("list_objects_v2")
    folder_prefix = prefix.rstrip("/") + "/"

    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))

    if not keys:
        try:
            client.head_object(Bucket=bucket, Key=prefix)
            keys = [prefix]
        except client.exceptions.ClientError:
            print(f"No objects found at s3://{bucket}/{prefix}")
            return

    print(f"  {len(keys)} object(s) to delete")
    if dry_run:
        for k in keys[:20]:
            print(f"  [dry-run] would delete: {k}")
        if len(keys) > 20:
            print(f"  ... and {len(keys) - 20} more")
        return

    batch_size = 1000
    deleted = 0
    for i in range(0, len(keys), batch_size):
        batch = keys[i : i + batch_size]
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        deleted += len(batch)
        print(f"  deleted {deleted}/{len(keys)}")

    print(f"  done: {deleted} objects deleted")


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download or upload files/folders to CSS (PDX storage).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download from CSS to local path")
    dl.add_argument("remote",
                    help="swift:// URL or bare path (e.g. recordings/v2d/...)")
    dl.add_argument("local", help="Local destination path")
    dl.add_argument("--dry-run", action="store_true",
                    help="Show what would be transferred without doing it")

    ls_p = sub.add_parser("ls", help="List contents of a remote directory")
    ls_p.add_argument("remote",
                      help="swift:// URL or bare path (e.g. recordings/v2d/...)")
    ls_p.add_argument("-r", "--recursive", action="store_true",
                      help="List all objects recursively instead of just immediate children")
    ls_p.add_argument("-n", "--limit", type=int, default=None,
                      help="Max number of entries to display")

    rm = sub.add_parser("delete", help="Delete a file or directory recursively from CSS")
    rm.add_argument("remote",
                    help="swift:// URL or bare path (e.g. recordings/v2d/...)")
    rm.add_argument("--dry-run", action="store_true",
                    help="Show what would be deleted without doing it")

    ul = sub.add_parser("upload", help="Upload from local path to CSS")
    ul.add_argument("local", help="Local file or directory to upload")
    ul.add_argument("remote",
                    help="swift:// URL or bare path (e.g. recordings/v2d/...)")
    ul.add_argument("--dry-run", action="store_true",
                    help="Show what would be transferred without doing it")

    args = parser.parse_args()
    client = _get_s3_client()
    bucket, prefix = _parse_swift_url(args.remote)

    display_url = f"swift://{_env_host()}/{_env_account()}/{bucket}/{prefix}"
    if args.command == "ls":
        print(f"Listing {display_url}")
        ls(client, bucket, prefix, recursive=args.recursive, limit=args.limit)
    elif args.command == "download":
        print(f"Downloading {display_url} -> {args.local}")
        download(client, bucket, prefix, Path(args.local), dry_run=args.dry_run)
    elif args.command == "delete":
        print(f"Deleting {display_url}")
        delete(client, bucket, prefix, dry_run=args.dry_run)
    elif args.command == "upload":
        print(f"Uploading {args.local} -> {display_url}")
        upload(client, bucket, prefix, Path(args.local), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
