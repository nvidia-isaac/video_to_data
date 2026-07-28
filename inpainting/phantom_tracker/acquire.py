"""Network-enabled acquisition for the Phantom tracking condition.

Inference never calls this module.  It downloads the official HaMeR demo
bundle, verifies its known SHA-256, and extracts only the files used by the
tracker.  Licensed MANO files are intentionally neither copied nor packaged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from .assets import HAMER_REQUIRED_SHA256
from .provenance import sha256_file, sha256_tree


HAMER_URL = "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
HAMER_TAR_SHA256 = "ccfb70abd672b64c3ea90891c808d4499cc36a37dd6cf86c561a665113aef11e"
HAMER_MEMBERS = tuple(HAMER_REQUIRED_SHA256)
HAMER_MEMBER_SHA256 = HAMER_REQUIRED_SHA256


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def _safe_extract_selected(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        missing = sorted(set(HAMER_MEMBERS) - set(members))
        if missing:
            raise RuntimeError(f"Official HaMeR archive is missing members: {missing}")
        for relative in HAMER_MEMBERS:
            member = members[relative]
            if not member.isfile():
                raise RuntimeError(f"Expected regular file in HaMeR archive: {relative}")
            target = destination / relative
            resolved = target.resolve()
            if destination.resolve() not in resolved.parents:
                raise RuntimeError(f"Unsafe HaMeR archive path: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read HaMeR archive member: {relative}")
            temporary = target.with_name(target.name + ".partial")
            with source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            os.replace(temporary, target)


def acquire_hamer(download_dir: Path, models_dir: Path) -> dict:
    archive = download_dir / "hamer_demo_data.tar.gz"
    if not archive.is_file() or sha256_file(archive) != HAMER_TAR_SHA256:
        _download(HAMER_URL, archive)
    actual_hash = sha256_file(archive)
    if actual_hash != HAMER_TAR_SHA256:
        raise RuntimeError(
            f"HaMeR archive SHA-256 mismatch: expected {HAMER_TAR_SHA256}, got {actual_hash}"
        )
    hamer_dir = models_dir / "hamer"
    expected = {member: hamer_dir / member for member in HAMER_MEMBERS}
    if not all(
        path.is_file() and sha256_file(path) == HAMER_MEMBER_SHA256[member]
        for member, path in expected.items()
    ):
        _safe_extract_selected(archive, hamer_dir)
    for member, path in expected.items():
        actual = sha256_file(path)
        if actual != HAMER_MEMBER_SHA256[member]:
            raise RuntimeError(
                f"Extracted HaMeR member SHA-256 mismatch for {member}: {actual}"
            )
    tree_hash, entries = sha256_tree(hamer_dir)
    return {
        "url": HAMER_URL,
        "archive_sha256": actual_hash,
        "selected_tree_sha256": tree_hash,
        "selected_files": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    record = acquire_hamer(args.download_dir.resolve(), args.models_dir.resolve())
    manifest = args.manifest or args.models_dir / "hamer_acquisition.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(manifest.name + ".partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
