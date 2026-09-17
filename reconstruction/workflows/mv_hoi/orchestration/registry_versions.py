"""Resolve MV HOI pipeline versions from the private NGC registry."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

try:
    from .db import parse_semver, validate_semver_gt
except ImportError:  # Direct script execution.
    from db import parse_semver, validate_semver_gt


NGC_NAMESPACE = "nvstaging/isaac-amr"
DOCKER_REGISTRY = "nvcr.io"
CANONICAL_REPOSITORY = "mv_hoi_mv_preprocess"

# Local image name, NGC repository name. Keep release membership centralized here.
MANAGED_IMAGES: tuple[tuple[str, str], ...] = (
    ("v2d_rosbag", "mv_hoi_rosbag"),
    ("v2d_mv_calibration", "mv_hoi_mv_calibration"),
    ("v2d_mv_preprocess", "mv_hoi_mv_preprocess"),
    ("v2d_face_detector", "mv_hoi_face_detector"),
    ("v2d_foundation_stereo", "mv_hoi_foundation_stereo"),
    ("v2d_grounding_dino", "mv_hoi_grounding_dino"),
    ("v2d_sam2", "mv_hoi_sam2"),
    ("v2d_foundation_pose", "mv_hoi_foundation_pose"),
    ("v2d_detectron2", "mv_hoi_detectron2"),
    ("v2d_sam3d_body", "mv_hoi_sam3d_body"),
    ("v2d_mv_postprocess", "mv_hoi_mv_postprocess"),
)

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class RegistryVersionError(RuntimeError):
    """Raised when a pipeline release cannot be resolved from NGC."""


def managed_repositories() -> tuple[str, ...]:
    return tuple(repository for _, repository in MANAGED_IMAGES)


def docker_repository(repository: str) -> str:
    return f"{DOCKER_REGISTRY}/{NGC_NAMESPACE}/{repository}"


def _extract_semver_tags(payload: object) -> list[str]:
    """Collect strict semver tags from an NGC JSON response."""
    tags: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, list):
            for child in value:
                visit(child)
            return
        if not isinstance(value, str):
            return

        candidates = (value, value.rsplit(":", 1)[-1])
        for candidate in candidates:
            if _SEMVER_RE.fullmatch(candidate):
                tags.add(candidate)

    visit(payload)
    return sorted(tags, key=parse_semver)


def list_repository_versions(repository: str) -> list[str]:
    """Return sorted semver tags for one repository using the authenticated NGC CLI."""
    image_pattern = f"{NGC_NAMESPACE}/{repository}:*"
    cmd = [
        "ngc",
        "registry",
        "image",
        "list",
        "--format_type",
        "json",
        image_pattern,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RegistryVersionError(
            "NGC CLI not found. Install `ngc`, add it to PATH, and authenticate it."
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "no error details"
        auth_hint = ""
        if any(
            marker in detail.lower()
            for marker in ("auth", "credential", "forbidden", "unauthorized")
        ):
            auth_hint = " Check the local NGC authentication configuration."
        raise RegistryVersionError(
            f"NGC query failed for {image_pattern} (exit {result.returncode}): "
            f"{detail}.{auth_hint}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RegistryVersionError(
            f"NGC returned malformed JSON for {image_pattern}: {exc}"
        ) from exc
    return _extract_semver_tags(payload)


def latest_registry_version() -> str | None:
    versions = list_repository_versions(CANONICAL_REPOSITORY)
    return versions[-1] if versions else None


def missing_repositories_for_version(
    version: str,
    repositories: Iterable[str] | None = None,
    known_versions: dict[str, Iterable[str]] | None = None,
) -> list[str]:
    """Return repositories that do not contain *version*."""
    parse_semver(version)
    known_versions = known_versions or {}
    repositories = tuple(repositories or managed_repositories())
    uncached = [
        repository for repository in repositories if repository not in known_versions
    ]
    fetched_versions: dict[str, Iterable[str]] = {}
    if uncached:
        with ThreadPoolExecutor(max_workers=min(5, len(uncached))) as executor:
            fetched_versions = dict(
                zip(uncached, executor.map(list_repository_versions, uncached))
            )

    missing: list[str] = []
    for repository in repositories:
        versions = known_versions.get(repository, fetched_versions.get(repository, ()))
        if version not in versions:
            missing.append(repository)
    return missing


def validate_release(version: str) -> str:
    """Validate that a semver release exists on every managed image."""
    try:
        missing = missing_repositories_for_version(version)
    except ValueError as exc:
        raise RegistryVersionError(str(exc)) from exc
    if missing:
        repositories = ", ".join(missing)
        raise RegistryVersionError(
            f"Pipeline version {version} is incomplete in NGC; missing from: "
            f"{repositories}"
        )
    return version


def resolve_submission_version(requested_version: str | None = None) -> str:
    """Resolve and validate the immutable image tag for one submit invocation."""
    if requested_version:
        return validate_release(requested_version)

    canonical_versions = list_repository_versions(CANONICAL_REPOSITORY)
    if not canonical_versions:
        raise RegistryVersionError(
            f"No semver tags found for {NGC_NAMESPACE}/{CANONICAL_REPOSITORY}. "
            "Publish the pipeline images with push_images.sh first."
        )
    version = canonical_versions[-1]
    missing = missing_repositories_for_version(
        version,
        known_versions={CANONICAL_REPOSITORY: canonical_versions},
    )
    if missing:
        raise RegistryVersionError(
            f"Latest pipeline version {version} is incomplete in NGC; missing from: "
            f"{', '.join(missing)}"
        )
    return version


def resolve_push_version(explicit_version: str | None = None) -> tuple[str | None, str]:
    """Return (latest remote version, version to publish)."""
    if explicit_version:
        try:
            parse_semver(explicit_version)
        except ValueError as exc:
            raise RegistryVersionError(str(exc)) from exc
    latest = latest_registry_version()
    if explicit_version:
        try:
            validate_semver_gt(explicit_version, latest)
        except ValueError as exc:
            raise RegistryVersionError(str(exc)) from exc
        return latest, explicit_version
    if latest is None:
        return None, "0.1.0"
    major, minor, patch = parse_semver(latest)
    return latest, f"{major}.{minor}.{patch + 1}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "managed-images", help="Print local and fully-qualified image repositories",
    )
    push_parser = subparsers.add_parser(
        "resolve-push", help="Resolve the next version to publish",
    )
    push_parser.add_argument("--version", help="Explicit semver to publish")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "managed-images":
        for local_image, repository in MANAGED_IMAGES:
            print(local_image, docker_repository(repository))
        return

    try:
        latest, version = resolve_push_version(args.version)
    except RegistryVersionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Latest NGC version: {latest or 'none'}", file=sys.stderr)
    print(version)


if __name__ == "__main__":
    main()
