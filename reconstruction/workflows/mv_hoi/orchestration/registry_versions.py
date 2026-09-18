"""Resolve MV HOI pipeline versions from the configured container registry."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

try:
    from .db import parse_semver, validate_semver_gt
except ImportError:  # Direct script execution.
    from db import parse_semver, validate_semver_gt


CANONICAL_REPOSITORY = "mv_hoi_mv_preprocess"

# Local image name, registry repository name. Keep release membership centralized here.
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
    """Raised when a pipeline release cannot be resolved from the configured registry."""


def managed_repositories() -> tuple[str, ...]:
    return tuple(repository for _, repository in MANAGED_IMAGES)


def image_registry(configured: str | None = None) -> str:
    """Resolve an explicit registry/namespace without a deployment default."""
    value = (configured or os.environ.get("V2D_IMAGE_REGISTRY", "")).strip().rstrip("/")
    if not value or value == "???":
        raise RegistryVersionError(
            "Set V2D_IMAGE_REGISTRY or dataset image_registry to your registry/namespace"
        )
    if "://" in value or any(char.isspace() for char in value):
        raise RegistryVersionError("image_registry must be a registry/namespace without a URL scheme")
    return value


def docker_repository(repository: str, registry: str | None = None) -> str:
    return f"{image_registry(registry)}/{repository}"


def _extract_semver_tags(payload: object) -> list[str]:
    """Collect strict semver tags from a registry JSON response."""
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


def list_repository_versions(repository: str, *, registry: str | None = None) -> list[str]:
    """Return sorted semver tags using the configured registry client."""
    registry = image_registry(registry)
    if registry.startswith("nvcr.io/"):
        image_pattern = f"{registry.removeprefix('nvcr.io/')}/{repository}:*"
        cmd = ["ngc", "registry", "image", "list", "--format_type", "json", image_pattern]
        client_name = "NGC"
    else:
        image_pattern = f"{registry}/{repository}"
        cmd = ["skopeo", "list-tags", f"docker://{image_pattern}"]
        client_name = "skopeo"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RegistryVersionError(
            f"{client_name} CLI not found. Install and authenticate "
            f"{cmd[0]} for registry {registry}."
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "no error details"
        auth_hint = ""
        if any(
            marker in detail.lower()
            for marker in ("auth", "credential", "forbidden", "unauthorized")
        ):
            auth_hint = " Check the local registry authentication configuration."
        # New namespaces have no canonical repository until the first push.
        # Only a registry NAME_UNKNOWN response is an empty tag set; a bare
        # HTTP 404, authorization failure or transport error is not evidence.
        if client_name == "skopeo" and not auth_hint and re.search(
            r'"code"\s*:\s*"NAME_UNKNOWN"'
            r'|\bname unknown:\s*repository (?:not found|name not known to registry)\b',
            detail, re.IGNORECASE,
        ):
            return []
        raise RegistryVersionError(
            f"{client_name} query failed for {image_pattern} (exit {result.returncode}): "
            f"{detail}.{auth_hint}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RegistryVersionError(
            f"{client_name} returned malformed JSON for {image_pattern}: {exc}"
        ) from exc
    return _extract_semver_tags(payload)


def latest_registry_version(*, registry: str | None = None) -> str | None:
    versions = list_repository_versions(CANONICAL_REPOSITORY, registry=registry)
    return versions[-1] if versions else None


def missing_repositories_for_version(
    version: str,
    repositories: Iterable[str] | None = None,
    known_versions: dict[str, Iterable[str]] | None = None,
    *, registry: str | None = None,
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
        fetch_versions = partial(list_repository_versions, registry=image_registry(registry))
        with ThreadPoolExecutor(max_workers=min(5, len(uncached))) as executor:
            fetched_versions = dict(
                zip(uncached, executor.map(fetch_versions, uncached))
            )

    missing: list[str] = []
    for repository in repositories:
        versions = known_versions.get(repository, fetched_versions.get(repository, ()))
        if version not in versions:
            missing.append(repository)
    return missing


def validate_release(version: str, *, registry: str | None = None) -> str:
    """Validate that a semver release exists on every managed image."""
    try:
        missing = missing_repositories_for_version(version, registry=registry)
    except ValueError as exc:
        raise RegistryVersionError(str(exc)) from exc
    if missing:
        repositories = ", ".join(missing)
        raise RegistryVersionError(
            f"Pipeline version {version} is incomplete in the configured registry; missing from: "
            f"{repositories}"
        )
    return version


def resolve_submission_version(
    requested_version: str | None = None, *, registry: str | None = None,
) -> str:
    """Resolve and validate the immutable image tag for one submit invocation."""
    registry = image_registry(registry)
    if requested_version:
        return validate_release(requested_version, registry=registry)

    canonical_versions = list_repository_versions(CANONICAL_REPOSITORY, registry=registry)
    if not canonical_versions:
        raise RegistryVersionError(
            f"No semver tags found for {docker_repository(CANONICAL_REPOSITORY, registry)}. "
            "Publish the pipeline images with push_images.sh first."
        )
    version = canonical_versions[-1]
    missing = missing_repositories_for_version(
        version,
        known_versions={CANONICAL_REPOSITORY: canonical_versions},
        registry=registry,
    )
    if missing:
        raise RegistryVersionError(
            f"Latest pipeline version {version} is incomplete in the configured registry; missing from: "
            f"{', '.join(missing)}"
        )
    return version


def resolve_push_version(
    explicit_version: str | None = None, *, registry: str | None = None,
) -> tuple[str | None, str]:
    """Return (latest remote version, version to publish)."""
    if explicit_version:
        try:
            parse_semver(explicit_version)
        except ValueError as exc:
            raise RegistryVersionError(str(exc)) from exc
    latest = latest_registry_version(registry=registry)
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
    parser.add_argument("--image-registry", help="Registry/namespace (or V2D_IMAGE_REGISTRY)")
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
    try:
        registry = image_registry(args.image_registry)
    except RegistryVersionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.command == "managed-images":
        for local_image, repository in MANAGED_IMAGES:
            print(local_image, docker_repository(repository, registry))
        return

    try:
        latest, version = resolve_push_version(args.version, registry=registry)
    except RegistryVersionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Latest registry version: {latest or 'none'}", file=sys.stderr)
    print(version)


if __name__ == "__main__":
    main()
