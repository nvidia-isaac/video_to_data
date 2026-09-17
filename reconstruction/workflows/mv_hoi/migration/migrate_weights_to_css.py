"""Publish the runtime MV-HOI weights as an immutable CSS release.

The command is intentionally dry-run by default.  Use ``--apply`` only after
reviewing the generated manifest.  Payloads are uploaded and read back before
``manifest.json`` is written as the release commit record.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from typing import BinaryIO, Iterable

import boto3

try:
    from ..orchestration.runtime import require_submit_authority
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from orchestration.runtime import require_submit_authority


DEFAULT_RELEASE = "20260722"
DEFAULT_RELEASE_URL = (
    "swift://pdx.s8k.io/AUTH_team-isaac/recordings/"
    "v2d/multiview_weights/releases/20260722"
)
MANIFEST_SCHEMA = "v2d.mv_hoi_weights_release.v1"
READ_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class PayloadSpec:
    source_path: str
    path: str
    role: str


PAYLOAD_SPECS = (
    PayloadSpec(
        "face_detector/face_detection_yunet_2023mar.onnx",
        "face_detector/face_detection_yunet_2023mar.onnx",
        "YuNet face detector",
    ),
    PayloadSpec(
        "foundation_stereo/deployable_foundationstereo_small_576x960_v2.0.onnx",
        "foundation_stereo/deployable_foundationstereo_small_576x960_v2.0.onnx",
        "Foundation Stereo ONNX",
    ),
    PayloadSpec(
        "grounding_dino/groundingdino_swint_ogc.pth",
        "grounding_dino/groundingdino_swint_ogc.pth",
        "Grounding DINO checkpoint",
    ),
    PayloadSpec(
        "sam2/sam2.1_hiera_large.pt",
        "sam2/sam2.1_hiera_large.pt",
        "SAM2 large checkpoint",
    ),
    PayloadSpec(
        "sam2/sam2.1_hiera_l.yaml",
        "sam2/sam2.1_hiera_l.yaml",
        "SAM2 large configuration",
    ),
    PayloadSpec(
        "foundation_pose/nvidia_tensorrt/deployable_v1.0/refiner_net.onnx",
        "foundation_pose/nvidia_tensorrt/deployable_v1.0/refiner_net.onnx",
        "commercial TAO FoundationPose refiner ONNX",
    ),
    PayloadSpec(
        "foundation_pose/nvidia_tensorrt/deployable_v1.0/score_net.onnx",
        "foundation_pose/nvidia_tensorrt/deployable_v1.0/score_net.onnx",
        "commercial TAO FoundationPose scorer ONNX",
    ),
    PayloadSpec(
        "foundation_pose/nvidia_tensorrt/deployable_v1.0/manifest.json",
        "foundation_pose/nvidia_tensorrt/deployable_v1.0/manifest.json",
        "commercial TAO FoundationPose model manifest",
    ),
    PayloadSpec(
        "detectron2/cascade_mask_rcnn_vitdet_b/model_final_435fa9.pkl",
        "detectron2/cascade_mask_rcnn_vitdet_b/model_final_435fa9.pkl",
        "Detectron2 ViTDet-B checkpoint",
    ),
    PayloadSpec(
        "sam3d_body/sam-3d-body-dinov3/model.ckpt",
        "sam3d_body/sam-3d-body-dinov3/model.ckpt",
        "SAM3D-Body checkpoint",
    ),
    PayloadSpec(
        "sam3d_body/sam-3d-body-dinov3/model_config.yaml",
        "sam3d_body/sam-3d-body-dinov3/model_config.yaml",
        "SAM3D-Body model configuration",
    ),
    PayloadSpec(
        "sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt",
        "sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt",
        "MHR body model",
    ),
)

DINO_SOURCE = Path("sam3d_body/torch_home/hub/facebookresearch_dinov3_main")
DINO_ARCHIVE_PATH = "sam3d_body/dinov3_repo.tar.gz"


@dataclass(frozen=True)
class PreparedPayload:
    path: str
    size: int
    sha256: str
    source_path: str
    role: str
    local_path: Path

    def manifest_entry(self) -> dict:
        result = asdict(self)
        result.pop("local_path")
        return result


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(READ_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def _excluded_archive_path(relative: Path) -> bool:
    return (
        any(part in {".git", "__pycache__"} for part in relative.parts)
        or relative.suffix in {".pyc", ".pyo"}
    )


def create_deterministic_dinov3_archive(source_dir: Path, output_path: Path) -> None:
    """Create a byte-stable gzip-compressed tar archive for offline torch.hub."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 source directory not found: {source_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                paths = [Path(".")] + sorted(
                    (
                        path.relative_to(source_dir)
                        for path in source_dir.rglob("*")
                        if not _excluded_archive_path(path.relative_to(source_dir))
                    ),
                    key=lambda path: path.as_posix(),
                )
                for relative in paths:
                    source = source_dir if relative == Path(".") else source_dir / relative
                    archive_name = Path("facebookresearch_dinov3_main")
                    if relative != Path("."):
                        archive_name /= relative
                    info = tar.gettarinfo(str(source), arcname=archive_name.as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    if info.isfile():
                        with source.open("rb") as payload:
                            tar.addfile(info, payload)
                    else:
                        tar.addfile(info)


def prepare_release(
    source_root: Path,
    work_dir: Path,
    *,
    release: str = DEFAULT_RELEASE,
    payload_specs: Iterable[PayloadSpec] = PAYLOAD_SPECS,
) -> tuple[list[PreparedPayload], dict, bytes]:
    source_root = source_root.resolve()
    prepared: list[PreparedPayload] = []
    seen_release_paths: set[str] = set()
    for spec in payload_specs:
        if PurePosixPath(spec.path).is_absolute() or ".." in PurePosixPath(spec.path).parts:
            raise ValueError(f"Unsafe release path: {spec.path}")
        if spec.path in seen_release_paths:
            raise ValueError(f"Duplicate release path: {spec.path}")
        seen_release_paths.add(spec.path)
        local_path = source_root / spec.source_path
        if not local_path.is_file():
            raise FileNotFoundError(f"Required runtime weight not found: {local_path}")
        prepared.append(
            PreparedPayload(
                path=spec.path,
                size=local_path.stat().st_size,
                sha256=sha256_file(local_path),
                source_path=spec.source_path,
                role=spec.role,
                local_path=local_path,
            )
        )

    dino_archive = work_dir / "dinov3_repo.tar.gz"
    create_deterministic_dinov3_archive(source_root / DINO_SOURCE, dino_archive)
    prepared.append(
        PreparedPayload(
            path=DINO_ARCHIVE_PATH,
            size=dino_archive.stat().st_size,
            sha256=sha256_file(dino_archive),
            source_path=DINO_SOURCE.as_posix(),
            role="deterministic offline DINOv3 torch.hub source archive",
            local_path=dino_archive,
        )
    )
    prepared.sort(key=lambda payload: payload.path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "release": release,
        "complete": True,
        "files": [payload.manifest_entry() for payload in prepared],
        "file_count": len(prepared),
        "total_bytes": sum(payload.size for payload in prepared),
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return prepared, manifest, manifest_bytes


def parse_swift_url(url: str) -> tuple[str, str, str]:
    stripped = url.rstrip("/").removeprefix("swift://")
    parts = stripped.split("/", 3)
    if len(parts) != 4 or not parts[1].startswith("AUTH_"):
        raise ValueError(
            "Expected swift://host/AUTH_account/container/prefix release URL"
        )
    return f"https://{parts[0]}", parts[2], parts[3].strip("/")


def _is_missing_object(error: Exception) -> bool:
    response = getattr(error, "response", {})
    code = str(response.get("Error", {}).get("Code", ""))
    return code in {"404", "NoSuchKey", "NotFound"}


def read_remote_bytes(client, bucket: str, key: str) -> bytes | None:
    try:
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as error:
        if _is_missing_object(error):
            return None
        raise


def remote_sha256(client, bucket: str, key: str) -> tuple[int, str] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as error:
        if _is_missing_object(error):
            return None
        raise
    body = response["Body"]
    return int(response.get("ContentLength", 0)), sha256_stream(body)


def _remote_key(prefix: str, relative_path: str) -> str:
    return f"{prefix.rstrip('/')}/{relative_path.lstrip('/')}"


def publish_release(
    client,
    bucket: str,
    prefix: str,
    payloads: Iterable[PreparedPayload],
    manifest_bytes: bytes,
    *,
    apply: bool,
) -> dict:
    """Publish or verify an immutable release, committing its manifest last."""
    manifest_key = _remote_key(prefix, "manifest.json")
    existing_manifest = read_remote_bytes(client, bucket, manifest_key)
    if existing_manifest is not None and existing_manifest != manifest_bytes:
        raise RuntimeError(
            f"Refusing to modify committed release with conflicting {manifest_key}"
        )

    result = {"uploaded": [], "resumed": [], "verified": [], "dry_run": not apply}
    for payload in payloads:
        key = _remote_key(prefix, payload.path)
        existing = remote_sha256(client, bucket, key)
        if existing is not None:
            remote_size, remote_digest = existing
            if (remote_size, remote_digest) != (payload.size, payload.sha256):
                raise RuntimeError(
                    f"Conflicting immutable object {key}: "
                    f"remote=({remote_size}, {remote_digest}), "
                    f"expected=({payload.size}, {payload.sha256})"
                )
            result["resumed"].append(payload.path)
            continue
        if not apply:
            result["uploaded"].append(payload.path)
            continue
        client.upload_file(
            str(payload.local_path),
            bucket,
            key,
            ExtraArgs={"Metadata": {"sha256": payload.sha256}},
        )
        result["uploaded"].append(payload.path)

    if not apply:
        return result

    # A complete readback is deliberate: an ETag is not a logical-content hash.
    for payload in payloads:
        key = _remote_key(prefix, payload.path)
        verified = remote_sha256(client, bucket, key)
        if verified != (payload.size, payload.sha256):
            raise RuntimeError(f"Full checksum verification failed for {key}")
        result["verified"].append(payload.path)

    if existing_manifest is None:
        client.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=manifest_bytes,
            ContentType="application/json",
            Metadata={"sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        )
    committed = read_remote_bytes(client, bucket, manifest_key)
    if committed != manifest_bytes:
        raise RuntimeError("Release manifest commit readback did not match")
    result["manifest_key"] = manifest_key
    return result


def build_client(release_url: str):
    endpoint, bucket, prefix = parse_swift_url(release_url)
    access_key = os.environ.get("CSS_ACCESS_KEY")
    secret_key = os.environ.get("CSS_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(
            "CSS_ACCESS_KEY and CSS_SECRET_KEY are required; "
            "source ~/secrets/setup_css_env.sh"
        )
    return (
        boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ),
        bucket,
        prefix,
    )


def parse_args() -> argparse.Namespace:
    reconstruction_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=reconstruction_root / "data" / "weights",
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--release-url", default=DEFAULT_RELEASE_URL)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upload, verify, and commit the release (default: dry-run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.release_url.rstrip("/").endswith(f"/{args.release}"):
        raise SystemExit("--release-url must end with /<release>")
    with tempfile.TemporaryDirectory(prefix="mv_hoi_weights_") as temp_dir:
        payloads, manifest, manifest_bytes = prepare_release(
            args.source_root, Path(temp_dir), release=args.release
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        if not args.apply:
            print("DRY RUN: no CSS objects will be written")
            return
        require_submit_authority("publish a CSS weight release")
        client, bucket, prefix = build_client(args.release_url)
        result = publish_release(
            client, bucket, prefix, payloads, manifest_bytes, apply=True
        )
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
