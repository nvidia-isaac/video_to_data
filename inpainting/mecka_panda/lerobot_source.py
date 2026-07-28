"""Read LeRobot v3 datasets from a local path or an S3-compatible bucket.

Only the parts one episode needs are transferred. A v3 shard packs hundreds of
episodes into one multi-hundred-megabyte parquet file and one multi-gigabyte
mp4, so the readers here resolve an episode to its parquet row groups and to a
video time range instead of downloading whole files.

No LeRobot dependency: the on-disk layout is read directly through the path
templates declared in `meta/info.json`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import pyarrow.fs as pafs
import pyarrow.parquet as pq

DEFAULT_ENDPOINT = "https://storage.googleapis.com"
DEFAULT_CREDENTIALS = "credentials/gcp_training.secret"
VIDEO_KEY = "observation.images.main"
SUPPORTED_CODEBASE_VERSIONS = ("v3.0",)


class LeRobotSourceError(RuntimeError):
    """Raised when a dataset does not match the expected LeRobot v3 layout."""


@dataclass(frozen=True)
class Credentials:
    """HMAC credentials for an S3-compatible endpoint."""

    access_key: str
    secret_key: str
    endpoint: str
    region: str

    @property
    def host(self) -> str:
        return urlparse(self.endpoint).netloc or self.endpoint


@dataclass(frozen=True)
class EpisodeRecord:
    """One row of `meta/episodes/**.parquet`, resolved for reading."""

    episode_index: int
    length: int
    duration_s: float
    task_id: str
    task_description: str
    data_chunk_index: int
    data_file_index: int
    row_start: int
    row_stop: int
    video_chunk_index: int
    video_file_index: int
    video_from_s: float
    video_to_s: float
    camera_intrinsics: tuple[float, ...]


def load_credentials(path: str | Path = DEFAULT_CREDENTIALS) -> Credentials:
    """Load credentials from `COSMOS_GCP_CHECKPOINT_CREDS` or a JSON file."""
    encoded = os.environ.get("COSMOS_GCP_CHECKPOINT_CREDS")
    if encoded:
        import base64

        payload = json.loads(base64.b64decode(encoded))
    else:
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise LeRobotSourceError(
                f"No credentials at {resolved}. See credentials/README.md for the "
                f"expected JSON fields, or set COSMOS_GCP_CHECKPOINT_CREDS."
            )
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    missing = {"aws_access_key_id", "aws_secret_access_key"} - payload.keys()
    if missing:
        raise LeRobotSourceError(f"Credentials are missing {sorted(missing)}")
    return Credentials(
        access_key=payload["aws_access_key_id"],
        secret_key=payload["aws_secret_access_key"],
        endpoint=payload.get("endpoint_url", DEFAULT_ENDPOINT),
        region=payload.get("region_name") or "auto",
    )


@dataclass(frozen=True)
class Source:
    """A dataset root, either local or on an S3-compatible endpoint."""

    filesystem: pafs.FileSystem
    root: str
    bucket: str | None
    credentials: Credentials | None

    def path(self, *parts: str) -> str:
        return "/".join((self.root, *parts))

    @property
    def is_remote(self) -> bool:
        return self.bucket is not None


def open_source(
    uri: str | Path, *, credentials: str | Path | Credentials | None = None
) -> Source:
    """Open a `s3://bucket/prefix` URI or a local directory for reading."""
    text = str(uri)
    if not text.startswith("s3://"):
        local = Path(text).expanduser().resolve()
        if not local.is_dir():
            raise LeRobotSourceError(f"Not a directory: {local}")
        return Source(pafs.LocalFileSystem(), str(local), None, None)

    parsed = urlparse(text)
    bucket = parsed.netloc
    prefix = parsed.path.strip("/")
    if not bucket:
        raise LeRobotSourceError(f"Missing bucket in {text!r}")

    if isinstance(credentials, Credentials):
        resolved = credentials
    else:
        resolved = load_credentials(credentials or DEFAULT_CREDENTIALS)

    # GCS omits the response checksum headers that recent AWS SDKs validate by
    # default, which fails every GetObject with a checksum mismatch.
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")

    filesystem = pafs.S3FileSystem(
        access_key=resolved.access_key,
        secret_key=resolved.secret_key,
        endpoint_override=resolved.host,
        region=resolved.region,
        scheme="https",
    )
    root = f"{bucket}/{prefix}" if prefix else bucket
    return Source(filesystem, root, bucket, resolved)


def load_info(source: Source) -> dict[str, Any]:
    """Read and version-check `meta/info.json`."""
    path = source.path("meta", "info.json")
    with source.filesystem.open_input_stream(path) as stream:
        info = json.loads(stream.readall().decode("utf-8"))
    version = info.get("codebase_version")
    if version not in SUPPORTED_CODEBASE_VERSIONS:
        raise LeRobotSourceError(
            f"Unsupported codebase_version {version!r} at {path}; "
            f"expected one of {SUPPORTED_CODEBASE_VERSIONS}"
        )
    return info


def _format_path(template: str, **values: Any) -> str:
    return template.format(**values)


def _episode_meta_paths(source: Source, info: dict[str, Any]) -> list[str]:
    template = info["episodes_path"]
    root = source.path(_format_path(template, chunk_index=0, file_index=0))
    # Strip the chunk and file components to get the metadata root itself, so
    # the listing does not depend on how many chunks the shard happens to have.
    prefix = root.rsplit("/", 2)[0]
    selector = pafs.FileSelector(prefix, recursive=True, allow_not_found=True)
    paths = [
        entry.path
        for entry in source.filesystem.get_file_info(selector)
        if entry.type == pafs.FileType.File and entry.path.endswith(".parquet")
    ]
    if not paths:
        raise LeRobotSourceError(f"No episode metadata under {prefix}")
    return sorted(paths)


def _record_from_row(row: dict[str, Any]) -> EpisodeRecord:
    video_prefix = f"videos/{VIDEO_KEY}/"
    intrinsics = tuple(float(value) for value in row["camera_intrinsics"])
    if len(intrinsics) != 8:
        raise LeRobotSourceError(
            f"camera_intrinsics must hold fx,fy,cx,cy,k1,k2,p1,p2; got {len(intrinsics)}"
        )
    return EpisodeRecord(
        episode_index=int(row["episode_index"]),
        length=int(row["length"]),
        duration_s=float(row["duration"]),
        task_id=str(row.get("task_id") or ""),
        task_description=str(row.get("task_description") or ""),
        data_chunk_index=int(row["data/chunk_index"]),
        data_file_index=int(row["data/file_index"]),
        row_start=int(row["dataset_from_index"]),
        row_stop=int(row["dataset_to_index"]),
        video_chunk_index=int(row[f"{video_prefix}chunk_index"]),
        video_file_index=int(row[f"{video_prefix}file_index"]),
        video_from_s=float(row[f"{video_prefix}from_timestamp"]),
        video_to_s=float(row[f"{video_prefix}to_timestamp"]),
        camera_intrinsics=intrinsics,
    )


def find_episode(
    source: Source, info: dict[str, Any], episode: int | None = None
) -> EpisodeRecord:
    """Resolve one episode's metadata, defaulting to the lowest index."""
    wanted = None if episode is None else int(episode)
    for path in _episode_meta_paths(source, info):
        with source.filesystem.open_input_file(path) as stream:
            table = pq.read_table(stream)
        for row in table.to_pylist():
            if wanted is None or int(row["episode_index"]) == wanted:
                return _record_from_row(row)
    raise LeRobotSourceError(f"Episode {episode!r} not present under {source.root}")


def read_episode_table(
    source: Source,
    info: dict[str, Any],
    record: EpisodeRecord,
    *,
    columns: list[str] | None = None,
) -> Any:
    """Read one episode's frames, touching only the row groups it spans."""
    path = source.path(
        _format_path(
            info["data_path"],
            chunk_index=record.data_chunk_index,
            file_index=record.data_file_index,
        )
    )
    with source.filesystem.open_input_file(path) as stream:
        parquet = pq.ParquetFile(stream)
        if columns is not None:
            # pyarrow silently drops unknown column names, which would turn a
            # renamed upstream column into missing data instead of an error.
            available = set(parquet.schema_arrow.names)
            absent = [name for name in columns if name not in available]
            if absent:
                raise LeRobotSourceError(f"Columns {absent} are not in {path}")
        metadata = parquet.metadata
        wanted: list[int] = []
        offset = 0
        for index in range(metadata.num_row_groups):
            rows = metadata.row_group(index).num_rows
            if offset < record.row_stop and offset + rows > record.row_start:
                wanted.append(index)
            offset += rows
        if not wanted:
            raise LeRobotSourceError(
                f"Rows [{record.row_start}, {record.row_stop}) fall outside {path}"
            )
        table = parquet.read_row_groups(wanted, columns=columns)
        group_start = sum(
            metadata.row_group(index).num_rows for index in range(wanted[0])
        )
    local_start = record.row_start - group_start
    return table.slice(local_start, record.row_stop - record.row_start)


def presign_url(
    credentials: Credentials, bucket: str, key: str, *, expires_seconds: int = 3600
) -> str:
    """Build an AWS SigV4 presigned GET URL, which GCS also accepts."""
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    scope = f"{date_stamp}/{credentials.region}/s3/aws4_request"
    canonical_uri = "/" + quote(f"{bucket}/{key}", safe="/~")
    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{credentials.access_key}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires_seconds),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_query = "&".join(
        f"{quote(key_, safe='-_.~')}={quote(value, safe='-_.~')}"
        for key_, value in sorted(query.items())
    )
    canonical_request = "\n".join(
        [
            "GET",
            canonical_uri,
            canonical_query,
            f"host:{credentials.host}\n",
            "host",
            "UNSIGNED-PAYLOAD",
        ]
    )
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )

    def sign(key_bytes: bytes, message: str) -> bytes:
        return hmac.new(key_bytes, message.encode("utf-8"), hashlib.sha256).digest()

    signing_key = sign(
        sign(
            sign(sign(f"AWS4{credentials.secret_key}".encode(), date_stamp), credentials.region),
            "s3",
        ),
        "aws4_request",
    )
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return (
        f"https://{credentials.host}{canonical_uri}"
        f"?{canonical_query}&X-Amz-Signature={signature}"
    )


def extract_episode_video(
    source: Source,
    info: dict[str, Any],
    record: EpisodeRecord,
    destination: str | Path,
    *,
    overwrite: bool = False,
    expires_seconds: int = 3600,
) -> Path:
    """Cut one episode out of its shared mp4 without downloading the whole file."""
    output = Path(destination).expanduser().resolve()
    if output.exists() and not overwrite:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    relative = _format_path(
        info["video_path"],
        video_key=VIDEO_KEY,
        chunk_index=record.video_chunk_index,
        file_index=record.video_file_index,
    )
    if source.is_remote:
        assert source.credentials is not None
        key = source.path(relative).split("/", 1)[1]
        source_url = presign_url(
            source.credentials,
            source.bucket or "",
            key,
            expires_seconds=expires_seconds,
        )
    else:
        source_url = str(Path(source.root) / relative)

    partial = output.with_suffix(output.suffix + ".partial")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{record.video_from_s:.6f}",
        "-to",
        f"{record.video_to_s:.6f}",
        "-i",
        source_url,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        # The temporary name ends in .partial, so the muxer cannot be inferred.
        "-f",
        "mp4",
        str(partial),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise LeRobotSourceError(
                f"ffmpeg failed to cut episode {record.episode_index}: "
                f"{completed.stderr.strip()[-2000:]}"
            )
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    return output
