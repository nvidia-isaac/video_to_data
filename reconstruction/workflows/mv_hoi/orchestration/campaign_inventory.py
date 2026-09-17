"""Build frozen legacy-revalidation and backlog campaign inventories."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Callable, Iterable

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

try:
    from . import db
    from .config_utils import (
        RECON_PIPELINE, REVALIDATION_PIPELINE,
        get_pipeline_input_path, get_legacy_export_path, load_config,
    )
    from .runtime import MV_HOI_DIR, state_path
except ImportError:
    import db
    from config_utils import (
        RECON_PIPELINE, REVALIDATION_PIPELINE,
        get_pipeline_input_path, get_legacy_export_path, load_config,
    )
    from runtime import MV_HOI_DIR, state_path


SCHEMA = "v2d.mv_hoi.campaign_inventory.v1"
CAMERAS_LEFT = (
    "back_stereo_camera_left", "front_stereo_camera_left",
    "left_stereo_camera_left", "right_stereo_camera_left",
)
CAMERAS_RIGHT = tuple(name.replace("_left", "_right") for name in CAMERAS_LEFT)
CAMERAS_RGB = CAMERAS_LEFT + CAMERAS_RIGHT
RGB_H5_FILES = frozenset(f"{camera}.h5" for camera in CAMERAS_RGB)
RGB_VIDEO_FILES = frozenset(f"{camera}.mp4" for camera in CAMERAS_RGB)
LEGACY_SINGLE_FILES = {
    "edex", "hoi_metadata.yaml", "poses.npy", "mhr_params_mv.pt",
    "soma_params.npz", "ground_plane.json", "tiled_hoi_overlay.mp4",
}
HASHED_SOURCE_SUFFIXES = (
    "/foundation_pose/poses.npy",
    "/foundation_pose/pose_valid_mask.npy",
    "/mv_preprocess/hoi_metadata.yaml",
    "/mv_preprocess/object_mesh/output_aligned.glb",
    "/mv_preprocess/object_mesh/output_symmetry.json",
    "/eval_chamfer_human/chamfer_metrics.json",
    "/check_object_mask/check_object_mask.json",
    "/poses.npy",
    "/pose_valid_mask.npy",
    "/hoi_metadata.yaml",
    "/object_mesh/output_aligned.glb",
    "/object_mesh/output_symmetry.json",
    "/mv_preprocess/object_bbox_source.txt",
)

# A reconstruction sequence can contain millions of diagnostic or per-frame
# objects which the abbreviated workflow never reads.  Freeze only the
# canonical artifacts consumed by revalidation and final export.  This keeps
# the inventory bounded while the controller still validates every selected
# object's size and ETag immediately before use and publication.
DATA_OUTPUT_SINGLE_FILES = {
    "intermediate_cleanup.json",
    "mv_preprocess/edex",
    "mv_preprocess/hoi_metadata.yaml",
    "mv_preprocess/object_bbox_source.txt",
    "foundation_pose/poses.npy",
    "foundation_pose/pose_valid_mask.npy",
    "sam3d_body/mhr_params_mv.pt",
    "sam3d_body/mhr_mesh_mv.pt",
    "export_soma/soma_params.npz",
    "estimate_ground_plane/ground_plane.json",
    "eval_chamfer_human/chamfer_metrics.json",
    "check_object_mask/check_object_mask.json",
    "check_accuracy/check_accuracy.json",
}
DATA_OUTPUT_CAMERA_FILES = {
    *(f"mv_preprocess/images/{camera}.h5" for camera in CAMERAS_RGB),
    *(f"mv_preprocess/videos/{camera}.mp4" for camera in CAMERAS_RGB),
    *(f"foundation_stereo/{camera}/depth.h5" for camera in CAMERAS_LEFT),
    *(f"sam2_object_masks/{camera}/0.h5" for camera in CAMERAS_LEFT),
    *(f"sam2_human_masks/{camera}/0.h5" for camera in CAMERAS_LEFT),
}
DATA_OUTPUT_EXACT_FILES = frozenset(DATA_OUTPUT_SINGLE_FILES | DATA_OUTPUT_CAMERA_FILES)


def _is_selected_data_output_path(path: str) -> bool:
    """Return whether *path* is part of the frozen revalidation input set."""
    if path in DATA_OUTPUT_SINGLE_FILES:
        return True
    parts = path.split("/")
    if len(parts) >= 3 and parts[:2] == ["mv_preprocess", "object_mesh"]:
        return True
    if len(parts) >= 3 and parts[:2] == ["mv_preprocess", "labeled_bboxes"]:
        return True
    if len(parts) == 3 and parts[:2] == ["mv_preprocess", "images"]:
        name = parts[2]
        return (
            name in RGB_H5_FILES
            or any(
                name.startswith(f"{camera}.ffv1.") and name.endswith(".mkv")
                for camera in CAMERAS_RGB
            )
        )
    if len(parts) == 4 and parts[:2] == ["mv_preprocess", "images"]:
        camera, name = parts[2:]
        return camera in CAMERAS_RGB and Path(name).suffix.lower() in {
            ".png", ".jpg", ".jpeg",
        }
    if len(parts) == 3 and parts[:2] == ["mv_preprocess", "videos"]:
        return parts[2] in RGB_VIDEO_FILES
    if len(parts) == 3 and parts[0] == "foundation_stereo":
        camera, name = parts[1:]
        return camera in CAMERAS_LEFT and (
            name == "depth.h5"
            or (name.startswith("depth.ffv1.") and name.endswith(".mkv"))
        )
    if len(parts) == 4 and parts[0] == "foundation_stereo":
        camera, directory, name = parts[1:]
        return (
            camera in CAMERAS_LEFT
            and directory == "depth"
            and Path(name).suffix.lower() in {".png", ".tif", ".tiff"}
        )
    if len(parts) == 3 and parts[0] in {"sam2_object_masks", "sam2_human_masks"}:
        camera, name = parts[1:]
        return camera in CAMERAS_LEFT and (
            name.endswith(".h5")
            or (".ffv1." in name and name.endswith(".mkv"))
        )
    if len(parts) == 4 and parts[0] in {
        "sam2_object_masks", "sam2_human_masks",
    }:
        camera, object_id, name = parts[1:]
        return (
            camera in CAMERAS_LEFT
            and object_id == "0"
            and Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    return False


def _parse_swift_url(url: str) -> tuple[str, str, str]:
    stripped = url.rstrip("/").removeprefix("swift://")
    parts = stripped.split("/", 3)
    if len(parts) < 3:
        raise ValueError("Swift URL must be swift://host/account/container/prefix")
    return f"https://{parts[0]}", parts[2], parts[3] if len(parts) == 4 else ""


def _client(swift_base: str, max_pool_connections: int = 64):
    endpoint, bucket, prefix = _parse_swift_url(swift_base)
    access = os.environ.get("CSS_ACCESS_KEY")
    secret = os.environ.get("CSS_SECRET_KEY")
    if not access or not secret:
        raise RuntimeError("Set CSS_ACCESS_KEY and CSS_SECRET_KEY")
    return boto3.client(
        "s3", endpoint_url=endpoint, aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=Config(
            max_pool_connections=max(10, max_pool_connections),
            retries={"mode": "adaptive", "max_attempts": 6},
        ),
    ), bucket, prefix.strip("/")


def _should_hash(key: str) -> bool:
    return (
        "/mv_preprocess/labeled_bboxes/" in key
        or key.endswith(HASHED_SOURCE_SUFFIXES)
    )


def _finalize_objects(
    client, bucket: str, objects: list[dict], *, workers: int = 64,
    content_hashes: bool = False,
) -> list[dict]:
    marker_indexes = {
        index for index, item in enumerate(objects)
        if item["key"].endswith("/mv_preprocess/object_bbox_source.txt")
    }
    hash_indexes = sorted(marker_indexes | (
        {
            index for index, item in enumerate(objects)
            if _should_hash(item["key"])
        }
        if content_hashes else set()
    ))

    def hash_object(index: int) -> tuple[int, str, str | None]:
        body = client.get_object(
            Bucket=bucket, Key=objects[index]["key"],
        )["Body"].read()
        content = (
            body.decode("utf-8").strip() if index in marker_indexes else None
        )
        return index, hashlib.sha256(body).hexdigest(), content

    if hash_indexes:
        with ThreadPoolExecutor(max_workers=min(workers, len(hash_indexes))) as executor:
            futures = [executor.submit(hash_object, index) for index in hash_indexes]
            for completed, future in enumerate(as_completed(futures), 1):
                index, digest, content = future.result()
                objects[index]["sha256"] = digest
                if content is not None:
                    objects[index]["content"] = content
                if completed % 1000 == 0:
                    print(
                        f"Hashed {completed}/{len(hash_indexes)} frozen object(s)",
                        file=sys.stderr,
                    )

    for record in objects:
        record["object_identity_sha256"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return objects


def _list(
    client, bucket: str, prefix: str, *, workers: int = 64,
    include_key: Callable[[str], bool] | None = None,
    finalize: bool = True, content_hashes: bool = False,
) -> list[dict]:
    objects: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            if include_key is not None and not include_key(item["Key"]):
                continue
            record = {
                "key": item["Key"],
                "size": int(item.get("Size", 0)),
                "etag": str(item.get("ETag", "")).strip('"'),
            }
            objects.append(record)
    return (
        _finalize_objects(
            client, bucket, objects, workers=workers,
            content_hashes=content_hashes,
        )
        if finalize else objects
    )


def _list_selected_data_outputs(
    client, bucket: str, root: str, sequences: Iterable[str], *,
    full_sequences: Iterable[str] = (), workers: int = 64,
    content_hashes: bool = False,
) -> list[dict]:
    """Fetch bounded workflow inputs without walking whole sequence trees."""
    root = root.rstrip("/")
    names = sorted(set(sequences))
    full = set(full_sequences)

    def head_if_present(key: str) -> dict | None:
        for attempt in range(9):
            try:
                response = client.head_object(Bucket=bucket, Key=key)
                break
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code", ""))
                status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                    return None
                if code in {"400", "BadRequest"} or status == 400:
                    try:
                        response = client.get_object(
                            Bucket=bucket, Key=key, Range="bytes=0-0",
                        )
                    except ClientError as get_exc:
                        get_code = str(
                            get_exc.response.get("Error", {}).get("Code", "")
                        )
                        get_status = get_exc.response.get(
                            "ResponseMetadata", {}
                        ).get("HTTPStatusCode")
                        if get_code in {"404", "NoSuchKey", "NotFound"} or get_status == 404:
                            return None
                        raise
                    content_range = str(response.get("ContentRange", ""))
                    if "/" not in content_range:
                        raise ValueError(
                            f"Invalid ranged GET Content-Range for {key}"
                        ) from exc
                    body = response.get("Body")
                    if body is not None:
                        body.read(1)
                        close = getattr(body, "close", None)
                        if close is not None:
                            close()
                    return {
                        "key": key,
                        "size": int(content_range.rsplit("/", 1)[1]),
                        "etag": str(response.get("ETag", "")).strip('"'),
                    }
                if code not in {"429", "SlowDown", "Throttling", "ThrottlingException"}:
                    raise
                if attempt == 8:
                    raise
                time.sleep(min(10.0, 0.25 * (2 ** attempt)) * random.uniform(0.75, 1.25))
        return {
            "key": key,
            "size": int(response.get("ContentLength", 0)),
            "etag": str(response.get("ETag", "")).strip('"'),
        }

    def list_sequence(name: str) -> list[dict]:
        sequence_root = f"{root}/{name}"
        selected: list[dict] = []
        if name in full:
            for relative_path in sorted(DATA_OUTPUT_EXACT_FILES):
                item = head_if_present(f"{sequence_root}/{relative_path}")
                if item is not None:
                    selected.append(item)
            prefixes = (
                "mv_preprocess/object_mesh",
                "mv_preprocess/labeled_bboxes",
                # These bounded prefixes cover both supported storage layouts:
                # metadata H5 + content-addressed FFV1 MKV sidecars, and legacy
                # per-frame image directories.  Filtering below excludes
                # unrelated reconstruction/debug payloads.
                "mv_preprocess/images",
                *(f"foundation_stereo/{camera}" for camera in CAMERAS_LEFT),
                *(f"sam2_object_masks/{camera}" for camera in CAMERAS_LEFT),
                *(f"sam2_human_masks/{camera}" for camera in CAMERAS_LEFT),
            )
        else:
            # Backlog preprocessing ignores old reconstruction outputs, but
            # the complete manual-label directory and its provenance marker
            # must remain frozen.
            marker = head_if_present(
                f"{sequence_root}/mv_preprocess/object_bbox_source.txt"
            )
            if marker is not None:
                selected.append(marker)
            prefixes = ("mv_preprocess/labeled_bboxes",)
        for relative_prefix in prefixes:
            for attempt in range(9):
                try:
                    listed = _list(
                        client, bucket, f"{sequence_root}/{relative_prefix}",
                        workers=1, finalize=False,
                    )
                    break
                except ClientError as exc:
                    code = str(exc.response.get("Error", {}).get("Code", ""))
                    if code not in {"429", "SlowDown", "Throttling", "ThrottlingException"}:
                        raise
                    if attempt == 8:
                        raise
                    time.sleep(
                        min(10.0, 0.25 * (2 ** attempt))
                        * random.uniform(0.75, 1.25)
                    )
            selected.extend(
                item for item in listed
                if _is_selected_data_output_path(
                    item["key"][len(sequence_root) + 1:]
                )
            )
        return selected

    objects: list[dict] = []
    if names:
        with ThreadPoolExecutor(max_workers=min(workers, len(names))) as executor:
            futures = {executor.submit(list_sequence, name): name for name in names}
            for completed, future in enumerate(as_completed(futures), 1):
                objects.extend(future.result())
                if completed % 250 == 0:
                    print(
                        f"Scanned selected data_output artifacts for "
                        f"{completed}/{len(names)} sequence(s)",
                        file=sys.stderr,
                    )
    objects = sorted(
        {item["key"]: item for item in objects}.values(),
        key=lambda item: item["key"],
    )
    return _finalize_objects(
        client, bucket, objects, workers=workers, content_hashes=content_hashes,
    )


def _by_sequence(objects: Iterable[dict], root: str) -> dict[str, list[dict]]:
    root = root.strip("/") + "/"
    result: dict[str, list[dict]] = {}
    for item in objects:
        key = item["key"]
        if not key.startswith(root):
            continue
        relative = key[len(root):]
        sequence, separator, path = relative.partition("/")
        if not separator or not path or sequence.startswith("_"):
            continue
        result.setdefault(sequence, []).append({**item, "relative_path": path})
    for values in result.values():
        values.sort(key=lambda item: item["relative_path"])
    return result


def _camera_stems(objects: list[dict], directory: str, extension: str) -> set[str]:
    prefix = directory.rstrip("/") + "/"
    stems: set[str] = set()
    for item in objects:
        path = item["relative_path"]
        if not path.startswith(prefix):
            continue
        remainder = path[len(prefix):]
        if "/" in remainder:
            camera, frame = remainder.split("/", 1)
            if Path(frame).suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".tif", ".tiff",
            }:
                stems.add(camera)
        elif path.endswith(extension):
            stems.add(Path(remainder).stem.split(".ffv1.", 1)[0])
    return stems


def _legacy_export_issues(objects: list[dict]) -> list[str]:
    paths = {item["relative_path"] for item in objects}
    issues = [f"missing_export:{name}" for name in sorted(LEGACY_SINGLE_FILES - paths)]
    for directory, cameras in (
        ("images", CAMERAS_LEFT), ("depth", CAMERAS_LEFT),
        ("object_masks", CAMERAS_LEFT), ("human_masks", CAMERAS_LEFT),
    ):
        stems = _camera_stems(objects, directory, ".h5")
        missing = sorted(set(cameras) - stems)
        if missing:
            issues.append(f"missing_export_{directory}:" + ",".join(missing))
    if not any(path.startswith("object_mesh/") for path in paths):
        issues.append("missing_export:object_mesh")
    return issues


def _data_output_issues(objects: list[dict]) -> list[str]:
    paths = {item["relative_path"] for item in objects}
    issues: list[str] = []
    required = {
        "mv_preprocess/edex",
        "mv_preprocess/hoi_metadata.yaml",
        "mv_preprocess/object_mesh/output_aligned.glb",
        "mv_preprocess/object_mesh/output_symmetry.json",
        "foundation_pose/poses.npy",
        "sam3d_body/mhr_params_mv.pt",
        "sam3d_body/mhr_mesh_mv.pt",
        "export_soma/soma_params.npz",
        "estimate_ground_plane/ground_plane.json",
        "eval_chamfer_human/chamfer_metrics.json",
        "check_object_mask/check_object_mask.json",
    }
    issues.extend(f"missing_output:{name}" for name in sorted(required - paths))
    for camera in CAMERAS_RGB:
        if (
            f"mv_preprocess/images/{camera}.h5" not in paths
            and not any(
                path.startswith(f"mv_preprocess/images/{camera}/")
                for path in paths
            )
        ):
            issues.append(f"missing_output_image:{camera}")
        if f"mv_preprocess/videos/{camera}.mp4" not in paths:
            issues.append(f"missing_output_video:{camera}")
    for camera in CAMERAS_LEFT:
        if (
            f"foundation_stereo/{camera}/depth.h5" not in paths
            and not any(
                path.startswith(f"foundation_stereo/{camera}/depth/")
                for path in paths
            )
        ):
            issues.append(f"missing_output_depth:{camera}")
        if not any(
            path.startswith(f"sam2_object_masks/{camera}/")
            and (path.endswith(".h5") or f"/{camera}/0/" in path)
            for path in paths
        ):
            issues.append(f"missing_output_object_mask:{camera}")
        if not any(
            path.startswith(f"sam2_human_masks/{camera}/")
            and (path.endswith(".h5") or f"/{camera}/0/" in path)
            for path in paths
        ):
            issues.append(f"missing_output_human_mask:{camera}")
    return issues


def _db_sequences(path: str, dataset: str) -> dict[str, dict]:
    connection = db.get_connection(path)
    try:
        rows = connection.execute(
            """SELECT s.id, s.sequence_name, s.object_id,
                      p.id AS preprocess_run_id, p.status AS preprocess_status,
                      r.id AS reconstruction_run_id, r.status AS reconstruction_status,
                      r.pipeline_version AS reconstruction_pipeline_version,
                      q.decision AS latest_qc_decision,
                      x.id AS export_run_id, x.status AS export_status,
                      x.reconstruction_run_id AS export_reconstruction_run_id,
                      x.authorization_type AS export_authorization
               FROM sequences s
               LEFT JOIN preprocess_runs p
                 ON p.sequence_id=s.id AND p.is_current=1
               LEFT JOIN reconstruction_runs r
                 ON r.sequence_id=s.id AND r.is_current=1
               LEFT JOIN qc_reviews q ON q.id=(
                 SELECT q2.id FROM qc_reviews q2
                 WHERE q2.reconstruction_run_id=r.id
                 ORDER BY q2.reviewed_at DESC, q2.id DESC LIMIT 1)
               LEFT JOIN export_runs x ON x.sequence_id=s.id AND x.is_current=1
               WHERE s.dataset=? AND s.sequence_kind='hoi'""",
            (dataset,),
        ).fetchall()
        return {row["sequence_name"]: dict(row) for row in rows}
    finally:
        connection.close()


def _load_accuracy_statuses(
    client, bucket: str, output_root: str, outputs: dict[str, list[dict]], *,
    workers: int,
) -> dict[str, str]:
    """Read legacy accuracy summaries concurrently for canary stratification."""
    expected = "check_accuracy/check_accuracy.json"
    keys = {
        name: f"{output_root.rstrip('/')}/{name}/{expected}"
        for name, objects in outputs.items()
        if any(item["relative_path"] == expected for item in objects)
    }

    def read(item: tuple[str, str]) -> tuple[str, str]:
        name, key = item
        try:
            payload = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
            return name, str(payload.get("status") or "unknown")
        except Exception:
            return name, "unreadable"

    statuses: dict[str, str] = {}
    if keys:
        with ThreadPoolExecutor(max_workers=min(workers, len(keys))) as executor:
            for name, status in executor.map(read, sorted(keys.items())):
                statuses[name] = status
    return statuses


def _has_legacy_pass_evidence(identity: dict) -> bool:
    if identity.get("reconstruction_status") != "SUCCEEDED":
        return False
    if identity.get("latest_qc_decision") == "PASS":
        return True
    return (
        identity.get("export_status") == "SUCCEEDED"
        and identity.get("export_reconstruction_run_id") is not None
        and identity.get("export_reconstruction_run_id")
        == identity.get("reconstruction_run_id")
    )


def select_canary(records: list[dict], count: int = 20) -> set[str]:
    """Deterministically maximize object and secondary-stratum coverage."""
    if len(records) < count:
        raise ValueError(f"Need at least {count} eligible sequences for the canary")
    sizes = sorted(record["source_size"] for record in records)
    quartiles = [sizes[(len(sizes) - 1) * index // 4] for index in range(1, 4)]
    candidates = []
    for record in records:
        band = sum(record["source_size"] > boundary for boundary in quartiles)
        candidates.append((
            record,
            record.get("object_id") or "unknown",
            bool(record.get("has_symmetry")),
            record.get("legacy_metric_status") or "unknown",
            band,
            hashlib.sha256(record["sequence"].encode()).hexdigest(),
        ))
    selected: set[str] = set()
    objects: set[str] = set()
    symmetries: set[bool] = set()
    metrics: set[str] = set()
    bands: set[int] = set()
    while len(selected) < count:
        remaining = [item for item in candidates if item[0]["sequence"] not in selected]
        if not remaining:
            break
        chosen = max(
            remaining,
            key=lambda item: (
                int(item[1] not in objects),
                int(item[2] not in symmetries),
                int(item[3] not in metrics),
                int(item[4] not in bands),
                item[5],
            ),
        )
        record, object_id, symmetry, metric, band, _ = chosen
        selected.add(record["sequence"])
        objects.add(object_id)
        symmetries.add(symmetry)
        metrics.add(metric)
        bands.add(band)
    return selected


def build_inventories(
    *, dataset: str, dataset_cfg: dict, db_path: str, cutover_at: str,
    client=None, bucket: str | None = None, base_prefix: str | None = None,
    workers: int = 64, content_hashes: bool = False,
) -> tuple[dict, dict, dict]:
    swift_base = dataset_cfg["swift_base"]
    if client is None:
        client, bucket, base_prefix = _client(
            swift_base, max_pool_connections=workers,
        )
    assert bucket is not None and base_prefix is not None
    export_root = f"{base_prefix}/{get_legacy_export_path(dataset_cfg)}"
    # Campaign inventories freeze the canonical legacy inputs consumed by the
    # revalidation workflow. Backlog preprocessing also uses this canonical
    # root, while each reconstruction writes beneath a unique
    # ``reconstruction_<request-id>`` child. Following reconstruction attempt
    # directories here would inventory the wrong lineage.
    legacy_output_path = get_pipeline_input_path(
        dataset_cfg, REVALIDATION_PIPELINE,
    )
    output_root = f"{base_prefix}/{legacy_output_path}"
    raw_root = f"{base_prefix}/{get_pipeline_input_path(dataset_cfg, RECON_PIPELINE)}"
    exports = _by_sequence(
        _list(
            client, bucket, export_root, workers=workers,
            content_hashes=content_hashes,
        ),
        export_root,
    )
    raw_inputs = _by_sequence(
        _list(
            client, bucket, raw_root, workers=workers,
            content_hashes=content_hashes,
        ),
        raw_root,
    )
    database = _db_sequences(db_path, dataset)
    names = sorted(set(database) | set(exports) | set(raw_inputs))
    full_output_sequences = {
        name for name, identity in database.items()
        if _has_legacy_pass_evidence(identity)
    }
    outputs = _by_sequence(
        _list_selected_data_outputs(
            client, bucket, output_root, names,
            full_sequences=full_output_sequences, workers=workers,
            content_hashes=content_hashes,
        ),
        output_root,
    )
    accuracy_statuses = _load_accuracy_statuses(
        client, bucket, output_root, outputs, workers=workers,
    )
    records: list[dict] = []
    for name in names:
        identity = database.get(name, {})
        export_objects = exports.get(name, [])
        output_objects = outputs.get(name, [])
        raw_objects = raw_inputs.get(name, [])
        pass_evidence = _has_legacy_pass_evidence(identity)
        issues = _legacy_export_issues(export_objects) if pass_evidence else []
        if pass_evidence:
            issues.extend(_data_output_issues(output_objects))
        if not pass_evidence:
            issues.append("missing_legacy_pass_evidence")
        output_paths = {item["relative_path"] for item in output_objects}
        cleaned = "intermediate_cleanup.json" in output_paths
        if cleaned:
            issues.append("FULL_RERUN_REQUIRED")
        # A cleanup commit deliberately invalidates abbreviated revalidation:
        # future campaign inventory routes the sequence through preprocessing
        # from raw input even when historical PASS evidence remains valid.
        eligible = pass_evidence and not cleaned
        preprocess_core = {
            "mv_preprocess/edex", "mv_preprocess/hoi_metadata.yaml",
            "mv_preprocess/object_mesh/output_aligned.glb",
            "mv_preprocess/object_mesh/output_symmetry.json",
        }
        preprocess_ready = (
            identity.get("preprocess_status") == "SUCCEEDED"
            and preprocess_core.issubset(output_paths)
            and all(
                f"mv_preprocess/images/{camera}.h5" in output_paths
                or any(
                    path.startswith(f"mv_preprocess/images/{camera}/")
                    for path in output_paths
                )
                for camera in CAMERAS_RGB
            )
            and all(f"mv_preprocess/videos/{camera}.mp4" in output_paths for camera in CAMERAS_RGB)
        )
        labeled_views = {
            Path(path).stem for path in output_paths
            if path.startswith("mv_preprocess/labeled_bboxes/") and path.endswith(".json")
        }
        object_bbox_source = next(
            (
                item.get("content")
                for item in output_objects
                if item["relative_path"]
                == "mv_preprocess/object_bbox_source.txt"
            ),
            None,
        )
        labels_ready = (
            object_bbox_source == "manual_labeled_bboxes"
            and set(CAMERAS_LEFT).issubset(labeled_views)
        )
        has_symmetry = "mv_preprocess/object_mesh/output_symmetry.json" in output_paths
        legacy_metric_status = accuracy_statuses.get(name)
        records.append({
            "sequence": name,
            "sequence_id": identity.get("id"),
            "object_id": identity.get("object_id"),
            "preprocess_run_id": identity.get("preprocess_run_id"),
            "reconstruction_run_id": identity.get("reconstruction_run_id"),
            "reconstruction_pipeline_version": identity.get(
                "reconstruction_pipeline_version"
            ),
            "export_run_id": identity.get("export_run_id"),
            "export_reconstruction_run_id": identity.get(
                "export_reconstruction_run_id"
            ),
            "export_authorization": identity.get("export_authorization"),
            "latest_qc_decision": identity.get("latest_qc_decision"),
            "route": "revalidation" if eligible else "backlog",
            "recommended_stage": "revalidation" if eligible else "preprocess",
            "preprocess_ready": preprocess_ready,
            "labeled_bboxes_ready": labels_ready,
            "object_bbox_source": object_bbox_source,
            "has_symmetry": has_symmetry,
            "legacy_metric_status": legacy_metric_status,
            "blocked_reasons": sorted(set(issues)),
            "source_size": sum(item["size"] for item in output_objects),
            "legacy_export_size": sum(item["size"] for item in export_objects),
            "data_output_objects": output_objects,
            "data_export_objects": export_objects,
            "raw_input_objects": raw_objects,
        })
    eligible = [record for record in records if record["route"] == "revalidation"]
    canary = select_canary(eligible) if len(eligible) >= 20 else set()
    common = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cutover_at": cutover_at,
        "dataset": dataset,
        "object_identity_protocol": (
            "size_etag_sha256_selected" if content_hashes else "size_etag"
        ),
        "source_prefixes": {
            "data_output": f"{swift_base.rstrip('/')}/{legacy_output_path}",
            "data_export": f"{swift_base.rstrip('/')}/{get_legacy_export_path(dataset_cfg)}",
            "raw_input": f"{swift_base.rstrip('/')}/{get_pipeline_input_path(dataset_cfg, RECON_PIPELINE)}",
        },
    }
    master = {**common, "kind": "cutover", "sequence_count": len(records), "sequences": records}
    revalidation_records = [
        {**record, "cohort": "CANARY" if record["sequence"] in canary else "BULK"}
        for record in eligible
    ]
    backlog_records = [record for record in records if record["route"] == "backlog"]
    revalidation = {
        **common, "kind": "legacy_revalidation",
        "sequence_count": len(revalidation_records), "sequences": revalidation_records,
    }
    backlog = {
        **common, "kind": "backlog_reprocessing",
        "sequence_count": len(backlog_records), "sequences": backlog_records,
    }
    return master, revalidation, backlog


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(f"{path}: {value['sequence_count']} sequence(s)")


def _write_sequence_records(directory: Path, inventory: dict) -> None:
    """Write small immutable inputs so each workflow avoids the full inventory."""
    directory.mkdir(parents=True, exist_ok=True)
    common = {
        key: value for key, value in inventory.items()
        if key not in {"sequences", "sequence_count"}
    }
    for record in inventory["sequences"]:
        name = record["sequence"]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Unsafe sequence name for record manifest: {name!r}")
        payload = {**common, "sequence_count": 1, "sequences": [record]}
        path = directory / f"{name}.json"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    print(f"{directory}: {inventory['sequence_count']} sequence manifest(s)")


def refresh_inventory_db_identities(
    output_dir: Path, *, dataset: str, db_path: str,
) -> tuple[dict, dict, dict]:
    """Refresh DB-assigned identity fields without rereading frozen CSS objects."""
    master_path = output_dir / "cutover_inventory.json"
    master = json.loads(master_path.read_text())
    if master.get("dataset") != dataset or master.get("kind") != "cutover":
        raise ValueError("Output directory does not contain the requested cutover inventory")
    database = _db_sequences(db_path, dataset)
    names = {record["sequence"] for record in master["sequences"]}
    if names != set(database):
        raise ValueError(
            f"Inventory/DB sequence mismatch: inventory_only={len(names - set(database))}, "
            f"db_only={len(set(database) - names)}"
        )
    refreshed = []
    identity_fields = (
        "id", "object_id", "preprocess_run_id", "reconstruction_run_id",
        "reconstruction_pipeline_version",
        "export_run_id", "export_reconstruction_run_id",
        "export_authorization", "latest_qc_decision",
    )
    for record in master["sequences"]:
        identity = database[record["sequence"]]
        expected_route = (
            "revalidation" if _has_legacy_pass_evidence(identity) else "backlog"
        )
        if record["route"] != expected_route:
            raise ValueError(
                f"Routing changed after cutover for {record['sequence']}: "
                f"{record['route']} -> {expected_route}"
            )
        updated = dict(record)
        for field in identity_fields:
            target = "sequence_id" if field == "id" else field
            updated[target] = identity.get(field)
        refreshed.append(updated)
    master = {
        **master,
        "db_identity_refreshed_at": datetime.now(timezone.utc).isoformat(),
        "sequences": refreshed,
    }
    revalidation_records = [
        record for record in refreshed if record["route"] == "revalidation"
    ]
    canary = select_canary(revalidation_records) if len(revalidation_records) >= 20 else set()
    revalidation_records = [
        {**record, "cohort": "CANARY" if record["sequence"] in canary else "BULK"}
        for record in revalidation_records
    ]
    common = {
        key: value for key, value in master.items()
        if key not in {"kind", "sequence_count", "sequences"}
    }
    revalidation = {
        **common, "kind": "legacy_revalidation",
        "sequence_count": len(revalidation_records),
        "sequences": revalidation_records,
    }
    backlog_records = [record for record in refreshed if record["route"] == "backlog"]
    backlog = {
        **common, "kind": "backlog_reprocessing",
        "sequence_count": len(backlog_records), "sequences": backlog_records,
    }
    _write(master_path, master)
    _write(output_dir / "legacy_revalidation_inventory.json", revalidation)
    _write(output_dir / "backlog_reprocessing_inventory.json", backlog)
    _write_sequence_records(output_dir / "revalidation_records", revalidation)
    return master, revalidation, backlog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--db", default=db.DB_PATH)
    parser.add_argument("--cutover-at", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--output-dir", type=Path, default=state_path("manifests"))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--refresh-db-only", action="store_true",
        help="Refresh DB IDs/object identities and canary cohorts in an existing inventory.",
    )
    parser.add_argument(
        "--content-hashes", action="store_true",
        help=(
            "Download and SHA-256 selected small objects in addition to the "
            "default CSS size+ETag identity checks."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = load_config(MV_HOI_DIR)
    dataset_cfg = config["datasets"][args.dataset]
    db.init_db(args.db)
    if args.refresh_db_only:
        refresh_inventory_db_identities(
            args.output_dir, dataset=args.dataset, db_path=args.db,
        )
        return
    master, revalidation, backlog = build_inventories(
        dataset=args.dataset, dataset_cfg=dataset_cfg, db_path=args.db,
        cutover_at=args.cutover_at, workers=args.workers,
        content_hashes=args.content_hashes,
    )
    _write(args.output_dir / "cutover_inventory.json", master)
    _write(args.output_dir / "legacy_revalidation_inventory.json", revalidation)
    _write(args.output_dir / "backlog_reprocessing_inventory.json", backlog)
    _write_sequence_records(args.output_dir / "revalidation_records", revalidation)


if __name__ == "__main__":
    main()
