# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WiLoR over a video: per-frame JSON list in <output_dir>/<frame:06d>.json.

The video is decoded with ffmpeg (cheaper + more robust than imageio across
container codecs). If ``--bboxes_dir`` is supplied, each frame's external
bboxes are read from ``<bboxes_dir>/<frame:06d>.json``.

Output schema per file: see ``image_to_hands.py``.

Usage:
    python -m v2d.wilor.lib.video_to_hands \\
        --video_path  /data/clip.mp4 \\
        --output_dir  /data/wilor \\
        --weights_dir /data/weights/wilor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from v2d.wilor.lib._wilor import (
    _PUBLIC_ARTIFACT_SHA256,
    _validate_weights,
    get_pipeline,
    run_wilor_detect,
    run_wilor_on_bboxes,
)
from v2d.wilor.lib import _wilor as wilor_implementation
from v2d.wilor.lib import image_to_hands as image_to_hands_implementation
from v2d.wilor.lib.image_to_hands import _load_external_bboxes


RUN_GENERATION_FILENAME = "run_generation.json"
RUN_GENERATION_SCHEMA = "v2d.wilor.video-to-hands-generation/v1"
WILOR_SOURCE_COMMIT = "ebec42f94c389070cdd7dda6fd1bf0b4a659c960"
WILOR_HF_REVISION = "b00adea9a6843bbb4c9042109c5eb29ab2a59dea"
_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _artifact_identity(value: dict) -> dict:
    return {
        "size_bytes": int(value["size_bytes"]),
        "sha256": str(value["sha256"]),
    }


def _weight_artifacts(weights_dir: Path) -> dict[str, dict]:
    _validate_weights(str(weights_dir))
    root = weights_dir.resolve() / "pretrained_models"
    names = ["MANO_RIGHT.pkl", *_PUBLIC_ARTIFACT_SHA256.keys()]
    return {name: _artifact(root / name) for name in sorted(names)}


def _implementation_artifacts() -> dict[str, dict]:
    paths = {
        "v2d/wilor/lib/video_to_hands.py": Path(__file__),
        "v2d/wilor/lib/_wilor.py": Path(str(wilor_implementation.__file__)),
        "v2d/wilor/lib/image_to_hands.py": Path(
            str(image_to_hands_implementation.__file__)
        ),
    }
    return {key: _artifact(path) for key, path in sorted(paths.items())}


def _bbox_artifacts(bboxes_dir: Path | None) -> dict | None:
    if bboxes_dir is None:
        return None
    bboxes_dir = bboxes_dir.resolve()
    if not bboxes_dir.is_dir():
        raise FileNotFoundError(f"Bbox directory not found: {bboxes_dir}")
    files = sorted(bboxes_dir.glob("*.json"))
    return {
        "directory": str(bboxes_dir),
        "files": {path.name: _artifact(path) for path in files},
    }


def _aggregate_outputs(paths: list[Path]) -> dict:
    digest = hashlib.sha256()
    files: dict[str, dict] = {}
    total = 0
    for path in sorted(paths, key=lambda value: value.name):
        artifact = _artifact(path)
        name = path.name.encode("utf-8")
        size = int(artifact["size_bytes"])
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[path.name] = _artifact_identity(artifact)
        total += size
    return {
        "files": files,
        "file_count": len(paths),
        "size_bytes": total,
        "aggregate_sha256": digest.hexdigest(),
        "aggregate_algorithm": (
            "sha256(concat(u64be(name_length),name_utf8,u64be(size),file_bytes) "
            "for lexical frame filenames)"
        ),
    }


def _static_identity(value: dict) -> dict:
    bboxes = value["sources"]["bboxes"]
    return {
        "execution_environment": value["execution_environment"],
        "source_revisions": value["source_revisions"],
        "source_video": _artifact_identity(value["sources"]["video"]),
        "bboxes": (
            None
            if bboxes is None
            else {
                name: _artifact_identity(artifact)
                for name, artifact in sorted(bboxes["files"].items())
            }
        ),
        "weights": {
            name: _artifact_identity(artifact)
            for name, artifact in sorted(value["weights"].items())
        },
        "implementation_sources": {
            name: _artifact_identity(artifact)
            for name, artifact in sorted(value["implementation_sources"].items())
        },
        "parameters": value["parameters"],
    }


def _generation_id(static_identity: dict, frame_names: list[str]) -> str:
    payload = {"identity": static_identity, "expected_frames": frame_names}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_complete_generation(output_dir: Path) -> dict:
    manifest_path = output_dir / RUN_GENERATION_FILENAME
    if not manifest_path.is_file():
        raise FileExistsError(
            f"Existing WiLoR output has no {RUN_GENERATION_FILENAME}; "
            "refusing to skip or mix generations"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid WiLoR run-generation manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RUN_GENERATION_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise RuntimeError("WiLoR run-generation manifest is not a complete v1 commit")
    return manifest


def _validate_resume(
    output_dir: Path,
    current_static: dict,
) -> dict:
    manifest = _load_complete_generation(output_dir)
    try:
        recorded_static = _static_identity(manifest)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"WiLoR manifest has incomplete top-level provenance: {exc}"
        ) from exc
    if manifest.get("static_identity") != recorded_static:
        raise RuntimeError(
            "WiLoR manifest top-level provenance is inconsistent with its "
            "static identity"
        )
    if recorded_static != current_static:
        raise RuntimeError(
            "Existing WiLoR output belongs to a different input/image/weight/"
            "parameter/source generation; refusing resume"
        )
    expected = manifest.get("expected_frames")
    if not isinstance(expected, dict) or not isinstance(
        expected.get("filenames"), list
    ):
        raise RuntimeError("WiLoR manifest omits the exact expected frame set")
    frame_names = expected["filenames"]
    if expected.get("count") != len(frame_names) or len(set(frame_names)) != len(
        frame_names
    ):
        raise RuntimeError("WiLoR manifest expected frame set is inconsistent")
    if manifest.get("generation_id") != _generation_id(recorded_static, frame_names):
        raise RuntimeError("WiLoR manifest generation ID is invalid")
    expected_entries = {RUN_GENERATION_FILENAME, *frame_names}
    actual_entries = {path.name for path in output_dir.iterdir()}
    if actual_entries != expected_entries:
        raise RuntimeError(
            "WiLoR output directory does not contain exactly the committed frame set"
        )
    frame_paths = [output_dir / name for name in frame_names]
    aggregate = _aggregate_outputs(frame_paths)
    if manifest.get("outputs") != aggregate:
        raise RuntimeError("WiLoR output hashes no longer match the run manifest")
    for path in frame_paths:
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid WiLoR frame JSON {path}: {exc}") from exc
        if not isinstance(records, list):
            raise RuntimeError(f"WiLoR frame JSON must be a list: {path}")
    return manifest


def _decode_video_to_frames(video_path: str, frames_dir: str) -> int:
    """Decode all frames of ``video_path`` to ``frames_dir/<06d>.png``. Return frame count."""
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_path,
            os.path.join(frames_dir, "%06d.png"),
        ],
        check=True,
    )
    files = sorted(os.listdir(frames_dir))
    return len(files)


def video_to_hands(
    video_path: str,
    output_dir: str,
    weights_dir: str,
    bboxes_dir: Optional[str] = None,
    image_id: str = "",
) -> dict:
    """Run one immutable WiLoR generation or strictly resume that generation.

    Frame JSON is produced in a private sibling directory and the entire
    directory is renamed into place only after all inputs and outputs have
    been rehashed.  Any pre-existing directory must be an exact, complete
    generation; legacy or mismatched output is refused without invoking the
    model.
    """

    if not _IMAGE_ID_PATTERN.fullmatch(image_id):
        raise ValueError(
            "image_id must be the immutable sha256:<64 lowercase hex> Docker ID"
        )
    video = Path(video_path).resolve()
    output = Path(output_dir).resolve()
    weights = Path(weights_dir).resolve()
    bboxes = Path(bboxes_dir).resolve() if bboxes_dir is not None else None
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    static_snapshot = {
        "execution_environment": {"container_image_id": image_id},
        "source_revisions": {
            "wilor_mini_commit": WILOR_SOURCE_COMMIT,
            "huggingface_revision": WILOR_HF_REVISION,
        },
        "sources": {
            "video": _artifact(video),
            "bboxes": _bbox_artifacts(bboxes),
        },
        "weights": _weight_artifacts(weights),
        "implementation_sources": _implementation_artifacts(),
        "parameters": {
            "inference_mode": "detector" if bboxes is None else "external_bboxes",
            "model_dtype": "torch.float16",
            "frame_decoder": "ffmpeg default video stream to RGB PNG",
            "frame_index_origin": 0,
            "detection_order": "descending score",
            "json_indent": 2,
        },
    }
    static_identity = _static_identity(static_snapshot)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"WiLoR output path is not a directory: {output}")
        return _validate_resume(output, static_identity)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".partial", dir=output.parent
        )
    )
    try:
        with tempfile.TemporaryDirectory() as decoded_dir:
            n = _decode_video_to_frames(str(video), decoded_dir)
            if n == 0:
                raise RuntimeError(f"ffmpeg decoded zero frames from {video}")
            frame_names = [f"{index:06d}.json" for index in range(n)]
            if bboxes is not None:
                actual_bbox_names = set(static_snapshot["sources"]["bboxes"]["files"])
                if actual_bbox_names != set(frame_names):
                    raise RuntimeError(
                        "External bbox directory must contain exactly one JSON for "
                        "every decoded frame"
                    )

            get_pipeline(str(weights))
            # ffmpeg writes 1-indexed PNGs; JSON is intentionally 0-indexed.
            for one_idx in tqdm(range(1, n + 1), desc="wilor", ncols=80, unit="frame"):
                frame_idx = one_idx - 1
                out_path = staging / frame_names[frame_idx]
                frame_path = Path(decoded_dir) / f"{one_idx:06d}.png"
                image = np.asarray(Image.open(frame_path).convert("RGB"))
                if bboxes is None:
                    records = run_wilor_detect(image, weights_dir=str(weights))
                else:
                    bb_path = bboxes / frame_names[frame_idx]
                    external_boxes, is_right = _load_external_bboxes(str(bb_path))
                    records = (
                        run_wilor_on_bboxes(
                            image,
                            external_boxes,
                            is_right,
                            weights_dir=str(weights),
                        )
                        if external_boxes
                        else []
                    )
                records.sort(key=lambda record: record["score"], reverse=True)
                _atomic_json(out_path, records)

        current_snapshot = {
            **static_snapshot,
            "sources": {
                "video": _artifact(video),
                "bboxes": _bbox_artifacts(bboxes),
            },
            "weights": _weight_artifacts(weights),
            "implementation_sources": _implementation_artifacts(),
        }
        if _static_identity(current_snapshot) != static_identity:
            raise RuntimeError(
                "WiLoR source inputs, weights, or implementation changed during "
                "inference; refusing the atomic generation commit"
            )
        frame_paths = [staging / name for name in frame_names]
        outputs = _aggregate_outputs(frame_paths)
        manifest = {
            "schema_version": RUN_GENERATION_SCHEMA,
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **static_snapshot,
            "static_identity": static_identity,
            "expected_frames": {"count": n, "filenames": frame_names},
            "generation_id": _generation_id(static_identity, frame_names),
            "outputs": outputs,
        }
        _atomic_json(staging / RUN_GENERATION_FILENAME, manifest)
        # Validate the exact staged directory and all hashes through the same
        # code used for future resumes before publishing it.
        validated = _validate_resume(staging, static_identity)
        os.replace(staging, output)
        return validated
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument(
        "--image_id",
        required=True,
        help="Immutable sha256:<64 hex> ID of the container executing inference.",
    )
    parser.add_argument(
        "--bboxes_dir",
        default=None,
        help="Optional dir of <frame:06d>.json bboxes; skip WiLoR's detector when set.",
    )
    args = parser.parse_args()
    video_to_hands(
        video_path=args.video_path,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        bboxes_dir=args.bboxes_dir,
        image_id=args.image_id,
    )


if __name__ == "__main__":
    main()
