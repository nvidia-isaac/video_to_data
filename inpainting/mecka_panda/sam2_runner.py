"""Run SAM2 with an already resolved immutable Docker image ID.

This runner intentionally lives with the MECKA pipeline.  The upstream SAM2
wrapper currently forwards its host-only ``image_id`` option to the
container's model CLI, which does not accept it.  Here the immutable ID is
used only as Docker's ``image=`` argument.
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

RUN_GENERATION_FILENAME = "run_generation.json"
OUTPUT_STATE_ABSENT = "absent"
OUTPUT_STATE_COMMITTED = "committed"
OUTPUT_STATE_INCOMPLETE = "incomplete"
_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
RUN_GENERATION_SCHEMA = "v2d.inpainting.sam2-host-generation/v1"


def _runtime() -> tuple[Callable[..., None], str]:
    """Import reconstruction helpers only inside its configured environment."""
    from v2d.docker.container import run_in_container
    from v2d.sam2.docker._config import MODULES_DIR

    return run_in_container, MODULES_DIR


def classify_existing_output(masks_dir: str) -> str:
    """Classify output without trusting a commit marker's contents."""
    output = Path(os.path.abspath(masks_dir))
    if not os.path.lexists(output):
        return OUTPUT_STATE_ABSENT
    if output.is_dir() and (output / RUN_GENERATION_FILENAME).is_file():
        return OUTPUT_STATE_COMMITTED
    return OUTPUT_STATE_INCOMPLETE


def _rewrite_prompts_for_container(prompts_path: str) -> tuple[str, str]:
    """Stage prompts and referenced masks in one read-only mount tree."""
    with open(prompts_path, encoding="utf-8") as stream:
        data = json.load(stream)
    prompts = data.get("prompts") or []
    tempdir = tempfile.mkdtemp(prefix="sam2_prompts_")
    try:
        source_root = os.path.dirname(os.path.abspath(prompts_path))
        staged_masks: dict[str, str] = {}
        for prompt in prompts:
            mask_path = prompt.get("mask_path")
            if not mask_path:
                continue
            source = (
                mask_path
                if os.path.isabs(mask_path)
                else os.path.join(source_root, mask_path)
            )
            source = os.path.abspath(source)
            relative = staged_masks.get(source)
            if relative is None:
                relative = os.path.join(
                    "prompt_assets",
                    f"{len(staged_masks):06d}",
                    os.path.basename(source),
                )
                destination = os.path.join(tempdir, relative)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy2(source, destination)
                staged_masks[source] = relative
            prompt["mask_path"] = f"/data/prompts_path/{relative}"

        rewritten_path = os.path.join(tempdir, "prompts.json")
        with open(rewritten_path, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
        return rewritten_path, tempdir
    except Exception:
        shutil.rmtree(tempdir, ignore_errors=True)
        raise


def _expected_object_ids(prompts_path: str) -> list[int]:
    with open(prompts_path, encoding="utf-8") as stream:
        data = json.load(stream)
    object_ids = sorted(
        {int(prompt["object_id"]) for prompt in (data.get("prompts") or [])}
    )
    if not object_ids:
        raise RuntimeError("SAM2 prompts contain no object IDs")
    return object_ids


def _source_frame_count(video_path: str) -> int:
    source = Path(video_path)
    if source.is_dir():
        frame_count = sum(
            child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg"}
            for child in source.iterdir()
        )
    elif source.suffix.lower() in {".h5", ".hdf5"}:
        raise RuntimeError(
            "Host-side SAM2 completeness validation does not support HDF5 sources"
        )
    elif source.is_file():
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(source),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        value = completed.stdout.strip()
        if completed.returncode or not value.isdecimal():
            detail = completed.stderr.strip() or value or "no frame count"
            raise RuntimeError(f"ffprobe could not count SAM2 source frames: {detail}")
        frame_count = int(value)
    else:
        raise FileNotFoundError(source)
    if frame_count <= 0:
        raise RuntimeError(f"SAM2 source has no frames: {video_path}")
    return frame_count


def _validate_outputs(
    output: Path,
    object_ids: list[int],
    frame_count: int,
    *,
    marker_expected: bool,
) -> dict[str, dict[str, object]]:
    expected_top_level = {str(object_id) for object_id in object_ids}
    if marker_expected:
        expected_top_level.add(RUN_GENERATION_FILENAME)
    if not output.is_dir():
        raise RuntimeError(f"SAM2 did not create its output directory: {output}")
    actual_top_level = {path.name for path in output.iterdir()}
    if actual_top_level != expected_top_level:
        raise RuntimeError(
            "SAM2 output does not contain exactly the expected object directories"
        )

    expected_names = [f"{index:06d}.png" for index in range(frame_count)]
    expected_name_set = set(expected_names)
    identities: dict[str, dict[str, object]] = {}
    for object_id in object_ids:
        object_dir = output / str(object_id)
        if object_dir.is_symlink() or not object_dir.is_dir():
            raise RuntimeError(f"Invalid SAM2 object directory: {object_dir}")
        if {path.name for path in object_dir.iterdir()} != expected_name_set:
            raise RuntimeError(
                f"SAM2 object {object_id} does not contain exactly "
                f"{frame_count} contiguous PNG frames"
            )
        digest = hashlib.sha256()
        total_bytes = 0
        for name in expected_names:
            path = object_dir / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Invalid SAM2 mask artifact: {path}")
            with path.open("rb") as stream:
                signature = stream.read(8)
                if signature != b"\x89PNG\r\n\x1a\n":
                    raise RuntimeError(f"SAM2 mask is not a PNG file: {path}")
                digest.update(name.encode("ascii"))
                digest.update(signature)
                total_bytes += len(signature)
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    total_bytes += len(chunk)
        identities[str(object_id)] = {
            "frame_count": frame_count,
            "size_bytes": total_bytes,
            "aggregate_sha256": digest.hexdigest(),
        }
    return identities


def _commit_generation(
    output: Path,
    *,
    image_id: str,
    prompts_path: str,
    object_ids: list[int],
    frame_count: int,
) -> None:
    outputs = _validate_outputs(output, object_ids, frame_count, marker_expected=False)
    prompts_digest = hashlib.sha256(Path(prompts_path).read_bytes()).hexdigest()
    manifest = {
        "schema_version": RUN_GENERATION_SCHEMA,
        "state": "complete",
        "container_image_id": image_id,
        "prompts_sha256": prompts_digest,
        "expected": {"object_ids": object_ids, "frame_count": frame_count},
        "outputs": outputs,
    }
    marker = output / RUN_GENERATION_FILENAME
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_committed_generation(
    output: Path,
    *,
    image_id: str,
    prompts_path: str,
    object_ids: list[int],
    frame_count: int,
) -> None:
    marker = output / RUN_GENERATION_FILENAME
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    expected = {"object_ids": object_ids, "frame_count": frame_count}
    prompts_digest = hashlib.sha256(Path(prompts_path).read_bytes()).hexdigest()
    if (
        manifest.get("schema_version") != RUN_GENERATION_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("container_image_id") != image_id
        or manifest.get("prompts_sha256") != prompts_digest
        or manifest.get("expected") != expected
    ):
        raise RuntimeError("Existing SAM2 generation does not match current inputs")
    outputs = _validate_outputs(output, object_ids, frame_count, marker_expected=True)
    if manifest.get("outputs") != outputs:
        raise RuntimeError("Existing SAM2 output hashes do not match its commit marker")


def run_video_to_masks(
    *,
    video_path: str,
    prompts_path: str,
    masks_dir: str,
    weights_dir: str,
    image_id: str,
    dev: bool = False,
    gpu: int = 0,
) -> None:
    """Run SAM2 by immutable image ID with strict, atomic host I/O."""
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a non-negative physical GPU index")
    if _IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise ValueError(
            "image_id must be an immutable sha256:<64 lowercase hex> Docker ID"
        )

    video_input = video_path
    input_directories = {"weights_dir"}
    input_files: set[str] = set()
    if os.path.isdir(video_path):
        input_directories.add("video_path")
    else:
        if not os.path.isfile(video_path) and os.path.isfile(video_path + ".h5"):
            video_input = video_path + ".h5"
        input_files.add("video_path")
    object_ids = _expected_object_ids(prompts_path)
    frame_count = _source_frame_count(video_input)
    output = Path(os.path.abspath(masks_dir))
    output_state = classify_existing_output(str(output))
    if output_state == OUTPUT_STATE_INCOMPLETE:
        raise FileExistsError(
            "Existing SAM2 output is not a committed generation; refusing "
            f"overwrite: {output}"
        )
    if output_state == OUTPUT_STATE_COMMITTED:
        _validate_committed_generation(
            output,
            image_id=image_id,
            prompts_path=prompts_path,
            object_ids=object_ids,
            frame_count=frame_count,
        )
        return

    rewritten_path, tempdir = _rewrite_prompts_for_container(prompts_path)
    private_parent: Path | None = None
    try:
        inputs = {
            "video_path": video_input,
            "prompts_path": rewritten_path,
            "weights_dir": weights_dir,
        }
        outputs: dict[str, str] = {}
        atomic_outputs: set[str] = set()
        output.parent.mkdir(parents=True, exist_ok=True)
        private_parent = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.container.", dir=output.parent)
        )
        outputs["masks_dir"] = str(private_parent / output.name)
        atomic_outputs.add("masks_dir")

        run_in_container, modules_dir = _runtime()
        run_in_container(
            image=image_id,
            module="v2d.sam2.lib.video_to_masks",
            inputs=inputs,
            outputs=outputs,
            dev=dev,
            modules_dir=modules_dir,
            gpu_device=gpu,
            env={"CUDA_VISIBLE_DEVICES": "0"},
            network_disabled=True,
            strict_io_isolation=True,
            input_directories=input_directories,
            input_files=input_files,
            atomic_output_directories=atomic_outputs,
        )
        if private_parent is not None:
            staged_output = Path(outputs["masks_dir"])
            _commit_generation(
                staged_output,
                image_id=image_id,
                prompts_path=prompts_path,
                object_ids=object_ids,
                frame_count=frame_count,
            )
            if os.path.lexists(output):
                raise FileExistsError(
                    "SAM2 output appeared during generation; refusing overwrite: "
                    f"{output}"
                )
            os.replace(staged_output, output)
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)
        if private_parent is not None:
            shutil.rmtree(private_parent, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--image_id", required=True)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    values: dict[str, Any] = vars(arguments)
    run_video_to_masks(**values)


if __name__ == "__main__":
    main()
