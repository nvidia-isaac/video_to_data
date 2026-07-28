"""Host orchestration for build, acquisition, and offline Phantom inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from . import IMAGE_NAME, MANOTORCH_COMMIT, PHANTOM_COMMIT, PHANTOM_HAMER_COMMIT
from .acquire import acquire_hamer
from .assets import (
    GROUNDING_DINO_REQUIRED_SHA256,
    verify_pinned_inference_assets,
)
from .build import build
from .provenance import sha256_file, sha256_tree


def _inspect_image() -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", IMAGE_NAME],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(result.stdout)[0]
    labels = record.get("Config", {}).get("Labels", {}) or {}
    if labels.get("org.opencontainers.image.revision") != PHANTOM_COMMIT:
        raise RuntimeError("Phantom image has an unexpected parent source revision")
    if labels.get("io.v2d.phantom-hamer.revision") != PHANTOM_HAMER_COMMIT:
        raise RuntimeError("Phantom image has an unexpected HaMeR source revision")
    if labels.get("io.v2d.manotorch.revision") != MANOTORCH_COMMIT:
        raise RuntimeError("Phantom image has an unexpected manotorch revision")
    return {"id": record["Id"], "labels": labels}


def acquire(download_dir: Path, models_dir: Path) -> dict[str, Any]:
    image = _inspect_image()
    hamer = acquire_hamer(download_dir, models_dir)
    dino_dir = models_dir / "grounding-dino-base"
    dino_dir.mkdir(parents=True, exist_ok=True)
    from .provenance import sha256_file

    dino_is_complete = all(
        (dino_dir / name).is_file() and sha256_file(dino_dir / name) == expected_hash
        for name, expected_hash in GROUNDING_DINO_REQUIRED_SHA256.items()
    )
    if not dino_is_complete:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-e",
                "HF_HOME=/tmp/huggingface",
                "-e",
                "HF_HUB_DISABLE_TELEMETRY=1",
                "-v",
                f"{dino_dir.resolve()}:/models/grounding-dino:rw",
                image["id"],
                "python",
                "-m",
                "inpainting.phantom_tracker.acquire_grounding_dino",
                "--output-dir",
                "/models/grounding-dino",
            ],
            check=True,
        )
    dino_hash, dino_files = sha256_tree(dino_dir)
    manifest = {
        "schema_version": "v2d.inpainting.phantom-acquisition/v1",
        "container_image_id": image["id"],
        "hamer": hamer,
        "grounding_dino": {"tree_sha256": dino_hash, "files": dino_files},
        "mano": "external licensed read-only input; not copied",
    }
    manifest_path = models_dir / "acquisition_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


def _mount(source: Path, target: str, read_only: bool) -> list[str]:
    resolved = str(source.resolve())
    if ":" in resolved:
        raise ValueError(f"Docker volume source contains unsupported colon: {resolved}")
    mode = ":ro" if read_only else ""
    # --mount uses commas as an option separator and therefore cannot express
    # the literal commas in TACO tuple directory names. -v receives this as a
    # single subprocess argument and safely supports both commas and spaces.
    return ["-v", f"{resolved}:{target}{mode}"]


def infer(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu != 0:
        raise ValueError("This investigation runner is locked to physical GPU 0")
    image = _inspect_image()
    video = args.video.resolve()
    intrinsics = args.intrinsics.resolve()
    models_dir = args.models_dir.resolve()
    mano_dir = args.mano_dir.resolve()
    output_dir = args.output_dir.resolve()
    required = (
        video,
        intrinsics,
        models_dir / "hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
        models_dir / "grounding-dino-base/config.json",
        mano_dir / "MANO_LEFT.pkl",
        mano_dir / "MANO_RIGHT.pkl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phantom input assets: {missing}")
    for source in (video, intrinsics):
        if source == output_dir or output_dir in source.parents:
            raise ValueError("Output directory must not alias or contain an input file")

    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--runtime=nvidia",
        "--gpus",
        "device=0",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,size=8g",
        "-e",
        "HF_HOME=/tmp/huggingface",
        "-e",
        "TORCH_HOME=/tmp/torch",
        "-e",
        "XDG_CACHE_HOME=/tmp/cache",
        "-e",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        "CUDA_VISIBLE_DEVICES=0",
    ]
    command += _mount(video, "/inputs/video.mp4", True)
    command += _mount(intrinsics, "/inputs/intrinsics.txt", True)
    command += _mount(models_dir / "hamer", "/models/hamer", True)
    command += _mount(
        models_dir / "grounding-dino-base", "/models/grounding-dino", True
    )
    command += _mount(mano_dir, "/models/mano", True)
    command += _mount(output_dir, "/output", False)
    command += [
        image["id"],
        "python",
        "-m",
        "inpainting.phantom_tracker.pipeline",
        "--video",
        "/inputs/video.mp4",
        "--intrinsics",
        "/inputs/intrinsics.txt",
        "--intrinsics-source-path",
        str(intrinsics),
        "--output-dir",
        "/output",
        "--grounding-dino-dir",
        "/models/grounding-dino",
        "--hamer-dir",
        "/models/hamer",
        "--mano-dir",
        "/models/mano",
        "--sequence-id",
        args.sequence_id,
        "--container-image-id",
        image["id"],
        "--box-threshold",
        str(args.box_threshold),
        "--text-threshold",
        str(args.text_threshold),
        "--min-valid-fraction",
        str(args.min_valid_fraction),
        "--max-ambiguous-fraction",
        str(args.max_ambiguous_fraction),
        "--batch-size",
        str(args.batch_size),
        "--minimum-box-area-fraction",
        str(args.minimum_box_area_fraction),
        "--maximum-box-area-fraction",
        str(args.maximum_box_area_fraction),
    ]
    if args.overwrite:
        command.append("--overwrite")
    plan = {
        "image": IMAGE_NAME,
        "container_image_id": image["id"],
        "network": "none",
        "physical_gpu": 0,
        "input_mounts_read_only": True,
        "video": str(video),
        "intrinsics": str(intrinsics),
        "models_dir": str(models_dir),
        "mano_dir": str(mano_dir),
        "output_dir": str(output_dir),
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    output_dir.mkdir(parents=True, exist_ok=True)
    # This host-side check is the final operation before the offline container
    # starts. The container repeats it directly before loading either network.
    verify_pinned_inference_assets(
        models_dir / "grounding-dino-base", models_dir / "hamer"
    )
    subprocess.run(command, check=True)

    from inpainting.contracts import validate_tracking_file

    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("state") != "complete":
        raise RuntimeError(
            f"Phantom run did not commit complete state: {metadata_path}"
        )
    if metadata.get("implementation", {}).get("container_image_id") != image["id"]:
        raise RuntimeError("Phantom metadata image ID does not match executed image")
    expected_names = {
        "raw": "phantom_raw_predictions.npz",
        "tracking": "tracking.npz",
        "overlay": "hand_overlay.mp4",
    }
    recorded_outputs = metadata.get("outputs")
    if not isinstance(recorded_outputs, dict) or set(recorded_outputs) != set(
        expected_names
    ):
        raise RuntimeError("Phantom metadata does not record the exact output bundle")
    for name, filename in expected_names.items():
        record = recorded_outputs[name]
        if not isinstance(record, dict) or record.get("filename") != filename:
            raise RuntimeError(f"Phantom metadata has an unexpected {name} filename")
        path = output_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Phantom output is missing: {path}")
        if record.get("bytes") != path.stat().st_size or record.get(
            "sha256"
        ) != sha256_file(path):
            raise RuntimeError(
                f"Phantom {name} output does not match its recorded fingerprint"
            )
    intrinsics_record = metadata.get("inputs", {}).get("intrinsics", {})
    if (
        intrinsics_record.get("source_path") != str(intrinsics)
        or intrinsics_record.get("bytes") != intrinsics.stat().st_size
        or intrinsics_record.get("sha256") != sha256_file(intrinsics)
    ):
        raise RuntimeError("Phantom metadata does not match the mounted intrinsics")
    validate_tracking_file(
        output_dir / "tracking.npz",
        expected_frames=int(metadata["geometry"]["frame_count"]),
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Build the pinned inference image")
    acquire_parser = subparsers.add_parser(
        "acquire", help="Network-enabled checkpoint/model acquisition"
    )
    acquire_parser.add_argument("--download-dir", type=Path, required=True)
    acquire_parser.add_argument("--models-dir", type=Path, required=True)
    infer_parser = subparsers.add_parser("infer", help="Run offline GPU-0 inference")
    infer_parser.add_argument("--video", type=Path, required=True)
    infer_parser.add_argument("--intrinsics", type=Path, required=True)
    infer_parser.add_argument("--models-dir", type=Path, required=True)
    infer_parser.add_argument("--mano-dir", type=Path, required=True)
    infer_parser.add_argument("--output-dir", type=Path, required=True)
    infer_parser.add_argument("--sequence-id", required=True)
    infer_parser.add_argument("--gpu", type=int, default=0)
    infer_parser.add_argument("--box-threshold", type=float, default=0.2)
    infer_parser.add_argument("--text-threshold", type=float, default=0.2)
    infer_parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    infer_parser.add_argument("--max-ambiguous-fraction", type=float, default=0.15)
    infer_parser.add_argument("--batch-size", type=int, default=16)
    infer_parser.add_argument("--minimum-box-area-fraction", type=float, default=0.001)
    infer_parser.add_argument("--maximum-box-area-fraction", type=float, default=0.12)
    infer_parser.add_argument("--overwrite", action="store_true")
    infer_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "acquire":
        print(
            json.dumps(
                acquire(args.download_dir.resolve(), args.models_dir.resolve()),
                indent=2,
            )
        )
    else:
        result = infer(args)
        if not args.dry_run:
            print(json.dumps(result["quality"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
