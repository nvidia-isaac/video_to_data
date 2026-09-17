#!/usr/bin/env python3
"""Read-only preflight for the local multi-view HOI reconstruction runner."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_IMAGES = (
    "v2d_rosbag",
    "v2d_mv_preprocess",
    "v2d_face_detector",
    "v2d_foundation_stereo",
    "v2d_grounding_dino",
    "v2d_sam2",
    "v2d_foundation_pose",
    "v2d_detectron2",
    "v2d_sam3d_body",
    "v2d_mv_postprocess",
)

REQUIRED_WEIGHT_DIRS = (
    "face_detector",
    "foundation_stereo",
    "grounding_dino",
    "sam2",
    "foundation_pose",
    "detectron2",
    "sam3d_body",
)


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _nonempty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    message = (result.stderr or result.stdout).strip()
    return result.returncode, message


def _load_prompt(metadata_path: Path) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is unavailable; run this helper with reconstruction/.venv/bin/python"
        ) from exc
    try:
        metadata = yaml.safe_load(metadata_path.read_text()) or {}
    except Exception as exc:
        raise RuntimeError(f"cannot parse {metadata_path}: {exc}") from exc
    object_metadata = metadata.get("object")
    if not isinstance(object_metadata, dict):
        return ""
    prompt = object_metadata.get("prompt")
    return prompt.strip() if isinstance(prompt, str) else ""


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", required=True, type=Path)
    parser.add_argument("--calibration-edex", required=True, type=Path)
    parser.add_argument("--object-mesh", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-runtime-checks",
        action="store_true",
        help="Validate files only; skip Docker, image, and GPU checks",
    )
    args = parser.parse_args()

    skill_dir = Path(__file__).resolve().parents[1]
    repo_root = skill_dir.parents[2]
    reconstruction_dir = repo_root / "reconstruction"
    errors: list[str] = []
    warnings: list[str] = []

    paths = {
        "sequence directory": args.sequence_dir,
        "calibration EDEX": args.calibration_edex,
        "object mesh": args.object_mesh,
        "output directory": args.output_dir,
    }
    for label, path in paths.items():
        if not path.is_absolute():
            errors.append(f"{label} must be absolute: {path}")

    sequence_dir = args.sequence_dir.resolve()
    calibration_edex = args.calibration_edex.resolve()
    object_mesh = args.object_mesh.resolve()
    output_dir = args.output_dir.resolve()

    runner = reconstruction_dir / "modules/v2d_pipelines/run_mv_hoi_reconstruction.py"
    runbook = reconstruction_dir / "docs/mv_hoi_local_pipeline.md"
    if not runner.is_file() or not runbook.is_file():
        errors.append(f"repository root could not be resolved from {skill_dir}")

    if not sequence_dir.is_dir():
        errors.append(f"sequence directory does not exist: {sequence_dir}")
        mcaps: list[Path] = []
    else:
        mcaps = sorted(path for path in sequence_dir.glob("*.mcap") if _nonempty_file(path))
        if not mcaps:
            errors.append(f"no nonempty .mcap file found directly under {sequence_dir}")

    metadata_path = sequence_dir / "hoi_metadata.yaml"
    if not _nonempty_file(metadata_path):
        errors.append(f"missing or empty metadata: {metadata_path}")
        prompt = ""
    else:
        try:
            prompt = _load_prompt(metadata_path)
        except RuntimeError as exc:
            errors.append(str(exc))
            prompt = ""
        if not prompt:
            errors.append(f"object.prompt is missing or empty in {metadata_path}")

    if not _nonempty_file(calibration_edex):
        errors.append(f"calibration EDEX is missing or empty: {calibration_edex}")
    else:
        try:
            json.loads(calibration_edex.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"calibration EDEX is not valid JSON: {exc}")

    if object_mesh.name != "output_aligned.glb":
        errors.append(f"object mesh must be named output_aligned.glb: {object_mesh}")
    if not _nonempty_file(object_mesh):
        errors.append(f"object mesh is missing or empty: {object_mesh}")
    symmetry_path = object_mesh.parent / "output_symmetry.json"
    if symmetry_path.exists():
        if not _nonempty_file(symmetry_path):
            errors.append(f"symmetry annotation is empty: {symmetry_path}")
        else:
            try:
                json.loads(symmetry_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"symmetry annotation is not valid JSON: {exc}")

    if output_dir.exists():
        errors.append(f"output directory must not already exist: {output_dir}")
    if _is_relative_to(output_dir, sequence_dir) or _is_relative_to(sequence_dir, output_dir):
        errors.append("sequence and output directories must not contain one another")
    if _is_relative_to(output_dir, object_mesh.parent) or _is_relative_to(
        object_mesh, output_dir
    ):
        errors.append("object mesh source and output directories must not overlap")

    weights_root = reconstruction_dir / "data/weights"
    for name in REQUIRED_WEIGHT_DIRS:
        weight_dir = weights_root / name
        if not _nonempty_dir(weight_dir):
            errors.append(f"required weight directory is missing or empty: {weight_dir}")

    soma_cache = weights_root / "sam3d_body/hf_home/hub/models--nvidia--soma-x"
    if not _nonempty_dir(soma_cache):
        warnings.append("SOMA-X cache is absent; first use requires network access")
    foundation_pose_weights = weights_root / "foundation_pose"
    if not any(foundation_pose_weights.rglob("*.engine")):
        warnings.append("FoundationPose TensorRT engines are absent; first use will build them")

    disk_parent = _existing_parent(output_dir.parent)
    if disk_parent.exists():
        free_gib = shutil.disk_usage(disk_parent).free / (1024**3)
    else:
        free_gib = 0.0
        warnings.append(f"could not determine free space for {output_dir}")

    if not args.skip_runtime_checks:
        if shutil.which("docker") is None:
            errors.append("docker is not installed or not on PATH")
        else:
            code, message = _run(["docker", "info"])
            if code:
                errors.append(f"docker is unavailable: {message}")
            for image in REQUIRED_IMAGES:
                code, _ = _run(["docker", "image", "inspect", image])
                if code:
                    errors.append(f"required Docker image is missing: {image}")
            code, message = _run(
                ["docker", "run", "--rm", "--gpus", "all", "v2d_foundation_pose", "nvidia-smi"]
            )
            if code:
                errors.append(f"Docker GPU check failed: {message}")
        if shutil.which("nvidia-smi") is None:
            errors.append("nvidia-smi is not installed or not on PATH")
        else:
            code, message = _run(["nvidia-smi", "-L"])
            if code:
                errors.append(f"host GPU check failed: {message}")

    print(f"repository: {repo_root}")
    print(f"sequence: {sequence_dir}")
    print(f"mcap files: {len(mcaps)}")
    print(f"object prompt: {prompt!r}")
    print(f"calibration EDEX: {calibration_edex}")
    print(f"object mesh: {object_mesh}")
    print(f"symmetry annotation: {'present' if symmetry_path.is_file() else 'absent'}")
    print(f"output: {output_dir}")
    print(f"free space: {free_gib:.1f} GiB")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if errors:
        print(f"preflight failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
