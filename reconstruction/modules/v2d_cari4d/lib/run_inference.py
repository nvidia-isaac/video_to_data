# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the complete CARI4D wild-video inference pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from v2d.cari4d.lib.download_weights import CARI4D_CHECKPOINT_RELATIVE_PATH, CARI4D_CHECKPOINT_SHA256, CARI4D_CONFIG_RELATIVE_PATH, CARI4D_MANIFEST_RELATIVE_PATH, CARI4D_REVISION, download_weights, require_moge2_model, sha256_file


SOURCE_ROOT = Path(__file__).resolve().parent / "cari4d"
SAM3D_SOURCE_ROOT = Path("/workspace/v2d_sam3d_body/lib")
PIPELINE_SCHEMA = "v2d.cari4d.wild_inference.v1"


def _file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _signature(command: Sequence[str], inputs: Sequence[Path]) -> dict[str, Any]:
    payload = {"command": list(command), "inputs": [_file_identity(path) for path in inputs]}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def _output_identities(outputs: Sequence[Path]) -> list[dict[str, Any]]:
    identities = []
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"CARI4D stage output is missing or empty: {path}")
        identities.append(_file_identity(path))
    return identities


def _stage_run(name: str, command: list[str], inputs: Sequence[Path], outputs: Sequence[Path], marker_root: Path, env: dict[str, str], overwrite: bool) -> dict[str, Any]:
    marker = marker_root / f"{name}.json"
    signature = _signature([value for value in command if value not in ("--overwrite", "--redo", "--redo-sam3d-cache")], inputs)
    if marker.is_file() and not overwrite:
        existing = json.loads(marker.read_text())
        outputs_available = all(path.is_file() and path.stat().st_size > 0 for path in outputs)
        if outputs_available and existing.get("signature") == signature and existing.get("outputs") == _output_identities(outputs):
            print(f"CARI4D_STAGE_REUSED {name}", flush=True)
            return {"name": name, "reused": True, "elapsed_seconds": 0.0, "outputs": existing["outputs"]}
        if outputs_available:
            raise ValueError(f"CARI4D stage {name} outputs do not match their current inputs; rerun with --overwrite")
    started = time.perf_counter()
    print(f"CARI4D_STAGE_STARTED {name}", flush=True)
    subprocess.run(command, cwd=SOURCE_ROOT, env=env, check=True)
    elapsed_seconds = time.perf_counter() - started
    output_identities = _output_identities(outputs)
    _atomic_json(marker, {"schema": "v2d.cari4d.stage.v1", "name": name, "signature": signature, "outputs": output_identities})
    print(f"CARI4D_STAGE_COMPLETED {name} elapsed_seconds={elapsed_seconds:.3f}", flush=True)
    return {"name": name, "reused": False, "elapsed_seconds": elapsed_seconds, "outputs": output_identities}


def _required_weights(weights_path: Path) -> dict[str, Path]:
    paths = {"checkpoint": weights_path / "cari4d" / CARI4D_CHECKPOINT_RELATIVE_PATH, "config": weights_path / "cari4d" / CARI4D_CONFIG_RELATIVE_PATH, "manifest": weights_path / "cari4d" / CARI4D_MANIFEST_RELATIVE_PATH, "sam3d_checkpoint": weights_path / "sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt", "mhr_model": weights_path / "sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt", "foundationpose_score": weights_path / "foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth", "foundationpose_refine": weights_path / "foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth"}
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required CARI4D weight is missing: {path}")
    paths["moge2_model"] = require_moge2_model(weights_path)
    if sha256_file(paths["checkpoint"]) != CARI4D_CHECKPOINT_SHA256:
        raise ValueError(f"CARI4D checkpoint SHA-256 mismatch: {paths['checkpoint']}")
    return paths


def _flag(command: list[str], enabled: bool, value: str) -> None:
    if enabled:
        command.append(value)


def run_inference(video_path: str, mask_h5_path: str, object_mesh_path: str, weights_path: str, output_dir: str, *, download_models: bool = True, expected_frames: int | None = None, device: str = "cuda", moge_devices: str | None = None, moge_batch_size: int = 8, sam3d_chunk_size: int = 16, sam3d_alignment_workers: int = 1, sam3d_refit_batch_size: int = 512, depth_alignment_render_batch_size: int = 4, depth_alignment_encoding_workers: int = 16, depth_alignment_write_batch_size: int = 64, foundationpose_iteration: int = 5, foundationpose_max_attempts: int = 5, coconet_stride: int = 96, coconet_render_batch_size: int = 32, coconet_crop_workers: int = 8, coconet_crop_buffer_count: int = 2, postopt_num_steps: int = 300, postopt_batch_size: int = 0, postopt_temporal_weight: float = 100.0, postopt_human_pose_prior_weight: float = 200.0, postopt_contact_activation_distance_m: float = 0.05, postopt_report_every: int = 10, postopt_diagnostics_every: int = 500, symmetric_object: bool = False, optimize_object_rotation: bool = False, render_batch_size: int = 4, overwrite: bool = False) -> Path:
    video_path, mask_h5_path, object_mesh_path, weights_path, output_dir = map(lambda value: Path(value).resolve(), (video_path, mask_h5_path, object_mesh_path, weights_path, output_dir))
    for path in (video_path, mask_h5_path, object_mesh_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    suffix = ".0.color.mp4"
    if not video_path.name.endswith(suffix):
        raise ValueError(f"CARI4D input video must end with {suffix}: {video_path}")
    sequence = video_path.name[:-len(suffix)]
    output_root = output_dir / sequence
    marker_root = output_root / ".stages"
    marker_root.mkdir(parents=True, exist_ok=True)
    if download_models:
        download_weights(weights_path)
    weights = _required_weights(weights_path)
    env = os.environ.copy()
    env.update({"PYTHONPATH": os.pathsep.join((str(SOURCE_ROOT), str(SAM3D_SOURCE_ROOT), str(Path("/workspace/v2d_foundation_pose/lib/FoundationPose")), env.get("PYTHONPATH", ""))), "HF_HOME": str(weights_path / "hf_home"), "TORCH_HOME": str(weights_path / "sam3d_body/torch_home"), "SAM3D_BODY_ROOT": str(SAM3D_SOURCE_ROOT), "MHR_ASSETS_ROOT": str(weights_path / "sam3d_body"), "FOUNDATIONPOSE_WEIGHTS_DIR": str(weights_path / "foundationpose/nvlabs_pytorch"), "OPENCV_IO_ENABLE_OPENEXR": "1", "PYOPENGL_PLATFORM": "egl", "HDF5_USE_FILE_LOCKING": "FALSE", "PYTHONUNBUFFERED": "1"})
    raw_depth = output_root / "depth/moge2/raw.npy"
    depth_intrinsics = output_root / "depth/moge2/intrinsics.pkl"
    depth_report = output_root / "depth/moge2/depth_report.json"
    export_root = output_root / "export"
    export_seq = export_root / sequence
    export_marker = export_seq / "wild_export.json"
    aligned_object_mesh = export_seq / "object_mesh/output_aligned.glb"
    mhr_init = output_root / "mhr_init" / f"{sequence}.pkl"
    sam3d_cache = output_root / "mhr_init" / f"{sequence}.sam3d.h5"
    aligned_depth = output_root / "depth/moge2/aligned.h5"
    foundationpose_init = output_root / "foundationpose" / f"{sequence}.pkl"
    coconet_bundle = output_root / "inference/coconet.pth"
    input_cache = output_root / "inference/materialized_inputs.h5"
    refined_bundle = output_root / "inference/refined.pth"
    refinement_checkpoint = output_root / "inference/refinement_checkpoint.pth"
    output_video = output_root / "inference" / f"{sequence}_step200000_reconstruction.mp4"
    comparison_video = output_root / "inference" / f"{sequence}_step200000_before_after.mp4"
    report_path = output_root / "pipeline_report.json"
    for path in (raw_depth, depth_intrinsics, depth_report, mhr_init, sam3d_cache, aligned_depth, foundationpose_init, coconet_bundle, input_cache, refined_bundle, refinement_checkpoint, output_video, comparison_video):
        path.parent.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    stages = []
    depth_command = [python, str(SOURCE_ROOT / "prep/mhr_wild_depth.py"), "--video", str(video_path), "--output-depth", str(raw_depth), "--output-intrinsics", str(depth_intrinsics), "--output-report", str(depth_report), "--device", device, "--batch-size", str(moge_batch_size), "--local-files-only"]
    if expected_frames is not None:
        depth_command += ["--expected-frames", str(expected_frames)]
    if moge_devices:
        depth_command += ["--devices", *[value.strip() for value in moge_devices.split(",") if value.strip()]]
    _flag(depth_command, overwrite, "--overwrite")
    stages.append(_stage_run("01_depth", depth_command, [video_path, weights["manifest"]], [raw_depth, depth_intrinsics, depth_report], marker_root, env, overwrite))
    export_command = [python, str(SOURCE_ROOT / "prep/prepare_mhr_wild_export.py"), "--video", str(video_path), "--mask-h5", str(mask_h5_path), "--object-mesh", str(object_mesh_path), "--intrinsics-file", str(depth_intrinsics), "--output-root", str(export_root)]
    _flag(export_command, overwrite, "--redo")
    stages.append(_stage_run("02_export", export_command, [video_path, mask_h5_path, object_mesh_path, depth_intrinsics], [export_marker, aligned_object_mesh], marker_root, env, overwrite))
    sam3d_command = [python, str(SOURCE_ROOT / "prep/run_sam3d_mhr_export.py"), str(export_seq), "--out-file", str(mhr_init), "--camera-id", "0", "--depth-root", str(export_seq), "--sam3d-ckpt", str(weights["sam3d_checkpoint"]), "--mhr-path", str(weights["mhr_model"]), "--chunk-size", str(sam3d_chunk_size), "--sam3d-cache", str(sam3d_cache), "--alignment-workers", str(sam3d_alignment_workers), "--refit-batch-size", str(sam3d_refit_batch_size), "--human-prompt-mode", "sam2_bbox_mask", "--no-align-to-gt-depth"]
    if overwrite:
        sam3d_command += ["--redo", "--redo-sam3d-cache"]
    stages.append(_stage_run("03_human_initialization", sam3d_command, [export_marker, weights["sam3d_checkpoint"], weights["mhr_model"]], [mhr_init, sam3d_cache], marker_root, env, overwrite))
    alignment_command = [python, str(SOURCE_ROOT / "prep/align_depth_to_mhr_wild.py"), str(export_seq), "--raw-depth", str(raw_depth), "--depth-report", str(depth_report), "--mhr-init", str(mhr_init), "--output", str(aligned_depth), "--render-batch-size", str(depth_alignment_render_batch_size), "--encoding-workers", str(depth_alignment_encoding_workers), "--write-batch-size", str(depth_alignment_write_batch_size)]
    _flag(alignment_command, overwrite, "--redo")
    stages.append(_stage_run("04_depth_alignment", alignment_command, [export_marker, raw_depth, depth_report, mhr_init], [aligned_depth], marker_root, env, overwrite))
    foundationpose_command = [python, str(SOURCE_ROOT / "prep/run_foundationpose_mhr_export.py"), str(export_seq), "--depth-root", str(aligned_depth), "--out-file", str(foundationpose_init), "--camera-id", "0", "--iteration", str(foundationpose_iteration), "--max-attempts", str(foundationpose_max_attempts), "--register-first-then-track", "--no-first-usable-frame-gt-rotation-oracle"]
    _flag(foundationpose_command, overwrite, "--redo")
    stages.append(_stage_run("05_object_pose", foundationpose_command, [export_marker, aligned_depth, weights["foundationpose_score"], weights["foundationpose_refine"]], [foundationpose_init], marker_root, env, overwrite))
    coconet_command = [python, str(SOURCE_ROOT / "tools/run_mhr_wild_inference.py"), str(export_seq), "--depth-h5", str(aligned_depth), "--mhr-init", str(mhr_init), "--foundationpose-file", str(foundationpose_init), "--config", str(weights["config"]), "--checkpoint", str(weights["checkpoint"]), "--output", str(coconet_bundle), "--stride", str(coconet_stride), "--render-batch-size", str(coconet_render_batch_size), "--crop-workers", str(coconet_crop_workers), "--crop-buffer-count", str(coconet_crop_buffer_count), "--input-cache", str(input_cache), "--device", device, "--offline-supervision-contract"]
    _flag(coconet_command, overwrite, "--overwrite")
    stages.append(_stage_run("06_coconet", coconet_command, [export_marker, aligned_depth, mhr_init, foundationpose_init, weights["config"], weights["checkpoint"]], [coconet_bundle, input_cache], marker_root, env, overwrite))
    postopt_command = [python, "-m", "learning.training.mhr_opt_refineout", "--bundle", str(coconet_bundle), "--object-mesh", str(aligned_object_mesh), "--out", str(refined_bundle), "--mode", "smplh_parity", "--num-steps", str(postopt_num_steps), "--batch-size", str(postopt_batch_size), "--device", device, "--postopt-checkpoint", str(refinement_checkpoint), "--mhr-assets-root", str(weights_path / "sam3d_body"), "--w-temporal", str(postopt_temporal_weight), "--w-human-pose-prior", str(postopt_human_pose_prior_weight), "--contact-activation-distance-m", str(postopt_contact_activation_distance_m), "--report-every", str(postopt_report_every), "--diagnostics-every", str(postopt_diagnostics_every)]
    _flag(postopt_command, symmetric_object, "--symmetric-object")
    _flag(postopt_command, optimize_object_rotation, "--optimize-object-rotation")
    stages.append(_stage_run("07_refinement", postopt_command, [coconet_bundle, aligned_object_mesh, weights["mhr_model"]], [refined_bundle, refinement_checkpoint], marker_root, env, overwrite))
    render_command = [python, str(SOURCE_ROOT / "tools/render_mhr_wild_inference.py"), str(export_seq), "--source-video", str(video_path), "--object-mesh", str(aligned_object_mesh), "--before-bundle", str(coconet_bundle), "--after-bundle", str(refined_bundle), "--output", str(output_video), "--comparison-output", str(comparison_video), "--device", device, "--batch-size", str(render_batch_size)]
    _flag(render_command, overwrite, "--overwrite")
    stages.append(_stage_run("08_render", render_command, [export_marker, video_path, aligned_object_mesh, coconet_bundle, refined_bundle], [output_video, comparison_video], marker_root, env, overwrite))
    report = {"schema": PIPELINE_SCHEMA, "verdict": "PASS", "sequence": sequence, "checkpoint": {**_file_identity(weights["checkpoint"]), "step": 200000, "sha256": CARI4D_CHECKPOINT_SHA256, "repository": CARI4D_REVISION}, "inputs": {"video": _file_identity(video_path), "masks": _file_identity(mask_h5_path), "object_mesh": _file_identity(object_mesh_path)}, "defaults": {"depth": "moge2", "human_prompt": "sam2_bbox_mask", "foundationpose": "register_first_then_track", "coconet_supervision": "checkpoint_embedded_offline", "postopt_frames": "full_clip" if postopt_batch_size == 0 else postopt_batch_size, "postopt_steps": postopt_num_steps, "postopt_temporal_weight": postopt_temporal_weight, "postopt_human_pose_prior_weight": postopt_human_pose_prior_weight, "postopt_contact_activation_distance_m": postopt_contact_activation_distance_m}, "stages": stages, "output_video": _file_identity(output_video), "comparison_video": _file_identity(comparison_video), "output_root": str(output_root)}
    _atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run complete CARI4D wild-video inference")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--mask_h5_path", required=True)
    parser.add_argument("--object_mesh_path", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skip_weight_download", action="store_true")
    parser.add_argument("--expected_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--moge_devices", default=None, help="Comma-separated CUDA devices")
    parser.add_argument("--moge_batch_size", type=int, default=8)
    parser.add_argument("--sam3d_chunk_size", type=int, default=16)
    parser.add_argument("--sam3d_alignment_workers", type=int, default=1)
    parser.add_argument("--sam3d_refit_batch_size", type=int, default=512)
    parser.add_argument("--depth_alignment_render_batch_size", type=int, default=4)
    parser.add_argument("--depth_alignment_encoding_workers", type=int, default=16)
    parser.add_argument("--depth_alignment_write_batch_size", type=int, default=64)
    parser.add_argument("--foundationpose_iteration", type=int, default=5)
    parser.add_argument("--foundationpose_max_attempts", type=int, default=5)
    parser.add_argument("--coconet_stride", type=int, default=96)
    parser.add_argument("--coconet_render_batch_size", type=int, default=32)
    parser.add_argument("--coconet_crop_workers", type=int, default=8)
    parser.add_argument("--coconet_crop_buffer_count", type=int, default=2)
    parser.add_argument("--postopt_num_steps", type=int, default=300)
    parser.add_argument("--postopt_batch_size", type=int, default=0)
    parser.add_argument("--postopt_temporal_weight", type=float, default=100.0)
    parser.add_argument("--postopt_human_pose_prior_weight", type=float, default=200.0)
    parser.add_argument("--postopt_contact_activation_distance_m", type=float, default=0.05)
    parser.add_argument("--postopt_report_every", type=int, default=10)
    parser.add_argument("--postopt_diagnostics_every", type=int, default=500)
    parser.add_argument("--symmetric_object", action="store_true")
    parser.add_argument("--optimize_object_rotation", action="store_true")
    parser.add_argument("--render_batch_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    values = vars(args)
    values["download_models"] = not values.pop("skip_weight_download")
    run_inference(**values)


if __name__ == "__main__":
    main()
