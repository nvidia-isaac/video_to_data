#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit exact frame and modality invariants across a GR00T run."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from groot_finetune.contracts import EmbodimentContract, load_embodiment_contract
from groot_finetune.task_profile import load_task_profile


@dataclass
class Audit:
    """Collect checks, warnings, and errors without hiding later failures."""

    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def require(self, condition: bool, success: str, failure: str) -> None:
        """Record one required invariant."""
        if condition:
            self.checks.append(success)
        else:
            self.errors.append(failure)


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def finite_tree(value: Any) -> bool:
    """Return whether every numeric leaf in a JSON-like value is finite."""
    if isinstance(value, dict):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, list):
        return all(finite_tree(child) for child in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True


def resolve(root: Path, value: Path | None) -> Path | None:
    """Resolve an optional path relative to the run root."""
    if value is None:
        return None
    return value if value.is_absolute() else root / value


def audit_selected(
    audit: Audit,
    directory: Path,
    episodes: int,
    frames: int,
    contract: EmbodimentContract,
) -> None:
    """Audit selected successful NPZ source episodes."""
    manifest_path = directory / "manifest.json"
    timeout_termination: str | None = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        audit.errors.append(
            f"{manifest_path}: failed to read selection manifest: {exc}"
        )
    else:
        expected = {
            "format": "joint_rollout_selection",
            "schema_version": 1,
            "embodiment_contract": contract.contract_id,
            "embodiment_contract_sha256": contract.sha256,
            "fps": contract.fps,
            "joint_names": list(contract.joint_names),
            "selected_episode_count": episodes,
        }
        for key, value in expected.items():
            audit.require(
                manifest.get(key) == value,
                f"selected manifest {key} matches the run contract",
                f"selected manifest {key}={manifest.get(key)!r}, expected {value!r}",
            )
        timeout_value = manifest.get("timeout_termination")
        if (
            isinstance(timeout_value, str)
            and timeout_value in contract.source_terminations
        ):
            timeout_termination = timeout_value
            audit.checks.append(
                "selected manifest declares a contract-allowed timeout termination"
            )
        else:
            audit.errors.append(
                f"selected manifest timeout_termination is missing or outside the contract: {timeout_value!r}"
            )
    paths = sorted(directory.glob("episode_*.npz"))
    audit.require(
        len(paths) == episodes,
        f"selected export has {episodes} episodes",
        f"selected export has {len(paths)} episodes, expected {episodes}",
    )
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "source_success" not in data.files:
                    audit.errors.append(f"{path}: missing source_success marker")
                    continue
                if not bool(np.asarray(data["source_success"]).item()):
                    audit.errors.append(f"{path}: source_success marker is false")
                reasons = (
                    tuple(
                        str(reason)
                        for reason in np.asarray(data["termination_reasons"])
                    )
                    if "termination_reasons" in data.files
                    else ()
                )
                if timeout_termination is None or timeout_termination not in reasons:
                    audit.errors.append(
                        f"{path}: source_success is true without timeout termination"
                    )
                lengths = {
                    key: int(np.asarray(data[key]).shape[0])
                    for key in ("joint_pos", "action_target", "object_pose")
                    if key in data.files
                }
                if set(lengths) != {"joint_pos", "action_target", "object_pose"}:
                    audit.errors.append(f"{path}: incomplete rollout time series")
                    continue
                if set(lengths.values()) != {frames}:
                    audit.errors.append(
                        f"{path}: expected every time series to have {frames} frames, got {lengths}"
                    )
                for key in data.files:
                    value = np.asarray(data[key])
                    if (
                        np.issubdtype(value.dtype, np.number)
                        and not np.isfinite(value).all()
                    ):
                        audit.errors.append(f"{path}: {key} contains non-finite values")
        except Exception as exc:
            audit.errors.append(f"{path}: failed to read NPZ: {exc}")
    if paths and not any(str(directory) in error for error in audit.errors):
        audit.checks.append(
            f"all selected episodes are timeout-eligible and exactly {frames} frames"
        )


def audit_hdf5(
    audit: Audit,
    *,
    path: Path,
    episodes: int,
    frames: int,
    contract: EmbodimentContract,
    expected_provenance: dict[str, Any],
) -> None:
    """Audit exact semantic HDF5 state, action, camera, and timing contracts."""
    initial_error_count = len(audit.errors)
    try:
        import h5py  # noqa: PLC0415
    except ImportError:
        audit.errors.append("h5py is required to audit --hdf5")
        return
    try:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                audit.errors.append(f"{path}: missing /data group")
                return
            data = handle["data"]
            for key, expected in expected_provenance.items():
                audit.require(
                    data.attrs.get(key) == expected,
                    f"HDF5 {key} matches the run contract",
                    f"HDF5 {key}={data.attrs.get(key)!r}, expected {expected!r}",
                )
            demos = sorted(
                (name for name in data if name.startswith("demo_")),
                key=lambda name: int(name.split("_")[-1]),
            )
            audit.require(
                len(demos) == episodes,
                f"HDF5 has {episodes} demos",
                f"HDF5 has {len(demos)} demos, expected {episodes}",
            )
            required_state_terms: dict[str, int] = {}
            for field in contract.state_fields:
                required_state_terms[field.source_term] = max(
                    required_state_terms.get(field.source_term, 0), field.end
                )
            common_camera_hw: tuple[int, int] | None = None
            for name in demos:
                demo = handle["data"][name]
                if "actions" not in demo or "obs" not in demo:
                    audit.errors.append(f"{path}:{name}: missing actions or obs")
                    continue
                actions = demo["actions"]
                if actions.shape != (frames, contract.action_dim):
                    audit.errors.append(
                        f"{path}:{name}/actions: got {actions.shape}, "
                        f"expected ({frames}, {contract.action_dim})"
                    )
                elif (
                    not np.issubdtype(actions.dtype, np.number)
                    or not np.isfinite(actions[...]).all()
                ):
                    audit.errors.append(
                        f"{path}:{name}/actions: expected finite numeric values"
                    )
                obs = demo["obs"]
                for term, width in required_state_terms.items():
                    if term not in obs:
                        audit.errors.append(
                            f"{path}:{name}/obs: missing required state term {term!r}"
                        )
                        continue
                    values = obs[term]
                    if values.shape != (frames, width):
                        audit.errors.append(
                            f"{path}:{name}/obs/{term}: got {values.shape}, "
                            f"expected ({frames}, {width})"
                        )
                    elif (
                        not np.issubdtype(values.dtype, np.number)
                        or not np.isfinite(values[...]).all()
                    ):
                        audit.errors.append(
                            f"{path}:{name}/obs/{term}: expected finite numeric values"
                        )
                for camera in contract.cameras:
                    term = camera.observation_term
                    if term not in obs:
                        audit.errors.append(
                            f"{path}:{name}/obs: missing required camera {term!r}"
                        )
                        continue
                    values = obs[term]
                    if (
                        values.ndim != 4
                        or values.shape[0] != frames
                        or values.shape[-1] != 3
                    ):
                        audit.errors.append(
                            f"{path}:{name}/obs/{term}: got {values.shape}, "
                            f"expected ({frames}, H, W, 3)"
                        )
                        continue
                    if values.dtype != np.uint8:
                        audit.errors.append(
                            f"{path}:{name}/obs/{term}: got {values.dtype}, expected uint8"
                        )
                    camera_hw = (int(values.shape[1]), int(values.shape[2]))
                    if common_camera_hw is None:
                        common_camera_hw = camera_hw
                    elif camera_hw != common_camera_hw:
                        audit.errors.append(
                            f"{path}:{name}/obs/{term}: resolution {camera_hw} "
                            f"does not match {common_camera_hw}"
                        )
                for term, dataset in demo["obs"].items():
                    if not dataset.shape or dataset.shape[0] != frames:
                        observed_frames = (
                            int(dataset.shape[0]) if dataset.shape else None
                        )
                        audit.errors.append(
                            f"{path}:{name}/obs/{term}: got {observed_frames} frames, "
                            f"expected {frames}"
                        )
    except Exception as exc:
        audit.errors.append(f"{path}: failed to audit HDF5: {exc}")
        return
    if len(audit.errors) == initial_error_count:
        audit.checks.append(
            f"HDF5 state, action, and camera terms satisfy the contract for exactly {frames} frames"
        )


def ffprobe_frames(path: Path) -> int:
    """Return the encoded frame count for one video."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if value and value != "N/A":
        return int(value)

    counted = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = counted.stdout.strip()
    if not value or value == "N/A":
        raise ValueError(f"ffprobe did not report a frame count for {path}")
    return int(value)


def audit_dataset(
    audit: Audit,
    *,
    directory: Path,
    episodes: int,
    frames: int,
    state_dim: int,
    action_dim: int,
    camera_count: int,
    skip_video_probe: bool,
    expected_provenance: dict[str, Any],
) -> None:
    """Audit LeRobot metadata, Parquet rows, and video frame counts."""
    info_path = directory / "meta" / "info.json"
    modality_path = directory / "meta" / "modality.json"
    stats_path = directory / "meta" / "stats.json"
    for path in (info_path, modality_path, stats_path):
        if not path.is_file():
            audit.errors.append(f"missing dataset artifact: {path}")
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text())
    for key, expected in expected_provenance.items():
        audit.require(
            info.get(key) == expected,
            f"dataset {key} matches the run contract",
            f"dataset {key}={info.get(key)!r}, expected {expected!r}",
        )
    audit.require(
        int(info.get("total_episodes", -1)) == episodes,
        f"dataset metadata has {episodes} episodes",
        f"dataset metadata total_episodes={info.get('total_episodes')}, expected {episodes}",
    )
    audit.require(
        int(info.get("total_frames", -1)) == episodes * frames,
        f"dataset metadata has {episodes * frames} frames",
        f"dataset metadata total_frames={info.get('total_frames')}, expected {episodes * frames}",
    )
    audit.require(
        int(info.get("total_videos", -1)) == episodes * camera_count,
        f"dataset metadata has {episodes * camera_count} videos",
        f"dataset metadata total_videos={info.get('total_videos')}, expected {episodes * camera_count}",
    )
    features = info.get("features", {})
    observed_state = features.get("observation.state", {}).get("shape")
    observed_action = features.get("action", {}).get("shape")
    audit.require(
        observed_state == [state_dim],
        f"dataset state dimension is {state_dim}",
        f"dataset state shape is {observed_state}, expected [{state_dim}]",
    )
    audit.require(
        observed_action == [action_dim],
        f"dataset action dimension is {action_dim}",
        f"dataset action shape is {observed_action}, expected [{action_dim}]",
    )
    if modality_path.is_file():
        modality = json.loads(modality_path.read_text())
        audit.require(
            len(modality.get("video", {})) == camera_count,
            f"modality config has {camera_count} video keys",
            f"modality config has {len(modality.get('video', {}))} video keys, expected {camera_count}",
        )
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        audit.require(
            finite_tree(stats),
            "dataset statistics are finite",
            "dataset statistics contain non-finite values",
        )

    parquet_paths = sorted((directory / "data").rglob("*.parquet"))
    audit.require(
        len(parquet_paths) == episodes,
        f"dataset has {episodes} Parquet episodes",
        f"dataset has {len(parquet_paths)} Parquet files, expected {episodes}",
    )
    try:
        from pyarrow import parquet  # noqa: PLC0415
    except ImportError:
        audit.warnings.append("pyarrow unavailable; skipped Parquet row counts")
    else:
        for path in parquet_paths:
            rows = parquet.ParquetFile(path).metadata.num_rows
            if rows != frames:
                audit.errors.append(f"{path}: has {rows} rows, expected {frames}")
        if parquet_paths and not any(
            "Parquet" in error or ".parquet:" in error for error in audit.errors
        ):
            audit.checks.append(f"every Parquet episode has exactly {frames} rows")

    video_paths = sorted((directory / "videos").rglob("*.mp4"))
    audit.require(
        len(video_paths) == episodes * camera_count,
        f"dataset has {episodes * camera_count} video files",
        f"dataset has {len(video_paths)} video files, expected {episodes * camera_count}",
    )
    if skip_video_probe:
        audit.warnings.append("skipped encoded dataset video frame probes")
    elif shutil.which("ffprobe") is None:
        audit.errors.append("ffprobe is required to audit dataset videos")
    else:
        for path in video_paths:
            try:
                encoded_frames = ffprobe_frames(path)
            except Exception as exc:
                audit.errors.append(f"{path}: ffprobe failed: {exc}")
                continue
            if encoded_frames != frames:
                audit.errors.append(
                    f"{path}: has {encoded_frames} encoded frames, expected {frames}"
                )
        if video_paths and not any(".mp4:" in error for error in audit.errors):
            audit.checks.append(
                f"every dataset video has exactly {frames} encoded frames"
            )


def audit_checkpoint(audit: Audit, directory: Path) -> None:
    """Audit the minimum contents of a serveable GR00T checkpoint."""
    expected = (
        "config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "statistics.json",
    )
    missing = [name for name in expected if not (directory / name).is_file()]
    shards = sorted(directory.glob("model-*-of-*.safetensors"))
    audit.require(
        not missing and bool(shards),
        f"checkpoint is complete with {len(shards)} model shards",
        f"checkpoint is incomplete: missing={missing}, model_shards={len(shards)}",
    )


def audit_eval(
    audit: Audit,
    *,
    path: Path,
    eval_episodes: int,
    expected_reset_frame: int | None,
    expected_finger_openness: float | None,
    expected_visual_mode: str | None,
    expected_provenance: dict[str, Any],
    expected_evaluator: str,
) -> None:
    """Audit structured task-level closed-loop results."""
    result = json.loads(path.read_text())
    for key, expected in expected_provenance.items():
        audit.require(
            result.get(key) == expected,
            f"closed-loop {key} matches the run contract",
            f"closed-loop {key}={result.get(key)!r}, expected {expected!r}",
        )
    audit.require(
        int(result.get("completed_episodes", -1)) == eval_episodes,
        f"closed-loop evaluation completed {eval_episodes} episodes",
        f"closed-loop completed={result.get('completed_episodes')}, expected {eval_episodes}",
    )
    successes = int(result.get("successful_episodes", -1))
    audit.require(
        0 <= successes <= eval_episodes,
        f"closed-loop task successes are bounded: {successes}/{eval_episodes}",
        f"invalid successful_episodes={successes}",
    )
    audit.require(
        int(result.get("unsuccessful_episodes", -1)) == eval_episodes - successes,
        "closed-loop success counts are internally consistent",
        "closed-loop unsuccessful_episodes does not complement successful_episodes",
    )
    expected_rate = successes / eval_episodes
    observed_rate = result.get("success_rate")
    audit.require(
        isinstance(observed_rate, (int, float))
        and math.isfinite(observed_rate)
        and math.isclose(float(observed_rate), expected_rate),
        "closed-loop success rate matches episode counts",
        f"closed-loop success_rate={observed_rate!r}, expected {expected_rate}",
    )
    if expected_reset_frame is not None:
        audit.require(
            result.get("reset_frame") == expected_reset_frame,
            f"closed-loop reset frame is {expected_reset_frame}",
            f"closed-loop reset frame is {result.get('reset_frame')}, expected {expected_reset_frame}",
        )
    if expected_finger_openness is not None:
        observed = result.get("reset_finger_openness")
        audit.require(
            observed == expected_finger_openness,
            f"closed-loop finger openness is {expected_finger_openness}",
            f"closed-loop finger openness is {observed}, expected {expected_finger_openness}",
        )
    if expected_visual_mode is not None:
        audit.require(
            result.get("visual_mode") == expected_visual_mode,
            f"closed-loop visual mode is {expected_visual_mode}",
            f"closed-loop visual mode is {result.get('visual_mode')}, expected {expected_visual_mode}",
        )
    camera_means = result.get("first_policy_camera_means", {})
    audit.require(
        bool(camera_means)
        and all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0
            for value in camera_means.values()
        ),
        "first closed-loop policy camera observations are nonblack",
        f"invalid first policy camera means: {camera_means}",
    )
    audit.require(
        result.get("success_evaluator") == expected_evaluator,
        f"closed-loop evaluator is {expected_evaluator}",
        f"closed-loop evaluator is {result.get('success_evaluator')!r}, expected {expected_evaluator!r}",
    )
    audit.require(
        result.get("success_evaluator_config", {}).get("id") == expected_evaluator,
        "closed-loop evaluator config matches its declared evaluator",
        "closed-loop evaluator config ID does not match success_evaluator",
    )
    for key in (
        "episode_lengths",
        "metric_sample_counts",
        "max_object_lift_m",
        "max_hold_steps",
    ):
        values = result.get(key)
        audit.require(
            isinstance(values, list)
            and len(values) == eval_episodes
            and finite_tree(values),
            f"closed-loop {key} has one finite value per episode",
            f"closed-loop {key} must contain {eval_episodes} finite values",
        )


def audit_success_videos(
    audit: Audit,
    directory: Path,
    frames: int,
) -> None:
    """Probe videos that were retained only after task success."""
    paths = sorted(directory.glob("*.mp4"))
    audit.require(
        bool(paths),
        f"found {len(paths)} successful episode videos",
        f"no successful episode videos under {directory}",
    )
    if not paths:
        return
    if shutil.which("ffprobe") is None:
        audit.errors.append("ffprobe is required to audit success videos")
        return
    for path in paths:
        try:
            encoded_frames = ffprobe_frames(path)
        except Exception as exc:
            audit.errors.append(f"{path}: ffprobe failed: {exc}")
            continue
        if encoded_frames != frames:
            audit.errors.append(
                f"{path}: has {encoded_frames} encoded frames, expected {frames}"
            )
    if not any(".mp4:" in error for error in audit.errors):
        audit.checks.append(
            f"every successful episode video has exactly {frames} frames"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=positive_int, required=True)
    parser.add_argument("--frames", type=positive_int, required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--task-profile", required=True)
    parser.add_argument("--selected-export", type=Path)
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-json", type=Path)
    parser.add_argument("--eval-episodes", type=positive_int)
    parser.add_argument("--expected-reset-frame", type=int)
    parser.add_argument("--expected-finger-openness", type=float)
    parser.add_argument(
        "--expected-visual-mode",
        choices=("training", "off"),
    )
    parser.add_argument("--success-videos", type=Path)
    parser.add_argument("--skip-video-probe", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run requested audits and exit nonzero when an invariant fails."""
    args = parse_args()
    contract = load_embodiment_contract(args.contract)
    task_profile = load_task_profile(args.task_profile)
    expected_provenance = {
        "schema_version": 1,
        "embodiment_contract": contract.contract_id,
        "embodiment_contract_sha256": contract.sha256,
        "task_profile": task_profile.task_id,
        "task_profile_sha256": task_profile.sha256,
    }
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root}")
    if (args.eval_json is None) != (args.eval_episodes is None):
        raise ValueError("--eval-json and --eval-episodes must be supplied together")

    audit = Audit()
    selected = resolve(root, args.selected_export)
    hdf5_path = resolve(root, args.hdf5)
    dataset = resolve(root, args.dataset)
    checkpoint = resolve(root, args.checkpoint)
    eval_json = resolve(root, args.eval_json)
    success_videos = resolve(root, args.success_videos)

    for label, path in (
        ("selected export", selected),
        ("HDF5", hdf5_path),
        ("dataset", dataset),
        ("checkpoint", checkpoint),
        ("evaluation JSON", eval_json),
        ("success videos", success_videos),
    ):
        if path is not None and not path.exists():
            audit.errors.append(f"{label} path does not exist: {path}")

    if selected is not None and selected.is_dir():
        audit_selected(audit, selected, args.episodes, args.frames, contract)
    if hdf5_path is not None and hdf5_path.is_file():
        audit_hdf5(
            audit,
            path=hdf5_path,
            episodes=args.episodes,
            frames=args.frames,
            contract=contract,
            expected_provenance=expected_provenance,
        )
    if dataset is not None and dataset.is_dir():
        audit_dataset(
            audit,
            directory=dataset,
            episodes=args.episodes,
            frames=args.frames,
            state_dim=contract.state_dim,
            action_dim=contract.action_dim,
            camera_count=len(contract.cameras),
            skip_video_probe=args.skip_video_probe,
            expected_provenance=expected_provenance,
        )
    if checkpoint is not None and checkpoint.is_dir():
        audit_checkpoint(audit, checkpoint)
    if eval_json is not None and eval_json.is_file():
        audit_eval(
            audit,
            path=eval_json,
            eval_episodes=args.eval_episodes,
            expected_reset_frame=args.expected_reset_frame,
            expected_finger_openness=args.expected_finger_openness,
            expected_visual_mode=args.expected_visual_mode,
            expected_provenance=expected_provenance,
            expected_evaluator=task_profile.evaluator.evaluator_id,
        )
    if success_videos is not None and success_videos.is_dir():
        audit_success_videos(audit, success_videos, args.frames)

    summary = {
        "root": str(root),
        "ok": not audit.errors,
        "checks": audit.checks,
        "warnings": audit.warnings,
        "errors": audit.errors,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_output is not None:
        output = resolve(root, args.json_output)
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    if audit.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
