# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Procedural release tests for the Vega expert and GR00T artifact chain."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import pytest
import torch

from groot_finetune.contracts import VEGA_SHARPA_JOINT
from groot_finetune.convert_to_gr00t import convert, images_to_video
from groot_finetune.source_policy import OnnxSourcePolicy
from groot_finetune.task_profile import (
    LiftHoldEvaluator,
    TargetObject,
    TaskProfile,
    write_task_profile,
)
from groot_finetune.tools.audit_groot_run import Audit, audit_hdf5
from groot_finetune.tools.select_successful_episodes import inspect_episode

REPOSITORY = Path(__file__).resolve().parents[2]
ROBOTIC_ROOT = REPOSITORY / "robotic_grounding"
POLICY = REPOSITORY / (
    "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/"
    "policies/e2e_example/tissue_box/vega_sharpa_policy.onnx"
)
FRAMES = 17
HEIGHT = 16
WIDTH = 16

PROFILE = TaskProfile(
    task_id="generated_object_lift_hold",
    object_prompt="a generated object",
    instruction="lift the generated object and hold it",
    target_object=TargetObject("primary"),
    evaluator=LiftHoldEvaluator(
        lift_threshold_m=0.10,
        hold_threshold_m=0.05,
        min_hold_steps=3,
    ),
)


@dataclass(frozen=True)
class GeneratedEpisode:
    """One deterministic in-memory episode used only by this test module."""

    index: int
    frames: int
    source_success: bool
    arm: np.ndarray
    finger: np.ndarray
    actions: np.ndarray
    object_pose: np.ndarray
    cameras: dict[str, np.ndarray]

    @property
    def state(self) -> np.ndarray:
        """Return state in the released modality order."""
        return np.concatenate(
            (
                self.arm[:, 0:7],
                self.arm[:, 7:14],
                self.finger[:, 0:22],
                self.finger[:, 22:44],
            ),
            axis=1,
        )


def _episode(index: int, *, frames: int, source_success: bool) -> GeneratedEpisode:
    timeline = np.arange(frames, dtype=np.float32)[:, None]
    arm = index * 10_000.0 + timeline * 100.0 + np.arange(14, dtype=np.float32)
    finger = index * 20_000.0 + timeline * 100.0 + np.arange(44, dtype=np.float32)
    actions = index * 30_000.0 + timeline * 100.0 + np.arange(58, dtype=np.float32)
    object_pose = np.zeros((frames, 1, 7), dtype=np.float32)
    object_pose[:, 0, 2] = 0.2 + timeline[:, 0] * 0.001
    object_pose[:, 0, 3] = 1.0
    cameras: dict[str, np.ndarray] = {}
    for camera_index, camera in enumerate(VEGA_SHARPA_JOINT.cameras):
        images = np.empty((frames, HEIGHT, WIDTH, 3), dtype=np.uint8)
        for frame in range(frames):
            value = (camera_index * 60 + index * 20 + frame) % 256
            images[frame, ..., 0] = value
            images[frame, ..., 1] = (value + 17) % 256
            images[frame, ..., 2] = (value + 31) % 256
        cameras[camera.observation_term] = images
    return GeneratedEpisode(
        index=index,
        frames=frames,
        source_success=source_success,
        arm=arm,
        finger=finger,
        actions=actions,
        object_pose=object_pose,
        cameras=cameras,
    )


def _provenance() -> dict[str, object]:
    return {
        "schema_version": 1,
        "embodiment_contract": VEGA_SHARPA_JOINT.contract_id,
        "embodiment_contract_sha256": VEGA_SHARPA_JOINT.sha256,
        "task_profile": PROFILE.task_id,
        "task_profile_sha256": PROFILE.sha256,
    }


def _write_source(root: Path, episodes: tuple[GeneratedEpisode, ...]) -> Path:
    root.mkdir(parents=True)
    manifest = {
        "format": "joint_rollout",
        "schema_version": 1,
        "embodiment_contract": VEGA_SHARPA_JOINT.contract_id,
        "embodiment_contract_sha256": VEGA_SHARPA_JOINT.sha256,
        "fps": VEGA_SHARPA_JOINT.fps,
        "joint_names": list(VEGA_SHARPA_JOINT.joint_names),
        "object_names": ["generated_object"],
        "timeout_termination": "timeout",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for episode in episodes:
        np.savez_compressed(
            root / f"episode_{episode.index:06d}.npz",
            joint_pos=episode.actions,
            action_target=episode.actions,
            object_pose=episode.object_pose,
            source_success=np.asarray(episode.source_success),
            termination_reasons=np.asarray(
                ["timeout"] if episode.source_success else ["robot_state_diverged"]
            ),
        )
    return root


def _write_recording(path: Path, episodes: tuple[GeneratedEpisode, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        data.attrs.update({**_provenance(), "fps": VEGA_SHARPA_JOINT.fps})
        for demo_index, episode in enumerate(episodes):
            demo = data.create_group(f"demo_{demo_index}")
            obs = demo.create_group("obs")
            obs.create_dataset("arm_joint_pos", data=episode.arm)
            obs.create_dataset("finger_joint_pos", data=episode.finger)
            for term, images in episode.cameras.items():
                obs.create_dataset(term, data=images)
            demo.create_dataset("actions", data=episode.actions)
    return path


def _run_module(
    *args: str | Path, check: bool = True
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROBOTIC_ROOT)
    return subprocess.run(
        (sys.executable, *map(str, args)),
        cwd=ROBOTIC_ROOT,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )


def _run_audit(root: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run_module(
        "-m",
        "groot_finetune.tools.audit_groot_run",
        "--root",
        root,
        "--episodes",
        "2",
        "--frames",
        str(FRAMES),
        "--contract",
        "vega_sharpa_joint",
        "--task-profile",
        root / "task_profile.json",
        "--selected-export",
        "selected",
        "--hdf5",
        "recording/data.h5",
        "--dataset",
        "gr00t_dataset",
        check=check,
    )


def test_released_onnx_contract_and_batching() -> None:
    """The committed expert must be materialized and serve the released action width."""
    prefix = POLICY.read_bytes()[:64]
    assert not prefix.startswith(b"version https://git-lfs.github.com/spec/v1")
    assert POLICY.stat().st_size > 1_000_000

    session = ort.InferenceSession(str(POLICY), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    assert len(inputs) == 1
    assert inputs[0].type == "tensor(float)"
    assert len(inputs[0].shape) == 2
    assert inputs[0].shape[-1] == 445
    action_output = next(
        (output for output in outputs if output.name == "actions"), outputs[0]
    )
    assert action_output.type == "tensor(float)"
    assert len(action_output.shape) == 2
    assert action_output.shape[-1] == VEGA_SHARPA_JOINT.action_dim

    policy = OnnxSourcePolicy(session, device="cpu")
    rng = np.random.default_rng(20260818)
    observations = torch.from_numpy(
        rng.uniform(-0.05, 0.05, size=(3, 445)).astype(np.float32)
    )
    first = policy({"policy": observations})
    second = policy({"policy": observations})
    assert first.shape == (3, VEGA_SHARPA_JOINT.action_dim)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


def test_generated_artifact_chain(tmp_path: Path) -> None:
    """Procedural episodes survive selection, conversion, and exact auditing."""
    episodes = (
        _episode(0, frames=FRAMES, source_success=True),
        _episode(1, frames=7, source_success=False),
        _episode(2, frames=FRAMES, source_success=True),
    )
    source = _write_source(tmp_path / "source", episodes)
    _run_module(
        "-m",
        "groot_finetune.tools.select_successful_episodes",
        "--input",
        source,
        "--output",
        tmp_path / "selected",
        "--target",
        "2",
        "--expected-frames",
        str(FRAMES),
    )
    selected_manifest = json.loads(
        (tmp_path / "selected/manifest.json").read_text(encoding="utf-8")
    )
    assert selected_manifest["source_successful_episode_count"] == 2
    assert [Path(item["source"]).name for item in selected_manifest["selected"]] == [
        "episode_000000.npz",
        "episode_000002.npz",
    ]

    selected_episodes = (episodes[0], episodes[2])
    recording = _write_recording(tmp_path / "recording/data.h5", selected_episodes)
    profile_path = tmp_path / "task_profile.json"
    write_task_profile(profile_path, PROFILE)
    dataset = tmp_path / "gr00t_dataset"
    convert(recording, dataset, VEGA_SHARPA_JOINT, PROFILE)
    stats = {
        "observation.state": {"mean": [0.0] * VEGA_SHARPA_JOINT.state_dim},
        "action": {"mean": [0.0] * VEGA_SHARPA_JOINT.action_dim},
    }
    stats_path = dataset / "meta/stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    audit = json.loads(_run_audit(tmp_path).stdout)
    assert audit["ok"]
    assert audit["errors"] == []

    parquet_paths = sorted((dataset / "data").rglob("*.parquet"))
    video_paths = sorted((dataset / "videos").rglob("*.mp4"))
    assert len(parquet_paths) == 2
    assert len(video_paths) == 6
    global_index = 0
    for episode_index, (path, episode) in enumerate(
        zip(parquet_paths, selected_episodes, strict=True)
    ):
        table = pq.read_table(path).to_pydict()
        np.testing.assert_allclose(
            np.asarray(table["observation.state"], dtype=np.float32), episode.state
        )
        np.testing.assert_allclose(
            np.asarray(table["action"], dtype=np.float32), episode.actions
        )
        np.testing.assert_allclose(
            table["timestamp"],
            np.arange(FRAMES, dtype=np.float32) / VEGA_SHARPA_JOINT.fps,
        )
        assert table["episode_index"] == [episode_index] * FRAMES
        assert table["index"] == list(range(global_index, global_index + FRAMES))
        assert table["next.done"] == [False] * (FRAMES - 1) + [True]
        assert table["task_index"] == [0] * FRAMES
        global_index += FRAMES

    original_stats = stats_path.read_text(encoding="utf-8")
    stats_path.write_text('{"mean": NaN}', encoding="utf-8")
    failed_stats = json.loads(_run_audit(tmp_path, check=False).stdout)
    assert not failed_stats["ok"]
    assert "non-finite" in "\n".join(failed_stats["errors"])
    stats_path.write_text(original_stats, encoding="utf-8")

    first_video = video_paths[0]
    images_to_video(
        selected_episodes[0].cameras[VEGA_SHARPA_JOINT.cameras[0].observation_term][
            :-1
        ],
        first_video,
        VEGA_SHARPA_JOINT.fps,
    )
    failed_video = json.loads(_run_audit(tmp_path, check=False).stdout)
    assert not failed_video["ok"]
    assert "encoded frames" in "\n".join(failed_video["errors"])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("missing_camera", "missing required camera"),
        ("nonfinite_action", "finite numeric values"),
        ("wrong_action_width", "expected (17, 58)"),
        ("wrong_state_width", "expected (17, 14)"),
        ("wrong_camera_dtype", "expected uint8"),
        ("wrong_camera_resolution", "resolution"),
        ("scalar_observation", "got None frames"),
    ),
)
def test_hdf5_audit_rejects_semantic_corruption(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    """Every corruption case starts from a fresh generated recording."""
    recording = _write_recording(
        tmp_path / "recording.h5",
        (_episode(0, frames=FRAMES, source_success=True),),
    )
    with h5py.File(recording, "a") as handle:
        demo = handle["data/demo_0"]
        if mutation == "missing_camera":
            del demo["obs/image_left_wrist"]
        elif mutation == "nonfinite_action":
            demo["actions"][0, 0] = np.nan
        elif mutation == "wrong_action_width":
            values = np.asarray(demo["actions"])[:, :-1]
            del demo["actions"]
            demo.create_dataset("actions", data=values)
        elif mutation == "wrong_state_width":
            values = np.asarray(demo["obs/arm_joint_pos"])[:, :-1]
            del demo["obs/arm_joint_pos"]
            demo["obs"].create_dataset("arm_joint_pos", data=values)
        elif mutation == "wrong_camera_dtype":
            values = np.asarray(demo["obs/image"], dtype=np.float32)
            del demo["obs/image"]
            demo["obs"].create_dataset("image", data=values)
        elif mutation == "wrong_camera_resolution":
            values = np.asarray(demo["obs/image_left_wrist"])[:, :, :-2, :]
            del demo["obs/image_left_wrist"]
            demo["obs"].create_dataset("image_left_wrist", data=values)
        elif mutation == "scalar_observation":
            demo["obs"].create_dataset("debug_scalar", data=np.float32(1.0))
        else:  # pragma: no cover - parametrization controls this value
            raise AssertionError(mutation)

    audit = Audit()
    audit_hdf5(
        audit,
        path=recording,
        episodes=1,
        frames=FRAMES,
        contract=VEGA_SHARPA_JOINT,
        expected_provenance=_provenance(),
    )
    assert expected_error in "\n".join(audit.errors)


def test_selector_rejects_timeout_marker_mismatch(tmp_path: Path) -> None:
    episode = _episode(0, frames=FRAMES, source_success=True)
    source = _write_source(tmp_path / "source", (episode,))
    path = source / "episode_000000.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["source_success"] = np.asarray(False)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="does not match timeout termination"):
        inspect_episode(path, FRAMES, "timeout")
