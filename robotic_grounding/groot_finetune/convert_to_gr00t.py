# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a contract-matched semantic HDF5 recording to LeRobot for GR00T."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from groot_finetune.closed_loop.embodiment import Q_REF_LEFT_WRIST, Q_REF_RIGHT_WRIST
from groot_finetune.contracts import (
    EmbodimentContract,
    FieldSpec,
    load_embodiment_contract,
)
from groot_finetune.task_profile import TaskProfile, load_task_profile

CHUNK_SIZE = 1000
QUAT_REF_MARGIN = 0.15


def _canonicalize_quat(
    arr: np.ndarray, start: int, q_ref: np.ndarray | None = None
) -> None:
    """Choose a continuous quaternion hemisphere in-place for one trajectory."""
    q = arr[:, start : start + 4]
    step_sign = np.sign(np.einsum("td,td->t", q[1:], q[:-1]))
    step_sign[step_sign == 0.0] = 1.0
    sign = np.concatenate([[1.0], np.cumprod(step_sign)])
    seed = (
        float(q[0] @ q_ref)
        if q_ref is not None
        else float(q[0, np.argmax(np.abs(q[0]))])
    )
    if seed < 0.0:
        sign = -sign
    q *= sign[:, None].astype(q.dtype)
    if q_ref is not None:
        margin = float((q @ q_ref).min())
        if margin < QUAT_REF_MARGIN:
            raise ValueError(
                "quaternion trajectory approaches the configured reference-hemisphere "
                f"boundary (minimum dot={margin:.3f})"
            )


def _transform_field(values: np.ndarray, field: FieldSpec) -> np.ndarray:
    result = np.asarray(values[:, field.start : field.end], dtype=np.float32).copy()
    if field.transform == "quaternion_right_ref":
        _canonicalize_quat(result, 0, Q_REF_RIGHT_WRIST)
    elif field.transform == "quaternion_left_ref":
        _canonicalize_quat(result, 0, Q_REF_LEFT_WRIST)
    return result


def _source_array(demo: h5py.Group, field: FieldSpec) -> np.ndarray:
    if field.source_term == "action_target":
        return np.asarray(demo["actions"])
    return np.asarray(demo["obs"][field.source_term])


def _assemble(demo: h5py.Group, fields: tuple[FieldSpec, ...]) -> np.ndarray:
    return np.concatenate(
        [_transform_field(_source_array(demo, field), field) for field in fields],
        axis=1,
    )


def build_modality_config(
    contract: EmbodimentContract,
) -> dict[str, Any]:
    """Build modality slices from the exact ordered embodiment contract."""

    def slices(fields: tuple[FieldSpec, ...]) -> dict[str, dict[str, int]]:
        offset = 0
        result: dict[str, dict[str, int]] = {}
        for field in fields:
            result[field.key] = {"start": offset, "end": offset + field.width}
            offset += field.width
        return result

    return {
        "state": slices(contract.state_fields),
        "action": slices(contract.action_fields),
        "video": {
            camera.key: {"original_key": f"observation.images.{camera.key}"}
            for camera in contract.cameras
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def _validate_provenance(
    data: h5py.Group,
    contract: EmbodimentContract,
    task_profile: TaskProfile,
) -> None:
    expected = {
        "schema_version": 1,
        "embodiment_contract": contract.contract_id,
        "embodiment_contract_sha256": contract.sha256,
        "task_profile": task_profile.task_id,
        "task_profile_sha256": task_profile.sha256,
        "fps": contract.fps,
    }
    mismatches = {
        name: {"expected": value, "actual": data.attrs.get(name)}
        for name, value in expected.items()
        if data.attrs.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"recording provenance does not match the requested contracts: {mismatches}"
        )


def _demo_names(data: h5py.Group) -> list[str]:
    names = sorted(
        (name for name in data if name.startswith("demo_")),
        key=lambda name: int(name.split("_")[1]),
    )
    if not names:
        raise ValueError("recording contains no demo_* groups")
    return names


def _validate_recording(
    data: h5py.Group,
    demo_names: list[str],
    contract: EmbodimentContract,
) -> tuple[int, int]:
    """Validate exact state, action, camera, dtype, and timing contracts."""
    common_hw: tuple[int, int] | None = None
    for name in demo_names:
        demo = data[name]
        if "obs" not in demo or "actions" not in demo:
            raise ValueError(f"{name}: expected obs and actions")
        actions = demo["actions"]
        if actions.ndim != 2 or actions.shape[1] != contract.action_dim:
            raise ValueError(
                f"{name}/actions must be (T, {contract.action_dim}), got {actions.shape}"
            )
        if (
            not np.issubdtype(actions.dtype, np.number)
            or not np.isfinite(actions[...]).all()
        ):
            raise ValueError(f"{name}/actions must contain finite numeric values")
        frames = int(actions.shape[0])
        if frames <= 0:
            raise ValueError(f"{name}: empty episode")
        required_terms: dict[str, int] = {}
        for field in contract.state_fields:
            required_terms[field.source_term] = max(
                required_terms.get(field.source_term, 0), field.end
            )
        for term, minimum_width in required_terms.items():
            if term not in demo["obs"]:
                raise ValueError(f"{name}: missing state observation {term!r}")
            values = demo["obs"][term]
            if values.ndim != 2 or values.shape != (frames, minimum_width):
                raise ValueError(
                    f"{name}/{term} must be {(frames, minimum_width)}, got {values.shape}"
                )
            if (
                not np.issubdtype(values.dtype, np.number)
                or not np.isfinite(values[...]).all()
            ):
                raise ValueError(f"{name}/{term} must contain finite numeric values")
        for camera in contract.cameras:
            if camera.observation_term not in demo["obs"]:
                raise ValueError(
                    f"{name}: missing required camera {camera.observation_term!r}"
                )
            values = demo["obs"][camera.observation_term]
            if values.ndim != 4 or values.shape[0] != frames or values.shape[-1] != 3:
                raise ValueError(
                    f"{name}/{camera.observation_term} must be (T, H, W, 3), got {values.shape}"
                )
            if values.dtype != np.uint8:
                raise ValueError(
                    f"{name}/{camera.observation_term} must be uint8 RGB, got {values.dtype}"
                )
            hw = (int(values.shape[1]), int(values.shape[2]))
            if common_hw is None:
                common_hw = hw
            elif common_hw != hw:
                raise ValueError(
                    f"{name}/{camera.observation_term} resolution {hw} != {common_hw}"
                )
    assert common_hw is not None
    return common_hw


def images_to_video(images: np.ndarray, output_path: Path, fps: int) -> None:
    """Write RGB uint8 frames to an MP4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = images.shape[1:3]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {output_path}")
    for image in images:
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    writer.release()


def convert(
    input_path: str | Path,
    output_path: str | Path,
    contract_value: str | Path | EmbodimentContract,
    task_profile_value: str | Path | TaskProfile,
) -> None:
    """Convert a recording only when every serialized contract matches exactly."""
    contract = (
        contract_value
        if isinstance(contract_value, EmbodimentContract)
        else load_embodiment_contract(contract_value)
    )
    task_profile = (
        task_profile_value
        if isinstance(task_profile_value, TaskProfile)
        else load_task_profile(task_profile_value)
    )
    input_path = Path(input_path)
    output_path = Path(output_path)
    meta_dir = output_path / "meta"
    modality = build_modality_config(contract)

    with h5py.File(input_path, "r") as recording:
        if "data" not in recording:
            raise ValueError("recording is missing the data group")
        data = recording["data"]
        _validate_provenance(data, contract, task_profile)
        demo_names = _demo_names(data)
        video_hw = _validate_recording(data, demo_names, contract)
        meta_dir.mkdir(parents=True, exist_ok=True)

        episodes_info: list[dict[str, Any]] = []
        global_index = 0
        for episode_index, name in enumerate(demo_names):
            demo = data[name]
            state = _assemble(demo, contract.state_fields)
            actions = _assemble(demo, contract.action_fields)
            if state.shape != (len(actions), contract.state_dim):
                raise ValueError(
                    f"{name}: assembled state shape {state.shape} does not match ({len(actions)}, {contract.state_dim})"
                )
            records = [
                {
                    "observation.state": state[index].tolist(),
                    "action": actions[index].tolist(),
                    "timestamp": float(index) / contract.fps,
                    "task_index": 0,
                    "episode_index": episode_index,
                    "index": global_index + index,
                    "next.reward": 0.0,
                    "next.done": index == len(state) - 1,
                }
                for index in range(len(state))
            ]
            chunk = episode_index // CHUNK_SIZE
            data_dir = output_path / "data" / f"chunk-{chunk:03d}"
            data_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(records),
                data_dir / f"episode_{episode_index:06d}.parquet",
            )
            for camera in contract.cameras:
                path = (
                    output_path
                    / "videos"
                    / f"chunk-{chunk:03d}"
                    / f"observation.images.{camera.key}"
                    / f"episode_{episode_index:06d}.mp4"
                )
                images_to_video(
                    np.asarray(demo["obs"][camera.observation_term]),
                    path,
                    contract.fps,
                )
            episodes_info.append(
                {
                    "episode_index": episode_index,
                    "tasks": [task_profile.instruction],
                    "length": len(state),
                }
            )
            global_index += len(state)

    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task_profile.instruction}) + "\n",
        encoding="utf-8",
    )
    (meta_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes_info),
        encoding="utf-8",
    )
    (meta_dir / "modality.json").write_text(
        json.dumps(modality, indent=2) + "\n", encoding="utf-8"
    )

    height, width = video_hw
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [contract.state_dim],
            "names": [field.key for field in contract.state_fields],
        },
        "action": {
            "dtype": "float32",
            "shape": [contract.action_dim],
            "names": [field.key for field in contract.action_fields],
        },
        "timestamp": {"dtype": "float32", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
    }
    for camera in contract.cameras:
        features[f"observation.images.{camera.key}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(contract.fps),
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "mp4v",
                "video.pix_fmt": "rgb24",
                "has_audio": False,
            },
        }
    episode_count = len(episodes_info)
    info = {
        "codebase_version": "v2.1",
        "robot_type": contract.contract_id,
        "schema_version": 1,
        "embodiment_contract": contract.contract_id,
        "embodiment_contract_sha256": contract.sha256,
        "task_profile": task_profile.task_id,
        "task_profile_sha256": task_profile.sha256,
        "total_episodes": episode_count,
        "total_frames": global_index,
        "total_tasks": 1,
        "total_videos": episode_count * len(contract.cameras),
        "total_chunks": (episode_count + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": contract.fps,
        "splits": {"train": f"0:{episode_count}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Parse command-line contracts and convert one recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--task-profile", required=True)
    args = parser.parse_args()
    convert(args.input, args.output, args.contract, args.task_profile)


if __name__ == "__main__":
    main()
