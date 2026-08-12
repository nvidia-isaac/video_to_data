# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for the `motion_v1` parquet pipeline.

Writes a motion_v1 parquet, loads it via SceneConfig + the unified reader,
and asserts the usual round-trip invariants.

Usage (pytest):
    pytest tests/test_motion_schema_parquet_integration.py

Usage (direct):
    python tests/test_motion_schema_parquet_integration.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path

from robotic_grounding.motion_schema import (
    SCHEMA_VERSION,
    MotionData,
    load_motion_data_parquet,
    save_motion_parquet,
)

try:
    from robotic_grounding.tasks.scene_utils.scene_config import SceneConfig
except ModuleNotFoundError:
    # Fallback for environments without IsaacLab/Omniverse modules where
    # importing robotic_grounding.tasks triggers heavy runtime dependencies.
    scene_config_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "robotic_grounding"
        / "robotic_grounding"
        / "tasks"
        / "scene_utils"
        / "scene_config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scene_config_fallback", scene_config_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load SceneConfig from {scene_config_path}"
        ) from None
    scene_config_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scene_config_module
    spec.loader.exec_module(scene_config_module)
    SceneConfig = scene_config_module.SceneConfig


def _write_test_mesh(mesh_path: Path) -> None:
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text(
        "\n".join(
            [
                "o object",
                "v 0.0 0.0 0.0",
                "v 0.1 0.0 0.0",
                "v 0.0 0.1 0.0",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )


def _write_test_urdf(urdf_path: Path, mesh_path: Path) -> None:
    urdf_path.write_text(
        f"""<?xml version="1.0"?>
<robot name="test_object">
  <link name="object">
    <visual>
      <geometry>
        <mesh filename="{mesh_path.resolve()}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{mesh_path.resolve()}"/>
      </geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )


def test_motion_v1_parquet_roundtrip_scene_config_and_loader(tmp_path: Path) -> None:
    """Write motion_v1 parquet, load via SceneConfig + unified reader."""
    mesh_path = tmp_path / "object" / "textured_mesh.obj"
    urdf_path = tmp_path / "object" / "textured_mesh.urdf"
    _write_test_mesh(mesh_path)
    _write_test_urdf(urdf_path, mesh_path)

    sequence_id = "seq_test_001"
    robot_name = "g1"
    t = 3
    md = MotionData(
        sequence_id=sequence_id,
        robot_name=robot_name,
        motion_kind="single_robot",
        source_dataset="soma",
        raw_motion_file=str(tmp_path / "nova_params_opt.pt"),
        fps=30.0,
        coord_frame="robot_base_z_up",
        robot_joint_names=["left_knee_joint", "right_knee_joint"],
        robot_root_position=[[0.0, 0.0, 0.8] for _ in range(t)],
        robot_root_wxyz=[[1.0, 0.0, 0.0, 0.0] for _ in range(t)],
        robot_joint_positions=[[0.1, -0.1] for _ in range(t)],
        ee_link_names=["left_hand_palm_link", "right_hand_palm_link"],
        ee_pose_w=[
            [
                [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
            ]
            for _ in range(t)
        ],
        object_name="seq_test_object",
        safe_object_name="seq_test_object",
        object_body_names=["object"],
        safe_object_body_names=["object"],
        object_mesh_paths=[str(mesh_path.resolve())],
        object_urdf_paths=[str(urdf_path.resolve())],
        object_mesh_radius=[0.1],
        object_articulation=[0.0 for _ in range(t)],
        object_root_axis_angle=[[0.0, 0.0, 0.0] for _ in range(t)],
        object_root_position=[[0.2, 0.3, 0.4] for _ in range(t)],
        object_body_position=[[[0.2, 0.3, 0.4]] for _ in range(t)],
        object_body_wxyz=[[[1.0, 0.0, 0.0, 0.0]] for _ in range(t)],
    )

    output_dir = tmp_path / "motion_v1_processed"
    save_motion_parquet(md, root_path=str(output_dir))

    # Round-trip via the unified reader.
    loaded = load_motion_data_parquet(
        str(output_dir / f"sequence_id={sequence_id}" / f"robot_name={robot_name}")
    )
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.object_body_names == ["object"]
    assert loaded.object_mesh_paths == [str(mesh_path.resolve())]
    assert loaded.object_urdf_paths == [str(urdf_path.resolve())]
    assert loaded.robot_joint_names == md.robot_joint_names
    assert tuple(loaded.robot_root_position.shape) == (t, 3)
    assert tuple(loaded.robot_joint_positions.shape) == (t, 2)

    # SceneConfig must still build from the same partition directory.
    partition_dir = (
        output_dir / f"sequence_id={sequence_id}" / f"robot_name={robot_name}"
    )
    scene_cfg = SceneConfig.from_motion_file(str(partition_dir))
    assert scene_cfg.object_body_names == ["object"]
    assert len(scene_cfg.scene_objects) == 1
    assert getattr(scene_cfg.scene_objects[0], "name", None) == "object"
    assert getattr(scene_cfg.scene_objects[0], "usd_path", None) == str(
        urdf_path.resolve()
    )


def _run_as_script() -> int:
    """Run this test file directly without pytest."""
    test_name = "test_motion_v1_parquet_roundtrip_scene_config_and_loader"
    print("=" * 72, flush=True)
    print("Running motion_v1 parquet integration test", flush=True)
    print(f"Test: {test_name}", flush=True)
    print("=" * 72, flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="motion_v1_test_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            print(f"[SETUP] Temporary directory: {tmp_path}", flush=True)
            test_motion_v1_parquet_roundtrip_scene_config_and_loader(tmp_path)
        print(f"[PASS] {test_name}", flush=True)
        return 0
    except Exception:
        print(f"[FAIL] {test_name}", flush=True)
        print("-" * 72, flush=True)
        traceback.print_exc()
        print("-" * 72, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
