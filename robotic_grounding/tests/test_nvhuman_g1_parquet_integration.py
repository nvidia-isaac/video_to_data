import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path

from robotic_grounding.retarget.data_logger import NvhumanG1Data
from robotic_grounding.retarget.params import G1_WHOLEBODY_TO_NVHUMAN_MAPPING

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
    # Register module before execution so decorators (e.g., @dataclass) can
    # resolve cls.__module__ via sys.modules during import-time processing.
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


def test_nvhuman_g1_parquet_roundtrip_scene_config_and_loader(tmp_path: Path) -> None:
    """Write NvhumanG1 parquet, then load via scene config and data loader API."""
    mesh_path = tmp_path / "object" / "textured_mesh.obj"
    urdf_path = tmp_path / "object" / "textured_mesh.urdf"
    _write_test_mesh(mesh_path)
    _write_test_urdf(urdf_path, mesh_path)

    sequence_id = "seq_test_001"
    robot_name = "g1"
    frame_task_errors = [0.0] * len(G1_WHOLEBODY_TO_NVHUMAN_MAPPING)

    logger_data = NvhumanG1Data(
        sequence_id=sequence_id,
        raw_motion_file=str(tmp_path / "nova_params_opt.pt"),
        robot_name=robot_name,
        fps=30.0,
        nvhuman_betas=[0.0] * 10,
        robot_joint_names=["left_knee_joint", "right_knee_joint"],
        robot_frame_names=["pelvis"],
        robot_frame_task_names=list(G1_WHOLEBODY_TO_NVHUMAN_MAPPING.keys()),
        source_to_robot_scale=1.0,
        object_name="seq_test_object",
        safe_object_name="seq_test_object",
        object_body_names=["object"],
        safe_object_body_names=["object"],
        object_mesh_paths=[str(mesh_path.resolve())],
        object_urdf_paths=[str(urdf_path.resolve())],
        object_mesh_radius=[0.1],
    )

    logger_data.log_timestep(
        nvhuman_joints=[[0.0, 0.0, 0.0] for _ in range(93)],
        nvhuman_joints_wxyz=[[1.0, 0.0, 0.0, 0.0] for _ in range(93)],
        nvhuman_head_translation=[0.0, 0.0, 1.0],
        nvhuman_head_wxyz=[1.0, 0.0, 0.0, 0.0],
        nvhuman_root_translation=[0.0, 0.0, 0.8],
        nvhuman_root_wxyz=[1.0, 0.0, 0.0, 0.0],
        robot_root_position=[0.0, 0.0, 0.8],
        robot_root_wxyz=[1.0, 0.0, 0.0, 0.0],
        robot_joint_positions=[0.1, -0.1],
        robot_frames=[[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]],
        robot_frame_task_errors=frame_task_errors,
        robot_ik_error=0.0,
        robot_num_optimization_iterations=1,
        object_articulation=0.0,
        object_root_axis_angle=[0.0, 0.0, 0.0],
        object_root_position=[0.2, 0.3, 0.4],
        object_body_position=[[0.2, 0.3, 0.4]],
        object_body_wxyz=[[1.0, 0.0, 0.0, 0.0]],
    )

    output_dir = tmp_path / "nvhuman_g1_processed"
    logger_data.save_to_parquet(
        root_path=str(output_dir),
        partition_cols=["sequence_id", "robot_name"],
    )

    loaded = NvhumanG1Data.from_parquet(
        root_path=str(output_dir),
        filters=[("sequence_id", "=", sequence_id), ("robot_name", "=", robot_name)],
    )
    assert loaded.object_body_names == ["object"]
    assert loaded.object_mesh_paths == [str(mesh_path.resolve())]
    assert loaded.object_urdf_paths == [str(urdf_path.resolve())]
    assert len(loaded.robot_joint_names) == len(loaded.robot_joint_positions[0])

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
    test_name = "test_nvhuman_g1_parquet_roundtrip_scene_config_and_loader"
    print("=" * 72, flush=True)
    print("Running NVHuman->G1 parquet integration test", flush=True)
    print(f"Test: {test_name}", flush=True)
    print("=" * 72, flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="nvhuman_g1_test_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            print(f"[SETUP] Temporary directory: {tmp_path}", flush=True)
            test_nvhuman_g1_parquet_roundtrip_scene_config_and_loader(tmp_path)
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
