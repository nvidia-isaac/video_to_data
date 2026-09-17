import ast
import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1]
SOURCE = MODULE / "lib/cari4d"


def _resolve_project_module(name: str) -> Path | None:
    candidate = SOURCE.joinpath(*name.split("."))
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    return None


def _top_level_project_imports(path: Path) -> set[Path]:
    imports = set()
    tree = ast.parse(path.read_text())
    for node in tree.body:
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        for name in names:
            resolved = _resolve_project_module(name)
            if resolved is not None:
                imports.add(resolved)
    return imports


def test_native_mhr_trainer_has_complete_top_level_project_import_closure():
    pending = [SOURCE / "learning/training/trainer.py", SOURCE / "learning/datasets/mhr_video_data.py", SOURCE / "learning/datasets/video_data.py", SOURCE / "scripts/build_mhr_dataset_index.py"]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        assert path.is_file(), path
        visited.add(path)
        pending.extend(_top_level_project_imports(path) - visited)
    required = {
        "learning/datasets/mhr_dataset_index.py",
        "learning/datasets/mhr_input_materialization.py",
        "learning/datasets/mhr_rank_local.py",
        "learning/datasets/mhr_tier_sampling.py",
        "learning/datasets/mhr_video_data.py",
        "learning/datasets/mhr_window_sampling.py",
        "learning/training/checkpoint_recovery.py",
        "learning/training/distributed_checkpoint.py",
        "learning/training/mhr_input_grid.py",
        "learning/training/mhr_object_pose_loss.py",
        "learning/training/nonfinite_loss.py",
        "learning/training/resume_state.py",
        "learning/training/runtime_optimization.py",
        "learning/training/source_snapshot.py",
        "learning/training/training_utils.py",
        "prep/mhr_foundationpose_training_tiers.py",
        "prep/mhr_packed_h5.py",
        "prep/mhr_render_shards.py",
        "render_h5_codec.py",
    }
    assert required <= {str(path.relative_to(SOURCE)) for path in visited}


def test_training_grid_uses_the_shared_minimal_mesh_renderer():
    source = (SOURCE / "learning/training/trainer.py").read_text()
    assert "from tools.mhr_mesh_renderer import NvdiffMeshRenderer" in source
    assert "build_mhr_input_grid" in source
    assert "tools.mhr_forward_viz" not in source


def test_commercial_moge2_training_identity_and_splits_are_pinned():
    config = (SOURCE / "learning/configs/mhr-daniel-commercial-moge2-behave79-val-fp16.yml").read_text()
    assert "expected_train_sequence_count: 2126" in config
    assert "mhr_xyz_anchor_type: root_joint_1" in config
    assert "mhr_spatial_normalization_type: human_height_2m" in config
    assert "mhr_interaction_trim_root:" in config and "data_export_3" in config
    assert "mhr_effective_mask_root:" in config
    assert "mhr_min_canonical_object_nonempty_frame_fraction: 0.10" in config
    assert "mhr_required_contact_revision: mhr-hand-surface-contact-v1" in config
    split = json.loads((SOURCE / "splits/daniel-hoi-train-commercial-moge2-2126.json").read_text())
    assert len(split["train"]) == len(set(split["train"])) == 2126
    excluded = {"2026-05-06_10-26-25_hatchet_rotate_08", "2026-04-04_13-15-04_potato_light_rotate_03", "2026-04-24_14-54-16_basketball_ball_pass_ankles_09"}
    assert excluded.isdisjoint(split["train"])
    assert len(json.loads((SOURCE / "splits/behave-date03-79.json").read_text())["test"]) == 79


def test_training_launcher_builds_an_accelerate_command(tmp_path):
    spec = importlib.util.spec_from_file_location("v2d_cari4d_run_training", MODULE / "lib/run_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "config.yml"
    config.write_text("body_model: mhr\n")
    command = module.build_training_command(config, 8, "2026-08-18-20-00-00", "2026-08-18-20-00-00", ("no_wandb=true",))
    assert command[1:5] == ["-m", "accelerate.commands.launch", "--num_processes", "8"]
    assert command[-4:] == [f"config={config.resolve()}", "exp_name=2026-08-18-20-00-00", "run_id=2026-08-18-20-00-00", "no_wandb=true"]
