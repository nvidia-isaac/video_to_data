# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-safe source contracts; run the full suite with v2d.cari4d.docker.run_tests."""

import ast
import subprocess
import sys
from pathlib import Path

from ._contract_helpers import MODULE, _function, _parameter_defaults


def test_host_wrapper_exposes_every_pipeline_parameter():
    library = _function(MODULE / "lib/run_inference.py", "run_inference")
    docker = _function(MODULE / "docker/run_inference.py", "run_inference")
    library_names = [value.arg for value in library.args.args + library.args.kwonlyargs]
    docker_names = [value.arg for value in docker.args.args + docker.args.kwonlyargs]
    assert docker_names == library_names + ["dev"]
    library_defaults = _parameter_defaults(library)
    docker_defaults = _parameter_defaults(docker)
    assert {key: value for key, value in docker_defaults.items() if key != "dev"} == library_defaults


def test_production_defaults_and_offline_checkpoint_contract():
    library = _function(MODULE / "lib/run_inference.py", "run_inference")
    defaults = _parameter_defaults(library)
    assert defaults["moge_batch_size"] == 8
    assert defaults["postopt_num_steps"] == 300
    assert defaults["postopt_batch_size"] == 0
    assert defaults["postopt_temporal_weight"] == 100.0
    assert defaults["postopt_human_pose_prior_weight"] == 200.0
    assert defaults["postopt_contact_activation_distance_m"] == 0.05
    source = (MODULE / "lib/run_inference.py").read_text()
    assert "--offline-supervision-contract" in source
    assert "--wandb-run-path" not in source
    assert '"sam2_bbox_mask"' in source
    assert '"--register-first-then-track"' in source
    assert '"--contact-activation-distance-m"' in source
    depth_backend = (MODULE / "lib/cari4d/prep/mhr_depth_backend.py").read_text()
    assert 'DEFAULT_MONOCULAR_DEPTH_BACKEND = "moge2"' in depth_backend


def test_foundationpose_debug_directory_defaults_to_output_tree(tmp_path):
    path = MODULE / "lib/cari4d/prep/run_foundationpose_mhr_export.py"
    source = path.read_text()
    resolver = _function(path, "_resolve_foundationpose_debug_dir")
    namespace = {"Path": Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[resolver], type_ignores=[])), str(path), "exec"), namespace)
    out_file = tmp_path / "foundationpose" / "sequence.pkl"
    assert namespace["_resolve_foundationpose_debug_dir"](out_file, None) == out_file.parent / "debug"
    assert namespace["_resolve_foundationpose_debug_dir"](out_file, tmp_path / "explicit") == tmp_path / "explicit"
    assert "debug_dir = _resolve_foundationpose_debug_dir(out_file, debug_dir)" in source
    assert 'parser.add_argument("--debug-dir", default=None)' in source
    assert "/mnt/data/xianghuix" not in source


def test_render_stage_outputs_final_and_three_stage_videos():
    pipeline_source = (MODULE / "lib/run_inference.py").read_text()
    renderer_path = MODULE / "lib/cari4d/tools/render_mhr_wild_inference.py"
    renderer_source = renderer_path.read_text()
    renderer = _function(renderer_path, "render_mhr_wild_inference")
    renderer_names = [value.arg for value in renderer.args.args + renderer.args.kwonlyargs]
    assert "before_bundle" in renderer_names
    assert "comparison_output" in renderer_names
    assert "--before-bundle" in pipeline_source
    assert "--comparison-output" in pipeline_source
    assert "_step200000_reconstruction.mp4" in pipeline_source
    assert "_step200000_before_after.mp4" in pipeline_source
    assert '"comparison_video": _file_identity(comparison_video)' in pipeline_source
    assert 'HUMAN_ALBEDO_HEX = "#C8D2D8"' in renderer_source
    assert 'MESH_BACKGROUND_HEX = "#202428"' in renderer_source
    assert "render_front_batch_constant_human_textured_object" in renderer_source
    assert "Initialization HOI" in renderer_source
    assert "CoCoNet prediction" in renderer_source
    assert "Contact-guided refinement" in renderer_source
    assert "frame ID" in renderer_source


def test_mask_packer_host_wrapper_exposes_every_parameter():
    library = _function(MODULE / "lib/pack_masks.py", "masks_pack_cari4d_h5")
    docker = _function(MODULE / "docker/run_pack_masks.py", "run_pack_masks")
    library_names = [value.arg for value in library.args.args + library.args.kwonlyargs]
    docker_names = [value.arg for value in docker.args.args + docker.args.kwonlyargs]
    assert docker_names == library_names + ["dev"]
    assert {key: value for key, value in _parameter_defaults(docker).items() if key != "dev"} == _parameter_defaults(library)


def test_exact_checkpoint_identity_is_pinned():
    source = (MODULE / "lib/download_weights.py").read_text()
    assert 'CARI4D_RUN_ID = "2026-08-25-09-35-57"' in source
    assert 'CARI4D_REVISION = "1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9"' in source
    assert 'CARI4D_CHECKPOINT_SHA256 = "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3"' in source
    assert 'MOGE2_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"' in source
    assert 'SAM3D_REVISION = "11aaa346c7204874a1cbafe3d39a979080b2c55a"' in source


def test_moge2_download_and_offline_preflight_share_runtime_cache(tmp_path):
    downloader_path = MODULE / "lib/download_weights.py"
    functions = {node.name: node for node in ast.parse(downloader_path.read_text()).body if isinstance(node, ast.FunctionDef)}
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    calls = []
    namespace = {"Path": Path, "MOGE2_REPO_ID": "Ruicheng/moge-2-vitl-normal", "MOGE2_REVISION": "revision", "MOGE2_MODEL_FILENAME": "model.pt", "snapshot_download": lambda **kwargs: calls.append(("snapshot", kwargs)), "hf_hub_download": lambda **kwargs: calls.append(("model", kwargs)) or str(model_path)}
    for name in ("_moge2_cache_dir", "require_moge2_model", "_download_moge2"):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[functions[name]], type_ignores=[])), str(downloader_path), "exec"), namespace)
    namespace["_download_moge2"](tmp_path)
    expected_cache = tmp_path / "hf_home" / "hub"
    assert calls[0] == ("snapshot", {"repo_id": "Ruicheng/moge-2-vitl-normal", "revision": "revision", "cache_dir": expected_cache})
    assert calls[1] == ("model", {"repo_id": "Ruicheng/moge-2-vitl-normal", "revision": "revision", "filename": "model.pt", "cache_dir": expected_cache, "local_files_only": True})
    assert 'paths["moge2_model"] = require_moge2_model(weights_path)' in (MODULE / "lib/run_inference.py").read_text()


def test_container_dependency_revisions_are_pinned():
    dockerfile = (MODULE / "docker/Dockerfile").read_text()
    constraints = (MODULE / "lib/pip-constraints.txt").read_text().splitlines()
    moge_project = (MODULE.parent / "v2d_moge/lib/pyproject.toml").read_text()
    assert "ENV PIP_CONSTRAINT=/workspace/v2d_cari4d/lib/pip-constraints.txt" in dockerfile
    assert constraints == ["numpy==1.26.3", "torch==2.5.1", "torchvision==0.20.1", "wis3d==1.0.1"]
    assert '"moge @ git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030"' in moge_project


def test_full_suite_runs_inside_the_cari4d_container():
    dockerfile = (MODULE / "docker/Dockerfile").read_text()
    runner = (MODULE / "docker/run_tests.py").read_text()
    readme = (MODULE / "README.md").read_text()
    assert "RUN pip install pytest==9.1.1" in dockerfile
    assert 'f"{MODULES_DIR}:/workspace"' in runner
    assert 'IMAGE_NAME, "python", "-m", "pytest", "tests", "-v"' in runner
    assert "python -m v2d.cari4d.docker.run_tests" in readme
    assert "python -m pytest modules/v2d_cari4d/tests/test_contract.py -v" in readme
    assert "test_runtime_contract.py" in readme


def test_host_contract_imports_without_container_dependencies():
    script = """
import importlib.abc
import runpy
import sys

class HostOnlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"h5py", "numpy", "torch", "cv2"}:
            raise ModuleNotFoundError(f"Container dependency imported by host contract: {fullname}")

sys.meta_path.insert(0, HostOnlyImports())
sys.path.insert(0, sys.argv[1])
runpy.run_module("tests.test_contract", run_name="host_contract_import_check")
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(MODULE)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pipeline_decoders_use_configured_mhr_asset_root():
    expected = 'MHRLayer.from_mhr_assets(mhr_assets_root=os.environ.get("MHR_ASSETS_ROOT")'
    assert expected in (MODULE / "lib/cari4d/prep/align_depth_to_mhr_wild.py").read_text()
    assert expected in (MODULE / "lib/cari4d/tools/run_mhr_wild_inference.py").read_text()


def test_human_part_material_is_generated_from_native_mhr_skinning():
    source_root = MODULE / "lib/cari4d"
    material_source = (source_root / "lib_mhr/human_texture.py").read_text()
    assert 'MHR_PART_TEXTURE_REVISION = "mhr-native-lbs-part-palette-v1"' in material_source
    assert "model.get_lbsw()" in material_source
    assert {path.name for path in (source_root / "lib_mhr/assets").iterdir()} == {"mhr_collision_proxy_4000v.npz", "mhr_hand_surface_spec.npz"}
