# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from v2d_hoi_object_reconstruction.docker import _container_steps


def _declared_dependencies(pyproject_text):
    match = re.search(
        r"(?ms)^dependencies\s*=\s*\[(.*?)^\]",
        pyproject_text,
    )
    assert match is not None
    return {
        re.split(r"[<>=!~ \[]", dependency, maxsplit=1)[0].lower()
        for dependency in re.findall(
            r'^\s*"([^"]+)"', match.group(1), flags=re.MULTILINE
        )
    }


def test_bundlesdf_launcher_has_no_scientific_host_dependencies():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    dependencies = _declared_dependencies(pyproject)
    assert dependencies.isdisjoint({
        "matplotlib",
        "numpy",
        "opencv-python",
        "opencv-python-headless",
        "scipy",
        "v2d-common",
    })
    assert {
        "v2d-docker",
        "v2d-cusfm-docker",
        "v2d-foundation-pose-docker",
        "v2d-grounding-dino-docker",
        "v2d-sam3d-docker",
    }.issubset(dependencies)
    assert "[project.scripts]" not in pyproject
    assert "[project.optional-dependencies]" not in pyproject
    assert re.search(
        r'(?m)^packages\s*=\s*\["v2d_hoi_object_reconstruction\.docker"\]$',
        pyproject,
    )

    guard = r'''
import builtins

blocked = {"cv2", "matplotlib", "numpy", "PIL", "scipy"}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"scientific host import attempted: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import v2d_hoi_object_reconstruction.docker.run_reconstruction
'''
    subprocess.run([sys.executable, "-c", guard], check=True)


def test_pipeline_invokes_container_python_as_packages():
    runner = (Path(__file__).parents[1] / "run_reconstruction.py").read_text()
    direct_script = re.compile(
        r'["\']python(?:3)?["\']\s*,\s*["\']/workspace/[^"\']+\.py["\']'
    )
    assert not direct_script.search(runner)

    tree = ast.parse(runner)
    host_lib_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("v2d_hoi_object_reconstruction.lib")
    ]
    assert host_lib_imports == []
    assert "/workspace" not in runner


def test_default_pipeline_config_is_packaged_with_host_launcher():
    docker_dir = Path(__file__).parents[1]
    assert (docker_dir / "data" / "configs" / "hoi_pipeline.yaml").is_file()

    pyproject = (docker_dir / "pyproject.toml").read_text()
    assert '"data/configs/*.yaml"' in pyproject


def test_grounding_dino_uses_canonical_container_wrapper(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _container_steps,
        "run_image_to_object_bboxes",
        lambda **kwargs: calls.append(kwargs),
    )

    _container_steps.run_grounding_dino(
        image_path="/job/ref_frame.jpg",
        output_path="/job/grounding_dino_bboxes.json",
        prompt="toy airplane",
        model_dir="/weights/grounding_dino",
        box_threshold=0.3,
    )

    assert calls == [{
        "image_path": "/job/ref_frame.jpg",
        "output_path": "/job/grounding_dino_bboxes.json",
        "prompt": "toy airplane",
        "model_dir": "/weights/grounding_dino",
        "box_threshold": 0.3,
    }]


def test_prepare_job_runs_in_cpu_only_hoi_container(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _container_steps,
        "run_in_container",
        lambda **kwargs: calls.append(kwargs),
    )

    _container_steps.prepare_job(
        image="v2d_hoi_object_reconstruction",
        input_dir="/input/session",
        job_dir="/output/job",
        fps=24,
        max_frames=10,
    )

    assert calls == [{
        "image": "v2d_hoi_object_reconstruction",
        "module": "v2d_hoi_object_reconstruction.lib.prepare_FP_folder",
        "inputs": {"input_dir": "/input/session"},
        "outputs": {"job_dir": "/output/job"},
        "extra_args": {"fps": 24, "max_frames": 10},
        "gpus": False,
    }]


def test_prepare_job_resolves_symlinked_dataset_before_mounting(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    link = tmp_path / "dataset"
    link.symlink_to(source, target_is_directory=True)
    calls = []
    monkeypatch.setattr(
        _container_steps,
        "run_in_container",
        lambda **kwargs: calls.append(kwargs),
    )

    _container_steps.prepare_job(
        image="v2d_hoi_object_reconstruction",
        input_dir=link,
        job_dir=tmp_path / "job",
    )

    assert calls[0]["inputs"]["input_dir"] == str(source)


def test_pose_conversion_runs_in_cpu_only_hoi_container(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _container_steps,
        "run_in_container",
        lambda **kwargs: calls.append(kwargs),
    )

    _container_steps.convert_poses_to_matrix(
        image="v2d_hoi_object_reconstruction",
        poses_dir="/output/job/poses",
    )

    assert calls == [{
        "image": "v2d_hoi_object_reconstruction",
        "module": "v2d_hoi_object_reconstruction.lib.convert_poses_to_matrix",
        "inputs": {"poses_dir": "/output/job/poses"},
        "outputs": {},
        "gpus": False,
    }]


def test_mask_postprocess_runs_in_cpu_only_hoi_container(tmp_path, monkeypatch):
    calls = []

    def fake_run_in_container(**kwargs):
        calls.append(kwargs)
        output_dir = tmp_path / "masks"
        output_dir.mkdir()
        (output_dir / "postprocess_summary.json").write_text(json.dumps({"frames": 7}))

    monkeypatch.setattr(
        _container_steps,
        "run_in_container",
        fake_run_in_container,
    )
    config = {
        "keep_largest_component": True,
        "min_component_area_px": 2000,
        "min_component_area_frac": 0.01,
        "open_px": 0,
        "erode_px": 3,
    }

    summary = _container_steps.postprocess_masks(
        image="v2d_hoi_object_reconstruction",
        input_dir=tmp_path / "masks_raw_sam2",
        output_dir=tmp_path / "masks",
        config=config,
    )

    assert summary == {"frames": 7}
    assert calls == [{
        "image": "v2d_hoi_object_reconstruction",
        "module": "v2d_hoi_object_reconstruction.lib.postprocess_masks",
        "inputs": {"input_dir": str(tmp_path / "masks_raw_sam2")},
        "outputs": {"output_dir": str(tmp_path / "masks")},
        "extra_args": config,
        "gpus": False,
    }]


def test_mask_postprocess_rejects_stale_summary(tmp_path, monkeypatch):
    output_dir = tmp_path / "masks"
    output_dir.mkdir()
    summary_path = output_dir / "postprocess_summary.json"
    summary_path.write_text(json.dumps({"frames": 99}))
    monkeypatch.setattr(
        _container_steps,
        "run_in_container",
        lambda **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="without producing"):
        _container_steps.postprocess_masks(
            image="v2d_hoi_object_reconstruction",
            input_dir=tmp_path / "masks_raw_sam2",
            output_dir=output_dir,
            config={},
        )

    assert not summary_path.exists()


def test_sam3d_frame_selection_runs_in_hoi_container(tmp_path, monkeypatch):
    calls = []
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    def fake_run_in_container(**kwargs):
        calls.append(kwargs)
        output_path = job_dir / "sam3d" / "selected_frames.json"
        output_path.parent.mkdir()
        output_path.write_text(json.dumps(["000010", "000020"]))

    monkeypatch.setattr(_container_steps, "run_in_container", fake_run_in_container)

    selected = _container_steps.select_sam3d_frames(
        image="v2d_hoi_object_reconstruction",
        job_dir=job_dir,
        bin_deg=60.0,
    )

    assert selected == ["000010", "000020"]
    assert calls == [{
        "image": "v2d_hoi_object_reconstruction",
        "module": "v2d_hoi_object_reconstruction.lib.select_sam3d_frames",
        "inputs": {"job_dir": str(job_dir)},
        "outputs": {
            "output_path": str(job_dir / "sam3d" / "selected_frames.json")
        },
        "extra_args": {"bin_deg": 60.0, "fallback_count": 6},
        "gpus": False,
    }]


def test_sam3d_srt_runs_in_hoi_container(tmp_path, monkeypatch):
    calls = []
    job_dir = tmp_path / "job"
    sam3d_dir = job_dir / "sam3d"
    sam3d_dir.mkdir(parents=True)
    selected_path = sam3d_dir / "selected_frames.json"
    selected_path.write_text(json.dumps(["000010"]))

    def fake_run_in_container(**kwargs):
        calls.append(kwargs)
        (sam3d_dir / "srt_run_summary.json").write_text(json.dumps({
            "outcomes": [{"frame_id": "000010", "elapsed": 1.5, "resumed": False}]
        }))

    monkeypatch.setattr(_container_steps, "run_in_container", fake_run_in_container)

    summary = _container_steps.run_sam3d_srt(
        image="v2d_hoi_object_reconstruction",
        job_dir=job_dir,
        use_depth=True,
        stage1_end_frame=925,
        max_views=25,
        maxiter=60,
        top_k=1,
        parallel=8,
        force=False,
    )

    assert summary["outcomes"][0]["frame_id"] == "000010"
    assert calls == [{
        "image": "v2d_hoi_object_reconstruction",
        "module": "v2d_hoi_object_reconstruction.lib.sam3d_srt",
        "inputs": {
            "job_dir": str(job_dir),
            "selected_frames": str(selected_path),
        },
        "outputs": {
            "summary_path": str(sam3d_dir / "srt_run_summary.json")
        },
        "extra_args": {
            "use_depth": True,
            "stage1_end_frame": 925,
            "max_views": 25,
            "maxiter": 60,
            "top_k": 1,
            "parallel": 8,
            "force": False,
        },
        "gpus": False,
    }]


def test_best_sam3d_selection_runs_in_hoi_container(tmp_path, monkeypatch):
    calls = []
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    def fake_run_in_container(**kwargs):
        calls.append(kwargs)
        output_dir = job_dir / "sam3d" / "best"
        output_dir.mkdir(parents=True)
        (output_dir / "best_frame.json").write_text(json.dumps({
            "best_frame": "000010",
            "best_artifacts": {
                "output_scaled_glb": "sam3d/best/output_scaled.glb"
            },
        }))

    monkeypatch.setattr(_container_steps, "run_in_container", fake_run_in_container)

    summary = _container_steps.select_best_sam3d(
        image="v2d_hoi_object_reconstruction",
        job_dir=job_dir,
    )

    assert summary == {
        "best_frame": "000010",
        "best_artifacts": {
            "output_scaled_glb": "sam3d/best/output_scaled.glb"
        },
    }
    assert not summary["best_artifacts"]["output_scaled_glb"].startswith("/data/")
    assert calls == [{
        "image": "v2d_hoi_object_reconstruction",
        "module": "v2d_hoi_object_reconstruction.lib.select_sam3d_best",
        "inputs": {"job_dir": str(job_dir)},
        "outputs": {},
        "gpus": False,
    }]


def test_mp4_stitching_runs_in_cpu_only_hoi_container(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _container_steps,
        "run_in_container",
        lambda **kwargs: calls.append(kwargs),
    )

    _container_steps.stitch_mp4(
        image="v2d_hoi_object_reconstruction",
        frames_dir="/output/job/render_frames",
        output_mp4="/output/job/render.mp4",
        fps=30,
    )

    assert calls == [{
        "image": "v2d_hoi_object_reconstruction",
        "module": "v2d_hoi_object_reconstruction.lib.stitch_mp4",
        "inputs": {"frames_dir": "/output/job/render_frames"},
        "outputs": {"output_mp4": "/output/job/render.mp4"},
        "extra_args": {"fps": 30},
        "gpus": False,
    }]
