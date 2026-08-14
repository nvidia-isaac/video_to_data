# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest

from v2d_cusfm.docker import gpu_compatibility as compatibility


def _gpu(index, name, compute_capability):
    return compatibility.GPUInfo(
        index=index,
        name=name,
        compute_capability=compute_capability,
    )


def test_parse_nvidia_smi_output():
    output = (
        "0, 8.6, NVIDIA RTX A6000\n"
        "1, 8.9, NVIDIA L40S\n"
    )

    assert compatibility.parse_nvidia_smi_output(output) == [
        _gpu(0, "NVIDIA RTX A6000", (8, 6)),
        _gpu(1, "NVIDIA L40S", (8, 9)),
    ]


@pytest.mark.parametrize(
    "output, expected_message",
    [
        ("", "found no NVIDIA GPUs"),
        ("GPU-0, 8.6, NVIDIA RTX A6000", "could not parse GPU index"),
        ("0, N/A, NVIDIA RTX A6000", "could not parse compute capability"),
        ("0, 8.6", "expected three fields"),
        ("0, 8.6, ", "empty GPU name"),
    ],
)
def test_parse_nvidia_smi_output_rejects_invalid_data(output, expected_message):
    with pytest.raises(compatibility.GPUCompatibilityError, match=expected_message):
        compatibility.parse_nvidia_smi_output(output)


def test_validate_accepts_validated_gpu_architectures():
    gpus = [
        _gpu(0, "NVIDIA RTX A6000", (8, 6)),
        _gpu(1, "NVIDIA L40S", (8, 9)),
    ]

    assert compatibility.validate_cusfm_gpu_compatibility(gpus) == gpus


def test_validate_rejects_sm120_with_actionable_error():
    gpu = _gpu(0, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", (12, 0))

    with pytest.raises(compatibility.GPUCompatibilityError) as exc_info:
        compatibility.validate_cusfm_gpu_compatibility([gpu])

    message = str(exc_info.value)
    assert "RTX PRO 6000 Blackwell" in message
    assert "compute capability 12.0 / sm_120" in message
    assert "v2d_cusfm" in message
    assert "TensorRT and cuVSLAM" in message


def test_validate_rejects_architectures_newer_than_sm120():
    gpu = _gpu(0, "Future NVIDIA GPU", (13, 0))

    with pytest.raises(compatibility.GPUCompatibilityError, match="SM 120 or newer"):
        compatibility.validate_cusfm_gpu_compatibility([gpu])


def test_validate_only_checks_selected_gpu_ids():
    gpus = [
        _gpu(0, "NVIDIA RTX PRO 6000 Blackwell", (12, 0)),
        _gpu(1, "NVIDIA RTX A6000", (8, 6)),
    ]

    assert compatibility.validate_cusfm_gpu_compatibility(gpus, [1, 1]) == [gpus[1]]


def test_validate_rejects_unknown_selected_gpu_id():
    with pytest.raises(
        compatibility.GPUCompatibilityError,
        match=r"Requested GPU ID\(s\) 1 .* visible GPU ID\(s\): 0",
    ):
        compatibility.validate_cusfm_gpu_compatibility(
            [_gpu(0, "NVIDIA RTX A6000", (8, 6))],
            [1],
        )


def test_query_nvidia_gpus_uses_compute_capability_query(monkeypatch):
    def fake_run(command, **kwargs):
        assert command == [
            "nvidia-smi",
            "--query-gpu=index,compute_cap,name",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "check": False,
        }
        return subprocess.CompletedProcess(command, 0, "0, 8.6, NVIDIA RTX A6000\n", "")

    monkeypatch.setattr(compatibility.subprocess, "run", fake_run)

    assert compatibility.query_nvidia_gpus() == [
        _gpu(0, "NVIDIA RTX A6000", (8, 6))
    ]


def test_query_nvidia_gpus_reports_missing_nvidia_smi(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(compatibility.subprocess, "run", fake_run)

    with pytest.raises(compatibility.GPUCompatibilityError, match="could not find nvidia-smi"):
        compatibility.query_nvidia_gpus()


def test_query_nvidia_gpus_reports_driver_query_failure(monkeypatch):
    monkeypatch.setattr(
        compatibility.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            9,
            "",
            "Failed to initialize NVML",
        ),
    )

    with pytest.raises(compatibility.GPUCompatibilityError) as exc_info:
        compatibility.query_nvidia_gpus()

    assert "nvidia-smi exit 9" in str(exc_info.value)
    assert "Failed to initialize NVML" in str(exc_info.value)


def test_cli_returns_nonzero_for_unsupported_gpu(monkeypatch, capsys):
    monkeypatch.setattr(
        compatibility,
        "query_nvidia_gpus",
        lambda: [_gpu(0, "NVIDIA RTX PRO 6000 Blackwell", (12, 0))],
    )

    assert compatibility.main([]) == 1
    assert "ERROR: Unsupported GPU architecture detected" in capsys.readouterr().err


def test_preflight_runs_before_build_and_pipeline_work():
    docker_dir = Path(__file__).parents[1]
    reconstruction_dir = Path(__file__).parents[4]
    build_script = (reconstruction_dir / "scripts" / "build_containers.sh").read_text()
    cusfm_runner = (docker_dir / "run_image_list_to_sfm.py").read_text()
    hoi_runner = (
        reconstruction_dir
        / "modules"
        / "v2d_hoi_object_reconstruction"
        / "docker"
        / "run_reconstruction.py"
    ).read_text()

    assert build_script.index(
        "modules/v2d_cusfm/docker/gpu_compatibility.py"
    ) < build_script.index("MODULES=(")
    assert cusfm_runner.index("require_compatible_cusfm_gpus()") < cusfm_runner.index(
        "os.makedirs(output_dir"
    )
    assert hoi_runner.index("require_compatible_cusfm_gpus(gpu_ids)") < hoi_runner.index(
        "os.makedirs(job_dir"
    )
