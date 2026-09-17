from __future__ import annotations

import threading
import time
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

LIB_DIR = Path(__file__).resolve().parents[1]
package = sys.modules.setdefault(
    "v2d.foundation_pose.lib", types.ModuleType("v2d.foundation_pose.lib")
)
package.__path__ = [str(LIB_DIR)]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trt_engine = _load_module(
    "v2d.foundation_pose.lib.trt_engine", LIB_DIR / "trt_engine.py"
)
backends = _load_module("v2d.foundation_pose.lib.backends", LIB_DIR / "backends.py")
download_weights = _load_module(
    "v2d.foundation_pose.lib.download_weights", LIB_DIR / "download_weights.py"
)


def _patch_engine_build(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    trtexec = tmp_path / "trtexec"
    trtexec.write_text("stub")
    monkeypatch.setattr(
        trt_engine, "_validate_onnx", lambda directory, spec: model_dir / spec.onnx_filename
    )
    monkeypatch.setattr(
        trt_engine, "engine_filename", lambda spec, **kwargs: f"{spec.name}.engine"
    )
    return model_dir, trtexec


def test_backend_names_and_default():
    assert backends.DEFAULT_BACKEND == "nvidia_tensorrt"
    assert backends.SUPPORTED_BACKENDS == ("nvidia_tensorrt", "nvlabs_pytorch")
    with pytest.raises(ValueError, match="Unsupported FoundationPose backend"):
        backends.create_predictor_backend("legacy_pytorch", "/unused")


def test_commercial_backend_never_silently_falls_back(monkeypatch):
    def missing_engines(_weights_dir):
        raise FileNotFoundError("commercial assets missing")

    monkeypatch.setattr(backends, "ensure_engines", missing_engines)
    with pytest.raises(FileNotFoundError, match="commercial assets missing"):
        backends.create_predictor_backend("nvidia_tensorrt", "/missing")


def test_engine_filename_contains_model_runtime_and_gpu_identity(monkeypatch):
    spec = trt_engine.MODEL_SPECS["refine"]
    filename = trt_engine.engine_filename(
        spec, trt_version="10.7.0.post1", compute_capability=(8, 9)
    )
    assert spec.sha256[:12] in filename
    assert trt_engine.MODEL_VERSION in filename
    assert "trt_10_7_0_post1" in filename
    assert "sm_89" in filename
    assert filename.endswith("_fp32.engine")


def test_tao_model_identity_and_graph_contract_are_pinned():
    assert trt_engine.MODEL_REGISTRY == "nvidia/tao/foundationpose"
    assert trt_engine.MODEL_VERSION == "deployable_v1.0"
    assert trt_engine.MODEL_LICENSE == "NVIDIA Open Model License"
    assert trt_engine.MODEL_SPECS["refine"].onnx_filename == "refiner_net.onnx"
    assert trt_engine.MODEL_SPECS["refine"].sha256 == (
        "4a4445d2506b1bfc0a8d62fa80f8780a7e37bb0df13bb09aa5dd04a03bf0c954"
    )
    assert trt_engine.MODEL_SPECS["refine"].input_names == ("inputA", "inputB")
    assert trt_engine.MODEL_SPECS["refine"].output_names == ("trans", "rot")
    assert trt_engine.MODEL_SPECS["score"].onnx_filename == "score_net.onnx"
    assert trt_engine.MODEL_SPECS["score"].sha256 == (
        "bb7208919557560ecb0e10a30612821f0025a2578190b17677bdf3653514a48d"
    )
    assert trt_engine.MODEL_SPECS["score"].output_names == ("score_logit",)


def test_build_engine_is_atomic_and_reuses_completed_engine(monkeypatch, tmp_path):
    model_dir, trtexec = _patch_engine_build(monkeypatch, tmp_path)
    cache_dir = tmp_path / "cache"
    calls = []

    def fake_run(command, check):
        assert check is True
        calls.append(command)
        save_arg = next(arg for arg in command if arg.startswith("--saveEngine="))
        Path(save_arg.split("=", 1)[1]).write_bytes(b"complete-engine")

    monkeypatch.setattr(trt_engine.subprocess, "run", fake_run)
    spec = trt_engine.MODEL_SPECS["refine"]

    result = trt_engine.build_engine(
        model_dir, spec, output_dir=cache_dir, trtexec_path=str(trtexec)
    )
    assert result.read_bytes() == b"complete-engine"
    assert not list(cache_dir.glob("*.tmp"))

    reused = trt_engine.build_engine(
        model_dir, spec, output_dir=cache_dir, trtexec_path=str(trtexec)
    )
    assert reused == result
    assert len(calls) == 1
    assert "--maxShapes=inputA:42x6x160x160,inputB:42x6x160x160" in calls[0]
    assert "--noTF32" in calls[0]


def test_concurrent_builders_publish_one_engine(monkeypatch, tmp_path):
    model_dir, trtexec = _patch_engine_build(monkeypatch, tmp_path)
    calls = 0
    calls_lock = threading.Lock()

    def fake_run(command, check):
        nonlocal calls
        assert check is True
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        save_arg = next(arg for arg in command if arg.startswith("--saveEngine="))
        Path(save_arg.split("=", 1)[1]).write_bytes(b"complete-engine")

    monkeypatch.setattr(trt_engine.subprocess, "run", fake_run)
    spec = trt_engine.MODEL_SPECS["score"]
    results = []

    def build():
        results.append(
            trt_engine.build_engine(model_dir, spec, trtexec_path=str(trtexec))
        )

    threads = [threading.Thread(target=build) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len(set(results)) == 1
    assert results[0].read_bytes() == b"complete-engine"


def test_ensure_engines_prefers_provisioned_then_writable_cache(monkeypatch, tmp_path):
    model_dir = tmp_path / "weights" / trt_engine.MODEL_SUBDIR
    model_dir.mkdir(parents=True)
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        trt_engine, "engine_filename", lambda spec, **kwargs: f"{spec.name}.engine"
    )
    monkeypatch.setattr(
        trt_engine,
        "_validate_onnx",
        lambda directory, spec: Path(directory) / spec.onnx_filename,
    )
    (model_dir / "refine.engine").write_bytes(b"prebuilt")
    builds = []

    def fake_build(source, spec, *, output_dir, force, trtexec_path):
        builds.append((Path(source), spec.name, Path(output_dir)))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result = Path(output_dir) / f"{spec.name}.engine"
        result.write_bytes(b"lazy")
        return result

    monkeypatch.setattr(trt_engine, "build_engine", fake_build)
    result = trt_engine.ensure_engines(
        tmp_path / "weights", cache_dir=cache_dir, trtexec_path="unused"
    )

    assert result["refine"] == model_dir / "refine.engine"
    assert result["score"] == cache_dir / "score.engine"
    assert builds == [(model_dir, "score", cache_dir)]


def test_missing_trtexec_fails_without_partial_engine(monkeypatch, tmp_path):
    model_dir, _ = _patch_engine_build(monkeypatch, tmp_path)
    spec = trt_engine.MODEL_SPECS["refine"]
    with pytest.raises(FileNotFoundError, match="trtexec not found"):
        trt_engine.build_engine(
            model_dir, spec, trtexec_path=str(tmp_path / "missing-trtexec")
        )
    assert not (model_dir / "refine.engine").exists()


def test_model_hash_mismatch_is_rejected(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    spec = trt_engine.MODEL_SPECS["refine"]
    (model_dir / spec.onnx_filename).write_bytes(b"not-the-model")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        trt_engine._validate_onnx(model_dir, spec)


def test_commercial_download_requires_explicit_eula_acceptance(tmp_path):
    with pytest.raises(RuntimeError, match="accepting the NVIDIA Open Model License"):
        download_weights.download_weights(
            str(tmp_path), backend="nvidia_tensorrt"
        )


def test_model_manifest_records_pinned_hashes(tmp_path):
    path = trt_engine.write_model_manifest(tmp_path)
    payload = path.read_text()
    assert trt_engine.MODEL_REGISTRY in payload
    assert trt_engine.MODEL_VERSION in payload
    assert trt_engine.MODEL_LICENSE in payload
    for spec in trt_engine.MODEL_SPECS.values():
        assert spec.onnx_filename in payload
        assert spec.sha256 in payload

    configuration_path = (
        Path(__file__).resolve().parents[4]
        / "workflows/mv_hoi/revalidation_configuration.example.json"
    )
    configuration = json.loads(configuration_path.read_text())
    assert trt_engine.sha256_file(path) == configuration["foundation_pose"][
        "checkpoint_manifest_sha256"
    ]


def test_commercial_download_uses_tao_registry_and_filenames(monkeypatch, tmp_path):
    downloads = []

    def fake_download(url, destination, expected_sha256):
        downloads.append((url, Path(destination), expected_sha256))

    monkeypatch.setattr(download_weights, "_download_verified", fake_download)
    monkeypatch.setattr(download_weights, "write_model_manifest", lambda _: None)
    download_weights.download_weights(
        str(tmp_path),
        backend="nvidia_tensorrt",
        accept_nvidia_model_eula=True,
    )

    assert [entry[0] for entry in downloads] == [
        f"{download_weights.NGC_BASE_URL}/refiner_net.onnx",
        f"{download_weights.NGC_BASE_URL}/score_net.onnx",
    ]
    assert all("/models/nvidia/tao/foundationpose/" in entry[0] for entry in downloads)
