"""TensorRT engine caching for the commercial FoundationPose models."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


MODEL_REGISTRY = "nvidia/tao/foundationpose"
MODEL_VERSION = "deployable_v1.0"
MODEL_LICENSE = "NVIDIA Open Model License"
MODEL_SUBDIR = Path("nvidia_tensorrt") / MODEL_VERSION
TRTEXEC_PATH = os.environ.get("TRTEXEC_PATH", "/usr/src/tensorrt/bin/trtexec")
ENGINE_CACHE_ENV = "FOUNDATIONPOSE_ENGINE_CACHE_DIR"
MIN_TENSORRT_VERSION = (10, 3)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    onnx_filename: str
    sha256: str
    input_names: tuple[str, str]
    output_names: tuple[str, ...]
    min_batch: int
    opt_batch: int
    max_batch: int


MODEL_SPECS = {
    "refine": ModelSpec(
        name="refine",
        onnx_filename="refiner_net.onnx",
        sha256="4a4445d2506b1bfc0a8d62fa80f8780a7e37bb0df13bb09aa5dd04a03bf0c954",
        input_names=("inputA", "inputB"),
        output_names=("trans", "rot"),
        min_batch=1,
        opt_batch=1,
        max_batch=42,
    ),
    "score": ModelSpec(
        name="score",
        onnx_filename="score_net.onnx",
        sha256="bb7208919557560ecb0e10a30612821f0025a2578190b17677bdf3653514a48d",
        input_names=("inputA", "inputB"),
        output_names=("score_logit",),
        min_batch=1,
        opt_batch=1,
        max_batch=252,
    ),
}


def commercial_model_dir(weights_dir: str | os.PathLike[str]) -> Path:
    """Resolve the commercial model directory from a FoundationPose weights root."""
    root = Path(weights_dir)
    if root.name == MODEL_VERSION and root.parent.name == "nvidia_tensorrt":
        return root
    return root / MODEL_SUBDIR


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensorrt_version() -> str:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is required for backend='nvidia_tensorrt'. "
            "Use the FoundationPose container or install TensorRT 10.3+."
        ) from exc
    version = trt.__version__
    numeric = []
    for part in version.split(".")[:2]:
        digits = "".join(character for character in part if character.isdigit())
        numeric.append(int(digits or 0))
    if tuple(numeric) < MIN_TENSORRT_VERSION:
        raise RuntimeError(
            "backend='nvidia_tensorrt' requires TensorRT 10.3 or newer; "
            f"found {version}"
        )
    return version


def _gpu_compute_capability() -> tuple[int, int]:
    try:
        import torch

        if torch.cuda.is_available():
            return tuple(int(v) for v in torch.cuda.get_device_capability())
    except (ImportError, RuntimeError):
        pass

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        major, minor = result.stdout.strip().splitlines()[0].split(".")
        return int(major), int(minor)
    raise RuntimeError("Cannot determine GPU compute capability")


def _version_token(version: str) -> str:
    return "_".join(part for part in version.replace("-", ".").split(".") if part)


def engine_filename(spec: ModelSpec, *, trt_version: str | None = None,
                    compute_capability: tuple[int, int] | None = None) -> str:
    trt_version = trt_version or _tensorrt_version()
    sm_major, sm_minor = compute_capability or _gpu_compute_capability()
    return (
        f"{spec.name}_{MODEL_VERSION}_{spec.sha256[:12]}"
        f"_trt_{_version_token(trt_version)}_sm_{sm_major}{sm_minor}_fp32.engine"
    )


def _validate_onnx(model_dir: Path, spec: ModelSpec) -> Path:
    path = model_dir / spec.onnx_filename
    if not path.is_file():
        raise FileNotFoundError(f"FoundationPose ONNX model not found: {path}")
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {spec.sha256}, got {actual}"
        )
    return path


def _shape_arg(spec: ModelSpec, batch: int) -> str:
    shape = f"{batch}x6x160x160"
    return ",".join(f"{name}:{shape}" for name in spec.input_names)


def build_engine(model_dir: str | os.PathLike[str], spec: ModelSpec, *,
                 output_dir: str | os.PathLike[str] | None = None,
                 force: bool = False, trtexec_path: str = TRTEXEC_PATH) -> Path:
    """Build one FP32 engine atomically, serializing concurrent builders."""
    model_dir = Path(model_dir)
    onnx_path = _validate_onnx(model_dir, spec)
    output_dir = Path(output_dir) if output_dir is not None else model_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"FoundationPose engine cache is not writable: {output_dir}. "
            f"Set {ENGINE_CACHE_ENV} to a writable directory or provision "
            "compatible prebuilt engines with the weights."
        ) from exc
    output_path = output_dir / engine_filename(spec)
    lock_path = output_dir / f".{output_path.name}.lock"

    try:
        lock_stream = open(lock_path, "a+")
    except OSError as exc:
        raise RuntimeError(
            f"FoundationPose engine cache is not writable: {output_dir}. "
            f"Set {ENGINE_CACHE_ENV} to a writable directory or provision "
            "compatible prebuilt engines with the weights."
        ) from exc
    with lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        if output_path.is_file() and output_path.stat().st_size > 0 and not force:
            return output_path
        if not os.path.isfile(trtexec_path):
            raise FileNotFoundError(
                f"trtexec not found at {trtexec_path}; set TRTEXEC_PATH or use the "
                "FoundationPose TensorRT container"
            )

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_dir
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        command = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={temporary_path}",
            f"--minShapes={_shape_arg(spec, spec.min_batch)}",
            f"--optShapes={_shape_arg(spec, spec.opt_batch)}",
            f"--maxShapes={_shape_arg(spec, spec.max_batch)}",
            "--noTF32",
            "--skipInference",
        ]
        try:
            subprocess.run(command, check=True)
            if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                raise RuntimeError(f"trtexec did not create a valid engine: {temporary_path}")
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return output_path


def ensure_engines(weights_dir: str | os.PathLike[str], *, force: bool = False,
                   trtexec_path: str = TRTEXEC_PATH,
                   cache_dir: str | os.PathLike[str] | None = None) -> dict[str, Path]:
    """Return compatible scorer/refiner engines, building missing variants."""
    model_dir = commercial_model_dir(weights_dir)
    configured_cache = cache_dir or os.environ.get(ENGINE_CACHE_ENV)
    build_dir = Path(configured_cache) if configured_cache else model_dir
    paths: dict[str, Path] = {}
    for name, spec in MODEL_SPECS.items():
        _validate_onnx(model_dir, spec)
        filename = engine_filename(spec)
        provisioned = model_dir / filename
        cached = build_dir / filename
        expected = cached if force else provisioned
        if not force and (not expected.is_file() or expected.stat().st_size == 0):
            expected = cached
        if force or not expected.is_file() or expected.stat().st_size == 0:
            expected = build_engine(
                model_dir,
                spec,
                output_dir=build_dir,
                force=force,
                trtexec_path=trtexec_path,
            )
        paths[name] = expected
    return paths


def runtime_manifest(weights_dir: str | os.PathLike[str],
                     engines: dict[str, Path]) -> dict:
    sm_major, sm_minor = _gpu_compute_capability()
    return {
        "backend": "nvidia_tensorrt",
        "model_registry": MODEL_REGISTRY,
        "model_version": MODEL_VERSION,
        "model_license": MODEL_LICENSE,
        "tensorrt_version": _tensorrt_version(),
        "gpu_compute_capability": f"{sm_major}.{sm_minor}",
        "models": {
            name: {
                **asdict(spec),
                "engine_filename": engines[name].name,
            }
            for name, spec in MODEL_SPECS.items()
        },
        "model_dir": str(commercial_model_dir(weights_dir)),
    }


def write_model_manifest(model_dir: str | os.PathLike[str]) -> Path:
    """Write the pinned, non-runtime model manifest used during provisioning."""
    model_dir = Path(model_dir)
    path = model_dir / "manifest.json"
    payload = {
        "schema": "v2d.foundation_pose.tao_model_manifest.v1",
        "model_registry": MODEL_REGISTRY,
        "model_version": MODEL_VERSION,
        "model_license": MODEL_LICENSE,
        "precision": "fp32",
        "models": {name: asdict(spec) for name, spec in MODEL_SPECS.items()},
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path
