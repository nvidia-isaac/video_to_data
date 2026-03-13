"""Export Foundation Stereo ONNX model to a TensorRT engine.

The engine filename is versioned by TensorRT version + GPU SM to allow caching
across multiple GPU architectures and TRT versions.
"""

import os
import subprocess

import tensorrt as trt
from cuda.bindings import runtime as cuda_runtime

TRTEXEC_PATH = '/usr/src/tensorrt/bin/trtexec'
BASE_MODEL_NAME = 'deployable_foundationstereo_small_576x960_v2.0'


def _get_gpu_sm() -> tuple[int, int]:
    """Return (major, minor) GPU compute capability of the current device.

    Tries cuda.bindings first; falls back to nvidia-smi if the runtime API
    returns an error (e.g. in CUDA forward-compatibility mode).
    """
    try:
        cuda_runtime.cudaFree(0)
        err, device_id = cuda_runtime.cudaGetDevice()
        if err == cuda_runtime.cudaError_t.cudaSuccess and device_id is not None:
            err, props = cuda_runtime.cudaGetDeviceProperties(int(device_id))
            if err == cuda_runtime.cudaError_t.cudaSuccess:
                return props.major, props.minor
    except Exception:
        pass

    # Fallback: parse compute capability from nvidia-smi
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        cap = result.stdout.strip().split('\n')[0]
        major, minor = cap.split('.')
        return int(major), int(minor)

    raise RuntimeError(
        "Cannot determine GPU SM version via cuda.bindings or nvidia-smi. "
        "Ensure a GPU is available."
    )


def get_versioned_engine_filename(base_model_name: str = BASE_MODEL_NAME) -> str:
    """Return a versioned engine filename encoding TRT version and GPU SM.

    Format: {model}_{trt_major}_{trt_minor}_{trt_patch}_{trt_build}_sm_{major}_{minor}.engine
    """
    parts = trt.__version__.split('.')
    parts += ['0'] * (4 - len(parts))
    trt_major, trt_minor, trt_patch, trt_build = parts[:4]

    sm_major, sm_minor = _get_gpu_sm()

    return (
        f"{base_model_name}_{trt_major}_{trt_minor}_{trt_patch}_{trt_build}"
        f"_sm_{sm_major}_{sm_minor}.engine"
    )


def get_engine_path(model_dir: str) -> str:
    """Return the full path to the versioned engine file in model_dir."""
    return os.path.join(model_dir, get_versioned_engine_filename())


def export_engine(onnx_path: str, output_dir: str, force: bool = False) -> str:
    """Convert Foundation Stereo ONNX to a TensorRT engine.

    Args:
        onnx_path: Path to the ONNX file.
        output_dir: Directory where the .engine file will be saved.
        force: Rebuild even if the engine already exists.

    Returns:
        Path to the generated engine file.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    os.makedirs(output_dir, exist_ok=True)
    engine_path = get_engine_path(output_dir)

    if os.path.exists(engine_path) and not force:
        print(f"Engine already exists (skipping): {engine_path}")
        return engine_path

    cmd = [
        TRTEXEC_PATH,
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        '--noDataTransfers',
    ]

    print(f"Building TensorRT engine (this may take several minutes)...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"trtexec failed (exit {result.returncode}). "
            "Check GPU memory, TRT version, and ONNX compatibility."
        )

    print(f"Engine saved: {engine_path}")
    return engine_path


def ensure_engine(model_dir: str) -> str:
    """Return an engine path, auto-exporting from ONNX if the engine is missing.

    Args:
        model_dir: Directory containing the ONNX and (optionally) the engine.

    Returns:
        Path to a valid TensorRT engine file.

    Raises:
        FileNotFoundError: If neither engine nor ONNX is present.
        RuntimeError: If trtexec fails.
    """
    engine_path = get_engine_path(model_dir)
    if os.path.exists(engine_path):
        return engine_path

    onnx_filename = f"{BASE_MODEL_NAME}.onnx"
    onnx_path = os.path.join(model_dir, onnx_filename)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"Neither engine nor ONNX found in {model_dir}.\n"
            f"Run download.sh to download the ONNX first."
        )

    print(f"Engine not found — exporting from ONNX: {onnx_path}")
    return export_engine(onnx_path, model_dir)
