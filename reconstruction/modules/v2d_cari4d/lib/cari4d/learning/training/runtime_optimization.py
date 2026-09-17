"""Opt-in distributed and compilation controls for training benchmarks."""

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from enum import Enum
from pathlib import Path

from omegaconf import OmegaConf

from learning.training.source_snapshot import iter_source_files, should_include_source_file


_COMPILE_TARGET_MODULES = {
    "rgb_dino": (("encoder_rgb", "forward_features"), ("encoderAB_rgb", "forward")),
    "rgb_xyzm_fuser": (("encoderAB_rgb_xyzm", "forward"),),
    "xyzm_pair_fuser": (("encoderAB_xyzm", "forward"),),
    "xyzm_dino": (("encoder_xyzm", "forward_features"), ("encoderAB_xyzm", "forward")),
}
_DEFAULT_COMPILE_TARGETS = ("rgb_dino",)
_INDUCTOR_CACHE_PROTOCOL_VERSION = 1
_INDUCTOR_NON_SEMANTIC_CONFIG_KEYS = {"ckpt_file"}


def bounded_batch_slices(total_size, max_chunk_size):
    total_size = int(total_size)
    if total_size < 0:
        raise ValueError(f"total_size must be non-negative, got {total_size}")
    if max_chunk_size is None:
        return (slice(0, total_size),)
    max_chunk_size = int(max_chunk_size)
    if max_chunk_size <= 0:
        raise ValueError(f"max_chunk_size must be positive or None, got {max_chunk_size}")
    return tuple(slice(start, min(start + max_chunk_size, total_size)) for start in range(0, total_size, max_chunk_size)) or (slice(0, 0),)


def _cfg_get(cfg, name, default):
    value = getattr(cfg, name, default)
    return default if value is None else value


def _normalize_identity_value(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Enum):
        return _normalize_identity_value(value.value)
    if isinstance(value, dict):
        return {str(key): _normalize_identity_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_identity_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        scalar = value.item()
        if scalar is not value:
            return _normalize_identity_value(scalar)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "value": str(value)}


def _identity_digest(value):
    encoded = json.dumps(_normalize_identity_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compile_config_identity(cfg):
    if OmegaConf.is_config(cfg):
        payload = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    elif dataclasses.is_dataclass(cfg):
        payload = dataclasses.asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        payload = vars(cfg)
    else:
        raise TypeError(f"Cannot derive compile configuration identity from {type(cfg)!r}")
    payload = dict(payload)
    for key in _INDUCTOR_NON_SEMANTIC_CONFIG_KEYS:
        payload.pop(key, None)
    return _identity_digest(payload)


def _fallback_source_identity(repo_root):
    digest = hashlib.sha256()
    included, _ = iter_source_files(repo_root)
    for path, relative_path, _ in included:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_code_identity(repo_root):
    repo_root = Path(repo_root).resolve()
    probe = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    if probe.returncode != 0:
        return _fallback_source_identity(repo_root)
    commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    diff = subprocess.run(["git", "-C", str(repo_root), "diff", "--binary", "--no-ext-diff", "HEAD", "--"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    untracked = subprocess.run(["git", "-C", str(repo_root), "ls-files", "--others", "--exclude-standard", "-z"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    for name, result in (("rev-parse HEAD", commit), ("git diff", diff), ("git ls-files", untracked)):
        if result.returncode != 0:
            raise RuntimeError(f"Failed to derive source identity with {name}: {result.stderr.decode('utf-8', errors='replace').strip()}")
    digest = hashlib.sha256()
    digest.update(commit.stdout.strip())
    digest.update(b"\0")
    digest.update(diff.stdout)
    for encoded_path in sorted(item for item in untracked.stdout.split(b"\0") if item):
        relative_path = encoded_path.decode("utf-8", errors="surrogateescape")
        path = repo_root / relative_path
        include, _ = should_include_source_file(path, repo_root)
        if include:
            digest.update(encoded_path)
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def inductor_runtime_identity(cfg, device=None):
    import torch

    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_device = device.type == "cuda" and torch.cuda.is_available()
    triton_version = None
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        pass
    return {
        "python": {"implementation": platform.python_implementation(), "version": sys.version.split()[0]},
        "platform": {"machine": platform.machine(), "system": platform.system()},
        "torch": {"version": torch.__version__, "git_version": getattr(torch.version, "git_version", None), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "build": _identity_digest(torch.__config__.show())},
        "triton": triton_version,
        "device": {"type": device.type, "name": torch.cuda.get_device_name(device) if cuda_device else None, "capability": torch.cuda.get_device_capability(device) if cuda_device else None},
        "compile": {"targets": list(compile_module_names(getattr(cfg, "torch_compile_targets", _DEFAULT_COMPILE_TARGETS))), "backend": _cfg_get(cfg, "torch_compile_backend", "inductor"), "mode": _cfg_get(cfg, "torch_compile_mode", "default"), "fullgraph": bool(_cfg_get(cfg, "torch_compile_fullgraph", False)), "optimize_ddp": bool(_cfg_get(cfg, "torch_compile_optimize_ddp", True))},
    }


def _backend_name(distributed):
    return str(distributed.get_backend()).lower().split(".")[-1]


def _broadcast_root_identity(value, accelerator):
    if int(accelerator.num_processes) == 1:
        return value
    import torch.distributed as distributed

    if not distributed.is_available() or not distributed.is_initialized():
        raise RuntimeError(f"Inductor cache identity broadcast requires an initialized process group for world_size={accelerator.num_processes}")
    values = [value if accelerator.is_main_process else None]
    kwargs = {"src": 0}
    if _backend_name(distributed) == "nccl":
        kwargs["device"] = accelerator.device
    distributed.broadcast_object_list(values, **kwargs)
    return values[0]


def _atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def configure_persistent_inductor_cache(cfg, exp_dir, accelerator, repo_root=None, env=None, source_identity_fn=source_code_identity, runtime_identity_fn=inductor_runtime_identity):
    if not compile_module_names(getattr(cfg, "torch_compile_targets", _DEFAULT_COMPILE_TARGETS)):
        return None
    env = os.environ if env is None else env
    repo_root = Path.cwd() if repo_root is None else Path(repo_root)
    root_identity = None
    if accelerator.is_main_process:
        try:
            root_identity = {"ok": True, "source": source_identity_fn(repo_root), "config": compile_config_identity(cfg)}
        except Exception as exc:
            root_identity = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    root_identity = _broadcast_root_identity(root_identity, accelerator)
    if not root_identity["ok"]:
        raise RuntimeError(f"Failed to configure persistent Inductor cache: {root_identity['error_type']}: {root_identity['error']}")
    runtime_identity = runtime_identity_fn(cfg, accelerator.device)
    identity_payload = {"protocol_version": _INDUCTOR_CACHE_PROTOCOL_VERSION, "source": root_identity["source"], "config": root_identity["config"], "runtime": runtime_identity, "world_size": int(accelerator.num_processes)}
    identity = _identity_digest(identity_payload)
    configured_root = getattr(cfg, "torch_inductor_cache_root", None)
    cache_root = Path(exp_dir) / "torchinductor-cache" if configured_root is None else Path(configured_root).expanduser()
    if not cache_root.is_absolute():
        cache_root = Path(exp_dir) / cache_root
    cache_dir = cache_root / f"v{_INDUCTOR_CACHE_PROTOCOL_VERSION}-{identity[:20]}" / f"rank-{int(accelerator.process_index):05d}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    env["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    (cache_dir / "triton").mkdir(parents=True, exist_ok=True)
    manifest = {"identity": identity, "cache_dir": str(cache_dir), "rank": int(accelerator.process_index), **identity_payload}
    _atomic_write_json(cache_dir / "cache_identity.json", manifest)
    return manifest


def ddp_options(cfg):
    return {
        "find_unused_parameters": bool(_cfg_get(cfg, "ddp_find_unused_parameters", False)),
        "static_graph": bool(_cfg_get(cfg, "ddp_static_graph", True)),
    }


def compile_module_names(targets):
    if not targets:
        return ()
    if isinstance(targets, str):
        targets = [targets]
    method_specs = []
    for target in targets:
        if target not in _COMPILE_TARGET_MODULES:
            choices = ", ".join(sorted(_COMPILE_TARGET_MODULES))
            raise ValueError(f"Unknown torch.compile target {target!r}; expected one of: {choices}")
        for method_spec in _COMPILE_TARGET_MODULES[target]:
            if method_spec not in method_specs:
                method_specs.append(method_spec)
    return tuple(method_specs)


def configure_torch_compile(cfg, dynamo_config):
    optimize_ddp = bool(_cfg_get(cfg, "torch_compile_optimize_ddp", True))
    dynamo_config.optimize_ddp = optimize_ddp
    return optimize_ddp


def apply_compile_targets(model, cfg, compile_fn=None, dynamo_config=None):
    method_specs = compile_module_names(getattr(cfg, "torch_compile_targets", _DEFAULT_COMPILE_TARGETS))
    if not method_specs:
        return ()
    if compile_fn is None:
        import torch
        configure_torch_compile(cfg, torch._dynamo.config)
        compile_fn = torch.compile
    elif dynamo_config is not None:
        configure_torch_compile(cfg, dynamo_config)
    kwargs = {
        "backend": _cfg_get(cfg, "torch_compile_backend", "inductor"),
        "mode": _cfg_get(cfg, "torch_compile_mode", "default"),
        "fullgraph": bool(_cfg_get(cfg, "torch_compile_fullgraph", False)),
    }
    compiled_names = []
    for module_name, method_name in method_specs:
        module = getattr(model, module_name)
        method = getattr(module, method_name)
        setattr(module, method_name, compile_fn(method, **kwargs))
        compiled_names.append(f"{module_name}.{method_name}")
    return tuple(compiled_names)
