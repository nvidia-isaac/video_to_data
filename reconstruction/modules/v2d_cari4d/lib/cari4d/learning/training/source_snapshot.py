import io
import json
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path


# Dataset payloads must never be stored in this repository. Directories named
# `datasets` are source packages and must remain eligible for source snapshots.
EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "wandb",
    "experiments",
    "outputs",
    "output",
    "debug",
    "logs",
    "data",
    "node_modules",
}
EXCLUDED_FILE_NAMES = {
    ".env",
    ".netrc",
    "credentials",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}
EXCLUDED_SUFFIXES = {
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".tar",
    ".tgz",
    ".gz",
    ".zip",
    ".sqsh",
    ".glb",
    ".obj",
    ".ply",
    ".png",
    ".jpg",
    ".jpeg",
}


def safe_wandb_artifact_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "run"


def _git_text(repo_root, *args):
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except Exception:
        return ""


def _git_metadata(repo_root):
    status = _git_text(repo_root, "status", "--short")
    return {
        "commit": _git_text(repo_root, "rev-parse", "HEAD"),
        "branch": _git_text(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def wandb_launch_metadata_from_env(env=None):
    env = os.environ if env is None else env
    metadata = {}
    env_keys = {
        "launch_command": "MHR_LAUNCH_COMMAND",
        "submit_command": "MHR_SUBMIT_COMMAND",
        "slurm_job_id": "SLURM_JOB_ID",
        "submit_script": "MHR_SUBMIT_SCRIPT",
    }
    for key, env_key in env_keys.items():
        value = env.get(env_key)
        if value:
            metadata[key] = str(value)
    if "slurm_job_id" not in metadata and env.get("SLURM_JOBID"):
        metadata["slurm_job_id"] = str(env["SLURM_JOBID"])
    return metadata


def save_resolved_config(cfg, exp_dir, filename="resolved_config.yaml"):
    from omegaconf import OmegaConf

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = exp_dir / filename
    OmegaConf.save(config=cfg, f=str(resolved_config_path), resolve=True)
    return resolved_config_path


def should_include_source_file(path, repo_root, max_file_bytes=2 * 1024 * 1024):
    path = Path(path)
    repo_root = Path(repo_root)
    if not path.is_file():
        return False, "not_file"
    rel = path.relative_to(repo_root)
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts[:-1]):
        return False, "excluded_dir"
    name = rel.name
    lower_name = name.lower()
    if lower_name in EXCLUDED_FILE_NAMES or lower_name.startswith(".env"):
        return False, "secret_name"
    if lower_name.endswith((".pem", ".key")):
        return False, "secret_suffix"
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False, "excluded_suffix"
    size = path.stat().st_size
    if size > max_file_bytes:
        return False, "too_large"
    return True, "included"


def iter_source_files(repo_root, max_file_bytes=2 * 1024 * 1024):
    repo_root = Path(repo_root).resolve()
    included = []
    excluded = []
    for path in sorted(repo_root.rglob("*")):
        include, reason = should_include_source_file(path, repo_root, max_file_bytes=max_file_bytes)
        rel = path.relative_to(repo_root).as_posix()
        if include:
            included.append((path, rel, path.stat().st_size))
        elif path.is_file():
            excluded.append({"path": rel, "reason": reason, "bytes": path.stat().st_size})
    return included, excluded


def create_source_snapshot_archive(repo_root, out_dir, run_name, max_file_bytes=2 * 1024 * 1024):
    repo_root = Path(repo_root).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_run = safe_wandb_artifact_name(run_name)
    archive_path = out_dir / f"{safe_run}_source_snapshot.tar.gz"
    manifest_path = out_dir / f"{safe_run}_source_manifest.json"
    included, excluded = iter_source_files(repo_root, max_file_bytes=max_file_bytes)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_root": str(repo_root),
        "run_name": str(run_name),
        "max_file_bytes": int(max_file_bytes),
        "included_count": len(included),
        "included_bytes": int(sum(size for _, _, size in included)),
        "excluded_count": len(excluded),
        "excluded_sample": excluded[:200],
        "git": _git_metadata(repo_root),
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path.write_text(manifest_text)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path, rel, _ in included:
            tar.add(path, arcname=rel, recursive=False)
        manifest_bytes = manifest_text.encode("utf-8")
        info = tarfile.TarInfo("source_manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(manifest_bytes))
    return archive_path, manifest


def log_wandb_source_snapshot(repo_root, exp_dir, run_name, wandb_run, wandb_module=None, max_file_bytes=2 * 1024 * 1024, launch_metadata=None, resolved_config_path=None):
    exp_dir = Path(exp_dir)
    marker_path = exp_dir / "wandb_source_snapshot_logged.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text())
        marker["logged"] = False
        return marker
    if wandb_module is None:
        import wandb as wandb_module
    archive_path, manifest = create_source_snapshot_archive(repo_root, exp_dir, run_name, max_file_bytes=max_file_bytes)
    artifact_name = f"{safe_wandb_artifact_name(run_name)}-code-backup"
    artifact_metadata = {
        "run_name": str(run_name),
        "git_commit": manifest["git"]["commit"],
        "git_branch": manifest["git"]["branch"],
        "git_dirty": manifest["git"]["dirty"],
        "included_count": manifest["included_count"],
        "included_bytes": manifest["included_bytes"],
        "excluded_count": manifest["excluded_count"],
        "max_file_bytes": manifest["max_file_bytes"],
    }
    if resolved_config_path is not None:
        artifact_metadata["resolved_config"] = "resolved_config.yaml"
    artifact_metadata.update(launch_metadata or {})
    artifact = wandb_module.Artifact(
        artifact_name,
        type="source-code",
        metadata=artifact_metadata,
        description=f"Source snapshot for W&B run {run_name}",
    )
    artifact.add_file(str(archive_path), name=archive_path.name)
    if resolved_config_path is not None:
        artifact.add_file(str(resolved_config_path), name="resolved_config.yaml")
    logged_artifact = wandb_run.log_artifact(artifact)
    logged_artifact.wait()
    marker = {
        "logged": True,
        "artifact_name": artifact_name,
        "archive_name": archive_path.name,
        "archive_path": str(archive_path),
        "manifest_path": str(exp_dir / f"{safe_wandb_artifact_name(run_name)}_source_manifest.json"),
        "included_count": manifest["included_count"],
        "included_bytes": manifest["included_bytes"],
        "excluded_count": manifest["excluded_count"],
    }
    if resolved_config_path is not None:
        marker["resolved_config"] = "resolved_config.yaml"
        marker["resolved_config_path"] = str(resolved_config_path)
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True))
    print(f"W&B source snapshot logged: artifact={artifact_name}, archive={archive_path}, files={manifest['included_count']}", flush=True)
    return marker
