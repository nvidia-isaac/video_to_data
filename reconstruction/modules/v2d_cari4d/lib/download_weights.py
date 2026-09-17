# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download every model required by the CARI4D wild-video pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from v2d.foundation_pose.lib.backends import NVLABS_PYTORCH
from v2d.foundation_pose.lib.download_weights import download_weights as download_foundationpose_weights


CARI4D_REPO_ID = "nvidia/cari4d_commercial"
CARI4D_REVISION = "1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9"
CARI4D_RUN_ID = "2026-08-25-09-35-57"
CARI4D_CHECKPOINT_RELATIVE_PATH = f"{CARI4D_RUN_ID}/step200000.pth"
CARI4D_CONFIG_RELATIVE_PATH = f"{CARI4D_RUN_ID}/resolved_config.yaml"
CARI4D_MANIFEST_RELATIVE_PATH = f"{CARI4D_RUN_ID}/manifest.json"
CARI4D_CHECKPOINT_SHA256 = "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3"
SAM3D_REPO_ID = "facebook/sam-3d-body-dinov3"
SAM3D_REVISION = "11aaa346c7204874a1cbafe3d39a979080b2c55a"
MOGE2_REPO_ID = "Ruicheng/moge-2-vitl-normal"
MOGE2_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MOGE2_MODEL_FILENAME = "model.pt"
DINOV3_REPO_URL = "https://github.com/facebookresearch/dinov3.git"
DINOV3_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
DINOV2_REPO_URL = "https://github.com/facebookresearch/dinov2.git"
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_CHECKPOINT_SHA256 = {"dinov2_vitb14_pretrain.pth": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73", "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_cari4d(output_dir: Path) -> None:
    destination = output_dir / "cari4d"
    snapshot_download(repo_id=CARI4D_REPO_ID, revision=CARI4D_REVISION, allow_patterns=[CARI4D_CHECKPOINT_RELATIVE_PATH, CARI4D_CONFIG_RELATIVE_PATH, CARI4D_MANIFEST_RELATIVE_PATH], local_dir=destination)
    checkpoint = destination / CARI4D_CHECKPOINT_RELATIVE_PATH
    config = destination / CARI4D_CONFIG_RELATIVE_PATH
    manifest_path = destination / CARI4D_MANIFEST_RELATIVE_PATH
    for path in (checkpoint, config, manifest_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"CARI4D model artifact is missing: {path}")
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != CARI4D_CHECKPOINT_SHA256:
        raise ValueError(f"CARI4D checkpoint SHA-256 mismatch: expected={CARI4D_CHECKPOINT_SHA256}, actual={actual_sha256}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("checkpoint", {}).get("sha256") != actual_sha256 or int(manifest.get("checkpoint", {}).get("step", -1)) != 200000:
        raise ValueError(f"CARI4D model manifest does not identify the downloaded step-200000 checkpoint: {manifest_path}")


def _download_sam3d(output_dir: Path) -> None:
    destination = output_dir / "sam3d_body" / "checkpoints" / "sam-3d-body-dinov3"
    snapshot_download(repo_id=SAM3D_REPO_ID, revision=SAM3D_REVISION, local_dir=destination)
    for relative in ("model.ckpt", "assets/mhr_model.pt"):
        path = destination / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"SAM 3D Body artifact is missing: {path}")


def _moge2_cache_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "hf_home" / "hub"


def require_moge2_model(output_dir: str | Path) -> Path:
    model_path = Path(hf_hub_download(repo_id=MOGE2_REPO_ID, revision=MOGE2_REVISION, filename=MOGE2_MODEL_FILENAME, cache_dir=_moge2_cache_dir(output_dir), local_files_only=True))
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"Required MoGe 2 model is missing or empty: {model_path}")
    return model_path


def _download_moge2(output_dir: Path) -> None:
    snapshot_download(repo_id=MOGE2_REPO_ID, revision=MOGE2_REVISION, cache_dir=_moge2_cache_dir(output_dir))
    require_moge2_model(output_dir)


def _checkout_torch_hub_repo(destination: Path, repository_url: str, revision: str) -> None:
    if destination.is_dir() and (destination / ".git").is_dir():
        actual = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        if actual == revision:
            return
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth=1", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
        return
    if destination.exists():
        raise FileExistsError(f"Torch Hub cache exists but is not a Git checkout: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", repository_url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)


def _download_torch_hub_models(output_dir: Path) -> None:
    torch_home = output_dir / "sam3d_body" / "torch_home"
    os.environ["TORCH_HOME"] = str(torch_home)
    dinov3 = torch_home / "hub" / "facebookresearch_dinov3_main"
    dinov2 = torch_home / "hub" / "facebookresearch_dinov2_main"
    _checkout_torch_hub_repo(dinov3, DINOV3_REPO_URL, DINOV3_REVISION)
    _checkout_torch_hub_repo(dinov2, DINOV2_REPO_URL, DINOV2_REVISION)
    import torch
    for model_name in ("dinov2_vitb14", "dinov2_vits14"):
        torch.hub.load(str(dinov2), model_name, source="local", pretrained=True)
    for filename, expected_sha256 in DINOV2_CHECKPOINT_SHA256.items():
        path = torch_home / "hub" / "checkpoints" / filename
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"DINOv2 checkpoint is missing or differs from the validated model: {path}")


def download_weights(output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(output_dir / "hf_home")
    _download_cari4d(output_dir)
    _download_sam3d(output_dir)
    _download_moge2(output_dir)
    _download_torch_hub_models(output_dir)
    download_foundationpose_weights(str(output_dir / "foundationpose"), backend=NVLABS_PYTORCH)
    result = {"checkpoint": str(output_dir / "cari4d" / CARI4D_CHECKPOINT_RELATIVE_PATH), "config": str(output_dir / "cari4d" / CARI4D_CONFIG_RELATIVE_PATH), "sam3d_assets": str(output_dir / "sam3d_body"), "foundationpose_weights": str(output_dir / "foundationpose" / NVLABS_PYTORCH), "hf_home": str(output_dir / "hf_home")}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all CARI4D inference weights")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    download_weights(args.output_dir)


if __name__ == "__main__":
    main()
