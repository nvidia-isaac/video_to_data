# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import getpass
import os
import subprocess
import tempfile
from pathlib import Path

DINOV3_REPO = "https://github.com/facebookresearch/dinov3.git"
DINOV3_CACHE_DIR = "torch_home/hub/facebookresearch_dinov3_main"
SAM3D_REQUIRED_ARTIFACTS = (
    "model.ckpt",
    "model_config.yaml",
    "assets/mhr_model.pt",
)
MOGE_REQUIRED_ARTIFACTS = ("model.pt",)
DINOV3_REQUIRED_ARTIFACTS = (
    "hubconf.py",
    "dinov3/__init__.py",
)


def _missing_artifacts(
    directory: Path,
    required_artifacts: tuple[str, ...],
) -> list[str]:
    """Return required files that are absent or empty."""
    return [
        relative_path
        for relative_path in required_artifacts
        if not (directory / relative_path).is_file()
        or (directory / relative_path).stat().st_size == 0
    ]


def _require_artifacts(
    name: str,
    directory: Path,
    required_artifacts: tuple[str, ...],
) -> None:
    missing = _missing_artifacts(directory, required_artifacts)
    if missing:
        raise RuntimeError(
            f"{name} download completed without required nonempty artifacts: "
            + ", ".join(missing)
        )


def _replace_with_validated_directory(source: Path, destination: Path) -> None:
    """Replace an incomplete destination while restoring it if replacement fails."""
    backup = source.parent / "incomplete_previous"
    had_destination = os.path.lexists(destination)
    if had_destination:
        destination.replace(backup)
    try:
        source.replace(destination)
    except Exception:
        if had_destination and not os.path.lexists(destination):
            backup.replace(destination)
        raise


def _ensure_hf_token() -> None:
    """Check for HF_TOKEN and prompt the user if not found."""
    if os.environ.get("HF_TOKEN"):
        return

    result = subprocess.run(
        ["hf", "whoami"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return

    print("No HF_TOKEN found and you are not logged in to Hugging Face.")
    print("SAM 3D Body (facebook/sam-3d-body-dinov3) is a gated repo that requires authentication.")
    print("Request access at: https://huggingface.co/facebook/sam-3d-body-dinov3")
    print()
    token = getpass.getpass("Enter your Hugging Face token (or Ctrl+C to abort): ")
    os.environ["HF_TOKEN"] = token


def download_weights(output_dir: str) -> None:
    """Download SAM3D-Body, MoGe-2, and DINOv3 repo into a single weights directory.

    Layout:
      output_dir/
        sam-3d-body-dinov3/   - SAM3D-Body checkpoints (gated HuggingFace repo)
        moge-2-vitb-normal/   - MoGe-2 weights
        torch_home/hub/       - DINOv3 repo clone (for torch.hub.load)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sam3d_dir = output_dir / "sam-3d-body-dinov3"
    if _missing_artifacts(sam3d_dir, SAM3D_REQUIRED_ARTIFACTS):
        _ensure_hf_token()
        print("Downloading SAM 3D Body checkpoints...")
        subprocess.run(
            ["hf", "download", "facebook/sam-3d-body-dinov3",
             "--local-dir", str(sam3d_dir)],
            check=True,
        )
        _require_artifacts(
            "SAM 3D Body", sam3d_dir, SAM3D_REQUIRED_ARTIFACTS,
        )
        print("SAM 3D Body checkpoints downloaded.")
    else:
        print(f"SAM 3D Body checkpoints already exist at {sam3d_dir}")

    moge_dir = output_dir / "moge-2-vitb-normal"
    if _missing_artifacts(moge_dir, MOGE_REQUIRED_ARTIFACTS):
        print("Downloading MoGe-2 weights...")
        subprocess.run(
            ["hf", "download", "Ruicheng/moge-2-vitb-normal",
             "--local-dir", str(moge_dir)],
            check=True,
        )
        _require_artifacts("MoGe-2", moge_dir, MOGE_REQUIRED_ARTIFACTS)
        print("MoGe-2 weights downloaded.")
    else:
        print(f"MoGe-2 weights already exist at {moge_dir}")

    dinov3_dir = output_dir / DINOV3_CACHE_DIR
    if _missing_artifacts(dinov3_dir, DINOV3_REQUIRED_ARTIFACTS):
        print("Cloning DINOv3 repo (for torch.hub cache)...")
        dinov3_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".dinov3-download-", dir=dinov3_dir.parent,
        ) as temporary_dir:
            clone_dir = Path(temporary_dir) / "repository"
            subprocess.run(
                ["git", "clone", DINOV3_REPO, str(clone_dir)],
                check=True,
            )
            _require_artifacts("DINOv3", clone_dir, DINOV3_REQUIRED_ARTIFACTS)
            _replace_with_validated_directory(clone_dir, dinov3_dir)
        print("DINOv3 repo cached.")
    else:
        print(f"DINOv3 repo already cached at {dinov3_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SAM3D-Body and MoGe-2 weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    args = parser.parse_args()
    download_weights(args.output_dir)
