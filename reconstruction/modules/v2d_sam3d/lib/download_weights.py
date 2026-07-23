# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Downloads all SAM3D weights into a single directory:
  weights_dir/
    hf-download/      - SAM3D checkpoints (facebook/sam-3d-objects)
    hf_home/          - MoGE v1 model cache (Ruicheng/moge-vitl)
    torch_home/hub/   - DINOv2 checkpoints (matches torch.hub cache layout)
"""
import argparse
import getpass
import hashlib
import os
from pathlib import Path
import subprocess
import urllib.request


DINOV2_URLS = {
    "dinov2_vitl14_reg4_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    "dinov2_vitb14_reg4_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
}

SAM3D_REPOSITORY = "facebook/sam-3d-objects"
SAM3D_REVISION = "2e73555018d2741ccd486e56c24fac41155a1dc6"
MOGE_REPOSITORY = "Ruicheng/moge-vitl"
MOGE_REVISION = "979e84da9415762c30e6c0cf8dc0962896c793df"
MOGE_MODEL_BYTES = 1_256_823_446
MOGE_MODEL_SHA256 = (
    "da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f"
)
SAM3D_CHECKPOINTS = {
    "slat_decoder_gs.ckpt": (
        171_476_155,
        "f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111",
    ),
    "slat_decoder_gs_4.ckpt": (
        170_269_801,
        "731a0eceaa47945b52aa27f650d695b2aea9cc70945751e5609e5cb5b49f0186",
    ),
    "slat_decoder_mesh.ckpt": (
        363_726_862,
        "85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33",
    ),
    "slat_decoder_mesh.pt": (
        363_728_714,
        "93333fcd57a3e36ded0b3bca6969e05ce2b35142029dadab514f41df46d2f985",
    ),
    "slat_encoder.ckpt": (
        173_263_986,
        "6485623145535f42c8afa4cbb68ab9953e54e2f0c1cb1eaf95dcb41051e10181",
    ),
    "slat_generator.ckpt": (
        4_906_537_684,
        "91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb",
    ),
    "ss_decoder.ckpt": (
        147_609_242,
        "6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340",
    ),
    "ss_encoder.ckpt": (
        119_085_402,
        "dcc47810ac568b11fe6e4821ea1c8d6b960dfbda3e5f68e94c19f44b3bf9e83b",
    ),
    "ss_generator.ckpt": (
        6_690_136_964,
        "225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146",
    ),
}
DINOV2_EXPECTED = {
    "dinov2_vitl14_reg4_pretrain.pth": (
        1_217_607_321,
        "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
    ),
    "dinov2_vitb14_reg4_pretrain.pth": (
        346_393_545,
        "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Checkpoint size mismatch for {path}: expected {expected_bytes}, "
            f"got {path.stat().st_size}"
        )
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch for {path}: expected "
            f"{expected_sha256}, got {actual}"
        )


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
    print("SAM 3D Objects (facebook/sam-3d-objects) is a gated repo that requires authentication.")
    print("Request access at: https://huggingface.co/facebook/sam-3d-objects")
    print()
    token = getpass.getpass("Enter your Hugging Face token (or Ctrl+C to abort): ")
    os.environ["HF_TOKEN"] = token


def download_weights(output_dir: str):
    hf_download_dir = os.path.join(output_dir, "hf-download")
    hf_home_dir = os.path.join(output_dir, "hf_home")
    torch_ckpt_dir = os.path.join(output_dir, "torch_home", "hub", "checkpoints")

    os.makedirs(hf_download_dir, exist_ok=True)
    os.makedirs(hf_home_dir, exist_ok=True)
    os.makedirs(torch_ckpt_dir, exist_ok=True)

    _ensure_hf_token()

    print(f"Downloading SAM 3D checkpoints to {hf_download_dir}...")
    subprocess.run([
        "hf", "download", SAM3D_REPOSITORY,
        "--revision", SAM3D_REVISION,
        "--include", "checkpoints/*",
        "--local-dir", hf_download_dir,
    ], check=True)
    for filename, (expected_bytes, expected_sha256) in SAM3D_CHECKPOINTS.items():
        _verify(
            Path(hf_download_dir) / "checkpoints" / filename,
            expected_bytes,
            expected_sha256,
        )

    print("Downloading MoGE v1 checkpoint (required for SAM3D)...")
    env = os.environ.copy()
    env["HF_HOME"] = hf_home_dir
    subprocess.run([
        "hf", "download", MOGE_REPOSITORY, "--revision", MOGE_REVISION,
    ], check=True, env=env)
    _verify(
        Path(hf_home_dir)
        / "hub"
        / "models--Ruicheng--moge-vitl"
        / "snapshots"
        / MOGE_REVISION
        / "model.pt",
        MOGE_MODEL_BYTES,
        MOGE_MODEL_SHA256,
    )

    for filename, url in DINOV2_URLS.items():
        dest = Path(torch_ckpt_dir) / filename
        if dest.exists():
            print(f"DINOv2 {filename} already exists, skipping.")
        else:
            print(f"Downloading DINOv2 {filename}...")
            urllib.request.urlretrieve(url, dest)
        _verify(dest, *DINOV2_EXPECTED[filename])

    print("All SAM3D weights downloaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all SAM3D weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for all weights")
    args = parser.parse_args()
    download_weights(args.output_dir)
