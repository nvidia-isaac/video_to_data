"""
Downloads all SAM3D weights into a single directory:
  weights_dir/
    hf-download/      - SAM3D checkpoints (facebook/sam-3d-objects)
    hf_home/          - MoGE v1 model cache (Ruicheng/moge-vitl)
    torch_home/hub/   - DINOv2 checkpoints (matches torch.hub cache layout)
"""
import argparse
import getpass
import os
import subprocess
import urllib.request


DINOV2_URLS = {
    "dinov2_vitl14_reg4_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    "dinov2_vitb14_reg4_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
}


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
        "hf", "download", "facebook/sam-3d-objects",
        "--local-dir", hf_download_dir,
    ], check=True)

    print(f"Downloading MoGE v1 checkpoint (required for SAM3D)...")
    env = os.environ.copy()
    env["HF_HOME"] = hf_home_dir
    subprocess.run([
        "hf", "download", "Ruicheng/moge-vitl",
    ], check=True, env=env)

    for filename, url in DINOV2_URLS.items():
        dest = os.path.join(torch_ckpt_dir, filename)
        if os.path.exists(dest):
            print(f"DINOv2 {filename} already exists, skipping.")
            continue
        print(f"Downloading DINOv2 {filename}...")
        urllib.request.urlretrieve(url, dest)

    print("All SAM3D weights downloaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all SAM3D weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for all weights")
    args = parser.parse_args()
    download_weights(args.output_dir)
