"""
Download NLF model weights and SMPL body models.

NLF weights: from GitHub releases (isarandi/nlf v0.3.2)
SMPL weights: from Google Drive (user-provided archive)

Expected output structure:
  output_dir/
    nlf_l_multi_0.3.2.torchscript   (NLF model)
    smpl/
      smplh/
        SMPLH_male.pkl, SMPLH_female.pkl
      SMPL_MALE.pkl, SMPL_FEMALE.pkl, SMPL_neutral.pkl (optional, at top level)
"""
import argparse
import glob
import os
import shutil
import subprocess
import zipfile
import tarfile


NLF_WEIGHTS_URL = "https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript"
NLF_WEIGHTS_FILENAME = "nlf_l_multi_0.3.2.torchscript"

SMPL_GDRIVE_FILE_ID = "1W-QY9qFUJSHLc2l_ITabBwDUQA_ypcZG"


def _download_nlf_weights(output_dir: str) -> None:
    weights_path = os.path.join(output_dir, NLF_WEIGHTS_FILENAME)
    if os.path.isfile(weights_path):
        print(f"NLF weights already exist at {weights_path}, skipping.")
        return

    print(f"Downloading NLF weights to {weights_path} ...")
    subprocess.run(
        ["wget", "-O", weights_path, NLF_WEIGHTS_URL],
        check=True,
    )


def _organize_smplh_files(smpl_dir: str, extract_dir: str) -> None:
    """Walk extracted files and organize SMPLH models into expected structure."""
    smplh_dir = os.path.join(smpl_dir, "smplh")
    os.makedirs(smplh_dir, exist_ok=True)

    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            flow = fname.lower()

            if "smplh" in flow and ("male" in flow or "female" in flow or "neutral" in flow):
                dest = os.path.join(smplh_dir, fname)
                if not os.path.exists(dest):
                    shutil.copy2(fpath, dest)
                    print(f"  Copied {fname} -> {dest}")

            if flow == "model.npz" or flow == "model.pkl":
                parent = os.path.basename(root).lower()
                if parent in ("male", "female", "neutral"):
                    gender_dir = os.path.join(smplh_dir, parent)
                    os.makedirs(gender_dir, exist_ok=True)
                    dest = os.path.join(gender_dir, fname)
                    if not os.path.exists(dest):
                        shutil.copy2(fpath, dest)
                        print(f"  Copied {parent}/{fname} -> {dest}")

    found = glob.glob(os.path.join(smplh_dir, "**/*"), recursive=True)
    found_files = [f for f in found if os.path.isfile(f)]
    if found_files:
        print(f"SMPLH model files organized in {smplh_dir}:")
        for f in sorted(found_files):
            print(f"  {os.path.relpath(f, smpl_dir)}")
    else:
        print(f"WARNING: No SMPLH model files found in the download.")
        print(f"Please manually place SMPLH model files into {smplh_dir}/")
        print(f"Expected: SMPLH_MALE.npz, SMPLH_FEMALE.npz")
        print(f"      or: male/model.npz, female/model.npz")


def _has_smpl_models(candidate_dir: str) -> bool:
    """Check if directory has SMPL/SMPLH model files."""
    smplh_dir = os.path.join(candidate_dir, "smplh")
    smplh_files = []
    if os.path.isdir(smplh_dir):
        smplh_files = glob.glob(os.path.join(smplh_dir, "**/*.npz"), recursive=True) + \
                      glob.glob(os.path.join(smplh_dir, "**/*.pkl"), recursive=True)
    smpl_at_root = any(
        os.path.isfile(os.path.join(candidate_dir, n))
        for n in ("SMPL_MALE.pkl", "SMPL_FEMALE.pkl", "SMPL_neutral.pkl",
                  "SMPLH_male.pkl", "SMPLH_female.pkl")
    )
    return bool(smplh_files) or smpl_at_root


def _download_smpl_weights(output_dir: str) -> None:
    smpl_dir = os.path.join(output_dir, "smpl")
    smplh_dir = os.path.join(smpl_dir, "smplh")

    if _has_smpl_models(smpl_dir):
        print(f"SMPL models already exist in {smpl_dir}, skipping.")
        return

    os.makedirs(smpl_dir, exist_ok=True)
    tmp_download = os.path.join(output_dir, "_smpl_download_tmp")
    os.makedirs(tmp_download, exist_ok=True)

    print(f"Downloading SMPL body models from Google Drive ...")
    subprocess.run(
        [
            "gdown",
            f"https://drive.google.com/uc?id={SMPL_GDRIVE_FILE_ID}",
            "-O", os.path.join(tmp_download, "smpl_download"),
        ],
        check=True,
    )

    downloaded = os.path.join(tmp_download, "smpl_download")
    if not os.path.exists(downloaded):
        print("ERROR: Download failed.")
        return

    if zipfile.is_zipfile(downloaded):
        print("Extracting zip archive ...")
        with zipfile.ZipFile(downloaded, 'r') as z:
            z.extractall(tmp_download)
        os.remove(downloaded)
        _organize_smplh_files(smpl_dir, tmp_download)
    elif tarfile.is_tarfile(downloaded):
        print("Extracting tar archive ...")
        with tarfile.open(downloaded) as t:
            t.extractall(tmp_download)
        os.remove(downloaded)
        _organize_smplh_files(smpl_dir, tmp_download)
    else:
        ext = None
        with open(downloaded, 'rb') as f:
            header = f.read(4)
        if header[:2] == b'\x93N':
            ext = '.npz'
        elif header[:2] == b'\x80\x02' or header[:1] == b'\x80':
            ext = '.pkl'

        if ext:
            dest = os.path.join(smplh_dir, f"SMPLH_MALE{ext}")
            os.makedirs(smplh_dir, exist_ok=True)
            shutil.move(downloaded, dest)
            print(f"Single model file saved to {dest}")
            print(f"NOTE: You may need to also download SMPLH_FEMALE{ext}")
        else:
            print(f"WARNING: Could not determine file type of downloaded SMPL file.")
            print(f"File saved at: {downloaded}")
            print(f"Please manually organize into {smplh_dir}/")

    shutil.rmtree(tmp_download, ignore_errors=True)


def download_weights(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    _download_nlf_weights(output_dir)
    _download_smpl_weights(output_dir)

    print("\nDownload complete.")
    print(f"  NLF weights: {os.path.join(output_dir, NLF_WEIGHTS_FILENAME)}")
    print(f"  SMPL models: {os.path.join(output_dir, 'smpl')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NLF and SMPL weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for weights")
    args = parser.parse_args()
    download_weights(args.output_dir)
