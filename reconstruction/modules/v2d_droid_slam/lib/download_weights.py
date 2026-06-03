# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Download the DROID-SLAM checkpoint (droid.pth) into a local weights folder.

Upstream hosts the file on Google Drive — we use gdown to fetch it.
"""
import argparse
import os

import gdown

DROID_GDRIVE_FILE_ID = "1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh"


def download_droid(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "droid.pth")
    if os.path.exists(out_path):
        print(f"droid.pth already present at {out_path} — skipping download")
        return
    url = f"https://drive.google.com/uc?id={DROID_GDRIVE_FILE_ID}"
    gdown.download(url, out_path, quiet=False)
    print(f"DROID-SLAM checkpoint downloaded to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DROID-SLAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for droid.pth")
    args = parser.parse_args()
    download_droid(args.output_dir)
