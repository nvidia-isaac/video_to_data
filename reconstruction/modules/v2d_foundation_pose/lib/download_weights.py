# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Download FoundationPose weights from Google Drive.

Scorer weights (2024-01-11-20-02-45): config.yml + model_best.pth
Refiner weights (2023-10-28-18-33-37): config.yml + model_best.pth

The FoundationPose predictors look for weights under FOUNDATIONPOSE_WEIGHTS_DIR
in subdirectories named by their run_name timestamps.
"""
import argparse
import os
import subprocess


SCORER_FOLDER_ID = "12Te_3TELLes5cim1d7F7EBTwUSe7iRBj"
SCORER_RUN_NAME = "2024-01-11-20-02-45"

REFINER_FOLDER_ID = "1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC"
REFINER_RUN_NAME = "2023-10-28-18-33-37"


def download_weights(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    scorer_dir = os.path.join(output_dir, SCORER_RUN_NAME)
    refiner_dir = os.path.join(output_dir, REFINER_RUN_NAME)

    scorer_ckpt = os.path.join(scorer_dir, "model_best.pth")
    if os.path.isfile(scorer_ckpt):
        print(f"Scorer weights already exist at {scorer_dir}, skipping.")
    else:
        print(f"Downloading scorer weights to {scorer_dir} ...")
        subprocess.run(
            [
                "gdown", "--folder",
                f"https://drive.google.com/drive/folders/{SCORER_FOLDER_ID}",
                "-O", scorer_dir,
            ],
            check=True,
        )

    refiner_ckpt = os.path.join(refiner_dir, "model_best.pth")
    if os.path.isfile(refiner_ckpt):
        print(f"Refiner weights already exist at {refiner_dir}, skipping.")
    else:
        print(f"Downloading refiner weights to {refiner_dir} ...")
        subprocess.run(
            [
                "gdown", "--folder",
                f"https://drive.google.com/drive/folders/{REFINER_FOLDER_ID}",
                "-O", refiner_dir,
            ],
            check=True,
        )

    print("All FoundationPose weights downloaded successfully.")
    print(f"  Scorer:  {scorer_dir}")
    print(f"  Refiner: {refiner_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FoundationPose weights from Google Drive")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for weights")
    args = parser.parse_args()
    download_weights(args.output_dir)
