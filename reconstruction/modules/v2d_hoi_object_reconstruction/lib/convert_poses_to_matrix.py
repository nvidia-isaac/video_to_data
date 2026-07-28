# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert FoundationPose quaternion JSON files to 4x4 matrices in place."""

import argparse
import json
from pathlib import Path

from v2d.common.datatypes import Transform3d


def convert_poses_to_matrix(poses_dir: str | Path) -> int:
    """Convert pose files and return the number changed."""
    converted = 0
    for pose_file in sorted(Path(poses_dir).glob("*.json")):
        pose = json.loads(pose_file.read_text())
        if isinstance(pose, list):
            continue
        matrix = Transform3d.from_dict(pose).to_matrix()
        pose_file.write_text(json.dumps(matrix.tolist()))
        converted += 1
    print(f"Converted {converted} poses to 4x4 matrices")
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert FoundationPose JSON files to 4x4 matrices in place"
    )
    parser.add_argument("--poses_dir", required=True)
    args = parser.parse_args()
    convert_poses_to_matrix(args.poses_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
