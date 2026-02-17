# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Verify that tips_distance data in processed parquet is present and non-zero."""

import numpy as np
from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import ManoSharpaData

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def main() -> None:
    """Load processed parquet and verify tips_distance data."""
    processed_dir = str(HUMAN_MOTION_DATA_DIR / "arctic_processed")
    data = ManoSharpaData.from_parquet(
        processed_dir,
        filters=[
            ("robot_name", "=", "sharpa_wave"),
            ("sequence_id", "contains", "box_grab"),
        ],
    )

    right = np.array(data.mano_right_tips_distance)  # (T, 5)
    left = np.array(data.mano_left_tips_distance)  # (T, 5)

    print(f"Sequence: {data.sequence_id}")
    print(f"Frames: {len(right)}")
    print(f"Right shape: {right.shape}, Left shape: {left.shape}")
    print(f"FPS: {data.fps}")

    # Frame count should match other time-series columns
    n_joints = len(data.mano_right_joints)
    print(f"Joint frames: {n_joints}")
    assert len(right) == n_joints, (
        f"tips_distance frame count ({len(right)}) != joints frame count ({n_joints}). "
        "Some frames may have been logged as None."
    )

    # Global stats
    print(
        f"\nRight - min: {right.min():.6f}, max: {right.max():.6f}, mean: {right.mean():.6f}"
    )
    print(
        f"Left  - min: {left.min():.6f}, max: {left.max():.6f}, mean: {left.mean():.6f}"
    )
    print(f"Right all-zero frames: {(right.sum(axis=1) == 0).sum()} / {len(right)}")
    print(f"Left  all-zero frames: {(left.sum(axis=1) == 0).sum()} / {len(left)}")

    # Per-finger stats
    print("\nPer-finger mean distance (meters):")
    for i, name in enumerate(FINGERS):
        print(
            f"  Right {name:7s}: {right[:, i].mean():.4f}  Left {name:7s}: {left[:, i].mean():.4f}"
        )

    # Assertions
    assert right.min() >= 0, "Negative distance found in right"
    assert left.min() >= 0, "Negative distance found in left"
    assert right.max() < 1.0, f"Implausibly large right distance: {right.max()}"
    assert left.max() < 1.0, f"Implausibly large left distance: {left.max()}"
    assert (right > 0).any(), "Right tips_distance is all zeros"
    assert (left > 0).any(), "Left tips_distance is all zeros"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
