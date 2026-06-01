# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for rosbag-to-EDEX extraction."""

import pathlib

import pydantic
from rosbags.typesys import get_typestore, Stores
from rosbags.typesys.store import Typestore


ROS_TYPESTORES = {
    "empty": Stores.EMPTY,
    "latest": Stores.LATEST,
    "noetic": Stores.ROS1_NOETIC,
    "dashing": Stores.ROS2_DASHING,
    "eloquent": Stores.ROS2_ELOQUENT,
    "foxy": Stores.ROS2_FOXY,
    "galactic": Stores.ROS2_GALACTIC,
    "humble": Stores.ROS2_HUMBLE,
    "iron": Stores.ROS2_IRON,
    "jazzy": Stores.ROS2_JAZZY,
    "kilted": Stores.ROS2_KILTED,
}


class Config(pydantic.BaseModel):
    """Configuration for the bag to EDEX converter."""

    rosbag_path: pathlib.Path
    output_path: pathlib.Path
    camera_info_topics: list[str]
    image_topics: list[str]
    imu_topic: str | None = None
    rig_frame: str = ""
    camera_optical_frames: list[str] | None = None
    imu_frame: str | None = None
    num_workers: int = -1
    sync_threshold_ns: int = int(0.001 * 10**9)
    output_width: int | None = None
    output_height: int | None = None
    output_format: str | None = None
    ros_distribution: str = "humble"
    pack_h5: bool = False
    remove_pngs_after_pack: bool = False

    @pydantic.model_validator(mode="after")
    def check_fields(self):
        if not self.rosbag_path.exists():
            raise ValueError(f"Path '{self.rosbag_path}' does not exist")
        if len(self.image_topics) != len(self.camera_info_topics):
            raise ValueError("Need same number of image topics as camera info topics.")
        if self.camera_optical_frames:
            if len(self.camera_optical_frames) != len(self.camera_info_topics):
                raise ValueError("Need same number of camera optical frames as camera info topics.")
        return self


def typestore_from_ros_distribution(ros_distribution: str) -> Typestore:
    """Get the rosbags typestore for a ROS distribution name."""
    if ros_distribution not in ROS_TYPESTORES:
        raise ValueError(f"Unknown ROS distribution: {ros_distribution}")
    return get_typestore(ROS_TYPESTORES[ros_distribution])
