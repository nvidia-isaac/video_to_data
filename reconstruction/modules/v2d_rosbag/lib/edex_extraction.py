# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full EDEX extraction including camera extrinsics from TF tree and optional IMU."""

import json
import logging
import math
import pathlib
import shutil
import sys
from typing import Any

import numpy as np
from pytransform3d import transform_manager
from rosbags import highlevel

from v2d.mv.rig.edex import (
    Camera,
    EDEXBody,
    EDEXHeader,
    EDEXMetadata,
    IMU,
)

from v2d.rosbag.lib.config import Config, typestore_from_ros_distribution
from v2d.rosbag.lib.image_extraction import (
    _get_camera_intrinsics,
    _get_first_message,
    _image_path,
    image_extract_from_rosbag,
)
from v2d.rosbag.lib.tf_extraction import tf_static_manager_from_rosbag


logger = logging.getLogger(__name__)


# Coordinate frame conversions
# fmt: off
CV_FROM_ROS = np.array([
    [ 0, -1, 0, 0],
    [ 0,  0, 1, 0],
    [-1,  0, 0, 0],
    [ 0,  0, 0, 1],
])
ROS_FROM_CV = np.linalg.inv(CV_FROM_ROS)
ROS_CAMERA_OPTICAL_FROM_CV_CAMERA_OPTICAL = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
])
# fmt: on


def _pose_matrix_to_edex(pose_matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 pose matrix to the 3x4 format EDEX expects."""
    assert pose_matrix.shape == (4, 4)
    assert math.isclose(np.linalg.det(pose_matrix), 1.0)
    return pose_matrix[0:3, :]


def _get_camera_metadata(
    camera_idx: int,
    camera_msg: Any,
    tf_manager: transform_manager.TransformManager,
    config: Config,
) -> Camera:
    """Create Camera metadata with intrinsics and extrinsic transform from TF tree."""
    if config.camera_optical_frames:
        camera_optical_frame = config.camera_optical_frames[camera_idx]
    else:
        camera_optical_frame = camera_msg.header.frame_id

    ros_rig_from_ros_camera = tf_manager.get_transform(camera_optical_frame, config.rig_frame)
    cv_rig_from_cv_camera = CV_FROM_ROS @ ros_rig_from_ros_camera @ ROS_CAMERA_OPTICAL_FROM_CV_CAMERA_OPTICAL

    return Camera(
        intrinsics=_get_camera_intrinsics(camera_msg, config),
        transform=_pose_matrix_to_edex(cv_rig_from_cv_camera),
    )


def _get_imu_metadata(
    imu_msg: Any,
    tf_manager: transform_manager.TransformManager,
    config: Config,
) -> IMU:
    """Create IMU metadata from TF tree."""
    if not config.imu_frame:
        config.imu_frame = imu_msg.header.frame_id

    ros_rig_from_ros_imu = tf_manager.get_transform(config.imu_frame, config.rig_frame)
    cv_rig_from_cv_imu = CV_FROM_ROS @ ros_rig_from_ros_imu @ ROS_FROM_CV
    return IMU(
        g=np.array([0.0, -9.81, 0.0], dtype=np.float32),
        measurements="imu.jsonl",
        transform=_pose_matrix_to_edex(cv_rig_from_cv_imu),
    )


def _extract_imu_stream(reader: highlevel.AnyReader, config: Config):
    """Extract IMU measurements from the bag to a JSONL file."""
    imu_path = config.output_path / "imu.jsonl"
    logger.info(f"Writing IMU data to '{imu_path}'.")
    with open(imu_path, "w", encoding="utf-8") as file:
        connections = [c for c in reader.connections if c.topic == config.imu_topic]
        for connection, _, rawdata in reader.messages(connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            imu_data = {
                "timestamp": msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
                "AngularVelocityX": msg.angular_velocity.x,
                "AngularVelocityY": msg.angular_velocity.y,
                "AngularVelocityZ": msg.angular_velocity.z,
                "LinearAccelerationX": msg.linear_acceleration.x,
                "LinearAccelerationY": msg.linear_acceleration.y,
                "LinearAccelerationZ": msg.linear_acceleration.z,
            }
            json.dump(imu_data, file)
            file.write("\n")


def _build_edex_metadata(
    reader: highlevel.AnyReader,
    tf_manager: transform_manager.TransformManager,
    config: Config,
    num_frames: int,
):
    """Create full EDEX metadata with camera extrinsics from TF tree."""
    camera_info_msgs = _get_first_message(reader, config.camera_info_topics)
    cameras_metadata = [
        _get_camera_metadata(idx, msg, tf_manager, config)
        for idx, msg in enumerate(camera_info_msgs)
    ]

    imu_metadata = None
    if config.imu_topic:
        imu_msg = _get_first_message(reader, [config.imu_topic])[0]
        imu_metadata = _get_imu_metadata(imu_msg, tf_manager, config)

    sequence_paths = [
        _image_path(pathlib.Path("images"), topic, 0)
        for topic in config.image_topics
    ]

    edex_header = EDEXHeader(
        version="0.9",
        frame_start=0,
        frame_end=num_frames,
        cameras=cameras_metadata,
        imu=imu_metadata,
    )
    edex_body = EDEXBody(
        frame_metadata="frame_metadata.jsonl",
        sequence=sequence_paths,
    )
    edex_metadata = EDEXMetadata(header=edex_header, body=edex_body)

    edex_path = config.output_path / "edex"
    logger.info(f"Writing full EDEX metadata to '{edex_path}'.")
    edex_metadata.write(edex_path)


def _log_rosbag_info(reader: highlevel.AnyReader):
    """Log topics and message types found in the rosbag."""
    logs = sorted(f"\t- {c.topic}: {c.msgtype}" for c in reader.connections)
    logger.info("Found the following topics in rosbag:\n" + "\n".join(logs))


def rosbag_to_edex(config: Config):
    """Full EDEX extraction: images + TF extrinsics + optional IMU.

    This is the main entry point for complete EDEX extraction from a rosbag.
    For images-only extraction (without extrinsics from TF), use
    ``rosbag_extract_images`` instead.
    """
    # Clean output
    shutil.rmtree(config.output_path / "images", ignore_errors=True)
    for name in ("edex", "frame_metadata.jsonl", "imu.jsonl"):
        (config.output_path / name).unlink(missing_ok=True)
    config.output_path.mkdir(parents=True, exist_ok=True)

    # Extract static TF tree
    tf_manager = tf_static_manager_from_rosbag(config.rosbag_path, config.ros_distribution)
    logger.info(
        "Found the following frames in rosbag:\n"
        + "\n".join(f"\t- {node}" for node in tf_manager.nodes)
    )
    if config.rig_frame and config.rig_frame not in tf_manager.nodes:
        logger.error(
            f"Rig frame '{config.rig_frame}' not found in rosbag. "
            f"Available frames: {list(tf_manager.nodes)}"
        )
        sys.exit(1)

    with highlevel.AnyReader(
        paths=[config.rosbag_path],
        default_typestore=typestore_from_ros_distribution(config.ros_distribution),
    ) as reader:
        _log_rosbag_info(reader)

        # Filter topics that exist in the bag
        bag_topics = [c.topic for c in reader.connections]
        image_topics = []
        camera_info_topics = []
        camera_optical_frames = []
        for idx, (img_topic, info_topic) in enumerate(
            zip(config.image_topics, config.camera_info_topics)
        ):
            if img_topic not in bag_topics:
                logger.warning(f"Topic '{img_topic}' not found in rosbag, skipping.")
            elif info_topic not in bag_topics:
                logger.warning(f"Topic '{info_topic}' not found in rosbag, skipping.")
            else:
                image_topics.append(img_topic)
                camera_info_topics.append(info_topic)
                if config.camera_optical_frames:
                    camera_optical_frames.append(config.camera_optical_frames[idx])

        config.image_topics = image_topics
        config.camera_info_topics = camera_info_topics
        config.camera_optical_frames = camera_optical_frames or None

        # Extract images and build EDEX
        num_frames = image_extract_from_rosbag(reader, config)
        _build_edex_metadata(reader, tf_manager, config, num_frames)

        if config.imu_topic:
            _extract_imu_stream(reader, config)

    logger.info(f"Finished extracting EDEX to '{config.output_path}'.")


def rosbag_extract_images(config: Config):
    """Images-only extraction: images + intrinsics EDEX (no extrinsics from TF).

    Use this when extrinsics will be calibrated separately.
    """
    shutil.rmtree(config.output_path / "images", ignore_errors=True)
    for name in ("edex", "frame_metadata.jsonl"):
        (config.output_path / name).unlink(missing_ok=True)
    config.output_path.mkdir(parents=True, exist_ok=True)

    with highlevel.AnyReader(
        paths=[config.rosbag_path],
        default_typestore=typestore_from_ros_distribution(config.ros_distribution),
    ) as reader:
        _log_rosbag_info(reader)

        bag_topics = [c.topic for c in reader.connections]
        image_topics = []
        camera_info_topics = []
        for img_topic, info_topic in zip(config.image_topics, config.camera_info_topics):
            if img_topic not in bag_topics:
                logger.warning(f"Topic '{img_topic}' not found in rosbag, skipping.")
            elif info_topic not in bag_topics:
                logger.warning(f"Topic '{info_topic}' not found in rosbag, skipping.")
            else:
                image_topics.append(img_topic)
                camera_info_topics.append(info_topic)

        config.image_topics = image_topics
        config.camera_info_topics = camera_info_topics

        from v2d.rosbag.lib.image_extraction import edex_build_from_rosbag
        num_frames = image_extract_from_rosbag(reader, config)
        edex_build_from_rosbag(reader, config, num_frames)

    logger.info(f"Finished extracting images to '{config.output_path}'.")
