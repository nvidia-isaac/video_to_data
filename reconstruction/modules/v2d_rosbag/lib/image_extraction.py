"""Extract and synchronize images from a ROS bag.

Handles both raw (sensor_msgs/msg/Image) and compressed
(sensor_msgs/msg/CompressedImage) image messages, using PyAV for decoding.
Uses a producer-consumer thread pool for parallel I/O.
"""

import concurrent.futures
import json
import logging
import os
import pathlib
import queue
import sys
import threading
import time
from typing import Any

import av
import numpy as np
import pandas as pd
from rosbags import highlevel

from v2d.mv.rig.edex import (
    Camera,
    DistortionModel,
    EDEXBody,
    EDEXHeader,
    EDEXMetadata,
    Intrinsics,
)

from v2d.rosbag.lib.config import Config, typestore_from_ros_distribution


logger = logging.getLogger(__name__)


DISTORTION_MODEL_ROS2EDEX = {
    "pinhole": DistortionModel.PINHOLE,
    "equidistant": DistortionModel.FISHEYE,
    "plumb_bob": DistortionModel.BROWN5K,
    "rational_polynomial": DistortionModel.POLYNOMIAL,
}


def _pyav_format_from_ros_encoding(encoding: str) -> tuple[str, int]:
    """Convert a ROS encoding to a PyAV format string and number of color channels."""
    ros_to_pyav = {
        "mono8": ("gray8", 1),
        "bgr8": ("bgr24", 3),
        "rgb8": ("rgb24", 3),
    }
    return ros_to_pyav[encoding]


def _pyav_codec_from_ros_format(fmt: str) -> str:
    """Convert a ROS image format to a PyAV codec string."""
    if "jpeg" in fmt:
        return "mjpeg"
    elif "png" in fmt:
        return "png"
    elif fmt == "h264":
        return "h264"
    elif fmt in ("hevc", "h265"):
        return "hevc"
    else:
        raise ValueError(f"Unknown ROS image message format: '{fmt}'.")


def _image_path(base_path: pathlib.Path, topic: str, frame_idx: int) -> pathlib.Path:
    """Get the path to an image file for a given topic and frame index."""
    topic_clean = topic.lstrip("/")
    if topic_clean == "":
        return base_path / f"{frame_idx:06d}.png"
    return base_path / f"{topic_clean}/{frame_idx:06d}.png"


def _producer(
    reader: highlevel.AnyReader,
    topics: list[str],
    width: int | None,
    height: int | None,
    format: str | None,
    images_base_path: pathlib.Path,
    frame_queue: queue.Queue,
    shutdown_event: threading.Event,
) -> pd.DataFrame:
    """Read messages from rosbag and enqueue decoded frames for writing."""
    if width or height:
        assert width and height, "Both width and height must be specified."

    logger.info(f"Writing images to '{images_base_path}'.")

    timestamps: dict[str, list[int]] = {}
    for topic in topics:
        timestamps[topic] = []
        _image_path(images_base_path, topic, 0).parent.mkdir(parents=True, exist_ok=True)

    with highlevel.AnyReader(
        paths=reader.paths,
        default_typestore=reader.default_typestore,
    ) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        num_messages = sum(c.msgcount for c in connections)
        decoders: dict[str, av.CodecContext] = {}

        for idx, (connection, _, rawdata) in enumerate(reader.messages(connections)):
            if idx % 100 == 0:
                pct = 100 * idx / max(num_messages, 1)
                sys.stdout.write(f"\rExtracting images... {pct:.1f}%")
                sys.stdout.flush()

            topic = connection.topic
            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.msgtype == "sensor_msgs/msg/Image":
                fmt, nch = _pyav_format_from_ros_encoding(msg.encoding)
                shape = (msg.height, msg.width) if nch == 1 else (msg.height, msg.width, nch)
                decoded_frame = av.VideoFrame.from_ndarray(msg.data.reshape(shape), format=fmt)

            elif connection.msgtype == "sensor_msgs/msg/CompressedImage":
                if topic not in decoders:
                    decoders[topic] = av.CodecContext.create(_pyav_codec_from_ros_format(msg.format), "r")
                    logger.info(f"Using codec '{decoders[topic].codec.name}' for topic '{topic}'.")
                try:
                    packet = av.packet.Packet(msg.data.tobytes())
                    decoded_frames = decoders[topic].decode(packet)
                except av.error.InvalidDataError:
                    logger.warning(f"Skipping message {idx} for '{topic}' due to InvalidDataError.")
                    continue

                if len(decoded_frames) == 0:
                    continue
                elif len(decoded_frames) == 1:
                    decoded_frame = decoded_frames[0]
                else:
                    raise ValueError(f"Expected 1 decoded frame, got {len(decoded_frames)}")
            else:
                raise ValueError(f"Unknown message type '{connection.msgtype}'")

            decoded_frame = decoded_frame.reformat(
                width=width or decoded_frame.width,
                height=height or decoded_frame.height,
                format=format or decoded_frame.format,
            )

            frame_idx = len(timestamps[topic])
            timestamps[topic].append(msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)
            frame_queue.put((images_base_path, topic, frame_idx, decoded_frame))

    print("")  # Finish progress line
    shutdown_event.set()
    logger.info("Finished extracting images from rosbag.")

    max_len = max(len(v) for v in timestamps.values())
    for v in timestamps.values():
        v += [-1] * (max_len - len(v))

    timestamp_df = pd.DataFrame(timestamps)
    timestamp_df.to_csv(images_base_path / "raw_timestamps.csv")
    return timestamp_df


def _consumer(thread_id: int, frame_queue: queue.Queue, shutdown_event: threading.Event) -> None:
    """Consume frames from the queue and write them to disk."""
    while True:
        try:
            images_base_path, topic, frame_idx, frame = frame_queue.get(timeout=1)
        except queue.Empty:
            if shutdown_event.is_set():
                return
            continue
        image_path = _image_path(images_base_path, topic, frame_idx)
        frame.to_image().save(str(image_path))


def image_extract_from_rosbag(
    reader: highlevel.AnyReader,
    config: Config,
) -> int:
    """Extract synchronized images from a rosbag.

    Returns the number of synchronized frames.
    """
    num_workers = config.num_workers
    if num_workers == -1:
        num_workers = 2 * (os.cpu_count() or 1)
    num_workers = max(num_workers, 2)

    shutdown_event = threading.Event()
    frame_queue: queue.Queue = queue.Queue(maxsize=num_workers * 2)

    start = time.time()
    logger.info(f"Starting thread pool with {num_workers} workers.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        producer_future = executor.submit(
            _producer, reader, config.image_topics,
            config.output_width, config.output_height, config.output_format,
            config.output_path / "images", frame_queue, shutdown_event,
        )
        consumer_futures = [
            executor.submit(_consumer, i, frame_queue, shutdown_event)
            for i in range(num_workers - 1)
        ]

        all_futures = [producer_future] + consumer_futures
        try:
            for future in concurrent.futures.as_completed(all_futures):
                future.result()
        except Exception as e:
            shutdown_event.set()
            raise e
        timestamps_df = producer_future.result()

    logger.info(f"Image extraction took {time.time() - start:.1f}s.")
    raw_counts = {
        topic: int((timestamps_df[topic] >= 0).sum())
        for topic in timestamps_df.columns
    }
    raw_logs = "\n".join(f"\t- {topic}: {count}" for topic, count in raw_counts.items())
    logger.info("Raw extracted frame count per topic:\n" + raw_logs)

    # Synchronize
    synced_df = _synchronize_images(timestamps_df, config.output_path / "images", config.sync_threshold_ns)
    num_frames = _extract_frame_metadata(synced_df, config)
    synced_counts = {
        topic: len(list(_image_path(config.output_path / "images", topic, 0).parent.glob("*.png")))
        for topic in synced_df.columns
    }
    synced_logs = "\n".join(f"\t- {topic}: {count}" for topic, count in synced_counts.items())
    logger.info("Synced on-disk frame count per topic:\n" + synced_logs)
    logger.info(f"Number of synced frames: {num_frames}")
    return num_frames


def _synchronize_images(
    timestamps_df: pd.DataFrame,
    images_base_path: pathlib.Path,
    sync_threshold_ns: int,
) -> pd.DataFrame:
    """Synchronize multi-camera images by timestamp, renaming files on disk."""
    topics = timestamps_df.columns
    front_idx = {topic: 0 for topic in topics}
    frame_idx = 0
    synced_timestamps: dict[str, list[int]] = {topic: [] for topic in topics}

    while all(front_idx[topic] < timestamps_df.shape[0] for topic in topics):
        front = [timestamps_df[topic][idx] for topic, idx in front_idx.items()]
        if any(v < 0 for v in front):
            break

        argmin = np.argmin(front)
        argmax = np.argmax(front)
        if front[argmax] - front[argmin] < sync_threshold_ns:
            for topic, old_frame_idx in front_idx.items():
                old_path = _image_path(images_base_path, topic, old_frame_idx)
                new_path = _image_path(images_base_path, topic, frame_idx)
                os.rename(old_path, new_path)
                synced_timestamps[topic].append(timestamps_df[topic][old_frame_idx])
            front_idx = {topic: front_idx[topic] + 1 for topic in topics}
            frame_idx += 1
        else:
            path = _image_path(images_base_path, topics[argmin], front_idx[topics[argmin]])
            path.unlink(missing_ok=True)
            front_idx[topics[argmin]] += 1

    # Final outputs are compacted to [0, frame_idx). Remove all tail files
    # regardless of per-topic cursor state to avoid stale leftovers.
    for topic in topics:
        for old_frame_idx in range(frame_idx, timestamps_df.shape[0]):
            _image_path(images_base_path, topic, old_frame_idx).unlink(missing_ok=True)

    synced_df = pd.DataFrame(synced_timestamps)
    synced_df.to_csv(images_base_path / "synced_timestamps.csv")
    return synced_df


def _extract_frame_metadata(synced_timestamps_df: pd.DataFrame, config: Config) -> int:
    """Write per-frame metadata JSONL. Returns number of frames."""
    topics = synced_timestamps_df.columns
    num_frames = synced_timestamps_df.shape[0]

    out_lines = []
    for frame_idx in range(num_frames):
        timestamps = synced_timestamps_df.iloc[frame_idx]
        cams_list = [
            {
                "id": cam_idx,
                "filename": str(_image_path(pathlib.Path("images"), topic, frame_idx)),
                "timestamp": int(timestamps.iloc[cam_idx]),
            }
            for cam_idx, topic in enumerate(topics)
        ]
        out_lines.append({"frame_id": frame_idx, "cams": cams_list})

    with (config.output_path / "frame_metadata.jsonl").open("w") as f:
        for line in out_lines:
            json.dump(line, f)
            f.write("\n")

    return num_frames


# ---------- EDEX metadata from CameraInfo ----------


def _get_distortion_model(distortion_model: str, distortion_params: np.ndarray):
    """Convert ROS distortion model name to EDEX DistortionModel."""
    assert distortion_model in DISTORTION_MODEL_ROS2EDEX, f"Unrecognized distortion model: '{distortion_model}'"
    if np.all(distortion_params == 0):
        logger.info("All distortion parameters are zero. Using pinhole model.")
        return DistortionModel.PINHOLE, np.array([], dtype=np.float32)
    return DISTORTION_MODEL_ROS2EDEX[distortion_model], distortion_params


def _get_camera_intrinsics(camera_msg: Any, config: Config) -> Intrinsics:
    """Extract camera intrinsics from a ROS CameraInfo message."""
    distortion_model, distortion_params = _get_distortion_model(
        camera_msg.distortion_model, camera_msg.d
    )

    width_ratio = config.output_width / camera_msg.width if config.output_width else 1.0
    height_ratio = config.output_height / camera_msg.height if config.output_height else 1.0

    sx = int(width_ratio * camera_msg.width)
    sy = int(height_ratio * camera_msg.height)

    fx = width_ratio * camera_msg.k[0]
    fy = height_ratio * camera_msg.k[4]
    cx = width_ratio * camera_msg.k[2]
    cy = height_ratio * camera_msg.k[5]

    projection = camera_msg.p.reshape(3, 4).copy()
    projection[0, :] *= width_ratio
    projection[1, :] *= height_ratio

    rectification = camera_msg.r.reshape(3, 3).copy()

    return Intrinsics(
        distortion_model=distortion_model,
        distortion_params=distortion_params,
        focal=np.array([fx, fy], dtype=np.float32),
        principal=np.array([cx, cy], dtype=np.float32),
        size=np.array([sx, sy], dtype=np.int32),
        projection=projection,
        rectification=rectification,
    )


def _get_first_message(reader: highlevel.AnyReader, topics: list[str]) -> list[object]:
    """Get the first message of every topic."""
    connections = [c for c in reader.connections if c.topic in topics]
    topic_and_first_msg = {}
    for connection, _, rawdata in reader.messages(connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        topic_and_first_msg[connection.topic] = msg
        if len(topic_and_first_msg) == len(topics):
            break
    return [topic_and_first_msg[topic] for topic in topics]


def edex_build_from_rosbag(
    reader: highlevel.AnyReader,
    config: Config,
    num_frames: int,
) -> None:
    """Create EDEX metadata with camera intrinsics from rosbag CameraInfo messages."""
    camera_info_msgs = _get_first_message(reader, config.camera_info_topics)
    cameras_metadata = []
    for msg in camera_info_msgs:
        camera = Camera(
            intrinsics=_get_camera_intrinsics(msg, config),
            transform=None,
        )
        cameras_metadata.append(camera)

    sequence_paths = [
        _image_path(pathlib.Path("images"), topic, 0)
        for topic in config.image_topics
    ]

    edex_header = EDEXHeader(
        version="0.9",
        frame_start=0,
        frame_end=num_frames,
        cameras=cameras_metadata,
    )
    edex_body = EDEXBody(
        frame_metadata="frame_metadata.jsonl",
        sequence=sequence_paths,
    )
    edex_metadata = EDEXMetadata(header=edex_header, body=edex_body)

    edex_path = config.output_path / "edex"
    logger.info(f"Writing EDEX metadata to '{edex_path}'.")
    edex_metadata.write(edex_path)
