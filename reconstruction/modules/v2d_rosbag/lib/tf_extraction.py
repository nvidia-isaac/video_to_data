"""Extract static and temporal transforms from a ROS bag's /tf and /tf_static topics."""

import collections
import pathlib

import pandas as pd
from pytransform3d import trajectories, transform_manager
from rosbags import highlevel

from v2d.rosbag.lib.config import typestore_from_ros_distribution


def _extract_tf_dataframe(rosbag_path: pathlib.Path, ros_distribution: str) -> pd.DataFrame:
    """Read TF messages from a bag and return a DataFrame of all transforms."""
    data = collections.defaultdict(list)
    with highlevel.AnyReader(
        paths=[rosbag_path],
        default_typestore=typestore_from_ros_distribution(ros_distribution),
    ) as reader:
        connections = [x for x in reader.connections if x.topic in ["/tf_static", "/tf"]]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            for tf in msg.transforms:
                data["topic"].append(connection.topic)
                data["sec"].append(tf.header.stamp.sec)
                data["nanosec"].append(tf.header.stamp.nanosec)
                data["parent"].append(tf.header.frame_id)
                data["child"].append(tf.child_frame_id)
                data["x"].append(tf.transform.translation.x)
                data["y"].append(tf.transform.translation.y)
                data["z"].append(tf.transform.translation.z)
                data["qw"].append(tf.transform.rotation.w)
                data["qx"].append(tf.transform.rotation.x)
                data["qy"].append(tf.transform.rotation.y)
                data["qz"].append(tf.transform.rotation.z)
    df = pd.DataFrame(data)
    df["time_s"] = df["sec"] + df["nanosec"] / 10**9
    return df


def tf_static_manager_from_rosbag(
    rosbag_path: pathlib.Path,
    ros_distribution: str,
) -> transform_manager.TransformManager:
    """Extract static transforms from a bag into a TransformManager."""
    df = _extract_tf_dataframe(rosbag_path, ros_distribution)
    df_static = df[df["topic"] == "/tf_static"].reset_index()
    transforms = trajectories.transforms_from_pqs(
        df_static[["x", "y", "z", "qw", "qx", "qy", "qz"]]
    )
    tf_manager = transform_manager.TransformManager()
    for index, row in df_static.iterrows():
        tf_manager.add_transform(row["child"], row["parent"], transforms[index])
    return tf_manager
