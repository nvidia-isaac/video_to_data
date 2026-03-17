"""Export a pose folder (and optionally camera intrinsics) to a single HDF5 file."""
import argparse
import os
from typing import Optional

import h5py
import numpy as np

from v2d.common.datatypes import CameraIntrinsics, Transform3d


def poses_to_hdf5(
    poses_dir: str,
    output_path: str,
    intrinsics_dir: Optional[str] = None,
) -> None:
    """Read per-frame pose JSON files and write them to an HDF5 file.

    Root datasets:
      - frame_id   : (N,)    int64   — numeric frame index
      - rotation   : (N, 4)  float64 — quaternion (wxyz)
      - translation: (N, 3)  float64 — translation in metres
      - scale      : (N, 3)  float64 — scale factors

    If intrinsics_dir is provided, an 'intrinsics' group is added:
      - intrinsics/fx     : (N,)  float64
      - intrinsics/fy     : (N,)  float64
      - intrinsics/cx     : (N,)  float64
      - intrinsics/cy     : (N,)  float64
      - intrinsics/width  : (N,)  int64
      - intrinsics/height : (N,)  int64

    Intrinsics are matched to poses by frame_id. If only a single JSON file
    exists in intrinsics_dir it is broadcast to all frames.

    Args:
        poses_dir:      Directory containing per-frame JSON pose files.
        output_path:    Destination HDF5 file path.
        intrinsics_dir: Optional directory containing per-frame (or single)
                        JSON intrinsics files.
    """
    pose_files = sorted(p for p in os.listdir(poses_dir) if p.endswith(".json"))
    if not pose_files:
        raise ValueError(f"No JSON files found in {poses_dir}")

    frame_ids, rotations, translations, scales = [], [], [], []
    for filename in pose_files:
        frame_id = int(os.path.splitext(filename)[0])
        pose = Transform3d.load(os.path.join(poses_dir, filename))
        frame_ids.append(frame_id)
        rotations.append(pose.rotation)
        translations.append(pose.translation)
        scales.append(pose.scale)

    intrinsics_per_frame: Optional[list[CameraIntrinsics]] = None
    if intrinsics_dir is not None:
        intr_files = sorted(p for p in os.listdir(intrinsics_dir) if p.endswith(".json"))
        if not intr_files:
            raise ValueError(f"No JSON files found in {intrinsics_dir}")
        if len(intr_files) == 1:
            shared = CameraIntrinsics.load(os.path.join(intrinsics_dir, intr_files[0]))
            intrinsics_per_frame = [shared] * len(frame_ids)
        else:
            intr_by_frame = {
                int(os.path.splitext(p)[0]): CameraIntrinsics.load(os.path.join(intrinsics_dir, p))
                for p in intr_files
            }
            intrinsics_per_frame = [intr_by_frame[fid] for fid in frame_ids]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("frame_id", data=np.array(frame_ids, dtype=np.int64))
        f.create_dataset("rotation", data=np.array(rotations, dtype=np.float64))
        f.create_dataset("translation", data=np.array(translations, dtype=np.float64))
        f.create_dataset("scale", data=np.array(scales, dtype=np.float64))

        if intrinsics_per_frame is not None:
            g = f.create_group("intrinsics")
            g.create_dataset("fx", data=np.array([k.fx for k in intrinsics_per_frame], dtype=np.float64))
            g.create_dataset("fy", data=np.array([k.fy for k in intrinsics_per_frame], dtype=np.float64))
            g.create_dataset("cx", data=np.array([k.cx for k in intrinsics_per_frame], dtype=np.float64))
            g.create_dataset("cy", data=np.array([k.cy for k in intrinsics_per_frame], dtype=np.float64))
            g.create_dataset("width", data=np.array([k.width for k in intrinsics_per_frame], dtype=np.int64))
            g.create_dataset("height", data=np.array([k.height for k in intrinsics_per_frame], dtype=np.int64))

    print(f"Wrote {len(frame_ids)} frames to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses_dir", help="Folder of per-frame pose JSON files")
    parser.add_argument("output_path", help="Output HDF5 file path")
    parser.add_argument("--intrinsics_dir", default=None, help="Folder of per-frame (or single) intrinsics JSON files")
    args = parser.parse_args()
    poses_to_hdf5(args.poses_dir, args.output_path, intrinsics_dir=args.intrinsics_dir)
