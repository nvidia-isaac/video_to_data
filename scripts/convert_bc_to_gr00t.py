"""Convert a collect_bc_data.py robomimic HDF5 -> a GR00T-N1.7 LeRobot-v2.1 dataset.

The output loads with GR00T's `LeRobotSingleDataset` / `ShardedSingleStepDataset` and is
usable by BOTH the stock GR00T-N1.7 (which consumes the task language) and the DINOv3
specialist (which ignores language) — same dataset, single video camera.

SOURCE (collect_bc_data.py): robomimic-style HDF5
  /data                       attrs: fps(=60), joint_names(JSON, 29 canonical), resolution_hw, ...
  /data/demo_<i>/obs/image    (T, H, W, 3) uint8     facing-the-robot camera
  /data/demo_<i>/obs/joint_pos(T, 29) float32        current joint pos (arm7 + fingers22)
  /data/demo_<i>/actions      (T, 29) float32        delta joint-position command
  /data/demo_<i>  attr num_samples = T

TARGET (GR00T LeRobot v2.1):
  meta/info.json meta/modality.json meta/episodes.jsonl meta/tasks.jsonl meta/stats.json
  data/chunk-000/episode_{i:06d}.parquet
      cols: observation.state[29], action[29], timestamp, frame_index, episode_index, index, task_index
  videos/chunk-000/observation.images.front/episode_{i:06d}.mp4   (H.264, fps from source)

  modality.json:
    state.joint_pos   [0:29]   action.joint_pos [0:29]
    video.front -> observation.images.front
    annotation.human.task_description -> task_index   (constant single task; specialist ignores it)

The custom IIWA14 + Sharpa robot is not a pretrained embodiment -> finetune with
`--embodiment-tag NEW_EMBODIMENT` (see the printed summary).

Run (GR00T venv, which has pandas/pyarrow/av/h5py):
  cd Isaac-GR00T && source .venv/bin/activate
  python /home/cning/simtoolreal_isaaclab/scripts/convert_bc_to_gr00t.py \
      --hdf5 /home/cning/simtoolreal_isaaclab/datasets/hammer_bc_success.hdf5 \
      --out  /home/cning/simtoolreal_isaaclab/datasets/hammer_gr00t_lerobot \
      --task "hammer the screw into the hole"
"""

import argparse
import json
import os

import h5py
import numpy as np

CHUNK_SIZE = 1000  # LeRobot episodes-per-chunk


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True, help="source collect_bc_data.py .hdf5")
    p.add_argument("--out", required=True, help="output LeRobot dataset dir (created)")
    p.add_argument("--task", default="manipulate the tool", help="single task/language string (specialist ignores it)")
    p.add_argument("--video_key", default="front", help="modality.json video alias (dir = observation.images.<key>)")
    p.add_argument("--robot_type", default="iiwa14_sharpa", help="info.json robot_type tag")
    p.add_argument("--fps", type=int, default=0, help="override fps (default: read from HDF5 /data attrs)")
    p.add_argument("--crf", type=int, default=20, help="H.264 quality (lower = better/larger)")
    p.add_argument("--preset", default="veryfast", help="x264 preset (ultrafast..veryslow); faster = quicker encode, larger files")
    p.add_argument("--max_demos", type=int, default=-1, help="convert only the first N demos (-1 = all)")
    return p.parse_args()


def encode_h264(path, frames, fps, crf, preset="veryfast"):
    """Encode (T,H,W,3) uint8 RGB -> H.264 mp4 via PyAV (bundled libx264)."""
    import av

    T, H, W, _ = frames.shape
    if H % 2 or W % 2:  # libx264 yuv420p needs even dims
        H, W = H - (H % 2), W - (W % 2)
        frames = frames[:, :H, :W]
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": str(crf), "preset": preset}
    for fr in frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")
        for pkt in stream.encode(vf):
            container.mux(pkt)
    for pkt in stream.encode():  # flush
        container.mux(pkt)
    container.close()
    return H, W


def feature_stats(arr):
    """arr (N, D) float -> dict of LeRobot per-dim stats (lists of length D)."""
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": arr.mean(0).tolist(),
        "std": (arr.std(0) + 1e-8).tolist(),
        "min": arr.min(0).tolist(),
        "max": arr.max(0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def main():
    args = parse_args()
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    f = h5py.File(args.hdf5, "r")
    data = f["data"]
    fps = args.fps or int(data.attrs.get("fps", 60))
    joint_names = json.loads(data.attrs["joint_names"]) if "joint_names" in data.attrs else [f"joint_{i}" for i in range(29)]
    state_dim = len(joint_names)
    demo_keys = sorted([k for k in data.keys() if k.startswith("demo_")], key=lambda s: int(s.split("_")[1]))
    if args.max_demos > 0:
        demo_keys = demo_keys[: args.max_demos]
    assert demo_keys, f"no demos found in {args.hdf5}/data (collection may still be running / no successes yet)"

    vkey = f"observation.images.{args.video_key}"
    out = args.out
    os.makedirs(f"{out}/meta", exist_ok=True)
    os.makedirs(f"{out}/data/chunk-000", exist_ok=True)
    os.makedirs(f"{out}/videos/chunk-000/{vkey}", exist_ok=True)
    has_wrist = "obs/image_wrist" in data[demo_keys[0]]   # 2nd (palm-facing) view, if collected with --wrist
    wkey = "observation.images.wrist"
    if has_wrist:
        os.makedirs(f"{out}/videos/chunk-000/{wkey}", exist_ok=True)

    all_state, all_action, all_ts = [], [], []
    episodes_meta = []
    global_index = 0
    H = W = Hw = Ww = None

    print(f"[convert] {len(demo_keys)} demos | fps={fps} | state/action dim={state_dim} -> {out}", flush=True)
    for ep, dk in enumerate(demo_keys):
        g = data[dk]
        img = g["obs/image"][:]                       # (T,H,W,3) uint8
        state = g["obs/joint_pos"][:].astype(np.float32)   # (T,29)
        action = g["actions"][:].astype(np.float32)        # (T,29)
        T = img.shape[0]
        teleport = g["teleport"][:].astype(np.int64) if "teleport" in g else np.zeros(T, np.int64)  # (T,) 1=teleport this step
        assert state.shape == (T, state_dim) and action.shape == (T, state_dim), \
            f"{dk}: shape mismatch img{img.shape} state{state.shape} action{action.shape}"

        # --- video ---
        vid_path = f"{out}/videos/chunk-000/{vkey}/episode_{ep:06d}.mp4"
        H, W = encode_h264(vid_path, img, fps, args.crf, args.preset)
        if has_wrist:
            wimg = g["obs/image_wrist"][:]
            Hw, Ww = encode_h264(f"{out}/videos/chunk-000/{wkey}/episode_{ep:06d}.mp4", wimg, fps, args.crf, args.preset)

        # --- parquet (one row per frame) ---
        ts = (np.arange(T, dtype=np.float32) / fps)
        table = pa.table({
            "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
            "action": pa.array(list(action), type=pa.list_(pa.float32())),
            "timestamp": pa.array(ts, type=pa.float32()),
            "frame_index": pa.array(np.arange(T), type=pa.int64()),
            "episode_index": pa.array(np.full(T, ep), type=pa.int64()),
            "index": pa.array(np.arange(global_index, global_index + T), type=pa.int64()),
            "task_index": pa.array(np.zeros(T), type=pa.int64()),
            "teleport": pa.array(teleport, type=pa.int64()),   # per-step teleport flag for chunk-loss masking
        })
        pq.write_table(table, f"{out}/data/chunk-000/episode_{ep:06d}.parquet")

        all_state.append(state); all_action.append(action); all_ts.append(ts)
        episodes_meta.append({"episode_index": ep, "tasks": [args.task], "length": int(T)})
        global_index += T
        if (ep + 1) % 25 == 0 or ep + 1 == len(demo_keys):
            print(f"[convert]  episode {ep + 1}/{len(demo_keys)}  (T={T}, {global_index} frames total)", flush=True)
    f.close()

    total_frames = global_index
    n_ep = len(demo_keys)

    # --- meta/modality.json (GR00T-specific) ---
    modality = {
        "state": {"joint_pos": {"start": 0, "end": state_dim}},
        "action": {"joint_pos": {"start": 0, "end": state_dim}},
        "video": {args.video_key: {"original_key": vkey},
                  **({"wrist": {"original_key": wkey}} if has_wrist else {})},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    json.dump(modality, open(f"{out}/meta/modality.json", "w"), indent=4)

    # --- meta/info.json ---
    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": n_ep,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": n_ep,
        "total_chunks": (n_ep + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim], "names": joint_names},
            "action": {"dtype": "float32", "shape": [state_dim], "names": [f"delta_{n}" for n in joint_names]},
            vkey: {
                "dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channels"],
                "info": {
                    "video.height": H, "video.width": W, "video.codec": "h264",
                    "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                    "video.fps": fps, "video.channels": 3, "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    if has_wrist:
        info["features"][wkey] = {
            "dtype": "video", "shape": [Hw, Ww, 3], "names": ["height", "width", "channels"],
            "info": {"video.height": Hw, "video.width": Ww, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "video.fps": fps, "video.channels": 3, "has_audio": False},
        }
        info["total_videos"] = n_ep * 2
    json.dump(info, open(f"{out}/meta/info.json", "w"), indent=4)

    # --- meta/episodes.jsonl + tasks.jsonl ---
    with open(f"{out}/meta/episodes.jsonl", "w") as fp:
        for e in episodes_meta:
            fp.write(json.dumps(e) + "\n")
    with open(f"{out}/meta/tasks.jsonl", "w") as fp:
        fp.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

    # --- meta/stats.json (required by the loader) ---
    S = np.concatenate(all_state, 0); A = np.concatenate(all_action, 0); TS = np.concatenate(all_ts, 0)[:, None]
    stats = {
        "observation.state": feature_stats(S),
        "action": feature_stats(A),
        "timestamp": feature_stats(TS),
    }
    json.dump(stats, open(f"{out}/meta/stats.json", "w"), indent=4)

    print(f"\n[convert] DONE -> {out}")
    print(f"  episodes={n_ep}  frames={total_frames}  video={W}x{H}@{fps}fps  state/action_dim={state_dim}")
    print("  finetune (custom robot -> NEW_EMBODIMENT):")
    print(f"    cd Isaac-GR00T && uv run bash examples/finetune.sh \\")
    print(f"      --base-model-path nvidia/GR00T-N1.7-3B --dataset-path {out} \\")
    print(f"      --embodiment-tag NEW_EMBODIMENT --output-dir /tmp/finetune_out")


if __name__ == "__main__":
    main()
