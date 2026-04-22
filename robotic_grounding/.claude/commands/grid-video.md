---
name: grid-video
description: Combine multiple video files into a grid layout where each loops, total length = max video duration.
user-invocable: true
allowed-tools:
  - Bash
---

# /grid-video — Combine Videos into a Looping Grid

Arrange N videos in a grid (each looping), with total duration equal to the longest individual video. Empty grid slots (when N doesn't fill the grid) are filled with black.

Inputs can be **W&B run names** (resolved from `experiments/eval_recordings/`) or **direct `.mp4` file paths**.

Arguments: `$ARGUMENTS`

**Usage:**
```
/grid-video <run_or_video1> <run_or_video2> ... [--output <output.mp4>] [--cell-size <WxH>]
```

Examples:
- `/grid-video 2026-04-13_10-14-06_exp54_v3_stage2_20k_mixer_grab_01 2026-04-15_12-02-40_exp67_stage2_espressomachine_grab_01`
- `/grid-video 2026-04-13_10-14-06_exp54_v3_stage2_20k_mixer_grab_01 a.mp4 b.mp4 --output my_grid.mp4`
- `/grid-video a.mp4 b.mp4 c.mp4 d.mp4 --cell-size 640x360`

---

## Step 1 — Parse arguments

From `$ARGUMENTS`:
- **Run names**: tokens matching `\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\S+` — resolve to video files (see Step 1.5)
- **Video files**: tokens that are existing file paths ending in `.mp4`, `.mov`, `.avi`, `.mkv`
- `--output <path>`: output file, default `grid_output.mp4`
- `--cell-size <WxH>`: override cell dimensions (e.g. `640x360`), default: use first video's native size

## Step 1.5 — Resolve run names to video files

For each run name token, find its video in `experiments/eval_recordings/<run_name>/`:

```bash
ls experiments/eval_recordings/<run_name>/*.mp4 2>/dev/null | sort -t_ -k1 -V | tail -1
```

This picks the highest-numbered model file (e.g. `<run_name>_model50000.pt` > `_model10000`). If multiple `.mp4` files exist, pick the one with the largest model number by sorting. If none found, abort with an error.

Print resolved paths:
```
[grid-video] Resolved: <run_name> → experiments/eval_recordings/<run_name>/<file>.mp4
```

After resolving all run names, combine with any directly-specified file paths into a final ordered video list.

Print parsed summary:
```
[grid-video] N videos, output=<path>, cell_size=<WxH or "auto">
```

## Step 2 — Get video durations and dimensions

`ffprobe`/`ffmpeg` are not available on the host — use cv2 via the Isaac Sim Python inside Docker. Video paths inside the container are `/workspace/video_to_data/robotic_grounding/<relative_path>`.

```bash
docker exec robotic-grounding-v2d-gpu1 /workspace/isaaclab/_isaac_sim/python.sh -c "
import cv2
videos = ['/workspace/video_to_data/robotic_grounding/<path1>', ...]
for path in videos:
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = frames / fps if fps > 0 else 0
    cap.release()
    print(f'{path}: {w}x{h} fps={fps} dur={dur:.2f}s')
"
```

Collect width, height, and duration for each video.

Report:
```
[grid-video] Durations: v1=Xs v2=Xs ... max=Xs
[grid-video] First video dimensions: WxH
```

## Step 3 — Compute grid layout

```
N      = number of input videos
cols   = ceil(sqrt(N))
rows   = ceil(N / cols)
empty  = cols * rows - N
```

Cell size: use `--cell-size` if given, otherwise use the first video's width x height.

Print:
```
[grid-video] Grid: <cols>x<rows> (<N> videos + <empty> empty slots), cell=<W>x<H>
```

## Step 4 — Build and run via Python script

`ffmpeg` is not on the host PATH. Use the bundled binary from `imageio_ffmpeg` inside Docker via Isaac Sim Python.

Write the script to `/tmp/build_grid_video.py` on the **host** (using the Write tool), then copy it into the container and run it. Use the container-side paths (`/workspace/video_to_data/robotic_grounding/...`) for all video and output paths.

**IMPORTANT**: Do not use f-strings with nested quotes in the script — use string concatenation to avoid shell quoting issues.

```python
import subprocess, math, os, sys
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

videos = ["/workspace/video_to_data/robotic_grounding/<path1>", "/workspace/video_to_data/robotic_grounding/<path2>", ...]
output = "/workspace/video_to_data/robotic_grounding/<output_path>"
W, H = <cell_width>, <cell_height>
max_duration = <max_duration_float>

N = len(videos)
cols = math.ceil(math.sqrt(N))
rows = math.ceil(N / cols)
total = cols * rows
empty = total - N

# Build input args: loop each real video; fill empty slots with black
inputs = []
for v in videos:
    inputs += ["-stream_loop", "-1", "-i", v]
for _ in range(empty):
    inputs += ["-f", "lavfi", "-i", f"color=black:size={W}x{H}:rate=30"]

# Scale each input to uniform cell size (letterbox to preserve aspect ratio)
scale_parts = []
for i in range(total):
    scale_parts.append(
        f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2[v{i}]"
    )

# xstack layout: pixel positions for each cell
layout_parts = []
for i in range(total):
    col = i % cols
    row = i // cols
    layout_parts.append(f"{col * W}_{row * H}")

xstack = "".join(f"[v{i}]" for i in range(total))
xstack += f"xstack=inputs={total}:layout={'|'.join(layout_parts)}[out]"

filter_complex = ";".join(scale_parts + [xstack])

cmd = ["ffmpeg", "-y"] + inputs + [
    "-filter_complex", filter_complex,
    "-map", "[out]",
    "-t", str(max_duration),
    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
    output,
]

print("[grid-video] Running ffmpeg ...")
result = subprocess.run(cmd)
if result.returncode != 0:
    sys.exit(1)
size = os.path.getsize(output)
print(f"[grid-video] Done: {output} ({size/1024/1024:.1f} MB)")
```

Copy and run inside Docker:
```bash
docker cp /tmp/build_grid_video.py robotic-grounding-v2d-gpu1:/tmp/build_grid_video.py
docker exec robotic-grounding-v2d-gpu1 /workspace/isaaclab/_isaac_sim/python.sh /tmp/build_grid_video.py
```

## Step 5 — Verify and report

```bash
ls -lh "<host_output_path>"
```

Print final result:
```
[grid-video] ✓ Grid video saved: <output>
  Grid: <cols>x<rows>  Videos: <N>  Duration: <Xs>
  File: <size>
```
